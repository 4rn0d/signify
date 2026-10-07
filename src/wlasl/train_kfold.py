"""Entraine un GRU sur les sequences WLASL, en validation croisee par signeur.

Pourquoi k-fold plutot qu'un decoupage train/val/test : avec 9 a 16 clips par
glose, un ensemble de validation ferait ~3 clips par classe. Choisir l'epoch
d'arret sur un signal aussi bruite revient a tirer au sort, et cela amputerait
encore l'entrainement. En k-fold, chaque clip est teste exactement une fois et
on obtient un ecart-type — indispensable ici, puisque sur l'alphabet des ecarts
de 0.2 point se sont reveles etre du bruit (+-0.58).

Budget d'epochs FIXE, identique pour tous les folds. Utiliser le fold de test
pour arreter l'entrainement serait une fuite deguisee en rigueur.

Les folds sont construits par SIGNEUR : les decoupages officiels de WLASL
partagent 93% de leurs signeurs entre train et test, ce qui mesure en partie
la reconnaissance de personnes plutot que de signes.
"""

import argparse
import collections
import glob
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "wlasl", "features")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "wlasl", "download_manifest.json")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "models", "wlasl_kfold.json")

# Mediane 69 frames, p75 87. 80 couvre la majorite sans trop de remplissage.
MAX_LEN = 80

HAND_POINTS = 21
HAND_BLOCK = HAND_POINTS * 3          # 63
LEFT_HAND = slice(0, HAND_BLOCK)
RIGHT_HAND = slice(HAND_BLOCK, HAND_BLOCK * 2)
LEFT_FLAG = HAND_BLOCK * 2
RIGHT_FLAG = HAND_BLOCK * 2 + 1
POSE_START = HAND_BLOCK * 2 + 2

# POSE_KEYPOINTS = [nez, epauleG, epauleD, coudeG, coudeD, poignetG, poignetD,
#                   hancheG, hancheD] -> paires gauche/droite a echanger.
POSE_MIRROR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8)]


def load_sequences(variants="all"):
    """variants : 'all', 'dominant' (variante majoritaire seulement) ou
    'split' (chaque variante devient une classe a part).

    WLASL etiquette plusieurs gestes differents sous une meme glose : 19 des
    50 gloses ont plusieurs variation_id, et 15 d'entre elles n'ont aucune
    variante majoritaire a 75%. Le modele apprend donc des exemples qui se
    contredisent. 'dominant' supprime la contradiction en perdant 14% des
    clips ; 'split' la supprime sans rien perdre, au prix de classes plus
    nombreuses et plus petites.
    """
    with open(MANIFEST_PATH, encoding="utf-8") as handle:
        variation = {
            r["video_id"]: r.get("variation_id", 0)
            for r in json.load(handle)
        }

    records = []
    for path in sorted(glob.glob(os.path.join(FEATURES_DIR, "*.npz"))):
        data = np.load(path, allow_pickle=True)
        video_id = str(data["video_id"])
        records.append(
            {
                "features": data["features"].astype(np.float32),
                "gloss": str(data["gloss"]),
                "signer_id": int(data["signer_id"]),
                "variation_id": int(variation.get(video_id, 0)),
            }
        )

    if variants == "all":
        return records

    if variants == "split":
        for record in records:
            record["gloss"] = f"{record['gloss']}_v{record['variation_id']}"
        return records

    # dominant : on garde, par glose, la variante la plus representee —
    # pas forcement variation_id 0 (candy est a 7 contre 11 dans l'autre sens).
    by_gloss = collections.defaultdict(collections.Counter)
    for record in records:
        by_gloss[record["gloss"]][record["variation_id"]] += 1

    keep = {g: counts.most_common(1)[0][0] for g, counts in by_gloss.items()}
    return [r for r in records if r["variation_id"] == keep[r["gloss"]]]


def pad_sequence(sequence, max_len=MAX_LEN):
    """Tronque au centre si trop long, complete par des zeros sinon.

    Tronquer au centre plutot qu'a la fin : un clip contient souvent une
    amorce et un retour au repos, le signe est au milieu.
    """
    length = len(sequence)

    if length >= max_len:
        start = (length - max_len) // 2
        return sequence[start:start + max_len]

    padding = np.zeros((max_len - length, sequence.shape[1]), dtype=np.float32)
    return np.concatenate([sequence, padding], axis=0)


def mirror(batch):
    """Miroir horizontal d'un lot de sequences.

    Negater x ne suffit pas : refleter une personne echange aussi sa main
    gauche et sa main droite, ainsi que les points de pose lateraux. Sans
    cet echange, le modele verrait une main droite rangee dans le canal
    gauche — exactement le genre d'incoherence qui avait empoisonne la
    fusion sur l'alphabet.
    """
    out = batch.copy()

    # x est l'indice 0 de chaque triplet (x, y, z).
    for block in (LEFT_HAND, RIGHT_HAND):
        out[..., block.start:block.stop:3] *= -1.0
    out[..., POSE_START::3] *= -1.0

    # Echange des deux mains, puis des drapeaux de presence.
    left = out[..., LEFT_HAND].copy()
    out[..., LEFT_HAND] = out[..., RIGHT_HAND]
    out[..., RIGHT_HAND] = left

    flags = out[..., LEFT_FLAG].copy()
    out[..., LEFT_FLAG] = out[..., RIGHT_FLAG]
    out[..., RIGHT_FLAG] = flags

    # Echange des points de pose lateraux.
    for a, b in POSE_MIRROR_PAIRS:
        ia, ib = POSE_START + a * 3, POSE_START + b * 3
        tmp = out[..., ia:ia + 3].copy()
        out[..., ia:ia + 3] = out[..., ib:ib + 3]
        out[..., ib:ib + 3] = tmp

    return out


def time_warp(sequence, rng, low=0.8, high=1.25):
    """Reechantillonne la sequence a une vitesse differente."""
    factor = rng.uniform(low, high)
    length = len(sequence)
    new_len = max(8, min(MAX_LEN, int(length * factor)))

    source = np.linspace(0, length - 1, new_len)
    warped = np.stack(
        [np.interp(source, np.arange(length), sequence[:, c])
         for c in range(sequence.shape[1])],
        axis=1,
    ).astype(np.float32)

    return pad_sequence(warped)


def augment_batch(batch, rng):
    out = np.empty_like(batch)

    for i, sequence in enumerate(batch):
        item = sequence

        if rng.random() < 0.5:
            item = time_warp(item, rng)

        # Abandon de frames : imite les echecs de detection, frequents ici
        # (la main n'est vue que dans ~60% des frames).
        if rng.random() < 0.3:
            item = item.copy()
            drop = rng.random(len(item)) < 0.1
            item[drop] = 0.0

        # Bruit leger sur les coordonnees.
        if rng.random() < 0.5:
            item = item + rng.normal(0, 0.01, item.shape).astype(np.float32)

        out[i] = item

    flip = rng.random(len(out)) < 0.5
    if flip.any():
        out[flip] = mirror(out[flip])

    return out


def add_velocity(batch):
    """Concatene les deltas image-a-image aux positions.

    Un signe se definit par le MOUVEMENT : certaines paires ne different que
    par la direction du deplacement, a configuration de main identique. Le
    GRU peut deduire la vitesse des positions successives, mais la lui donner
    explicitement lui evite de l'apprendre.

    Les frames de remplissage restent nulles des deux cotes pour que le
    masquage fonctionne encore, et la vitesse est annulee a la frontiere
    entre vraie frame et remplissage : sinon la transition produirait un
    delta enorme et purement artificiel.
    """
    real = np.abs(batch).sum(axis=2) > 0

    velocity = np.zeros_like(batch)
    velocity[:, 1:] = batch[:, 1:] - batch[:, :-1]

    valid = real[:, 1:] & real[:, :-1]
    velocity[:, 1:][~valid] = 0.0
    velocity[:, 0] = 0.0

    return np.concatenate([batch, velocity], axis=2)


def build_model(num_classes, units=96, feature_dim=155, cell="gru"):
    """cell : 'gru', 'lstm' ou 'stack' (une couche de chaque).

    GRU et LSTM resolvent le meme probleme (gradients qui s'evanouissent dans
    un RNN) et ne different que par la presence d'un etat de cellule separe.
    Les empiler n'ajoute donc pas d'information : c'est teste ici pour le
    verifier, pas parce qu'on l'attend.
    """
    if cell == "lstm":
        first = layers.LSTM(units, return_sequences=True, dropout=0.3)
        second = layers.LSTM(units, dropout=0.3)
    elif cell == "stack":
        first = layers.GRU(units, return_sequences=True, dropout=0.3)
        second = layers.LSTM(units, dropout=0.3)
    else:
        first = layers.GRU(units, return_sequences=True, dropout=0.3)
        second = layers.GRU(units, dropout=0.3)

    model = models.Sequential([
        layers.Input(shape=(MAX_LEN, feature_dim)),
        # Les frames de remplissage sont entierement nulles ; une vraie frame
        # ne l'est jamais (la pose est detectee dans 97% des frames au moins).
        layers.Masking(mask_value=0.0),
        layers.Bidirectional(first),
        layers.Bidirectional(second),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def signer_folds(records, k, seed=0):
    """Repartit les SIGNEURS en k groupes de taille (en clips) comparable."""
    counts = collections.Counter(r["signer_id"] for r in records)

    # Glouton : le signeur le plus gros va au groupe le plus leger.
    folds = [[] for _ in range(k)]
    loads = [0] * k

    for signer, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        target = loads.index(min(loads))
        folds[target].append(signer)
        loads[target] += count

    return [set(f) for f in folds], loads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--units", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--velocity", action="store_true",
                        help="ajoute les deltas image-a-image (155 -> 310)")
    parser.add_argument("--tag", default="")
    parser.add_argument("--cell", default="gru", choices=("gru", "lstm", "stack"))
    parser.add_argument("--variants", default="all",
                        choices=("all", "dominant", "split"))
    args = parser.parse_args()

    records = load_sequences(args.variants)
    glosses = sorted({r["gloss"] for r in records})
    index = {g: i for i, g in enumerate(glosses)}

    features = np.stack([pad_sequence(r["features"]) for r in records])
    labels = np.array([index[r["gloss"]] for r in records], dtype=np.int32)
    signers = np.array([r["signer_id"] for r in records])

    print(f"{len(records)} sequences, {len(glosses)} gloses, "
          f"{len(set(signers))} signeurs")
    feature_dim = features.shape[2] * (2 if args.velocity else 1)
    print(f"entree : ({features.shape[1]}, {feature_dim}), "
          f"{args.units} unites {args.cell}, budget {args.epochs} epochs\n")

    folds, loads = signer_folds(records, args.folds, args.seed)
    print(f"taille des folds (clips) : {loads}\n")

    rng = np.random.default_rng(args.seed)
    scores = []
    histories = []

    for fold_index, held in enumerate(folds, start=1):
        test_mask = np.isin(signers, list(held))
        train_x, train_y = features[~test_mask], labels[~test_mask]
        test_x, test_y = features[test_mask], labels[test_mask]

        tf.keras.utils.set_random_seed(args.seed + fold_index)
        model = build_model(len(glosses), args.units, feature_dim, args.cell)

        eval_x = add_velocity(test_x) if args.velocity else test_x

        best = 0.0
        curve = []

        for epoch in range(args.epochs):
            order = rng.permutation(len(train_x))
            batch_x = augment_batch(train_x[order], rng)
            if args.velocity:
                # Apres augmentation : le time-warp change les vitesses.
                batch_x = add_velocity(batch_x)
            model.fit(
                batch_x, train_y[order],
                batch_size=args.batch, epochs=1, verbose=0,
            )

            # Mesure a chaque epoch pour TRACER la courbe uniquement.
            # L'arret reste au budget fixe : s'arreter sur ce signal
            # reviendrait a selectionner sur le test.
            _, accuracy = model.evaluate(eval_x, test_y, verbose=0)
            curve.append(round(float(accuracy), 4))
            best = max(best, accuracy)

        final = curve[-1]
        scores.append(final)
        histories.append(curve)

        print(f"  fold {fold_index}/{args.folds}  "
              f"train {len(train_x):>4}  test {len(test_x):>4}  "
              f"final {final:.4f}   (meilleur vu {best:.4f})", flush=True)

    scores = np.array(scores)
    # Moyenne de fin de plateau : lire une seule epoch fait dependre le
    # resultat de l'epoch sur laquelle on tombe (ecart de 2 points mesure).
    tail = np.array([np.mean(c[-30:]) for c in histories])
    print(f"\n{'=' * 60}")
    print(f"  accuracy (budget fixe, {args.epochs} epochs)")
    print(f"    plateau (30 dernieres) {tail.mean():.4f}  "
          f"ecart-type {tail.std():.4f}   <-- metrique retenue")
    print(f"    moyenne {scores.mean():.4f}  ecart-type {scores.std():.4f}")
    print(f"    folds   {', '.join(f'{s:.3f}' for s in scores)}")
    print(f"    hasard  {1 / len(glosses):.4f}")

    out_path = (RESULTS_PATH.replace(".json", args.tag + ".json")
                if args.tag else RESULTS_PATH)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "folds": args.folds,
                "epochs": args.epochs,
                "units": args.units,
                "glosses": len(glosses),
                "sequences": len(records),
                "cell": args.cell,
                "variants": args.variants,
                "velocity": bool(args.velocity),
                "units": args.units,
                "plateau_mean": float(tail.mean()),
                "plateau_std": float(tail.std()),
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "per_fold": [float(s) for s in scores],
                "curves": histories,
            },
            handle, ensure_ascii=False, indent=1,
        )
    print(f"\n  resultats : {out_path}")


if __name__ == "__main__":
    main()
