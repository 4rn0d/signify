"""Entraine le modele final sur TOUTES les sequences et le sauvegarde.

train_kfold.py mesure la precision attendue mais ne garde aucun poids : son
role est d'estimer, pas de produire un modele. L'estimation faite
(0.6254 +- 0.0450 en decoupage par signeur), on reentraine sur l'ensemble
des donnees — garder un fold de cote ne servirait qu'a repeter une mesure
deja obtenue, en se privant de 20% des exemples.

La configuration retenue est celle mesuree a 0.6675 en validation croisee
par signeur, soit 33x le hasard :

  --features-dir data/wlasl/features_rtm07   RTMPose, seuil de main 0.7
  --variants dominant   une seule variante de signe par glose (+5.6)
  --rich                23 scalaires derives de x,y (+5.9)
  --confidence          la confiance de chaque main, en continu (+1.6)
  --label-smoothing 0.1
  --seeds 3             ensemble de 3 modeles
  --pretrain-vocab 100  pre-entrainement sur un vocabulaire elargi

Pas de --no-z : RTMPose ne rend que x,y, il n y a pas de profondeur a
retirer. Le seuil de 0.7 est deja applique a l extraction, donc rien a
filtrer ici.

La TTA (moyenne avec l'image miroir) est appliquee a l'inference, dans la
demo : elle ne change rien a l'entrainement.

Attention : le score inscrit dans la fiche vient du k-fold, PAS de ce
modele. Un modele evalue sur ses propres donnees d'entrainement afficherait
un chiffre sans signification (0.815 contre 0.37 reels, mesure plus tot).
"""

import argparse
import collections
import datetime
import json
import os

import numpy as np
import tensorflow as tf

from train_kfold import (
    MAX_LEN,
    Layout,
    add_handshape,
    add_rich_features,
    augment_batch,
    build_model,
    drop_z_indices,
    load_sequences,
    pad_sequence,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
META_PATH = os.path.join(MODELS_DIR, "wlasl_gru.meta.json")
# La configuration exacte de ce script — pile complete ET opposition du pouce
# — n a jamais ete mesuree en k-fold. On cite donc la mesure la PLUS PROCHE,
# la pile complete sans l opposition, et on dit dans la fiche ce qui manque.
# Citer wlasl_kfold_shape8 (0.5998) serait pire : c est l opposition SANS la
# pile, donc une sous-estimation d environ 8 points.
KFOLD_PATH = os.path.join(MODELS_DIR, "wlasl_kfold_rtm07_everything.json")
KFOLD_CAVEAT = (
    "mesure sur la pile complete SANS l opposition du pouce. Celle-ci valait "
    "+0.0106 (5/5 folds, t=+4.06) mesuree separement sans la pile, donc "
    "attendre environ 0.68 — la combinaison des deux n a pas ete mesuree."
)


def model_path(seed_index, extractor="rtmpose"):
    """Un nom par extracteur.

    Les deux configurations ecrivaient au MEME chemin, et comme *.keras est
    ignore par git, entrainer l une ecrasait definitivement les poids de
    l autre. C est arrive une fois.
    """
    suffix = "" if extractor == "mediapipe" else f"_{extractor}"
    return os.path.join(MODELS_DIR, f"wlasl_gru{suffix}_s{seed_index}.keras")


def meta_path(extractor="rtmpose"):
    suffix = "" if extractor == "mediapipe" else f"_{extractor}"
    return os.path.join(MODELS_DIR, f"wlasl_gru{suffix}.meta.json")


def train_one(seed, features, labels, num_classes, args, pretrain=None,
              keep_columns=None, rich=False, layout=None, handshape=False):
    """features arrive en 155 colonnes.

    Le retrait de la profondeur se fait JUSTE avant le modele : mirror(),
    appele par augment_batch(), code en dur les indices des deux mains, des
    drapeaux et de la pose sur la disposition complete. Couper d'abord
    decale tout et leve une erreur de forme — exactement l'erreur deja
    corrigee dans train_kfold.py, puis reintroduite ici.
    """
    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)

    def to_model(batch):
        # Les features derivees se calculent sur la disposition COMPLETE
        # (indices des mains et de la pose codes en dur), donc avant la
        # selection de colonnes.
        # MEME ORDRE que train_kfold.to_model : d abord les 23 scalaires
        # derives, ensuite les 8 d opposition du pouce. Un ordre different
        # donnerait au modele les bonnes valeurs dans les mauvaises colonnes.
        pieces = []
        if rich:
            pieces.append(add_rich_features(batch, layout)[:, :, -23:])
        if handshape:
            pieces.append(add_handshape(batch, layout)[:, :, -8:])

        if keep_columns is not None:
            batch = batch[:, :, keep_columns]

        if pieces:
            batch = np.concatenate([batch] + pieces, axis=-1)

        return batch

    feature_dim = ((len(keep_columns) if keep_columns is not None
                    else features.shape[2])
                   + (23 if rich else 0) + (8 if handshape else 0))

    model = build_model(num_classes, args.units, feature_dim,
                        "gru", args.label_smoothing)

    if pretrain is not None:
        pre_x, pre_y, pre_classes = pretrain
        head = build_model(pre_classes, args.units, feature_dim,
                           "gru", args.label_smoothing)

        for _ in range(args.pretrain_epochs):
            order = rng.permutation(len(pre_x))
            batch = to_model(augment_batch(pre_x[order], rng, layout))
            target = pre_y[order]
            if args.label_smoothing > 0:
                target = tf.keras.utils.to_categorical(target, pre_classes)
            head.fit(batch, target, batch_size=args.batch, epochs=1, verbose=0)

        # Tout sauf la couche de sortie, dont le nombre de classes differe.
        for target_layer, source_layer in zip(model.layers[:-1], head.layers[:-1]):
            if source_layer.get_weights():
                target_layer.set_weights(source_layer.get_weights())

    for epoch in range(args.epochs):
        order = rng.permutation(len(features))
        batch = to_model(augment_batch(features[order], rng, layout))
        target = labels[order]
        if args.label_smoothing > 0:
            target = tf.keras.utils.to_categorical(target, num_classes)

        history = model.fit(batch, target, batch_size=args.batch,
                            epochs=1, verbose=0)

        if (epoch + 1) % 50 == 0 or epoch == args.epochs - 1:
            print(f"    epoch {epoch + 1}/{args.epochs}  "
                  f"train_acc {history.history['accuracy'][0]:.3f}", flush=True)

    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--units", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab", type=int, default=50)
    parser.add_argument("--variants", default="dominant")
    parser.add_argument("--features-dir",
                        default=os.path.join(PROJECT_ROOT, "data", "wlasl",
                                             "features_rtm07"),
                        help="dossier de features ; par defaut RTMPose a 0.7")
    parser.add_argument("--handshape", action="store_true", default=True,
                        help="8 scalaires d opposition du pouce (+1.1 points, "
                             "5/5 folds)")
    parser.add_argument("--confidence", action="store_true", default=True,
                        help="confiance de chaque main en valeur continue")
    parser.add_argument("--no-z", action="store_true", default=False,
                        help="MediaPipe seulement : RTMPose n a pas de z")
    parser.add_argument("--rich", action="store_true", default=True,
                        help="23 scalaires derives de x,y")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--pretrain-vocab", type=int, default=100)
    parser.add_argument("--pretrain-epochs", type=int, default=60)
    args = parser.parse_args()

    records = load_sequences(args.variants, args.features_dir,
                             with_confidence=args.confidence)

    counts = collections.Counter(r["gloss"] for r in records)
    target_glosses = {g for g, _ in counts.most_common(args.vocab)}

    glosses = sorted(target_glosses)
    index = {g: i for i, g in enumerate(glosses)}

    selected = [r for r in records if r["gloss"] in target_glosses]
    features = np.stack([pad_sequence(r["features"]) for r in selected])
    labels = np.array([index[r["gloss"]] for r in selected], dtype=np.int32)

    keep_columns = drop_z_indices(features.shape[2]) if args.no_z else None
    feature_dim = ((len(keep_columns) if keep_columns is not None
                    else features.shape[2])
                   + (23 if args.rich else 0) + (8 if args.handshape else 0))

    # La disposition ne peut pas etre devinee quand il y a une queue : 106
    # colonnes se lisent aussi bien comme 9 points de pose plus 2 confiances
    # que comme 10 points de pose, et le second ferait negater une confiance
    # au miroir. Voir Layout.from_dim.
    tail = 2 if args.confidence else 0
    layout = Layout.from_dim(features.shape[2] - tail)
    layout.tail = tail

    print(f"{len(selected)} sequences, {len(glosses)} gloses, "
          f"{feature_dim} dimensions")
    print(f"{args.seeds} modeles x {args.epochs} epochs\n")

    pretrain = None
    if args.pretrain_vocab:
        wide = {g for g, _ in counts.most_common(args.pretrain_vocab)}
        extra = [r for r in records if r["gloss"] in wide]
        # records vient DEJA du bon dossier et des bonnes colonnes : pas de
        # second load_sequences, qui relirait le dossier par defaut.
        wide_index = {g: i for i, g in enumerate(sorted(wide))}

        pre_x = np.stack([pad_sequence(r["features"]) for r in extra])

        pretrain = (
            pre_x,
            np.array([wide_index[r["gloss"]] for r in extra], dtype=np.int32),
            len(wide),
        )
        print(f"pre-entrainement : {len(extra)} sequences, {len(wide)} gloses\n")

    extractor = "rtmpose" if "rtm" in args.features_dir else "mediapipe"

    saved = []
    for seed_index in range(args.seeds):
        print(f"  modele {seed_index + 1}/{args.seeds}")
        model = train_one(args.seed + seed_index * 100, features, labels,
                          len(glosses), args, pretrain, keep_columns,
                          args.rich, layout, args.handshape)
        path = model_path(seed_index, extractor)
        model.save(path)
        saved.append(os.path.relpath(path, PROJECT_ROOT))

    expected = None
    if os.path.exists(KFOLD_PATH):
        with open(KFOLD_PATH, encoding="utf-8") as handle:
            kfold = json.load(handle)
        curves = np.array(kfold["curves"])
        plateau = curves[:, -30:].mean(axis=1)
        expected = {
            "mean": round(float(plateau.mean()), 4),
            "std": round(float(plateau.std()), 4),
            "folds": kfold["folds"],
            "chance": round(1 / len(glosses), 4),
            "measured_config": os.path.basename(KFOLD_PATH),
            "caveat": KFOLD_CAVEAT if args.handshape else None,
        }

    metadata = {
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": "ensemble de GRU bidirectionnels 2 couches sur sequences de landmarks",
        "models": saved,
        "classes": glosses,
        "num_classes": len(glosses),
        "sequences": len(selected),
        "clips_per_gloss": {g: counts[g] for g in glosses},
        "max_len": MAX_LEN,
        "feature_dim": int(feature_dim),
        "drop_z": bool(args.no_z),
        "rich": bool(args.rich),
        "confidence": bool(args.confidence),
        "handshape": bool(args.handshape),
        "features_dir": os.path.relpath(args.features_dir, PROJECT_ROOT),
        "extractor": extractor,
        "keep_columns": keep_columns.tolist() if keep_columns is not None else None,
        "seeds": args.seeds,
        "pretrain_epochs": args.pretrain_epochs,
        "units": args.units,
        "epochs": args.epochs,
        "label_smoothing": args.label_smoothing,
        "variants": args.variants,
        "pretrain_vocab": args.pretrain_vocab,
        "expected_accuracy": expected,
        "accuracy_note": (
            "expected_accuracy vient de la validation croisee par signeur "
            "(train_kfold.py, configuration identique), pas de ce modele : "
            "celui-ci est entraine sur toutes les donnees et n'a pas "
            "d'ensemble de test propre."
        ),
    }

    with open(meta_path(extractor), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"\n  {len(saved)} modeles sauvegardes")
    print(f"  fiche : {meta_path(extractor)}")
    if expected:
        print(f"  precision attendue (k-fold) : {expected['mean']:.4f} "
              f"+- {expected['std']:.4f}  (hasard {expected['chance']})")


if __name__ == "__main__":
    main()
