"""Entraine le modele final sur TOUTES les sequences et le sauvegarde.

train_kfold.py mesure la precision attendue mais ne garde aucun poids : son
role est d'estimer, pas de produire un modele. L'estimation faite
(0.6254 +- 0.0450 en decoupage par signeur), on reentraine sur l'ensemble
des donnees — garder un fold de cote ne servirait qu'a repeter une mesure
deja obtenue, en se privant de 20% des exemples.

La configuration retenue est celle qui a gagne les 5 folds :

  --variants dominant   une seule variante de signe par glose (+5.6)
  --no-z                sans la profondeur, peu fiable sur une camera (+11.6
                        avec le lissage et la TTA)
  --label-smoothing 0.1
  --seeds 3             ensemble de 3 modeles (+5.3 avec le pre-entrainement)
  --pretrain-vocab 100  pre-entrainement sur un vocabulaire elargi

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
KFOLD_PATH = os.path.join(MODELS_DIR, "wlasl_kfold_everything.json")


def model_path(seed_index):
    return os.path.join(MODELS_DIR, f"wlasl_gru_s{seed_index}.keras")


def train_one(seed, features, labels, num_classes, args, pretrain=None,
              keep_columns=None, rich=False):
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
        extra = add_rich_features(batch)[:, :, -23:] if rich else None

        if keep_columns is not None:
            batch = batch[:, :, keep_columns]

        if extra is not None:
            batch = np.concatenate([batch, extra], axis=-1)

        return batch

    feature_dim = (len(keep_columns) if keep_columns is not None
                   else features.shape[2]) + (23 if rich else 0)

    model = build_model(num_classes, args.units, feature_dim,
                        "gru", args.label_smoothing)

    if pretrain is not None:
        pre_x, pre_y, pre_classes = pretrain
        head = build_model(pre_classes, args.units, feature_dim,
                           "gru", args.label_smoothing)

        for _ in range(args.pretrain_epochs):
            order = rng.permutation(len(pre_x))
            batch = to_model(augment_batch(pre_x[order], rng))
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
        batch = to_model(augment_batch(features[order], rng))
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
    parser.add_argument("--no-z", action="store_true", default=True)
    parser.add_argument("--rich", action="store_true", default=True,
                        help="23 scalaires derives de x,y")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--pretrain-vocab", type=int, default=100)
    parser.add_argument("--pretrain-epochs", type=int, default=60)
    args = parser.parse_args()

    records = load_sequences(args.variants)

    counts = collections.Counter(r["gloss"] for r in records)
    target_glosses = {g for g, _ in counts.most_common(args.vocab)}

    glosses = sorted(target_glosses)
    index = {g: i for i, g in enumerate(glosses)}

    selected = [r for r in records if r["gloss"] in target_glosses]
    features = np.stack([pad_sequence(r["features"]) for r in selected])
    labels = np.array([index[r["gloss"]] for r in selected], dtype=np.int32)

    keep_columns = drop_z_indices(features.shape[2]) if args.no_z else None
    feature_dim = (len(keep_columns) if keep_columns is not None
                   else features.shape[2]) + (23 if args.rich else 0)

    print(f"{len(selected)} sequences, {len(glosses)} gloses, "
          f"{feature_dim} dimensions")
    print(f"{args.seeds} modeles x {args.epochs} epochs\n")

    pretrain = None
    if args.pretrain_vocab:
        wide = {g for g, _ in counts.most_common(args.pretrain_vocab)}
        extra = [r for r in records if r["gloss"] in wide]
        wide_index = {g: i for i, g in enumerate(sorted(wide))}

        pre_x = np.stack([pad_sequence(r["features"]) for r in extra])

        pretrain = (
            pre_x,
            np.array([wide_index[r["gloss"]] for r in extra], dtype=np.int32),
            len(wide),
        )
        print(f"pre-entrainement : {len(extra)} sequences, {len(wide)} gloses\n")

    saved = []
    for seed_index in range(args.seeds):
        print(f"  modele {seed_index + 1}/{args.seeds}")
        model = train_one(args.seed + seed_index * 100, features, labels,
                          len(glosses), args, pretrain, keep_columns, args.rich)
        path = model_path(seed_index)
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
        "keep_columns": keep_columns.tolist() if keep_columns is not None else None,
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

    with open(META_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"\n  {len(saved)} modeles sauvegardes")
    print(f"  fiche : {META_PATH}")
    if expected:
        print(f"  precision attendue (k-fold) : {expected['mean']:.4f} "
              f"+- {expected['std']:.4f}  (hasard {expected['chance']})")


if __name__ == "__main__":
    main()
