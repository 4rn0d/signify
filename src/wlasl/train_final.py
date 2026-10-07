"""Entraine le modele final sur TOUTES les sequences et le sauvegarde.

train_kfold.py mesure la precision attendue mais ne garde aucun poids : son
role est d'estimer, pas de produire un modele. Une fois l'estimation faite
(0.37 +- 0.04 en decoupage par signeur), on reentraine sur l'ensemble des
donnees — garder un fold de cote ne servirait qu'a repeter une mesure deja
obtenue, en se privant de 20% des exemples.

Le budget d'epochs vient de la courbe k-fold : plateau atteint vers 140-150.

Attention : le score inscrit dans la fiche est celui du k-fold, PAS une
mesure de ce modele-ci. Un modele evalue sur ses propres donnees
d'entrainement afficherait un chiffre sans signification.
"""

import argparse
import collections
import json
import os

import numpy as np
import tensorflow as tf

from train_kfold import (
    MAX_LEN,
    augment_batch,
    build_model,
    load_sequences,
    pad_sequence,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "wlasl_gru.keras")
META_PATH = os.path.join(PROJECT_ROOT, "models", "wlasl_gru.meta.json")
KFOLD_PATH = os.path.join(PROJECT_ROOT, "models", "wlasl_kfold.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--units", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    records = load_sequences()
    glosses = sorted({r["gloss"] for r in records})
    index = {g: i for i, g in enumerate(glosses)}

    features = np.stack([pad_sequence(r["features"]) for r in records])
    labels = np.array([index[r["gloss"]] for r in records], dtype=np.int32)

    print(f"{len(records)} sequences, {len(glosses)} gloses")
    print(f"entree {features.shape[1:]}, {args.epochs} epochs\n")

    tf.keras.utils.set_random_seed(args.seed)
    model = build_model(len(glosses), args.units, features.shape[2])

    rng = np.random.default_rng(args.seed)

    for epoch in range(args.epochs):
        order = rng.permutation(len(features))
        history = model.fit(
            augment_batch(features[order], rng),
            labels[order],
            batch_size=args.batch,
            epochs=1,
            verbose=0,
        )

        if (epoch + 1) % 25 == 0 or epoch == args.epochs - 1:
            print(
                f"  epoch {epoch + 1}/{args.epochs}  "
                f"train_acc {history.history['accuracy'][0]:.3f}  "
                f"loss {history.history['loss'][0]:.3f}",
                flush=True,
            )

    model.save(MODEL_PATH)

    expected = None
    if os.path.exists(KFOLD_PATH):
        with open(KFOLD_PATH, encoding="utf-8") as handle:
            kfold = json.load(handle)
        curves = np.array(kfold["curves"])
        tail = curves[:, -30:].mean(axis=1)
        expected = {"mean": round(float(tail.mean()), 4),
                    "std": round(float(tail.std()), 4),
                    "folds": kfold["folds"]}

    counts = collections.Counter(r["gloss"] for r in records)

    metadata = {
        "created": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": "GRU bidirectionnel 2 couches sur sequences de landmarks",
        "classes": glosses,
        "num_classes": len(glosses),
        "sequences": len(records),
        "clips_per_gloss": {g: counts[g] for g in glosses},
        "max_len": MAX_LEN,
        "feature_dim": int(features.shape[2]),
        "units": args.units,
        "epochs": args.epochs,
        "expected_accuracy": expected,
        "accuracy_note": (
            "expected_accuracy vient de la validation croisee par signeur "
            "(train_kfold.py), pas de ce modele : celui-ci est entraine sur "
            "toutes les donnees et n'a pas d'ensemble de test propre."
        ),
    }

    with open(META_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"\n  modele : {MODEL_PATH}")
    print(f"  fiche  : {META_PATH}")
    if expected:
        print(f"  precision attendue (k-fold) : "
              f"{expected['mean']:.4f} +- {expected['std']:.4f}")


if __name__ == "__main__":
    main()
