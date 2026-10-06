"""Compare CNN seul / landmarks seuls / fusion, a conditions identiques.

Point methodologique important : les landmarks ne sont disponibles que
pour ~83% des images (MediaPipe ne detecte pas toujours de main). Comparer
la fusion sur ce sous-ensemble au CNN sur 100% des images melangerait deux
effets. Les trois modeles sont donc entraines et evalues exactement sur
les memes echantillons : ceux qui ont a la fois une image et des landmarks.

Le decoupage reste temporel (voir data_loader), sans quoi tous les scores
seraient gonfles de la meme facon.
"""

import csv
import datetime
import json
import os
import sys

import numpy as np
import tensorflow as tf

from data_loader import BATCH_SIZE, IMG_SIZE, grouped_split
from landmark_extractor import NUM_LANDMARKS, normalize_landmarks
from model_cnn import build_cnn_model
from model_fusion import build_fusion_model
from model_landmark import build_landmark_model


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(
    PROJECT_ROOT, "data", "processed", "asl_alphabet_segmented"
)

LANDMARKS_CSV = os.path.join(PROJECT_ROOT, "data", "landmarks", "landmarks.csv")

RESULTS_PATH = os.path.join(PROJECT_ROOT, "models", "comparison.json")

FUSION_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "sign_model_fusion.keras")
FUSION_META_PATH = os.path.join(PROJECT_ROOT, "models", "sign_model_fusion.meta.json")

EPOCHS = 40
PATIENCE = 5


def load_landmarks():
    """{(classe, fichier): vecteur normalise de 63 valeurs}."""
    table = {}

    with open(LANDMARKS_CSV, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            coords = np.array(
                [
                    float(row[f"{axis}{i}"])
                    for i in range(NUM_LANDMARKS)
                    for axis in ("x", "y", "z")
                ],
                dtype=np.float32,
            )

            table[(row["class"], row["filename"])] = (
                normalize_landmarks(coords).reshape(-1)
            )

    return table


def build_arrays(files, labels, table):
    """Garde les echantillons ayant image ET landmarks.

    L'ordre de retour (files, landmarks, labels) doit correspondre a la
    signature de make_dataset : il est deballe avec *.
    """
    kept_files, kept_labels, kept_landmarks = [], [], []

    for path, label in zip(files, labels):
        key = (
            os.path.basename(os.path.dirname(path)),
            os.path.basename(path),
        )

        if key in table:
            kept_files.append(path)
            kept_labels.append(label)
            kept_landmarks.append(table[key])

    return (
        kept_files,
        np.array(kept_landmarks, dtype=np.float32),
        np.array(kept_labels, dtype=np.int32),
    )


def _decode(path, landmarks, label):
    img = tf.io.decode_image(
        tf.io.read_file(path), channels=3, expand_animations=False
    )
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return tf.cast(img, tf.float32) / 255.0, landmarks, label


def _check_shapes(files, landmarks, labels):
    if landmarks.ndim != 2 or landmarks.shape[1] != NUM_LANDMARKS * 3:
        raise ValueError(
            f"landmarks devrait etre (n, {NUM_LANDMARKS * 3}), "
            f"recu {landmarks.shape} — arguments inverses ?"
        )
    if labels.ndim != 1:
        raise ValueError(
            f"labels devrait etre (n,), recu {labels.shape} — "
            "arguments inverses ?"
        )
    if not (len(files) == len(landmarks) == len(labels)):
        raise ValueError(
            f"tailles incoherentes : {len(files)} fichiers, "
            f"{len(landmarks)} landmarks, {len(labels)} labels"
        )


def make_dataset(files, landmarks, labels, inputs, shuffle):
    """inputs : 'image', 'landmarks' ou 'both'.

    En mode 'landmarks' on ne touche pas aux fichiers image : decoder des
    JPEG pour les jeter ensuite rendrait cet entrainement inutilement
    limite par les entrees/sorties.
    """
    _check_shapes(files, landmarks, labels)

    if inputs == "landmarks":
        ds = tf.data.Dataset.from_tensor_slices((landmarks, labels))

        if shuffle:
            ds = ds.shuffle(
                len(labels), seed=123, reshuffle_each_iteration=True
            )

        return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    ds = tf.data.Dataset.from_tensor_slices((files, landmarks, labels))

    if shuffle:
        ds = ds.shuffle(len(files), seed=123, reshuffle_each_iteration=True)

    ds = ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)

    if inputs == "image":
        ds = ds.map(lambda i, l, y: (i, y), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(
            lambda i, l, y: ({"image": i, "landmarks": l}, y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def run(name, model, train_parts, val_parts, inputs):
    train_ds = make_dataset(*train_parts, inputs=inputs, shuffle=True)
    val_ds = make_dataset(*val_parts, inputs=inputs, shuffle=False)

    print(f"\n{'=' * 60}\n{name}  ({model.count_params():,} parametres)\n{'=' * 60}")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=PATIENCE,
                restore_best_weights=True,
                verbose=1,
            )
        ],
        verbose=2,
    )

    scores = history.history["val_accuracy"]
    best = int(np.argmax(scores))

    # restore_best_weights=True : le modele rendu ici est celui du meilleur
    # epoch, pas du dernier.
    return {
        "val_accuracy": round(float(scores[best]), 4),
        "best_epoch": best + 1,
        "epochs_run": len(scores),
        "params": int(model.count_params()),
    }, model


def main():
    (train_files, train_labels), (val_files, val_labels), class_names = (
        grouped_split(DATA_DIR)
    )

    table = load_landmarks()
    print(f"{len(table)} jeux de landmarks charges.")

    tr = build_arrays(train_files, train_labels, table)
    va = build_arrays(val_files, val_labels, table)

    print(
        f"Echantillons communs : {len(tr[2])} train / {len(va[2])} val "
        f"(sur {len(train_files)} / {len(val_files)} images)"
    )

    n = len(class_names)

    # --only fusion : reentraine uniquement la fusion (pour la sauvegarder)
    # sans repayer les deux baselines.
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]

    results = {}

    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, encoding="utf-8") as handle:
            results = json.load(handle)

    if only in (None, "cnn"):
        results["cnn"], _ = run(
            "CNN seul (image)", build_cnn_model(num_classes=n), tr, va, "image"
        )

    if only in (None, "landmark"):
        results["landmark"], _ = run(
            "Landmarks seuls", build_landmark_model(num_classes=n), tr, va, "landmarks"
        )

    if only in (None, "fusion"):
        results["fusion"], fusion_model = run(
            "FUSION image + landmarks",
            build_fusion_model(num_classes=n),
            tr,
            va,
            "both",
        )

        fusion_model.save(FUSION_MODEL_PATH)

        with open(FUSION_META_PATH, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "created": datetime.datetime.now()
                    .astimezone()
                    .isoformat(timespec="seconds"),
                    "model": "fusion (image + landmarks)",
                    "split": "temporal (grouped by frame index)",
                    "inputs": {
                        "image": "64x64x3, segmente sur fond blanc, /255",
                        "landmarks": (
                            "63 = 21 points x (x, y, z) en ESPACE PIXEL, "
                            "normalises poignet-origine et echelle unitaire"
                        ),
                    },
                    "val_accuracy": results["fusion"]["val_accuracy"],
                    "best_epoch": results["fusion"]["best_epoch"],
                    "epochs_run": results["fusion"]["epochs_run"],
                    "train_samples": len(tr[2]),
                    "val_samples": len(va[2]),
                    "num_classes": n,
                    "img_size": IMG_SIZE,
                    "compared_with": {
                        "cnn_only": results.get("cnn", {}).get("val_accuracy"),
                        "landmark_only": results.get("landmark", {}).get(
                            "val_accuracy"
                        ),
                    },
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

        print(f"\nModele fusion : {FUSION_MODEL_PATH}")
        print(f"Fiche         : {FUSION_META_PATH}")

    print(f"\n{'=' * 60}\nRESULTATS (memes echantillons, decoupage temporel)\n{'=' * 60}")
    print(f"{'modele':<26}{'val_accuracy':>14}{'epoch':>8}{'params':>12}")
    for key, label in [
        ("cnn", "CNN seul"),
        ("landmark", "Landmarks seuls"),
        ("fusion", "FUSION"),
    ]:
        if key not in results:
            continue
        r = results[key]
        print(
            f"{label:<26}{r['val_accuracy']:>14.4f}{r['best_epoch']:>8}"
            f"{r['params']:>12,}"
        )

    with open(RESULTS_PATH, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    print(f"\nResultats : {RESULTS_PATH}")


if __name__ == "__main__":
    main()
