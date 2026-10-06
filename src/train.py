"""Entraine le CNN sur le dataset et sauvegarde le modele + la liste des classes."""

import datetime
import json
import os
import subprocess

import tensorflow as tf

from data_loader import BATCH_SIZE, IMG_SIZE, grouped_split, load_image_dataset
from model_cnn import build_cnn_model


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "asl_alphabet_segmented",
)

# EarlyStopping decide de l'arret reel ; cette borne est juste un plafond.
EPOCHS = 40

# En decoupage temporel, val_loss remonte des l'epoch 2 alors que
# val_accuracy continue de progresser par a-coups (meilleur score observe
# a l'epoch 13). On surveille donc val_accuracy, pas val_loss, sinon on
# s'arrete trop tot sur une metrique qui ne mesure plus le bon objectif.
EARLY_STOPPING_PATIENCE = 5

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "sign_model.keras"
)

CLASSES_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "class_names.json"
)

# Fiche d'identite du modele : quel decoupage, quel score, quand.
# Sans ca, deux .keras sont indistinguables sur le disque.
META_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "sign_model.meta.json"
)

# Export TFLite : ~96x plus rapide que model.predict() sur une image seule
# (0.4 ms au lieu de 36 ms), pour le temps reel dans webcam_demo.py.
TFLITE_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "sign_model.tflite"
)


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def train():

    train_ds, val_ds, class_names = load_image_dataset(DATA_DIR)

    (train_files, _), (val_files, _), _ = grouped_split(DATA_DIR)

    print(f"Classes détectées ({len(class_names)}):")
    print(class_names)

    model = build_cnn_model(
        input_shape=(64, 64, 3),
        num_classes=len(class_names)
    )

    model.summary()

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    callbacks = [
        # Sauvegarde le MEILLEUR epoch, pas le dernier.
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        # restore_best_weights : le modele en memoire (et donc l'export
        # TFLite plus bas) correspond au meilleur epoch, pas au dernier.
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    best_epoch = int(
        max(
            range(len(history.history["val_accuracy"])),
            key=lambda i: history.history["val_accuracy"][i],
        )
    )
    best_acc = history.history["val_accuracy"][best_epoch]

    # --------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------

    model.save(MODEL_PATH)

    metadata = {
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "split": "temporal (grouped by frame index)",
        "split_note": (
            "Les images de validation sont les dernieres frames de chaque "
            "classe. Un decoupage aleatoire donnerait un score gonfle "
            "(0.9938 mesure) car les frames voisines sont quasi identiques."
        ),
        "val_accuracy": round(float(best_acc), 4),
        "best_epoch": best_epoch + 1,
        "epochs_run": len(history.history["val_accuracy"]),
        "epochs_max": EPOCHS,
        "train_images": len(train_files),
        "val_images": len(val_files),
        "num_classes": len(class_names),
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with open(
        CLASSES_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            class_names,
            f,
            ensure_ascii=False,
            indent=2
        )

    # Export TFLite pour l'inference temps reel
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    with open(TFLITE_PATH, "wb") as f:
        f.write(converter.convert())

    print()
    print("==============================")
    print("ENTRAÎNEMENT TERMINÉ")
    print("==============================")
    print(f"Modèle  : {MODEL_PATH}")
    print(f"Classes : {CLASSES_PATH}")
    print(f"Fiche   : {META_PATH}")
    print(f"TFLite  : {TFLITE_PATH}")
    print(
        f"Meilleur epoch : {best_epoch + 1}/{len(history.history['val_accuracy'])} "
        f"— val_accuracy {best_acc:.4f}"
    )


if __name__ == "__main__":
    train()