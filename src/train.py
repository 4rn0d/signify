"""Entraine le CNN sur le dataset et sauvegarde le modele + la liste des classes."""

import json
import os

import tensorflow as tf

from data_loader import load_image_dataset
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

EPOCHS = 15

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

# Export TFLite : ~96x plus rapide que model.predict() sur une image seule
# (0.4 ms au lieu de 36 ms), pour le temps reel dans webcam_demo.py.
TFLITE_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "sign_model.tflite"
)


def train():

    train_ds, val_ds, class_names = load_image_dataset(DATA_DIR)

    print(f"Classes détectées ({len(class_names)}):")
    print(class_names)

    model = build_cnn_model(
        input_shape=(64, 64, 3),
        num_classes=len(class_names)
    )

    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
    )

    # --------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True
    )

    model.save(MODEL_PATH)

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
    print(f"TFLite  : {TFLITE_PATH}")


if __name__ == "__main__":
    train()