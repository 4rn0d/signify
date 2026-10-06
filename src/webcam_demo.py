"""Ouvre la webcam, detecte la main et affiche la lettre predite en direct.

Deux points a ne pas casser :

1. Meme pretraitement image qu'a l'entrainement. Le CNN a ete entraine sur
   des mains detourees sur fond blanc (HandSegmenter + GrabCut). Lui envoyer
   un crop brut de la webcam fait chuter la precision de 0.985 a 0.384.

2. Meme espace de coordonnees pour les landmarks. MediaPipe normalise x et y
   par la largeur et la hauteur de l'image. Le dataset est carre (200x200),
   donc isotrope ; une webcam 640x480 ne l'est pas et etirerait la geometrie
   de 4:3 (decalage mesure : 0.11, soit ~5% de l'amplitude des coordonnees).
   HandSegmenter.last_landmarks est donc deja en espace pixel.
"""

import json
import os

import cv2
import numpy as np
import tensorflow as tf

from hand_segmenter import GRABCUT_REALTIME, HandSegmenter
from landmark_extractor import normalize_landmarks


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

CLASSES_PATH = os.path.join(MODELS_DIR, "class_names.json")

# Fusion image + landmarks : meilleur score mesure (0.9431 contre 0.8722
# pour le CNN seul, a echantillons identiques). Utilise si present.
FUSION_MODEL_PATH = os.path.join(MODELS_DIR, "sign_model_fusion.keras")
FUSION_META_PATH = os.path.join(MODELS_DIR, "sign_model_fusion.meta.json")

# Repli CNN seul.
MODEL_PATH = os.path.join(MODELS_DIR, "sign_model.keras")
TFLITE_PATH = os.path.join(MODELS_DIR, "sign_model.tflite")
META_PATH = os.path.join(MODELS_DIR, "sign_model.meta.json")

# doit correspondre a la taille utilisee a l'entrainement
IMG_SIZE = 64

# Le modele n'a pas de classe "nothing" : il choisit toujours une des 28
# lettres, meme sur une main au repos. En dessous de ce seuil on affiche "?"
# plutot qu'une lettre clignotante.
CONFIDENCE_THRESHOLD = 80.0

# Squelette de la main MediaPipe : 5 doigts partant du poignet (0),
# plus la ligne de paume reliant la base des doigts.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # pouce
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (0, 9), (9, 10), (10, 11), (11, 12),       # majeur
    (0, 13), (13, 14), (14, 15), (15, 16),     # annulaire
    (0, 17), (17, 18), (18, 19), (19, 20),     # auriculaire
    (5, 9), (9, 13), (13, 17),                 # paume
]

# BGR
COLOR_BONE = (150, 220, 140)
COLOR_JOINT = (60, 90, 235)
COLOR_OK = (120, 210, 90)
COLOR_UNSURE = (60, 150, 235)
COLOR_NONE = (70, 70, 220)


def read_meta(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_classifier():
    """Retourne (predict, class_names, description).

    predict(crop_bgr, landmarks_px) -> vecteur de probabilites.
    La branche landmarks est ignoree par les modeles image seule, pour que
    la boucle d'affichage n'ait pas a connaitre le type de modele.
    """
    with open(CLASSES_PATH, encoding="utf-8") as handle:
        class_names = json.load(handle)

    if os.path.exists(FUSION_MODEL_PATH):
        model = tf.keras.models.load_model(FUSION_MODEL_PATH)

        @tf.function
        def forward(image, landmarks):
            return model(
                {"image": image, "landmarks": landmarks}, training=False
            )

        def predict(crop_bgr, landmarks_px):
            image = prepare_image(crop_bgr)
            landmarks = prepare_landmarks(landmarks_px)
            return forward(
                tf.constant(image), tf.constant(landmarks)
            ).numpy()[0]

        meta = read_meta(FUSION_META_PATH) or {}
        description = (
            f"FUSION image+landmarks | val_accuracy={meta.get('val_accuracy', '?')} "
            f"| {meta.get('created', '?')}"
        )
        return predict, class_names, description

    # --- repli : CNN seul -------------------------------------------------
    meta = read_meta(META_PATH) or {}
    description = (
        f"CNN seul | split={meta.get('split', '?')} "
        f"| val_accuracy={meta.get('val_accuracy', '?')}"
        "  (sign_model_fusion.keras absent : lance train_fusion.py)"
    )

    if os.path.exists(TFLITE_PATH):
        interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH, num_threads=4)
        interpreter.allocate_tensors()

        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]

        def predict(crop_bgr, landmarks_px):
            interpreter.set_tensor(input_index, prepare_image(crop_bgr))
            interpreter.invoke()
            return interpreter.get_tensor(output_index)[0]

        return predict, class_names, "TFLite " + description

    model = tf.keras.models.load_model(MODEL_PATH)

    @tf.function
    def forward(image):
        return model(image, training=False)

    def predict(crop_bgr, landmarks_px):
        return forward(tf.constant(prepare_image(crop_bgr))).numpy()[0]

    return predict, class_names, "Keras " + description


def prepare_image(crop_bgr):
    """crop_for_model() a deja redimensionne en 64x64 (INTER_AREA), comme a
    la preparation du dataset. Il ne reste que BGR -> RGB et le /255."""
    img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return np.expand_dims(img.astype("float32") / 255.0, axis=0)


def prepare_landmarks(landmarks_px):
    """(21, 3) en pixels -> (1, 63) normalise poignet-origine, echelle 1."""
    if landmarks_px is None:
        return np.zeros((1, 63), dtype=np.float32)

    return normalize_landmarks(landmarks_px).reshape(1, -1).astype("float32")


def draw_landmarks(frame, landmarks_px):
    """Dessine le squelette de la main sur l'image."""
    if landmarks_px is None:
        return

    points = landmarks_px[:, :2].astype(np.int32)

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            tuple(points[start]),
            tuple(points[end]),
            COLOR_BONE,
            2,
            cv2.LINE_AA,
        )

    for index, (x, y) in enumerate(points):
        # Poignet un peu plus gros : c'est l'origine de la normalisation.
        radius = 5 if index == 0 else 3
        cv2.circle(frame, (int(x), int(y)), radius, COLOR_JOINT, -1, cv2.LINE_AA)


def draw_preview(frame, crop):
    """Apercu de ce que le CNN recoit reellement, en haut a droite.

    Quand une prediction est fausse, cette vignette dit tout de suite si
    c'est la segmentation ou le modele qui a echoue.
    """
    size = 128
    margin = 10
    preview = cv2.resize(crop, (size, size), interpolation=cv2.INTER_NEAREST)

    x1 = frame.shape[1] - size - margin
    y1 = margin

    frame[y1:y1 + size, x1:x1 + size] = preview
    cv2.rectangle(
        frame, (x1 - 1, y1 - 1), (x1 + size, y1 + size), COLOR_BONE, 1
    )


def run_webcam_demo():
    predict, class_names, description = load_classifier()
    print(f"Modele : {description}")

    segmenter = HandSegmenter(num_hands=1, **GRABCUT_REALTIME)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Impossible d'ouvrir la webcam.")
        segmenter.close()
        return

    print("Appuie sur 'q' pour quitter, 'l' pour masquer le squelette.")
    show_landmarks = True

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # effet miroir, plus naturel

            cleaned, bbox, found = segmenter.process(frame)
            landmarks_px = segmenter.last_landmarks

            if found and bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BONE, 1)

                if show_landmarks:
                    draw_landmarks(frame, landmarks_px)

                crop = segmenter.crop_for_model(
                    cleaned, bbox, output_size=IMG_SIZE
                )

                if crop is not None:
                    predictions = predict(crop, landmarks_px)

                    best = int(np.argmax(predictions))
                    confidence = float(predictions[best]) * 100

                    if confidence >= CONFIDENCE_THRESHOLD:
                        text = f"{class_names[best]} ({confidence:.0f}%)"
                        color = COLOR_OK
                    else:
                        text = f"? ({confidence:.0f}%)"
                        color = COLOR_UNSURE

                    cv2.putText(
                        frame, text, (x1, max(y1 - 12, 28)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA,
                    )

                    draw_preview(frame, crop)
            else:
                cv2.putText(
                    frame, "Aucune main detectee", (24, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_NONE, 2, cv2.LINE_AA,
                )

            cv2.imshow("Sign Language Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("l"):
                show_landmarks = not show_landmarks
    finally:
        cap.release()
        cv2.destroyAllWindows()
        segmenter.close()


if __name__ == "__main__":
    run_webcam_demo()
