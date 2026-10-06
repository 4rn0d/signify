"""Ouvre la webcam, detecte la main et affiche la lettre predite en direct.

IMPORTANT : ce demo applique exactement le meme pretraitement que
prepare_segmented_dataset.py, c'est-a-dire HandSegmenter (segmentation
GrabCut + fond blanc) puis crop_for_model(). Le CNN a ete entraine sur des
mains detourees sur fond blanc : lui envoyer un crop brut de la webcam
(avec le vrai arriere-plan) fait chuter la precision de ~0.98 a ~0.38.

HandSegmenter fait deja tourner MediaPipe en interne, il n'y a donc plus
besoin d'un HandLandmarker separe ici.
"""

import json
import os

import cv2
import numpy as np
import tensorflow as tf

from hand_segmenter import GRABCUT_REALTIME, HandSegmenter


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "sign_model.keras",
)

CLASSES_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "class_names.json",
)

# TFLite : 0.4 ms par image contre 36 ms pour model.predict(), pour des
# sorties numeriquement identiques (ecart max mesure 5e-11). Genere par
# train.py ; on retombe sur le .keras s'il est absent.
TFLITE_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "sign_model.tflite",
)

# doit correspondre a la taille utilisee a l'entrainement
IMG_SIZE = 64

# Le modele n'a plus de classe "nothing" : il choisit toujours une des 28
# lettres, meme sur une main au repos. En dessous de ce seuil on n'affiche
# rien plutot qu'une lettre clignotante.
CONFIDENCE_THRESHOLD = 80.0


def load_classifier():
    """Retourne une fonction predict(crop_64x64_bgr) -> vecteur de probabilites.

    Prefere TFLite (temps reel). Sinon, utilise le modele Keras appele
    directement : model(x) et non model.predict(), ce dernier ayant une
    surcharge d'environ 36 ms par appel sur une seule image.
    """
    with open(CLASSES_PATH, encoding="utf-8") as f:
        class_names = json.load(f)

    if os.path.exists(TFLITE_PATH):
        interpreter = tf.lite.Interpreter(
            model_path=TFLITE_PATH,
            num_threads=4,
        )
        interpreter.allocate_tensors()

        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]

        def predict(batch):
            interpreter.set_tensor(input_index, batch)
            interpreter.invoke()
            return interpreter.get_tensor(output_index)[0]

        print("Inference : TFLite")
    else:
        print(
            "Inference : Keras (sign_model.tflite absent, "
            "relance train.py pour le generer)"
        )
        model = tf.keras.models.load_model(MODEL_PATH)

        @tf.function
        def _forward(batch):
            return model(batch, training=False)

        def predict(batch):
            return _forward(tf.constant(batch)).numpy()[0]

    return predict, class_names


def preprocess_for_model(crop_bgr):
    """Normalise le crop deja produit par HandSegmenter.crop_for_model().

    crop_for_model() a deja fait le redimensionnement en 64x64 (INTER_AREA),
    comme a la preparation du dataset. Il ne reste que BGR -> RGB (les JPEG du
    dataset sont relus en RGB par image_dataset_from_directory) et le /255.
    """
    img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)  # -> (1, 64, 64, 3)


def run_webcam_demo():
    predict, class_names = load_classifier()

    # Reglages GrabCut temps reel : masque calcule en 128 px et 2 iterations
    # au lieu de la pleine resolution et 4. Mesure sur 280 images reelles :
    # precision 0.944 -> 0.939, segmentation 30.6 ms -> 19.1 ms.
    segmenter = HandSegmenter(num_hands=1, **GRABCUT_REALTIME)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Impossible d'ouvrir la webcam.")
        segmenter.close()
        return

    print("Appuie sur 'q' pour quitter.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # effet miroir, plus naturel

            # Meme chaine que prepare_segmented_dataset.py
            cleaned, bbox, found = segmenter.process(frame)

            if found and bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                crop = segmenter.crop_for_model(
                    cleaned,
                    bbox,
                    output_size=IMG_SIZE,
                )

                if crop is not None:
                    predictions = predict(preprocess_for_model(crop))

                    best_idx = int(np.argmax(predictions))
                    confidence = float(predictions[best_idx]) * 100

                    if confidence >= CONFIDENCE_THRESHOLD:
                        text = f"{class_names[best_idx]} ({confidence:.1f}%)"
                        color = (0, 255, 0)
                    else:
                        text = f"? ({confidence:.1f}%)"
                        color = (0, 165, 255)

                    cv2.putText(
                        frame, text, (x1, max(y1 - 15, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2,
                    )

                    # Apercu de ce que le CNN recoit reellement (coin haut droit)
                    preview = cv2.resize(
                        crop, (128, 128), interpolation=cv2.INTER_NEAREST
                    )
                    frame[10:138, frame.shape[1] - 138:frame.shape[1] - 10] = preview
            else:
                cv2.putText(
                    frame, "Aucune main detectee", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
                )

            cv2.imshow("Sign Language Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        segmenter.close()


if __name__ == "__main__":
    run_webcam_demo()
