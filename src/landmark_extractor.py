"""Extraction des landmarks MediaPipe vers un CSV, pour le modele landmark.

Pourquoi un CSV separe plutot que de relancer MediaPipe a l'entrainement :
l'extraction coute ~86 ms par image (~103 min pour les 72k images en
mono-processus), alors que l'entrainement sur les coordonnees ne prend que
quelques secondes. On paie donc l'extraction une seule fois.

On traite les images RAW correspondant aux images segmentees existantes :
les deux branches du modele de fusion voient ainsi exactement les memes
echantillons, avec le meme nom de fichier comme cle.
"""

import csv
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(
    PROJECT_ROOT, "data", "raw", "asl_alphabet_train", "asl_alphabet_train"
)

SEGMENTED_DIR = os.path.join(
    PROJECT_ROOT, "data", "processed", "asl_alphabet_segmented"
)

OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "landmarks", "landmarks.csv")

HAND_LANDMARKER_PATH = os.path.join(
    PROJECT_ROOT, "models", "hand_landmarker.task"
)

NUM_LANDMARKS = 21

# MediaPipe ne detecte pas de main sur ~16% des images d'entrainement.
# Abaisser ce seuil recupere quelques pourcents (0.5 -> 0.839, 0.3 -> 0.862).
MIN_DETECTION_CONFIDENCE = 0.3

# Un HandLandmarker par processus : l'objet n'est ni picklable ni partageable.
_landmarker = None


def _init_worker():
    global _landmarker
    _landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=os.path.abspath(HAND_LANDMARKER_PATH)
            ),
            # IMAGE et non VIDEO : chaque image est un echantillon independant.
            # En mode VIDEO le tracking d'une image deborderait sur la suivante.
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
        )
    )


def extract_landmarks_from_image(image_path, landmarker=None):
    """Retourne un tableau (21, 3) de coordonnees (x, y, z), ou None.

    x et y sont normalises par MediaPipe entre 0 et 1 relativement a l'image ;
    z est une profondeur relative au poignet.
    """
    detector = landmarker or _landmarker

    if detector is None:
        raise RuntimeError(
            "Aucun HandLandmarker : passe-en un, ou utilise "
            "extract_landmarks_from_dataset()."
        )

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    result = detector.detect(
        mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        )
    )

    if not result.hand_landmarks:
        return None

    return np.array(
        [[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]],
        dtype=np.float32,
    )


def normalize_landmarks(points):
    """Rend les coordonnees invariantes a la position et a la distance.

    Sans cette etape, bouger la main dans le cadre ou s'approcher de la
    camera change toutes les valeurs et le modele n'apprend rien de stable.

    - origine au poignet (landmark 0)
    - echelle divisee par la distance maximale au poignet
    """
    points = np.asarray(points, dtype=np.float32).reshape(NUM_LANDMARKS, 3)

    centered = points - points[0]

    scale = np.linalg.norm(centered, axis=1).max()

    if scale < 1e-6:
        return centered

    return centered / scale


def _process_one(args):
    class_name, filename, path = args
    points = extract_landmarks_from_image(path)

    if points is None:
        return class_name, filename, None

    return class_name, filename, points.reshape(-1).tolist()


def _collect_targets():
    """Images RAW correspondant aux images segmentees existantes."""
    targets = []

    for class_name in sorted(os.listdir(SEGMENTED_DIR)):
        seg_dir = os.path.join(SEGMENTED_DIR, class_name)

        if not os.path.isdir(seg_dir):
            continue

        for filename in os.listdir(seg_dir):
            raw_path = os.path.join(RAW_DIR, class_name, filename)

            if os.path.exists(raw_path):
                targets.append((class_name, filename, raw_path))

    return targets


def extract_landmarks_from_dataset(output_csv=OUTPUT_CSV, workers=8):
    """Extrait les landmarks de tout le dataset vers un CSV."""
    targets = _collect_targets()

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    header = (
        ["class", "filename"]
        + [f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")]
    )

    found = 0
    missing = 0

    print(f"{len(targets)} images a traiter, {workers} processus.")

    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)

        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker
        ) as pool:
            for index, (class_name, filename, coords) in enumerate(
                pool.map(_process_one, targets, chunksize=64), start=1
            ):
                if coords is None:
                    missing += 1
                else:
                    found += 1
                    writer.writerow([class_name, filename] + coords)

                if index % 5000 == 0:
                    print(
                        f"  {index}/{len(targets)} — "
                        f"{found} avec main, {missing} sans",
                        flush=True,
                    )

    print(
        f"\nTermine : {found} landmarks ecrits, {missing} images sans main "
        f"detectee ({found / max(len(targets), 1):.1%} de reussite)."
    )
    print(f"CSV : {output_csv}")

    return found, missing


if __name__ == "__main__":
    extract_landmarks_from_dataset()
