"""Demo webcam pour la reconnaissance de mots (WLASL50).

Fonctionnement : on enregistre une fenetre de ~2.8 s a la demande, puis on
classe la sequence entiere. On ne classe pas en continu, car segmenter un
flux de signes est un probleme a part entiere : savoir OU commence et finit
un signe est plus difficile que de reconnaitre le signe lui-meme.

Point critique : les features sont produites par frame_features() importee
de extract_landmarks.py, la fonction qui a servi a l'entrainement. Les
reimplementer ici serait reintroduire exactement le bug qui, sur l'alphabet,
faisait chuter la precision de 0.985 a 0.384 — un pretraitement different
entre entrainement et inference.

Precision attendue : environ 0.37 en top-1 sur 50 classes (hasard : 0.02),
mesuree en validation croisee par signeur. On affiche donc le top-5, qui est
bien plus informatif a ce niveau de precision. Et ce chiffre vient de
signeurs du dataset : sur une personne nouvelle, devant une webcam, attendez
nettement moins.
"""

import json
import os
import sys
import time

import cv2
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mediapipe as mp

from extract_landmarks import (
    FEATURE_DIM,
    HAND_POINTS,
    POSE_KEYPOINTS,
    _new_landmarker,
    frame_features,
)
from train_kfold import MAX_LEN, pad_sequence


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "wlasl_gru.keras")
META_PATH = os.path.join(PROJECT_ROOT, "models", "wlasl_gru.meta.json")

# Les clips d'entrainement durent 2.8 s en mediane. On enregistre la meme
# duree en secondes plutot qu'un nombre fixe de frames : la webcam peut
# tourner a 30 fps la ou le dataset est a 25, et c'est la duree du geste qui
# doit correspondre, pas le compte de frames.
RECORD_SECONDS = 2.8

TOP_K = 5

# Squelette de main (meme table que la demo alphabet).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# Paires de pose a relier : epaules, bras.
POSE_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

COLOR_BONE = (150, 220, 140)
COLOR_JOINT = (60, 90, 235)
COLOR_POSE = (210, 180, 90)
COLOR_REC = (60, 60, 240)
COLOR_OK = (120, 210, 90)
COLOR_DIM = (160, 160, 160)


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(
            f"Modele absent : {MODEL_PATH}\n"
            "Lance d'abord : python src/wlasl/train_final.py"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(META_PATH, encoding="utf-8") as handle:
        meta = json.load(handle)

    @tf.function
    def predict(batch):
        return model(batch, training=False)

    return predict, meta


def draw_skeleton(frame, result):
    height, width = frame.shape[:2]

    pose = result.pose_landmarks
    if pose:
        for a, b in POSE_CONNECTIONS:
            if a < len(pose) and b < len(pose):
                pa = (int(pose[a].x * width), int(pose[a].y * height))
                pb = (int(pose[b].x * width), int(pose[b].y * height))
                cv2.line(frame, pa, pb, COLOR_POSE, 2, cv2.LINE_AA)

    for hand in (result.left_hand_landmarks, result.right_hand_landmarks):
        if not hand:
            continue
        points = [(int(p.x * width), int(p.y * height)) for p in hand]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, points[a], points[b], COLOR_BONE, 2, cv2.LINE_AA)
        for index, point in enumerate(points):
            cv2.circle(frame, point, 5 if index == 0 else 3, COLOR_JOINT, -1, cv2.LINE_AA)


def draw_panel(frame, lines, title=None):
    height, width = frame.shape[:2]
    panel_height = 34 + 30 * len(lines)
    top = height - panel_height

    region = frame[top:height, 0:width]
    cv2.addWeighted(np.full_like(region, 22), 0.65, region, 0.35, 0, region)
    cv2.line(frame, (0, top), (width, top), COLOR_BONE, 1)

    y = top + 26
    if title:
        cv2.putText(frame, title, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, COLOR_DIM, 1, cv2.LINE_AA)
        y += 28

    for text, colour, scale in lines:
        cv2.putText(frame, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, colour, 2, cv2.LINE_AA)
        y += 30


def classify(predict, sequence, classes):
    padded = pad_sequence(np.array(sequence, dtype=np.float32))
    probabilities = predict(tf.constant(padded[None, ...])).numpy()[0]

    order = np.argsort(probabilities)[::-1][:TOP_K]
    return [(classes[i], float(probabilities[i])) for i in order]


def run():
    predict, meta = load_model()
    classes = meta["classes"]

    expected = meta.get("expected_accuracy") or {}
    print(f"Modele : {meta['num_classes']} signes, "
          f"precision attendue {expected.get('mean', '?')} "
          f"(+-{expected.get('std', '?')}, validation croisee par signeur)")
    print("ESPACE : enregistrer un signe  |  l : squelette  |  q : quitter")
    print(f"\nSignes connus :\n  {', '.join(classes)}\n")

    landmarker = _new_landmarker()
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        print("Impossible d'ouvrir la webcam.")
        landmarker.close()
        return

    recording = False
    started_at = 0.0
    sequence = []
    results_top = []
    show_skeleton = True
    timestamp = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)

            timestamp += 33
            result = landmarker.detect_for_video(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                ),
                timestamp,
            )

            if show_skeleton:
                draw_skeleton(frame, result)

            person_visible = bool(result.pose_landmarks)

            if recording:
                features, found = frame_features(result)
                sequence.append(features)

                elapsed = time.time() - started_at
                remaining = max(RECORD_SECONDS - elapsed, 0.0)

                # Barre de progression de l'enregistrement.
                width = frame.shape[1]
                filled = int(width * min(elapsed / RECORD_SECONDS, 1.0))
                cv2.rectangle(frame, (0, 0), (filled, 8), COLOR_REC, -1)
                cv2.putText(frame, f"ENREGISTREMENT  {remaining:.1f}s",
                            (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            COLOR_REC, 2, cv2.LINE_AA)

                if elapsed >= RECORD_SECONDS:
                    recording = False
                    if len(sequence) >= 8:
                        results_top = classify(predict, sequence, classes)
                    else:
                        results_top = []
                    sequence = []

            if results_top:
                lines = []
                for rank, (word, score) in enumerate(results_top):
                    colour = COLOR_OK if rank == 0 else COLOR_DIM
                    scale = 0.85 if rank == 0 else 0.6
                    lines.append((f"{rank + 1}. {word:<14} {score * 100:5.1f}%",
                                  colour, scale))
                draw_panel(frame, lines, "top 5")
            elif not recording:
                hint = ("ESPACE pour enregistrer" if person_visible
                        else "place-toi dans le cadre (buste visible)")
                colour = COLOR_OK if person_visible else COLOR_REC
                draw_panel(frame, [(hint, colour, 0.7)])

            cv2.imshow("WLASL50 - reconnaissance de mots", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("l"):
                show_skeleton = not show_skeleton
            if key == ord(" ") and not recording and person_visible:
                recording = True
                started_at = time.time()
                sequence = []
                results_top = []
    finally:
        capture.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    run()
