"""Visualiseur RTMW3D : voir si sa profondeur vaut quelque chose.

Outil de jugement, pas de classification. Le z de MediaPipe nous avait coute
7.8 points et le retirer les avait rendus ; RTMW3D est entraine sur des jeux
3D, donc ce n est pas le meme z — a verifier plutot qu a supposer.

Mesure hors ligne sur 4 clips du dataset, unites ramenees aux metres via la
largeur d epaules :

                        amplitude z / taille de main   gigue image a image
  accident_s10                   0.50                       0.016 m
  accident_s11_632               1.74                       0.041 m
  accident_s11_634               0.64                       0.139 m

Un rapport superieur a 1 est impossible : la profondeur interne a la main y
depasse la taille de la main. Le z du CORPS, lui, se tenait : moins de
0.035 m d ecart entre les deux epaules face camera. D ou cet outil, qui
separe les deux echelles au lieu de donner une moyenne.

ATTENTION : le z est relatif a une racine, ce n est pas une distance a la
camera. Sa valeur absolue ne veut donc pas dire grand chose ; ce qui se teste
est son EVOLUTION. Avancez une main vers la camera et regardez la courbe :
elle doit bouger franchement, dans un sens constant, sans sauts.

Touches : q quitter | c couleur par profondeur | n valeurs | t courbe | h aide
"""

import argparse
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpu_setup import enable_cuda_dlls

enable_cuda_dlls()


# COCO-WholeBody
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
LEFT_HAND = slice(91, 112)
RIGHT_HAND = slice(112, 133)
FACE = slice(23, 91)

BODY_LINKS = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
              (5, 11), (6, 12), (11, 12), (0, 1), (0, 2), (1, 3), (2, 4)]
HAND_LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
              (15, 16), (0, 17), (17, 18), (18, 19), (19, 20),
              (5, 9), (9, 13), (13, 17)]
TIPS = [4, 8, 12, 16, 20]

SCORE_THRESHOLD = 0.3

# Largeur d epaules d un adulte, pour convertir les pixels en metres et
# comparer le z (en metres) a des tailles mesurees a l ecran.
SHOULDER_METRES = 0.38

C_OK = (120, 210, 90)
C_WARN = (60, 150, 235)
C_BAD = (60, 60, 240)
C_DIM = (165, 165, 165)
C_TEXT = (235, 235, 235)
C_PLOT_L = (150, 220, 140)
C_PLOT_R = (120, 190, 240)


def depth_colour(z, lo, hi):
    """Proche = chaud, loin = froid. Rend le z lisible d un coup d oeil."""
    if hi - lo < 1e-9:
        t = 0.5
    else:
        t = float(np.clip((z - lo) / (hi - lo), 0.0, 1.0))
    # t=0 (le plus proche) -> rouge ; t=1 (le plus loin) -> bleu
    return (int(60 + 180 * t), int(90 + 60 * (1 - abs(t - 0.5) * 2)), int(240 - 180 * t))


def panel(frame, lines, x, y, width=250):
    height = 8 + 20 * len(lines)
    region = frame[max(y, 0):y + height, x:x + width]
    if region.size:
        cv2.addWeighted(np.full_like(region, 20), 0.6, region, 0.4, 0, region)
    for index, (text, colour) in enumerate(lines):
        cv2.putText(frame, text, (x + 8, y + 18 + 20 * index),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)


def sparkline(frame, series, x, y, width, height, colour, label):
    """Courbe d une grandeur dans le temps, avec son amplitude ecrite."""
    region = frame[y:y + height, x:x + width]
    if region.size:
        cv2.addWeighted(np.full_like(region, 20), 0.6, region, 0.4, 0, region)
    cv2.putText(frame, label, (x + 6, y + 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, colour, 1, cv2.LINE_AA)
    values = [v for v in series if v is not None]
    if len(values) < 2:
        return
    lo, hi = min(values), max(values)
    span = max(hi - lo, 1e-6)
    points = []
    step = width / max(len(series) - 1, 1)
    for index, value in enumerate(series):
        if value is None:
            continue
        px = int(x + index * step)
        py = int(y + height - 6 - (value - lo) / span * (height - 24))
        points.append((px, py))
    if len(points) >= 2:
        cv2.polylines(frame, [np.array(points, dtype=np.int32)], False,
                      colour, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{span:.3f} m", (x + width - 70, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_DIM, 1, cv2.LINE_AA)


def run(camera):
    from rtmlib import Wholebody3d

    print("chargement de RTMW3D-x (369 Mo)...")
    model = Wholebody3d(backend="onnxruntime", device="cuda")
    provider = model.pose_model.session.get_providers()[0]
    print(f"provider : {provider}")
    if provider != "CUDAExecutionProvider":
        print("  ATTENTION : CPU, ce sera tres lent")
    print(f"z_range : {getattr(model.pose_model, 'z_range', '?')}")
    print("q quitter | c couleur | n valeurs | t courbe | h aide")

    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        print("Impossible d ouvrir la webcam.")
        return

    show_colour = True
    show_numbers = True
    show_plot = True
    show_help = False

    times = deque(maxlen=30)
    left_z = deque(maxlen=160)
    right_z = deque(maxlen=160)
    ratios = deque(maxlen=90)       # amplitude z dans la main / taille de main
    tip_jitter = deque(maxlen=90)
    previous_tips = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            started = time.perf_counter()
            keypoints, scores, _, keypoints_2d = model(frame)
            times.append(time.perf_counter() - started)

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            if len(keypoints) == 0:
                panel(frame, [("aucune personne detectee", C_BAD)], 10, 10, 260)
                left_z.append(None)
                right_z.append(None)
                previous_tips = None
            else:
                # keypoints[..., :2] est dans le repere de l ENTREE du modele ;
                # seul keypoints_2d est en pixels. Dessiner avec le premier
                # donnerait un squelette plausible et faux.
                flat = np.asarray(keypoints_2d)[0].copy()
                flat[:, 0] = width - flat[:, 0]
                z = np.asarray(keypoints)[0][:, 2]
                confidence = np.asarray(scores)[0]

                lo, hi = float(z.min()), float(z.max())

                if show_colour:
                    for a, b in BODY_LINKS:
                        if min(confidence[a], confidence[b]) < SCORE_THRESHOLD:
                            continue
                        cv2.line(frame, tuple(flat[a].astype(int)),
                                 tuple(flat[b].astype(int)),
                                 depth_colour((z[a] + z[b]) / 2, lo, hi), 2, cv2.LINE_AA)
                    for block in (LEFT_HAND, RIGHT_HAND):
                        if confidence[block].mean() < SCORE_THRESHOLD:
                            continue
                        hand, hand_z = flat[block], z[block]
                        for a, b in HAND_LINKS:
                            cv2.line(frame, tuple(hand[a].astype(int)),
                                     tuple(hand[b].astype(int)),
                                     depth_colour((hand_z[a] + hand_z[b]) / 2, lo, hi),
                                     2, cv2.LINE_AA)
                    for index in list(range(17)) + list(range(91, 133)):
                        if confidence[index] >= SCORE_THRESHOLD:
                            cv2.circle(frame, tuple(flat[index].astype(int)), 3,
                                       depth_colour(z[index], lo, hi), -1)

                shoulder_px = float(np.linalg.norm(flat[L_SHOULDER] - flat[R_SHOULDER]))
                metres_per_px = SHOULDER_METRES / max(shoulder_px, 1e-6)

                # --- les trois controles de bon sens ---
                shoulder_gap = float(z[L_SHOULDER] - z[R_SHOULDER])

                rows = []
                for label, block, wrist in (("G", LEFT_HAND, L_WRIST),
                                            ("D", RIGHT_HAND, R_WRIST)):
                    if confidence[block].mean() < SCORE_THRESHOLD:
                        rows.append((f"main {label} : absente", C_DIM))
                        continue
                    hand_z = z[block]
                    hand_xy = flat[block] * metres_per_px
                    size = float(np.linalg.norm(hand_xy.max(axis=0) - hand_xy.min(axis=0)))
                    span = float(hand_z.max() - hand_z.min())
                    ratio = span / max(size, 1e-6)
                    ratios.append(ratio)
                    rows.append((
                        f"main {label}  z {hand_z[0]:+.3f}  amplitude {span:.3f} m",
                        C_TEXT,
                    ))
                    rows.append((
                        f"   amplitude/taille {ratio:.2f}"
                        + ("  IMPOSSIBLE" if ratio > 1.0 else ""),
                        C_BAD if ratio > 1.0 else (C_WARN if ratio > 0.6 else C_OK),
                    ))

                left_z.append(float(z[L_WRIST]) if confidence[L_WRIST] >= SCORE_THRESHOLD else None)
                right_z.append(float(z[R_WRIST]) if confidence[R_WRIST] >= SCORE_THRESHOLD else None)

                tips = z[LEFT_HAND][TIPS].mean() - z[LEFT_HAND][0]
                if previous_tips is not None:
                    tip_jitter.append(abs(float(tips - previous_tips)))
                previous_tips = tips

                lines = [
                    (f"z global  {lo:+.3f} a {hi:+.3f}  ({hi - lo:.3f} m)", C_TEXT),
                    (f"nez {z[NOSE]:+.3f}   epaules {z[L_SHOULDER]:+.3f} / "
                     f"{z[R_SHOULDER]:+.3f}", C_DIM),
                    (f"ecart epaules {shoulder_gap:+.3f} m  (doit etre ~0 de face)",
                     C_OK if abs(shoulder_gap) < 0.05 else C_WARN),
                ]
                if ratios:
                    mean_ratio = float(np.mean(ratios))
                    lines.append((
                        f"amplitude/taille moyenne {mean_ratio:.2f}"
                        + ("  >1 = impossible" if mean_ratio > 1.0 else ""),
                        C_BAD if mean_ratio > 1.0 else (C_WARN if mean_ratio > 0.6 else C_OK),
                    ))
                if tip_jitter:
                    jitter = float(np.mean(tip_jitter))
                    lines.append((
                        f"gigue doigts-poignet {jitter:.4f} m",
                        C_OK if jitter < 0.01 else (C_WARN if jitter < 0.03 else C_BAD),
                    ))
                panel(frame, lines, 10, 10, 400)

                if show_numbers and rows:
                    panel(frame, rows, 10, height - 8 - 20 * len(rows) - 8, 400)

            if show_plot:
                sparkline(frame, list(left_z), width - 270, 100, 260, 70,
                          C_PLOT_L, "z poignet G")
                sparkline(frame, list(right_z), width - 270, 180, 260, 70,
                          C_PLOT_R, "z poignet D")

            fps = 1.0 / max(np.mean(times), 1e-6)
            panel(frame, [(f"{fps:4.1f} fps  {np.mean(times)*1000:3.0f} ms", C_DIM),
                          ("RTMW3D-x", C_DIM)], width - 185, 10, 175)

            if show_help:
                panel(frame, [("c  couleur par z", C_DIM), ("n  valeurs", C_DIM),
                              ("t  courbes", C_DIM), ("q  quitter", C_DIM)],
                      width - 185, 260, 175)

            cv2.imshow("RTMW3D - profondeur", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                show_colour = not show_colour
            if key == ord("n"):
                show_numbers = not show_numbers
            if key == ord("t"):
                show_plot = not show_plot
            if key == ord("h"):
                show_help = not show_help
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    run(args.camera)
