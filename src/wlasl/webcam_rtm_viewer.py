"""Visualiseur RTMPose en direct : tout ce que le modele voit.

Outil de diagnostic, pas de classification. Il affiche les 133 points du
corps entier, les confiances par region, et les scalaires derives qui
servent a l entrainement — de quoi juger si la detection tient quand la
main pivote, ce que MediaPipe ne faisait pas.

Mesure sur 20 clips du dataset :

                    >=1 main   2 mains   gigue
  MediaPipe             0.61      0.31   0.0266
  RTMPose corps entier  0.96      0.87   0.0220

Touches : q quitter | t seuil | p boite | f visage | n valeurs | s squelette | h aide
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
from hand_identity import inside_fraction, resolve

enable_cuda_dlls()


# COCO-WholeBody : 0-16 corps, 17-22 pieds, 23-90 visage,
#                  91-111 main gauche, 112-132 main droite
BODY = slice(0, 17)
FEET = slice(17, 23)
FACE = slice(23, 91)
LEFT_HAND = slice(91, 112)
RIGHT_HAND = slice(112, 133)

BODY_LINKS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),        # epaules et bras
    (5, 11), (6, 12), (11, 12),                      # torse
    (0, 1), (0, 2), (1, 3), (2, 4),                  # tete
]

HAND_LINKS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

FINGERS = [("pouce", [1, 2, 3, 4]), ("index", [5, 6, 7, 8]),
           ("majeur", [9, 10, 11, 12]), ("annul", [13, 14, 15, 16]),
           ("auric", [17, 18, 19, 20])]

SCORE_THRESHOLD = 0.3

# Poignets de pose, qui tranchent quand les deux mains sont predites au meme
# endroit, et seuil de recouvrement en largeurs d epaules.
POSE_LEFT_WRIST = 9
POSE_RIGHT_WRIST = 10
# RTMPose est top-down : la boite personne est ramenee a l entree du modele,
# qui a appris sur des corps ENTIERS remplissant cette boite. Un buste en
# webcam viole cette relation d echelle. Prolonger la boite vers le bas la
# retablit, au prix d une main plus petite dans l entree.
PAD_FACTORS = (1.0, 1.5, 2.0, 2.5)

# Seuil de confiance au-dessus duquel une main compte comme presente. Celui
# du dessin reste a SCORE_THRESHOLD : on veut voir les points meme douteux,
# mais ne PAS les donner au modele.
#
#   seuil   mains gardees   gigue du repliement
#   0.3          91%            0.0688
#   0.5          76%            0.0565
#   0.7          60%            0.0395   <- retenu pour l extraction
#   0.8          49%            0.0273
#   MediaPipe    47%            0.0444
PRESENCE_THRESHOLDS = (0.3, 0.5, 0.7, 0.8)

# Part minimale des 21 points qui doit tomber dans l image pour qu une main
# compte comme presente.
INSIDE_THRESHOLD = 0.5

# BGR
C_BODY = (210, 180, 90)
C_HAND_L = (150, 220, 140)
C_HAND_R = (120, 190, 240)
C_JOINT = (60, 90, 235)
C_FACE = (170, 170, 170)
C_OK = (120, 210, 90)
C_WARN = (60, 150, 235)
C_BAD = (60, 60, 240)
C_DIM = (165, 165, 165)
C_TEXT = (235, 235, 235)


def curl(hand, chain):
    """Repliement : bout-a-base divise par la longueur deployee.

    1.0 = doigt tendu, ~0.3 = replie. Invariant a la rotation de la main,
    contrairement aux coordonnees brutes — c est ce qui en fait la feature
    derivee la plus utile.
    """
    straight = np.linalg.norm(hand[chain[-1]] - hand[chain[0]])
    along = sum(np.linalg.norm(hand[chain[i + 1]] - hand[chain[i]])
                for i in range(len(chain) - 1))
    return straight / max(along, 1e-6)


def stretch_box(box, factor):
    """Prolonge la boite vers le bas, le haut du corps restant en place."""
    x1, y1, x2, y2 = box[:4]
    return np.array([x1, y1, x2, y1 + (y2 - y1) * factor], dtype=np.float32)


class CurlJitter:
    """Variation image a image du repliement, moyennee sur une fenetre.

    Une main reelle bouge de facon lisse : ce qui varie vite est du bruit
    d estimation. C est la grandeur qui separe les deux modes (S/B 2.80
    contre 4.01 sur les clips du dataset), et celle dont dependait le gain
    le plus important du projet, les features de forme de main.
    """

    def __init__(self, window=60):
        self.previous = None
        self.deltas = deque(maxlen=window)
        self.levels = deque(maxlen=window)

    def update(self, curls):
        if curls is None:
            self.previous = None
            return
        if self.previous is not None:
            self.deltas.append(float(np.abs(curls - self.previous).mean()))
        self.levels.append(curls)
        self.previous = curls

    @property
    def jitter(self):
        return float(np.mean(self.deltas)) if self.deltas else float("nan")

    @property
    def ratio(self):
        """Etalement sur gigue : le rapport signal/bruit de la feature."""
        if len(self.levels) < 5 or not self.deltas:
            return float("nan")
        spread = float(np.stack(self.levels).std(axis=0).mean())
        return spread / max(self.jitter, 1e-9)


class Hysteresis:
    """Ne change d avis qu apres N frames d accord.

    Sans cela l affichage papillonne : la decision se prend image par image,
    et une quasi-egalite suffit a la faire basculer a chaque frame.
    """

    def __init__(self, state=False, frames=3):
        self.state = state
        self.frames = frames
        self.pending = 0

    def update(self, observed):
        if observed == self.state:
            self.pending = 0
        else:
            self.pending += 1
            if self.pending >= self.frames:
                self.state = observed
                self.pending = 0
        return self.state


def draw_links(frame, points, scores, links, colour, thickness=2):
    for a, b in links:
        if scores[a] < SCORE_THRESHOLD or scores[b] < SCORE_THRESHOLD:
            continue
        cv2.line(frame, tuple(points[a].astype(int)), tuple(points[b].astype(int)),
                 colour, thickness, cv2.LINE_AA)


def panel(frame, lines, x, y, width=250):
    """Bandeau semi-transparent avec des lignes (texte, couleur)."""
    height = 8 + 20 * len(lines)
    region = frame[max(y, 0):y + height, x:x + width]
    if region.size:
        cv2.addWeighted(np.full_like(region, 20), 0.6, region, 0.4, 0, region)

    for index, (text, colour) in enumerate(lines):
        cv2.putText(frame, text, (x + 8, y + 18 + 20 * index),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)


def run(mode, camera, log_path=None):
    from rtmlib import Wholebody

    model = Wholebody(mode=mode, backend="onnxruntime", device="cuda")
    provider = model.pose_model.session.get_providers()[0]

    print(f"RTMPose {mode} | {provider}")
    if provider != "CUDAExecutionProvider":
        print("  ATTENTION : CPU, environ 20x plus lent")
    print("q quitter | t seuil | p boite | f visage | n valeurs | s squelette | h aide")

    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        print("Impossible d ouvrir la webcam.")
        return

    show_face = True
    show_numbers = True
    show_skeleton = True
    show_help = False

    times = deque(maxlen=30)
    seen = deque(maxlen=120)          # taux de detection des deux mains

    left_hold = Hysteresis()
    right_hold = Hysteresis()


    jitter_track = {"G": CurlJitter(), "D": CurlJitter()}

    pad_index = 0
    presence_index = 0
    coverage = deque(maxlen=240)      # mains vues, pour le reglage courant

    def forget():
        """Les deux mesures dependent du reglage : on repart a zero pour ne
        pas melanger deux reglages dans une seule moyenne."""
        coverage.clear()
        for track in jitter_track.values():
            track.deltas.clear()
            track.levels.clear()
            track.previous = None

    log = None
    if log_path:
        log = open(log_path, "w", encoding="utf-8")
        print("t,left_score,right_score,gap,d_ll,d_lr,d_rr,d_rl,"
              "wrist_l,wrist_r,left_claim,right_claim,conflict,"
              "left_inside,right_inside,"
              "raw_left,raw_right,left_ok,right_ok", file=log)
        print(f"journal : {log_path}")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            # L inference tourne sur l image NON retournee : sur une image
            # miroir, RTMPose voit une personne inversee et echange ses
            # etiquettes gauche/droite. On deduit d abord, puis on retourne
            # l image ET les coordonnees x pour l affichage, ce qui garde
            # l effet miroir sans fausser les etiquettes.
            started = time.perf_counter()
            boxes = model.det_model(frame)
            box = None
            if len(boxes):
                box = stretch_box(boxes[0], PAD_FACTORS[pad_index])
                keypoints, scores = model.pose_model(frame, bboxes=[box])
            else:
                keypoints, scores = [], []
            times.append(time.perf_counter() - started)

            frame = cv2.flip(frame, 1)
            if box is not None:
                box = np.array([frame.shape[1] - box[2], box[1],
                                frame.shape[1] - box[0], box[3]])
            if len(keypoints):
                keypoints = np.asarray(keypoints).copy()
                keypoints[..., 0] = frame.shape[1] - keypoints[..., 0]

            height, width = frame.shape[:2]

            if len(keypoints) == 0:
                panel(frame, [("aucune personne detectee", C_BAD)], 10, 10, 260)
                seen.append(0)
            else:
                points = np.asarray(keypoints)[0]
                confidence = np.asarray(scores)[0]

                left_score = float(confidence[LEFT_HAND].mean())
                right_score = float(confidence[RIGHT_HAND].mean())
                body_score = float(confidence[BODY].mean())
                face_score = float(confidence[FACE].mean())

                presence = PRESENCE_THRESHOLDS[presence_index]
                raw_left = left_score >= presence
                raw_right = right_score >= presence

                shoulders = np.linalg.norm(points[5] - points[6])
                scale = max(shoulders, 1e-6)

                gap = float(np.linalg.norm(
                    points[LEFT_HAND] - points[RIGHT_HAND], axis=1).mean()) / scale
                lh, rh = points[LEFT_HAND][0], points[RIGHT_HAND][0]
                d_ll = float(np.linalg.norm(lh - points[POSE_LEFT_WRIST]) / scale)
                d_lr = float(np.linalg.norm(lh - points[POSE_RIGHT_WRIST]) / scale)
                d_rr = float(np.linalg.norm(rh - points[POSE_RIGHT_WRIST]) / scale)
                d_rl = float(np.linalg.norm(rh - points[POSE_LEFT_WRIST]) / scale)
                wrist_scores = (float(confidence[POSE_LEFT_WRIST]),
                                float(confidence[POSE_RIGHT_WRIST]))

                # RTMPose rend toujours 21 points par main, meme quand une
                # seule est levee : les points de la main absente se posent
                # alors sur celle qui est visible.
                #
                # On ne demande pas "les deux jeux se recouvrent-ils" : un
                # recouvrement lache passe sous n importe quel seuil, et en ASL
                # les mains se touchent pour de vrai. On demande "quel poignet
                # de pose chaque jeu revendique-t-il", les poignets etant
                # estimes separement par le squelette de pose. Si les deux
                # revendiquent le MEME, l un est une invention : celui dont le
                # nom ne correspond pas au poignet revendique. Le depart est
                # donc determine, et non un arbitrage entre deux distances
                # presque egales — c est cela qui faisait alterner l etiquette
                # d une frame a l autre.
                # Une main dont la plupart des points sont hors cadre n est
                # pas exploitable : on la traite comme absente plutot que de
                # nourrir le modele avec des coordonnees extrapolees.
                left_inside = float(inside_fraction(points[LEFT_HAND], width, height))
                right_inside = float(inside_fraction(points[RIGHT_HAND], width, height))

                if left_inside < INSIDE_THRESHOLD:
                    raw_left = False
                if right_inside < INSIDE_THRESHOLD:
                    raw_right = False

                left_claim = 0 if d_ll <= d_lr else 1
                right_claim = 1 if d_rr <= d_rl else 0

                raw_left, raw_right, conflict = resolve(
                    raw_left, raw_right, lh, rh,
                    points[POSE_LEFT_WRIST], points[POSE_RIGHT_WRIST],
                )

                left_ok = left_hold.update(raw_left)
                right_ok = right_hold.update(raw_right)
                merged = conflict

                if log:
                    print(
                        f"{time.time():.3f},{left_score:.4f},{right_score:.4f},"
                        f"{gap:.4f},{d_ll:.4f},{d_lr:.4f},{d_rr:.4f},{d_rl:.4f},"
                        f"{wrist_scores[0]:.4f},{wrist_scores[1]:.4f},"
                        f"{left_claim},{right_claim},{int(conflict)},"
                        f"{left_inside:.3f},{right_inside:.3f},"
                        f"{int(raw_left)},{int(raw_right)},"
                        f"{int(left_ok)},{int(right_ok)}",
                        file=log,
                    )

                seen.append(1 if (left_ok and right_ok) else 0)

                coverage.append(1 if left_ok else 0)
                coverage.append(1 if right_ok else 0)

                if box is not None and show_skeleton:
                    cv2.rectangle(frame, (int(box[0]), int(box[1])),
                                  (int(box[2]), int(box[3])), C_DIM, 1)

                if show_skeleton:
                    if show_face and face_score >= SCORE_THRESHOLD:
                        for point in points[FACE].astype(int):
                            cv2.circle(frame, tuple(point), 1, C_FACE, -1)

                    draw_links(frame, points[BODY], confidence[BODY], BODY_LINKS, C_BODY, 2)
                    for index in range(17):
                        if confidence[index] >= SCORE_THRESHOLD:
                            cv2.circle(frame, tuple(points[index].astype(int)), 3, C_JOINT, -1)

                    for block, colour, visible in (
                        (LEFT_HAND, C_HAND_L, left_ok),
                        (RIGHT_HAND, C_HAND_R, right_ok),
                    ):
                        if not visible:
                            continue
                        hand, hand_scores = points[block], confidence[block]
                        draw_links(frame, hand, hand_scores, HAND_LINKS, colour, 2)
                        for index, point in enumerate(hand):
                            if hand_scores[index] >= SCORE_THRESHOLD:
                                cv2.circle(frame, tuple(point.astype(int)),
                                           4 if index == 0 else 2, C_JOINT, -1)

                # --- confiances ---
                def grade(value):
                    return C_OK if value > 0.6 else (C_WARN if value > 0.3 else C_BAD)

                rate = sum(seen) / max(len(seen), 1)
                lines = [
                    (f"corps   {body_score:.2f}", grade(body_score)),
                    (f"visage  {face_score:.2f}", grade(face_score)),
                    (f"main G  {left_score:.2f}", grade(left_score)),
                    (f"main D  {right_score:.2f}", grade(right_score)),
                    (f"2 mains {rate:.0%} des frames", grade(rate)),
                ]
                lines.append((
                    f"seuil {presence:.1f}  boite x{PAD_FACTORS[pad_index]}  "
                    f"vues {sum(coverage) / max(len(coverage), 1):.0%}",
                    C_TEXT,
                ))
                if merged:
                    lines.append(("main inventee ecartee", C_WARN))
                for label, part in (("G", left_inside), ("D", right_inside)):
                    if part < 1.0:
                        lines.append((
                            f"main {label} hors cadre {1.0 - part:.0%}",
                            C_BAD if part < INSIDE_THRESHOLD else C_WARN,
                        ))
                panel(frame, lines, 10, 10, 230)

                # --- scalaires derives, ceux qui servent a l entrainement ---
                if show_numbers and body_score >= SCORE_THRESHOLD:
                    centre = (points[5] + points[6]) / 2.0

                    rows = []
                    for label, block, visible in (("G", LEFT_HAND, left_ok),
                                                  ("D", RIGHT_HAND, right_ok)):
                        if not visible:
                            jitter_track[label].update(None)
                            rows.append((f"main {label} : absente", C_DIM))
                            continue
                        hand = points[block]
                        values = np.array([curl(hand, c) for _, c in FINGERS])
                        jitter_track[label].update(values)
                        curls = " ".join(f"{v:.2f}" for v in values)
                        rows.append((f"main {label} repli {curls}", C_TEXT))
                        palm = np.linalg.norm(hand[5] - hand[17]) / scale
                        direction = hand[9] - hand[0]
                        angle = np.degrees(np.arctan2(-direction[1], direction[0]))
                        rows.append((f"   paume {palm:.2f}  angle {angle:+.0f}deg", C_DIM))
                        rows.append((f"   au buste {np.linalg.norm(hand[0]-centre)/scale:.2f}"
                                     f"  au nez {np.linalg.norm(hand[0]-points[0])/scale:.2f}", C_DIM))

                    if left_ok and right_ok:
                        gap = np.linalg.norm(points[LEFT_HAND][0] - points[RIGHT_HAND][0]) / scale
                        rows.append((f"ecart mains {gap:.2f}", C_TEXT))

                    claimed = ("G" if left_claim == 0 else "D") + ("G" if right_claim == 0 else "D")
                    rows.append((f"ecart 21 pts {gap:.3f}   revendique {claimed}",
                                 C_BAD if conflict else C_DIM))
                    rows.append((f"main G -> poignet G {d_ll:.2f}  D {d_lr:.2f}", C_DIM))
                    rows.append((f"main D -> poignet D {d_rr:.2f}  G {d_rl:.2f}", C_DIM))
                    rows.append((f"conf poignets G {wrist_scores[0]:.2f} D {wrist_scores[1]:.2f}",
                                 C_DIM))
                    for label in ("G", "D"):
                        track = jitter_track[label]
                        if track.deltas:
                            rows.append((
                                f"gigue repli {label} {track.jitter:.4f}   "
                                f"S/B {track.ratio:.2f}",
                                C_TEXT,
                            ))
                    rows.append((f"epaules {shoulders:.0f}px (= echelle 1.0)", C_DIM))
                    panel(frame, rows, 10, height - 8 - 20 * len(rows) - 8, 330)

            fps = 1.0 / max(np.mean(times), 1e-6)
            panel(frame, [(f"{fps:4.1f} fps   {np.mean(times)*1000:3.0f} ms", C_DIM),
                          (f"mode {mode}", C_DIM),
                          ("comparer les modes", C_DIM),
                          ("entre eux", C_DIM)],
                  width - 185, 10, 175)

            if show_help:
                panel(frame, [("t  seuil de presence", C_DIM),
                              ("p  cadrage de la boite", C_DIM), ("f  visage", C_DIM),
                              ("n  valeurs", C_DIM), ("s  squelette", C_DIM),
                              ("q  quitter", C_DIM)],
                      width - 185, 40, 175)

            cv2.imshow("RTMPose - corps entier", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("f"):
                show_face = not show_face
            if key == ord("n"):
                show_numbers = not show_numbers
            if key == ord("s"):
                show_skeleton = not show_skeleton
            if key == ord("h"):
                show_help = not show_help
            if key == ord("p"):
                pad_index = (pad_index + 1) % len(PAD_FACTORS)
                forget()
                print(f"boite x{PAD_FACTORS[pad_index]}")
            if key == ord("t"):
                presence_index = (presence_index + 1) % len(PRESENCE_THRESHOLDS)
                forget()
                print(f"seuil de presence {PRESENCE_THRESHOLDS[presence_index]}")
    finally:
        if log:
            log.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="balanced",
                        choices=("performance", "lightweight", "balanced"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--log", default=None,
                        help="journal CSV par frame, pour regler les seuils sur "
                             "la vraie camera plutot que sur les clips du dataset")
    args = parser.parse_args()
    run(args.mode, args.camera, args.log)
