"""Extrait les sequences de landmarks avec RTMPose (corps entier) sur GPU.

Pourquoi : le modele de main de MediaPipe se degrade des que la main pivote,
ce qui arrive en permanence en ASL, et RTMPose trouve les mains beaucoup plus
souvent (couverture 84% contre 47%).

ATTENTION a ce que cette couverture veut dire. Ces chiffres mesurent si une
main est TROUVEE, pas si elle est BIEN ESTIMEE, et les confondre a coute 14.9
points : a seuil 0.3, les frames que MediaPipe rejetait sont 3.9x plus
bruitees, et sa faible couverture etait en realite un filtre de qualite. Sur
les MEMES frames les deux detecteurs se valent (S/B 2.60 contre 2.59).

Le seuil de presence est donc ce qui compte ici, pas le detecteur. Voir
HAND_SCORE_THRESHOLD plus bas et docs/WLASL.md section 7.

Sortie identique a extract_landmarks.py : un .npz par clip, (T, 104).

Le modele utilise ici (Wholebody, RTMW-x) ne rend que des coordonnees 2D.
rtmlib expose aussi Wholebody3d (RTMW3D-x), qui rend bien un z : ce n est
donc pas une limite de RTMPose mais le choix du modele. Mesure sur 4 clips,
avec les unites ramenees aux metres via la largeur d epaules :

                                      amplitude z / taille de main   gigue
  accident_s10                                   0.50              0.016 m
  accident_s11_632                               1.74              0.041 m
  accident_s11_634                               0.64              0.139 m

Un rapport de 1.74 signifie une profondeur interne a la main plus grande
que la main elle-meme, ce qui est impossible, et 0.139 m de gigue image a
image sur une grandeur qui devrait etre lisse rappelle exactement ce qui
avait disqualifie le z de MediaPipe (+7.8 points en le retirant). Le z du
CORPS, lui, se tient : l ecart de profondeur entre les deux epaules reste
sous 0.035 m face camera. Une piste non testee est donc de ne garder le z
que sur les 9 points de pose, jamais sur les mains.

Les 104 dimensions correspondent a la configuration 2D actuelle.
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

from gpu_setup import enable_cuda_dlls
from hand_identity import inside_fraction, resolve

enable_cuda_dlls()


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WLASL_DIR = os.path.join(PROJECT_ROOT, "data", "wlasl")
MANIFEST_PATH = os.path.join(WLASL_DIR, "download_manifest.json")
FEATURES_DIR = os.path.join(WLASL_DIR, "features_rtm")
REPORT_PATH = os.path.join(WLASL_DIR, "extract_rtm_report.json")

# COCO-WholeBody : 0-16 corps, 17-22 pieds, 23-90 visage,
#                  91-111 main gauche, 112-132 main droite
LEFT_HAND = slice(91, 112)
RIGHT_HAND = slice(112, 133)

# Equivalents COCO des points de pose retenus pour MediaPipe :
# nez, epaules, coudes, poignets, hanches.
POSE_KEYPOINTS = [0, 5, 6, 7, 8, 9, 10, 11, 12]
COCO_LEFT_SHOULDER = 5
COCO_RIGHT_SHOULDER = 6
COCO_LEFT_WRIST = 9
COCO_RIGHT_WRIST = 10

# Part minimale des 21 points qui doit tomber dans l image pour qu une main
# compte comme presente. RTMPose place volontiers des points hors cadre.
INSIDE_THRESHOLD = 0.5

HAND_POINTS = 21
FEATURE_DIM = HAND_POINTS * 2 * 2 + 2 + len(POSE_KEYPOINTS) * 2   # 104

# Confiance moyenne au-dessus de laquelle une main est consideree presente.
#
# 0.3 etait un mauvais choix, et il a coute 14.9 points. A ce seuil RTMPose
# annonce une main dans 84% des frames contre 47% pour MediaPipe, mais ces
# 37 points supplementaires sont 3.9x plus bruites (gigue du repliement
# 0.1408 contre 0.0359). La faible couverture de MediaPipe n etait pas une
# faiblesse : c etait un filtre de qualite, et le remplacer par 0.3 revenait
# a nourrir le modele de mains inexploitables.
#
# Sur les MEMES frames les deux detecteurs se valent (S/B 2.60 contre 2.59),
# donc le detecteur n etait pas le probleme, le seuil l etait.
#
#   seuil   mains gardees   gigue
#   0.3          91.0%     0.0688
#   0.6          66.4%     0.0478
#   0.7          60.2%     0.0395   <- domine MediaPipe sur les deux axes
#   0.8          48.6%     0.0273
#   MediaPipe    47.0%     0.0444
HAND_SCORE_THRESHOLD = 0.7
POSE_SCORE_THRESHOLD = 0.3

MIN_SECONDS = 0.4
MAX_SECONDS = 10.0
MAX_FRAMES = 150
MIN_DETECT_RATIO = 0.3


def window_for(record, real_fps, frame_count):
    """(debut, fin) dans le referentiel du fichier telecharge.

    Les intervalles du metadata sont indexes sur la video d ORIGINE, et 39%
    des clips coupes ont ete reencodes a 30 fps alors que le metadata annonce
    25 : on convertit donc en secondes avant de revenir aux frames.
    """
    if record["frame_end"] == -1:
        return 0, int(frame_count) - 1

    meta_fps = record["fps"] or 25
    start = max(int(round((record["frame_start"] - 1) / meta_fps * real_fps)), 0)
    end = min(int(round(record["frame_end"] / meta_fps * real_fps)), int(frame_count) - 1)

    if end <= start:
        return 0, int(frame_count) - 1

    return start, end


def hand_confidences(scores):
    """(confiance main gauche, confiance main droite)."""
    return (float(np.asarray(scores)[LEFT_HAND].mean()),
            float(np.asarray(scores)[RIGHT_HAND].mean()))


def frame_features(keypoints, scores, width, height):
    """Vecteur (104,) ancre au corps. Retourne (vecteur, personne_vue).

    Origine au milieu des epaules, echelle = largeur d epaules : la position
    des mains par rapport au torse est porteuse de sens en ASL (le meme
    geste au front ou au menton sont deux signes differents), contrairement
    a l alphabet ou elle etait du bruit.
    """
    features = np.zeros(FEATURE_DIM, dtype=np.float32)

    left_shoulder = keypoints[COCO_LEFT_SHOULDER]
    right_shoulder = keypoints[COCO_RIGHT_SHOULDER]

    if min(scores[COCO_LEFT_SHOULDER], scores[COCO_RIGHT_SHOULDER]) < POSE_SCORE_THRESHOLD:
        return features, False

    origin = (left_shoulder + right_shoulder) / 2.0
    scale = float(np.linalg.norm(left_shoulder - right_shoulder))

    if scale < 1e-6:
        return features, False

    def normalise(points):
        return ((np.asarray(points, dtype=np.float64) - origin) / scale).astype(np.float32)

    left_visible = float(scores[LEFT_HAND].mean()) >= HAND_SCORE_THRESHOLD
    right_visible = float(scores[RIGHT_HAND].mean()) >= HAND_SCORE_THRESHOLD

    # Une main dont la plupart des points tombent hors de l image n est pas
    # exploitable : mieux vaut l annoncer absente que nourrir le modele de
    # coordonnees extrapolees.
    if left_visible and inside_fraction(keypoints[LEFT_HAND], width, height) < INSIDE_THRESHOLD:
        left_visible = False
    if right_visible and inside_fraction(keypoints[RIGHT_HAND], width, height) < INSIDE_THRESHOLD:
        right_visible = False

    # Le modele predit toujours 21 points par main, meme quand une seule est
    # visible, et les pose alors souvent sur celle qui est la. Le critere et
    # son justificatif sont dans hand_identity.py, partage avec les demos.
    left_visible, right_visible, _ = resolve(
        left_visible, right_visible,
        keypoints[LEFT_HAND][0], keypoints[RIGHT_HAND][0],
        keypoints[COCO_LEFT_WRIST], keypoints[COCO_RIGHT_WRIST],
    )

    offset = 0
    for hand, visible in ((LEFT_HAND, left_visible), (RIGHT_HAND, right_visible)):
        if visible:
            features[offset:offset + HAND_POINTS * 2] = normalise(keypoints[hand]).reshape(-1)
        offset += HAND_POINTS * 2

    features[offset] = 1.0 if left_visible else 0.0
    features[offset + 1] = 1.0 if right_visible else 0.0
    offset += 2

    features[offset:] = normalise(keypoints[POSE_KEYPOINTS]).reshape(-1)

    return features, True


def process_clip(model, record):
    path = os.path.join(PROJECT_ROOT, record["path"])
    out_path = os.path.join(FEATURES_DIR, f"{record['video_id']}.npz")

    if os.path.exists(out_path):
        return "cached", None

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        return "unreadable", None

    real_fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)

    if real_fps <= 0 or frame_count <= 0:
        capture.release()
        return "bad_metadata", None

    start, end = window_for(record, real_fps, frame_count)
    seconds = (end - start + 1) / real_fps

    if seconds < MIN_SECONDS or seconds > MAX_SECONDS:
        capture.release()
        return "suspicious_window", round(seconds, 2)

    wanted = list(range(start, end + 1))
    if len(wanted) > MAX_FRAMES:
        step = len(wanted) / MAX_FRAMES
        wanted = [wanted[int(i * step)] for i in range(MAX_FRAMES)]
    wanted_set = set(wanted)

    sequence = []
    confidences = []
    detected = 0

    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    index = start

    try:
        while index <= end:
            ok, frame = capture.read()
            if not ok:
                break

            if index in wanted_set:
                height, width = frame.shape[:2]
                keypoints, scores = model(frame)

                if len(keypoints) == 0:
                    sequence.append(np.zeros(FEATURE_DIM, dtype=np.float32))
                    confidences.append((0.0, 0.0))
                else:
                    vector, found = frame_features(
                        np.asarray(keypoints)[0], np.asarray(scores)[0], width, height
                    )
                    sequence.append(vector)
                    confidences.append(hand_confidences(np.asarray(scores)[0]))
                    detected += found

            index += 1
    except Exception as error:
        capture.release()
        return "error", f"{type(error).__name__}: {str(error)[:90]}"
    finally:
        capture.release()

    if not sequence:
        return "no_frames", None

    ratio = detected / len(sequence)
    if ratio < MIN_DETECT_RATIO:
        return "no_person", round(ratio, 2)

    np.savez_compressed(
        out_path,
        features=np.array(sequence, dtype=np.float32),
        hand_scores=np.array(confidences, dtype=np.float32),
        hand_threshold=HAND_SCORE_THRESHOLD,
        gloss=record["gloss"],
        signer_id=record["signer_id"],
        video_id=record["video_id"],
        detect_ratio=ratio,
    )

    return "ok", len(sequence)


def main():
    global HAND_SCORE_THRESHOLD, FEATURES_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mode", default="balanced",
                        choices=("performance", "lightweight", "balanced"))
    parser.add_argument("--hand-score", type=float, default=HAND_SCORE_THRESHOLD,
                        help="confiance minimale pour declarer une main presente")
    parser.add_argument("--out-dir", default=FEATURES_DIR,
                        help="ecrire ailleurs, pour comparer deux reglages "
                             "sans ecraser les features existantes")
    args = parser.parse_args()

    HAND_SCORE_THRESHOLD = args.hand_score
    FEATURES_DIR = args.out_dir

    from rtmlib import Wholebody

    os.makedirs(FEATURES_DIR, exist_ok=True)

    with open(MANIFEST_PATH, encoding="utf-8") as handle:
        manifest = json.load(handle)

    have = [r for r in manifest if r["status"] in ("ok", "cached") and r["path"]]

    counts = {}
    for record in have:
        counts[record["gloss"]] = counts.get(record["gloss"], 0) + 1

    keep = {g for g, _ in sorted(counts.items(), key=lambda kv: -kv[1])[: args.vocab]}
    todo = [r for r in have if r["gloss"] in keep]

    if args.limit:
        todo = todo[: args.limit]

    model = Wholebody(mode=args.mode, backend="onnxruntime", device="cuda")
    provider = model.pose_model.session.get_providers()[0]

    print(f"WLASL{args.vocab} : {len(todo)} clips")
    print(f"seuil de presence des mains : {HAND_SCORE_THRESHOLD}")
    print(f"sortie : {FEATURES_DIR}")
    print(f"provider : {provider}")
    if provider != "CUDAExecutionProvider":
        print("  ATTENTION : execution sur CPU, ce sera ~20x plus lent")
    print(f"dimension par frame : {FEATURE_DIM}\n")

    results = []
    tally = {}
    started = time.time()

    for index, record in enumerate(todo, start=1):
        status, detail = process_clip(model, record)
        tally[status] = tally.get(status, 0) + 1
        results.append({"video_id": record["video_id"], "status": status, "detail": detail})

        if index % 25 == 0 or index == len(todo):
            elapsed = time.time() - started
            remaining = (len(todo) - index) / max(index / elapsed, 1e-6)
            print(
                f"  {index}/{len(todo)}  "
                + "  ".join(f"{k} {v}" for k, v in sorted(tally.items()))
                + f"   ~{remaining / 60:.0f} min restantes",
                flush=True,
            )

    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=1)

    good = tally.get("ok", 0) + tally.get("cached", 0)
    print(f"\n{'=' * 60}")
    for status, count in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  {status:<20}{count:>6}")
    print(f"  {'-' * 26}")
    print(f"  {'exploitables':<20}{good:>6}/{len(todo)}")
    print(f"\n  sequences : {FEATURES_DIR}")


if __name__ == "__main__":
    main()
