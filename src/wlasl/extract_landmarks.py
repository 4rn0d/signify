"""Extrait les sequences de landmarks MediaPipe Holistic des clips WLASL.

Sortie : un .npz par clip, contenant (T, FEATURE_DIM) float32.

Trois pieges verifies sur les donnees reelles :

1. Les intervalles de frames sont exprimes dans le referentiel de la video
   D'ORIGINE. 39% des clips coupes ont ete reencodes a 30 fps alors que le
   metadata annonce 25 : utiliser les indices bruts decale la fenetre de 29 s
   en mediane, pour un signe qui dure 2.5 s. On convertit donc en SECONDES
   via le fps du metadata, puis vers les frames du fichier reel.

2. 88 clips durent plus de 10 s (jusqu'a 460 s) et contiennent plusieurs
   signes. Tous ont un intervalle explicite ; les 1233 autres sont de vrais
   clips courts (mediane 2.4 s) qu'on prend en entier.

3. Normalisation ancree au CORPS, pas au poignet. Pour l'alphabet, la
   position de la main dans le cadre etait du bruit. Ici c'est du sens : le
   meme geste au front ou au menton sont deux signes differents. On prend le
   milieu des epaules comme origine et la largeur d'epaules comme echelle,
   ce qui annule la distance a la camera en conservant la position des mains
   par rapport au torse.

Main absente (signe a une main, occlusion) : zeros + un drapeau de presence,
pour que le modele distingue "pas de main" de "main a l'origine".

API : mediapipe 1.x a supprime mediapipe.solutions ; on utilise
mediapipe.tasks.vision.HolisticLandmarker, ou les landmarks sont des listes
plates (pose_landmarks[11].x) et une liste vide signifie non detecte.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WLASL_DIR = os.path.join(PROJECT_ROOT, "data", "wlasl")
MANIFEST_PATH = os.path.join(WLASL_DIR, "download_manifest.json")
FEATURES_DIR = os.path.join(WLASL_DIR, "features")
EXTRACT_REPORT = os.path.join(WLASL_DIR, "extract_report.json")

HOLISTIC_MODEL = os.path.join(PROJECT_ROOT, "models", "holistic_landmarker.task")

# Haut du corps uniquement : les jambes n'apportent rien a un signe.
POSE_KEYPOINTS = [
    0,          # nez
    11, 12,     # epaules
    13, 14,     # coudes
    15, 16,     # poignets
    23, 24,     # hanches
]

HAND_POINTS = 21
FEATURE_DIM = HAND_POINTS * 3 * 2 + 2 + len(POSE_KEYPOINTS) * 3

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

# Garde-fous : une fenetre hors de ces bornes signale un intervalle douteux.
MIN_SECONDS = 0.4
MAX_SECONDS = 10.0

# Au-dela, on sous-echantillonne : un signe ne demande pas 300 frames.
MAX_FRAMES = 150

# En dessous de ce taux de frames ou une personne est detectee, le clip est
# inexploitable (cadrage, luminosite, ou ce n'est pas un signeur).
MIN_DETECT_RATIO = 0.3

def _new_landmarker():
    """Un HolisticLandmarker NEUF par clip.

    Deux raisons de ne pas le reutiliser entre clips :
    - le mode VIDEO exige des timestamps croissants, et chaque clip repart
      de zero : reutiliser l'objet leve "Input timestamp must be
      monotonically increasing" ;
    - le suivi temporel se propagerait d'un clip a l'autre, alors qu'ils
      montrent des signeurs et des scenes differents. C'est exactement le
      piege rencontre sur l'alphabet, ou le mode VIDEO faisait deborder le
      suivi entre images sans rapport.
    """
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    return vision.HolisticLandmarker.create_from_options(
        vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.abspath(HOLISTIC_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
        )
    )


def window_for(record, real_fps, frame_count):
    """(frame_debut, frame_fin) dans le referentiel du FICHIER telecharge."""
    if record["frame_end"] == -1:
        return 0, int(frame_count) - 1

    meta_fps = record["fps"] or 25

    # Indices d'origine -> secondes -> indices du fichier reel.
    start_s = (record["frame_start"] - 1) / meta_fps
    end_s = record["frame_end"] / meta_fps

    start = max(int(round(start_s * real_fps)), 0)
    end = min(int(round(end_s * real_fps)), int(frame_count) - 1)

    if end <= start:
        return 0, int(frame_count) - 1

    return start, end


def frame_features(result):
    """Vecteur par frame, ancre au corps. Retourne (vecteur, personne_vue)."""
    features = np.zeros(FEATURE_DIM, dtype=np.float32)

    pose = result.pose_landmarks
    if not pose or len(pose) <= RIGHT_SHOULDER:
        return features, False

    left = np.array([pose[LEFT_SHOULDER].x, pose[LEFT_SHOULDER].y, pose[LEFT_SHOULDER].z])
    right = np.array([pose[RIGHT_SHOULDER].x, pose[RIGHT_SHOULDER].y, pose[RIGHT_SHOULDER].z])

    origin = (left + right) / 2.0
    scale = float(np.linalg.norm(left[:2] - right[:2]))

    if scale < 1e-6:
        return features, False

    def normalise(points):
        return ((np.asarray(points, dtype=np.float64) - origin) / scale).astype(np.float32)

    offset = 0
    for hand in (result.left_hand_landmarks, result.right_hand_landmarks):
        if hand:
            coords = [[p.x, p.y, p.z] for p in hand]
            features[offset:offset + HAND_POINTS * 3] = normalise(coords).reshape(-1)
        offset += HAND_POINTS * 3

    features[offset] = 1.0 if result.left_hand_landmarks else 0.0
    features[offset + 1] = 1.0 if result.right_hand_landmarks else 0.0
    offset += 2

    coords = [[pose[i].x, pose[i].y, pose[i].z] for i in POSE_KEYPOINTS]
    features[offset:offset + len(POSE_KEYPOINTS) * 3] = normalise(coords).reshape(-1)

    return features, True


def process_clip(record):
    import mediapipe as mp

    path = os.path.join(PROJECT_ROOT, record["path"])
    out_path = os.path.join(FEATURES_DIR, f"{record['video_id']}.npz")

    if os.path.exists(out_path):
        return record["video_id"], "cached", None

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        return record["video_id"], "unreadable", None

    real_fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)

    if real_fps <= 0 or frame_count <= 0:
        capture.release()
        return record["video_id"], "bad_metadata", None

    start, end = window_for(record, real_fps, frame_count)
    seconds = (end - start + 1) / real_fps

    if seconds < MIN_SECONDS or seconds > MAX_SECONDS:
        capture.release()
        return record["video_id"], "suspicious_window", round(seconds, 2)

    wanted = list(range(start, end + 1))
    if len(wanted) > MAX_FRAMES:
        step = len(wanted) / MAX_FRAMES
        wanted = [wanted[int(i * step)] for i in range(MAX_FRAMES)]
    wanted_set = set(wanted)

    landmarker = _new_landmarker()

    sequence = []
    detected = 0

    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    index = start

    try:
        while index <= end:
            ok, frame = capture.read()
            if not ok:
                break

            if index in wanted_set:
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                # VIDEO exige des timestamps croissants ; l'indice de frame
                # converti en millisecondes les rend monotones et reels.
                # Timestamps relatifs au DEBUT de la fenetre : le
                # landmarker est neuf, il attend une suite partant de zero.
                result = landmarker.detect_for_video(
                    image, int((index - start) * 1000 / real_fps)
                )
                features, found = frame_features(result)
                sequence.append(features)
                detected += found

            index += 1
    except Exception as error:
        capture.release()
        return record["video_id"], "error", f"{type(error).__name__}: {str(error)[:90]}"
    finally:
        capture.release()
        landmarker.close()

    if not sequence:
        return record["video_id"], "no_frames", None

    ratio = detected / len(sequence)
    if ratio < MIN_DETECT_RATIO:
        return record["video_id"], "no_person", round(ratio, 2)

    np.savez_compressed(
        out_path,
        features=np.array(sequence, dtype=np.float32),
        gloss=record["gloss"],
        signer_id=record["signer_id"],
        video_id=record["video_id"],
        detect_ratio=ratio,
    )

    return record["video_id"], "ok", len(sequence)


def select_clips(manifest, vocab):
    have = [r for r in manifest if r["status"] in ("ok", "cached") and r["path"]]

    counts = {}
    for record in have:
        counts[record["gloss"]] = counts.get(record["gloss"], 0) + 1

    keep = {g for g, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:vocab]}
    return [r for r in have if r["gloss"] in keep]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=int, default=50)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(HOLISTIC_MODEL):
        raise SystemExit(
            f"Modele absent : {HOLISTIC_MODEL}\n"
            "Telecharge-le depuis "
            "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
            "holistic_landmarker/float16/latest/holistic_landmarker.task"
        )

    os.makedirs(FEATURES_DIR, exist_ok=True)

    with open(MANIFEST_PATH, encoding="utf-8") as handle:
        manifest = json.load(handle)

    todo = select_clips(manifest, args.vocab)
    if args.limit:
        todo = todo[: args.limit]

    print(f"WLASL{args.vocab} : {len(todo)} clips, {args.workers} processus")
    print(f"dimension par frame : {FEATURE_DIM}\n")

    results = []
    tally = {}
    started = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (video_id, status, detail) in enumerate(
            pool.map(process_clip, todo, chunksize=2), start=1
        ):
            tally[status] = tally.get(status, 0) + 1
            results.append({"video_id": video_id, "status": status, "detail": detail})

            if index % 25 == 0 or index == len(todo):
                elapsed = time.time() - started
                remaining = (len(todo) - index) / max(index / elapsed, 1e-6)
                print(
                    f"  {index}/{len(todo)}  "
                    + "  ".join(f"{k} {v}" for k, v in sorted(tally.items()))
                    + f"   ~{remaining / 60:.0f} min restantes",
                    flush=True,
                )

    with open(EXTRACT_REPORT, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=1)

    good = tally.get("ok", 0) + tally.get("cached", 0)
    print(f"\n{'=' * 60}")
    for status, count in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  {status:<20}{count:>6}")
    print(f"  {'-' * 26}")
    print(f"  {'exploitables':<20}{good:>6}/{len(todo)}")
    print(f"\n  sequences : {FEATURES_DIR}")
    print(f"  rapport   : {EXTRACT_REPORT}")


if __name__ == "__main__":
    main()
