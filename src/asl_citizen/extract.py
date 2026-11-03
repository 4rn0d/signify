"""Extraction d ASL Citizen, avec le pipeline EXACT de la demo.

Trois choix, chacun tire d une erreur deja payee sur WLASL.

1. Le detecteur est celui de la demo, TensorRT compris (`use_tensorrt`,
   importe tel quel). Sur WLASL, l entrainement tournait sur CUDA et la demo
   sur TensorRT : 4.4% des frames divergeaient, et TF32 interdisait de les
   reconcilier. Ici les deux cotes executent le meme moteur, donc la derive
   disparait par construction.

2. Mode BRUT : les deux mains toujours ecrites, avec leurs confiances, leur
   part hors cadre et le drapeau de conflit. Le filtrage se fait a
   l entrainement, par gate_hands, reglable sans reextraire.

3. TOUTES les frames, avec leur horodatage reel, et rien d autre. Sur WLASL
   l extraction plafonnait a 150 frames et pad_sequence gardait les 80 du
   milieu : a 15 fps ces 80 frames couvrent 5.3 s, et le milieu du clip n est
   pas forcement le milieu du signe. Le reechantillonnage a 30 fps et le
   placement de la fenetre se font au chargement, pour pouvoir les regler
   sans reextraire. L horodatage vient du decodeur (CAP_PROP_POS_MSEC) : des
   videos enregistrees dans un navigateur ont souvent une cadence variable,
   que index / fps ne saurait pas decrire.

    python src/asl_citizen/extract.py --ranks 1-100      # la cible
    python src/asl_citizen/extract.py --ranks 101-400    # l echafaudage

Reprend la ou il s est arrete : un clip deja ecrit est saute.
"""

import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "wlasl"))

from gpu_setup import enable_cuda_dlls  # noqa: E402

enable_cuda_dlls()

import extract_rtmpose  # noqa: E402
from extract_rtmpose import (FEATURE_DIM, HAND_POINTS, frame_features,  # noqa: E402
                             hand_confidences, hand_point_confidences)
from webcam_demo_rtm import use_tensorrt  # noqa: E402

CITIZEN = os.path.join(PROJECT_ROOT, "data", "asl_citizen", "ASL_Citizen")
RANKED = os.path.join(PROJECT_ROOT, "data", "asl_citizen", "ranked_signs.json")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "asl_citizen", "features")
REPORT = os.path.join(PROJECT_ROOT, "data", "asl_citizen", "extract_report.json")

MIN_DETECT_RATIO = 0.3


def parse_ranks(text):
    low, high = (int(x) for x in text.split("-"))
    return low, high


def load_clips(low, high):
    with open(RANKED, encoding="utf-8") as handle:
        ranked = json.load(handle)
    rank = {s["gloss"]: i + 1 for i, s in enumerate(ranked)}
    clips = []
    for split in ("train", "val", "test"):
        with open(os.path.join(CITIZEN, "splits", f"{split}.csv"),
                  encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                r = rank.get(row["Gloss"])
                if r is None or not low <= r <= high:
                    continue
                clips.append({
                    "file": row["Video file"],
                    "gloss": row["Gloss"],
                    "code": row["ASL-LEX Code"],
                    # P12 -> 12 : le harnais attend un entier, comme WLASL.
                    "signer_id": int(row["Participant ID"].lstrip("Pp")),
                    "split": split,
                    "rank": r,
                })
    # Les signes les mieux classes d abord : si l extraction est
    # interrompue, ce qui est fait est ce qui sert en premier.
    clips.sort(key=lambda c: (c["rank"], c["file"]))
    return clips


def frame_vector(model, frame):
    """Exactement ce que Source.process de la demo calcule, avant filtrage."""
    height, width = frame.shape[:2]
    keypoints, scores = model(frame)
    if len(keypoints) == 0:
        return (np.zeros(FEATURE_DIM, dtype=np.float32), (0.0, 0.0),
                np.zeros(HAND_POINTS * 2, dtype=np.float32), (0.0, 0.0),
                False, False)
    points = np.asarray(keypoints)[0]
    confidence = np.asarray(scores)[0]
    vector, found = frame_features(points, confidence, width, height)
    return (vector, hand_confidences(confidence),
            hand_point_confidences(confidence),
            getattr(frame_features, "last_inside", (0.0, 0.0)),
            getattr(frame_features, "last_conflict", False), found)


def process(model, clip, detector):
    stem = os.path.splitext(clip["file"])[0]
    os.makedirs(FEATURES_DIR, exist_ok=True)
    out = os.path.join(FEATURES_DIR, f"{stem}.npz")
    if os.path.exists(out):
        return "cached", None

    capture = cv2.VideoCapture(os.path.join(CITIZEN, "videos", clip["file"]))
    if not capture.isOpened():
        return "unreadable", None
    fps = capture.get(cv2.CAP_PROP_FPS)

    vectors, scores, points, insides, conflicts, stamps = [], [], [], [], [], []
    detected = 0
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            # Horodatage du decodeur, pas index / fps : voir le docstring.
            stamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if stamp <= 0.0 and index > 0 and fps > 0:
                stamp = index / fps
            vector, score, point, inside, conflict, found = frame_vector(model, frame)
            vectors.append(vector)
            scores.append(score)
            points.append(point)
            insides.append(inside)
            conflicts.append(conflict)
            stamps.append(stamp)
            detected += bool(found)
            index += 1
    except Exception as error:
        return "error", f"{type(error).__name__}: {str(error)[:90]}"
    finally:
        capture.release()

    if not vectors:
        return "no_frames", None
    ratio = detected / len(vectors)
    if ratio < MIN_DETECT_RATIO:
        return "no_person", round(ratio, 2)

    # Ecriture ATOMIQUE : un processus interrompu en pleine ecriture laissait
    # un .npz tronque, que la reprise prenait pour un clip fini (elle saute
    # tout fichier existant) et que le chargement faisait ensuite echouer.
    # On ecrit a cote, sans l extension .npz que glob ramasserait, puis on
    # renomme : le fichier final existe entier ou n existe pas.
    partial = out + ".tmp"
    with open(partial, "wb") as handle:
        np.savez_compressed(
        handle,
        features=np.array(vectors, dtype=np.float32),
        stamps=np.array(stamps, dtype=np.float64),
        hand_scores=np.array(scores, dtype=np.float32),
        hand_point_scores=np.array(points, dtype=np.float32),
        hand_inside=np.array(insides, dtype=np.float32),
        hand_conflict=np.array(conflicts, dtype=bool),
        hand_threshold=extract_rtmpose.HAND_SCORE_THRESHOLD,
        raw_mode=True,
        detector=detector,
        gloss=clip["gloss"],
        asl_lex_code=clip["code"],
        signer_id=clip["signer_id"],
        video_id=stem,
        split=clip["split"],
        rank=clip["rank"],
        fps=fps,
        detect_ratio=ratio,
    )
    os.replace(partial, out)
    return "ok", len(vectors)


def build_model(tensorrt=True):
    """Le modele de la demo, ou un arret net : jamais un repli silencieux.

    Deux replis ont deja coute cher. Le 30 octobre, `pip install pymupdf` a
    tire le paquet onnxruntime CPU, qui a ecrase onnxruntime-gpu sans un
    message : tout tournait 20x plus lentement. Et si TensorRT echoue,
    use_tensorrt retombe sur CUDA, ce qui va pour une demo mais pas pour une
    extraction : la demo, elle, tournerait sur TensorRT, et les deux cotes
    divergeraient sur 4.4% des frames sans que rien ne le signale.
    """
    from rtmlib import Wholebody

    extract_rtmpose.RAW_MODE = True
    model = Wholebody(mode="balanced", backend="onnxruntime", device="cuda")

    pose = model.pose_model.session.get_providers()[0]
    if pose != "CUDAExecutionProvider":
        raise SystemExit(
            f"Le modele de pose tourne sur {pose}, pas sur CUDA. Le paquet "
            f"onnxruntime CPU a probablement ecrase onnxruntime-gpu. Reparer : "
            f"pip uninstall -y onnxruntime, puis "
            f"pip install --force-reinstall --no-deps onnxruntime-gpu==1.22.0")

    if not tensorrt:
        return model, "cuda"
    if not use_tensorrt(model):
        raise SystemExit(
            "TensorRT n a pas pris le detecteur. Extraire sur CUDA pendant que "
            "la demo tourne sur TensorRT ferait diverger 4.4% des frames. "
            "Reparer TensorRT, ou passer --no-trt ET lancer la demo sans --trt.")
    return model, "tensorrt"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ranks", default="1-100",
                        help="rangs de frequence ASL-LEX a extraire, ex. 1-100")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reverse", action="store_true",
                        help="traiter les clips en partant de la fin. Un second "
                             "processus en sens inverse double le debit sans "
                             "interrompre le premier : chacun saute ce que "
                             "l autre a deja ecrit, et ils se rejoignent.")
    parser.add_argument("--shard", default="0/1",
                        help="i/n : ne traiter qu un clip sur n, a partir du "
                             "i-eme. Plusieurs processus en parallele, chacun "
                             "son i, occupent le GPU que l extraction seule "
                             "laisse a moitie inactif entre deux frames. Le "
                             "calcul par frame ne change pas : les features "
                             "sont identiques.")
    parser.add_argument("--no-trt", action="store_true",
                        help="detecteur sur CUDA ; la demo devra alors tourner "
                             "SANS --trt pour rester identique")
    args = parser.parse_args()

    os.makedirs(FEATURES_DIR, exist_ok=True)
    clips = load_clips(*parse_ranks(args.ranks))
    shard, shards = (int(x) for x in args.shard.split("/"))
    # Un sur n dans l ordre des rangs : chaque processus avance lui aussi des
    # signes les mieux classes vers les moins bien classes.
    clips = clips[shard::shards]
    if args.reverse:
        clips = clips[::-1]
    if args.limit:
        clips = clips[:args.limit]
    # Un rapport par processus : plusieurs processus reecrivant le meme
    # fichier s effaceraient l un l autre.
    report_path = (REPORT if shards == 1
                   else REPORT.replace(".json", f"_shard{shard}of{shards}.json"))
    if args.reverse:
        report_path = report_path.replace(".json", "_reverse.json")

    model, detector = build_model(tensorrt=not args.no_trt)
    print(f"{len(clips)} clips, rangs {args.ranks}, detecteur {detector}", flush=True)

    report = {}
    if os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as handle:
            report = json.load(handle)

    started = time.time()
    done = 0
    for position, clip in enumerate(clips, 1):
        status, detail = process(model, clip, detector)
        report[clip["file"]] = {"status": status, "detail": detail}
        if status == "ok":
            done += 1
        if position % 100 == 0 or position == len(clips):
            elapsed = time.time() - started
            rate = elapsed / max(done, 1)
            left = rate * sum(1 for c in clips[position:])
            print(f"  {position}/{len(clips)}  {done} extraits  "
                  f"{rate:.2f} s/clip  reste ~{left / 60:.0f} min", flush=True)
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump(report, handle)

    counts = {}
    for clip in clips:
        s = report[clip["file"]]["status"]
        counts[s] = counts.get(s, 0) + 1
    print(f"  bilan : {counts}")


if __name__ == "__main__":
    main()
