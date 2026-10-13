"""Extrait des clips de reference pour les 50 signes que le modele connait.

But : donner aux signeurs qui vont enregistrer de nouvelles donnees la
VARIANTE EXACTE que le modele a apprise. WLASL etiquette plusieurs gestes
differents sous une meme glose — 19 des 50 gloses ont plusieurs
variation_id — et l entrainement n a garde que la variante majoritaire de
chacune (c est ce qui valait +5.6 points). Enregistrer une autre variante
produirait des donnees que le modele considere comme une autre classe.

La liste des gloses vient de la fiche du modele, pas d un recalcul : c est
elle qui dit ce que le modele connait reellement.

Les clips sont decoupes sur la fenetre du signe : 88 clips du dataset font
jusqu a 460 s et contiennent plusieurs signes, et personne ne regardera
4 minutes pour en voir un.
"""

import argparse
import collections
import json
import os
import shutil

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WLASL_DIR = os.path.join(PROJECT_ROOT, "data", "wlasl")
MANIFEST_PATH = os.path.join(WLASL_DIR, "download_manifest.json")
FEATURES_DIR = os.path.join(WLASL_DIR, "features_rtm07")
META_PATH = os.path.join(PROJECT_ROOT, "models", "wlasl_gru_rtmpose.meta.json")
OUTPUT_DIR = os.path.join(WLASL_DIR, "reference")

PADDING_SECONDS = 0.4          # un peu avant et apres, pour voir l amorce


def window_for(record, real_fps, frame_count):
    """Meme conversion que l extraction : via les SECONDES.

    Les intervalles du metadata sont indexes sur la video d origine, et 39%
    des clips coupes ont ete reencodes a 30 fps alors que le metadata annonce
    25. Utiliser les indices bruts se trompe de 29 s en mediane.
    """
    if record["frame_end"] == -1:
        return 0, int(frame_count) - 1

    meta_fps = record["fps"] or 25
    start = max(int(round((record["frame_start"] - 1) / meta_fps * real_fps)), 0)
    end = min(int(round(record["frame_end"] / meta_fps * real_fps)), int(frame_count) - 1)
    if end <= start:
        return 0, int(frame_count) - 1
    return start, end


def dominant_variants(classes):
    """Pour chaque glose, la variante que l entrainement a conservee.

    Meme regle que load_sequences : la variante la PLUS REPRESENTEE parmi les
    clips reellement extraits, pas forcement variation_id 0 (candy est a 7
    contre 11 dans l autre sens).
    """
    with open(MANIFEST_PATH, encoding="utf-8") as handle:
        variation = {r["video_id"]: r.get("variation_id", 0) for r in json.load(handle)}

    by_gloss = collections.defaultdict(collections.Counter)
    quality = collections.defaultdict(list)

    import glob
    for path in sorted(glob.glob(os.path.join(FEATURES_DIR, "*.npz"))):
        data = np.load(path, allow_pickle=True)
        gloss = str(data["gloss"])
        if gloss not in classes:
            continue
        video_id = str(data["video_id"])
        variant = int(variation.get(video_id, 0))
        by_gloss[gloss][variant] += 1
        quality[(gloss, variant)].append((float(data["detect_ratio"]), video_id))

    keep = {g: c.most_common(1)[0][0] for g, c in by_gloss.items()}
    return keep, by_gloss, quality


def trim(record, destination, padding=PADDING_SECONDS):
    """Ecrit la fenetre du signe dans un fichier a part. True si ecrit."""
    source = os.path.join(PROJECT_ROOT, record["path"])
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        return False

    fps = capture.get(cv2.CAP_PROP_FPS)
    count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or count <= 0 or width <= 0:
        capture.release()
        return False

    start, end = window_for(record, fps, count)
    start = max(int(start - padding * fps), 0)
    end = min(int(end + padding * fps), int(count) - 1)

    # Clip deja court et non decoupe : une copie suffit, sans reencodage.
    if record["frame_end"] == -1 and count / fps <= 8.0:
        capture.release()
        shutil.copyfile(source, destination)
        return True

    writer = cv2.VideoWriter(destination, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    index = start
    written = 0
    while index <= end:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
        index += 1
    writer.release()
    capture.release()
    return written > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=OUTPUT_DIR)
    parser.add_argument("--per-gloss", type=int, default=2,
                        help="clips de reference par signe, de signeurs differents")
    args = parser.parse_args()

    with open(META_PATH, encoding="utf-8") as handle:
        meta = json.load(handle)
    classes = sorted(meta["classes"])

    with open(MANIFEST_PATH, encoding="utf-8") as handle:
        manifest = json.load(handle)
    by_id = {r["video_id"]: r for r in manifest}

    keep, by_gloss, quality = dominant_variants(set(classes))
    os.makedirs(args.out_dir, exist_ok=True)

    multi = [g for g in classes if len(by_gloss.get(g, {})) > 1]
    print(f"{len(classes)} gloses, dont {len(multi)} avec plusieurs variantes\n")

    rows = []
    for gloss in classes:
        variant = keep.get(gloss)
        if variant is None:
            print(f"  {gloss}: aucun clip extrait, ignore")
            continue

        # Les clips les mieux detectes d abord, et un seul par signeur pour
        # que les references montrent des personnes differentes.
        candidates = sorted(quality[(gloss, variant)], reverse=True)
        chosen, seen_signers = [], set()
        for ratio, video_id in candidates:
            record = by_id.get(video_id)
            if record is None or not record.get("path"):
                continue
            if record["signer_id"] in seen_signers:
                continue
            destination = os.path.join(
                args.out_dir, f"{gloss}_{len(chosen) + 1}.mp4")
            if trim(record, destination):
                chosen.append((video_id, record["signer_id"], ratio,
                               os.path.basename(destination), record.get("url")))
                seen_signers.add(record["signer_id"])
            if len(chosen) >= args.per_gloss:
                break

        others = sorted(by_gloss[gloss])
        rows.append({
            "gloss": gloss,
            "variant": variant,
            "variants_seen": others,
            "clips_in_variant": by_gloss[gloss][variant],
            "clips_total": sum(by_gloss[gloss].values()),
            "references": chosen,
        })
        flag = "" if len(others) == 1 else f"  <- {len(others)} variantes"
        print(f"  {gloss:<14} variante {variant}  "
              f"{by_gloss[gloss][variant]}/{sum(by_gloss[gloss].values())} clips  "
              f"{len(chosen)} reference(s){flag}")

    with open(os.path.join(args.out_dir, "reference.json"), "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=1)

    print(f"\n{sum(len(r['references']) for r in rows)} clips ecrits dans {args.out_dir}")
    return rows


if __name__ == "__main__":
    main()
