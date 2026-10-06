"""Construit le dataset segmente (main detouree sur fond blanc) depuis les
images brutes.

Parallelisation : un processus par CLASSE, pas par image. HandSegmenter
tourne en RunningMode.VIDEO, donc le suivi temporel se propage d'une image
a la suivante — c'est ce qui fait grimper le taux de detection (81% contre
52% en mode IMAGE sur les memes fichiers). Decouper par image casserait
cette continuite ; decouper par classe la preserve, chaque worker traitant
une classe entiere dans l'ordre des indices de frame.

Resolution : le dataset d'origine est en 64x64. Les backbones pre-entraines
(MobileNetV2 et consorts) n'acceptent que [96, 128, 160, 192, 224] ; les
sources faisant 200x200, 128 est le bon compromis (on reduit presque
toujours, donc aucun detail invente).

Usage:
    python src/prepare_segmented_dataset.py
    python src/prepare_segmented_dataset.py --size 128 --out asl_alphabet_segmented_128
"""

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

from hand_segmenter import HandSegmenter


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "asl_alphabet_train"
    / "asl_alphabet_train"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_OUTPUT_NAME = "asl_alphabet_segmented"
DEFAULT_SIZE = 64

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}

JPEG_QUALITY = 95

_INDEX_RE = re.compile(r"(\d+)\.[A-Za-z]+$")


def frame_index(path):
    """Indice de frame extrait du nom ('A123.jpg' -> 123), -1 sinon."""
    match = _INDEX_RE.search(path.name)
    return int(match.group(1)) if match else -1


def process_class(args):
    """Traite une classe entiere dans un processus dedie."""
    class_name, output_dir, output_size = args

    source_class_dir = SOURCE_DIR / class_name
    output_class_dir = Path(output_dir) / class_name
    output_class_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        (
            p
            for p in source_class_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda p: (frame_index(p), p.name),
    )

    segmenter = HandSegmenter()

    success = 0
    failed = 0

    try:
        for image_path in images:
            image = cv2.imread(str(image_path))

            if image is None:
                failed += 1
                continue

            cleaned, bbox, found = segmenter.process(image)

            if not found or bbox is None:
                failed += 1
                continue

            cropped = segmenter.crop_for_model(
                cleaned,
                bbox,
                output_size=output_size,
            )

            if cropped is None:
                failed += 1
                continue

            cv2.imwrite(
                str(output_class_dir / f"{image_path.stem}.jpg"),
                cropped,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )

            success += 1
    finally:
        segmenter.close()

    return class_name, len(images), success, failed


def prepare_dataset(output_name=DEFAULT_OUTPUT_NAME, output_size=DEFAULT_SIZE, workers=8):
    output_dir = PROCESSED_DIR / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = sorted(p.name for p in SOURCE_DIR.iterdir() if p.is_dir())

    print(f"{len(classes)} classes trouvées.")
    print(f"Sortie     : {output_dir}")
    print(f"Resolution : {output_size}x{output_size}")
    print(f"Processus  : {workers}\n")

    jobs = [(name, str(output_dir), output_size) for name in classes]

    total = 0
    success = 0
    failed = 0
    done = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for class_name, count, ok, ko in pool.map(process_class, jobs):
            total += count
            success += ok
            failed += ko
            done += 1
            print(
                f"  [{done}/{len(classes)}] {class_name:<6} "
                f"{ok}/{count} segmentées",
                flush=True,
            )

    print("\n==============================")
    print("PREPARATION TERMINÉE")
    print("==============================")
    print(f"Total    : {total}")
    print(f"Réussies : {success} ({success / max(total, 1):.1%})")
    print(f"Échecs   : {failed}")
    print(f"Sortie   : {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--out", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    prepare_dataset(
        output_name=args.out,
        output_size=args.size,
        workers=args.workers,
    )
