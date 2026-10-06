import os
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "asl_alphabet_segmented"
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


def prepare_dataset():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    segmenter = HandSegmenter()

    total = 0
    success = 0
    failed = 0

    try:
        classes = sorted(
            [
                p
                for p in SOURCE_DIR.iterdir()
                if p.is_dir()
            ]
        )

        print(f"{len(classes)} classes trouvées.")

        for class_dir in classes:
            output_class_dir = (
                OUTPUT_DIR / class_dir.name
            )

            output_class_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            images = sorted(
                [
                    p
                    for p in class_dir.iterdir()
                    if p.suffix.lower()
                    in IMAGE_EXTENSIONS
                ]
            )

            print(
                f"\nClasse {class_dir.name}: "
                f"{len(images)} images"
            )

            for index, image_path in enumerate(images):
                total += 1

                image = cv2.imread(
                    str(image_path)
                )

                if image is None:
                    failed += 1
                    continue

                cleaned, bbox, found = (
                    segmenter.process(image)
                )

                if not found or bbox is None:
                    failed += 1
                    continue

                cropped = (
                    segmenter.crop_for_model(
                        cleaned,
                        bbox,
                        output_size=64,
                    )
                )

                if cropped is None:
                    failed += 1
                    continue

                output_path = (
                    output_class_dir
                    / f"{image_path.stem}.jpg"
                )

                cv2.imwrite(
                    str(output_path),
                    cropped,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        95,
                    ],
                )

                success += 1

                if index % 250 == 0:
                    print(
                        f"  {index}/{len(images)}"
                    )

    finally:
        segmenter.close()

    print("\n==============================")
    print("PREPARATION TERMINÉE")
    print("==============================")
    print(f"Total   : {total}")
    print(f"Réussies: {success}")
    print(f"Échecs  : {failed}")
    print(f"Sortie  : {OUTPUT_DIR}")


if __name__ == "__main__":
    prepare_dataset()