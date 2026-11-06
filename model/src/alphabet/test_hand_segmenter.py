import cv2

from hand_segmenter import HandSegmenter


def main():
    print("Démarrage du test...", flush=True)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERREUR : impossible d'ouvrir la webcam.", flush=True)
        return

    print("Webcam démarrée.", flush=True)
    print("Appuie sur Q pour quitter.", flush=True)

    segmenter = HandSegmenter()

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Impossible de lire une frame.", flush=True)
                break

            cleaned, bbox, found = segmenter.process(frame)

            display = cleaned.copy()

            if found and bbox is not None:
                x1, y1, x2, y2 = bbox

                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

            # Affichage côte à côte
            original = frame

            display = cv2.resize(
                display,
                (original.shape[1], original.shape[0]),
            )

            combined = cv2.hconcat([
                original,
                display,
            ])

            cv2.putText(
                combined,
                "Original",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                combined,
                "Main isolee",
                (original.shape[1] + 20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow(
                "Hand segmentation",
                combined,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        print("Fermeture...", flush=True)

        segmenter.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()