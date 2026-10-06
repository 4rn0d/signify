import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "hand_landmarker.task",
)


# GrabCut coute O(pixels) x iterations et domine le temps de segmentation.
# Valeurs par defaut = comportement historique (pleine resolution, 4 iterations),
# pour que prepare_segmented_dataset.py reste reproductible a l'identique.
# webcam_demo.py demande explicitement les reglages rapides.
GRABCUT_FULL_QUALITY = dict(grabcut_max_side=None, grabcut_iters=4)
GRABCUT_REALTIME = dict(grabcut_max_side=128, grabcut_iters=2)


class HandSegmenter:
    """
    Detects a hand with MediaPipe and extracts the real hand pixels
    using GrabCut initialized from MediaPipe landmarks.

    Output:
        - real hand pixels preserved
        - background replaced with white
    """

    def __init__(
        self,
        num_hands=1,
        min_detection_confidence=0.5,
        min_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        grabcut_max_side=None,
        grabcut_iters=4,
    ):
        self.num_hands = num_hands
        self.grabcut_max_side = grabcut_max_side
        self.grabcut_iters = grabcut_iters

        self._ensure_model()

        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=self.model_path
            ),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.landmarker = HandLandmarker.create_from_options(options)
        self.timestamp_ms = 0

        # Landmarks de la derniere image traitee, en ESPACE PIXEL (21, 3).
        # MediaPipe renvoie x et y normalises par la largeur et la hauteur :
        # sur une image carree (le dataset est en 200x200) c'est isotrope,
        # mais sur une webcam 640x480 cela etire la geometrie de 4:3. On
        # reconvertit donc en pixels ici, une fois, pour que le modele
        # landmark voie la meme forme qu'a l'entrainement.
        self.last_landmarks = None

    # ================================================================
    # MediaPipe model
    # ================================================================

    def _ensure_model(self):
        self.model_path = os.path.abspath(MODEL_PATH)

        if os.path.exists(self.model_path):
            return

        os.makedirs(
            os.path.dirname(self.model_path),
            exist_ok=True,
        )

        print("Téléchargement du modèle MediaPipe Hand Landmarker...")
        print(f"Destination : {self.model_path}")

        urllib.request.urlretrieve(
            MODEL_URL,
            self.model_path,
        )

        print("Modèle MediaPipe téléchargé.")

    # ================================================================
    # Main processing
    # ================================================================

    def process(self, frame_bgr):
        """
        Input:
            frame_bgr : OpenCV BGR image

        Returns:
            cleaned : full image with white background
            bbox    : square crop around hand
            found   : True/False
        """

        if frame_bgr is None:
            return None, None, False

        height, width = frame_bgr.shape[:2]

        # ------------------------------------------------------------
        # MediaPipe
        # ------------------------------------------------------------

        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        self.timestamp_ms += 1

        result = self.landmarker.detect_for_video(
            mp_image,
            self.timestamp_ms,
        )

        if not result.hand_landmarks:
            self.last_landmarks = None
            return frame_bgr.copy(), None, False

        # Première main détectée
        landmarks = result.hand_landmarks[0]

        # z suit approximativement l'echelle de x chez MediaPipe : on le
        # multiplie donc par la largeur, comme x.
        self.last_landmarks = np.array(
            [
                [lm.x * width, lm.y * height, lm.z * width]
                for lm in landmarks
            ],
            dtype=np.float32,
        )

        points = np.array(
            [
                [
                    int(np.clip(lm.x * width, 0, width - 1)),
                    int(np.clip(lm.y * height, 0, height - 1)),
                ]
                for lm in landmarks
            ],
            dtype=np.int32,
        )

        # ------------------------------------------------------------
        # Bounding box basée uniquement sur les landmarks
        # ------------------------------------------------------------

        x1 = int(points[:, 0].min())
        x2 = int(points[:, 0].max())

        y1 = int(points[:, 1].min())
        y2 = int(points[:, 1].max())

        hand_w = x2 - x1
        hand_h = y2 - y1

        if hand_w < 5 or hand_h < 5:
            self.last_landmarks = None
            return frame_bgr.copy(), None, False

        # Petite marge autour de la main.
        # Plus petite que dans l'ancienne version pour éviter
        # d'inclure trop de background.
        margin = int(max(hand_w, hand_h) * 0.18)

        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)

        x2 = min(width, x2 + margin)
        y2 = min(height, y2 + margin)

        bbox = (x1, y1, x2, y2)

        # ------------------------------------------------------------
        # Crop carré
        # ------------------------------------------------------------

        square_bbox = self._square_bbox(
            bbox,
            width,
            height,
        )

        sx1, sy1, sx2, sy2 = square_bbox

        crop = frame_bgr[
            sy1:sy2,
            sx1:sx2,
        ].copy()

        if crop.size == 0:
            self.last_landmarks = None
            return frame_bgr.copy(), None, False

        # ------------------------------------------------------------
        # IMPORTANT:
        # Convertir les landmarks vers les coordonnées du crop.
        #
        # C'était un problème dans l'ancien code:
        # on soustrayait bbox.x1/bbox.y1 alors que le crop réel
        # commençait à square_bbox.x1/square_bbox.y1.
        # ------------------------------------------------------------

        crop_points = points.copy()

        crop_points[:, 0] -= sx1
        crop_points[:, 1] -= sy1

        crop_h, crop_w = crop.shape[:2]

        crop_points[:, 0] = np.clip(
            crop_points[:, 0],
            0,
            crop_w - 1,
        )

        crop_points[:, 1] = np.clip(
            crop_points[:, 1],
            0,
            crop_h - 1,
        )

        # ------------------------------------------------------------
        # GrabCut
        # ------------------------------------------------------------

        segmented = self._grabcut_hand(
            crop,
            crop_points,
            self.grabcut_max_side,
            self.grabcut_iters,
        )

        if segmented is None:
            self.last_landmarks = None
            return frame_bgr.copy(), None, False

        # ------------------------------------------------------------
        # Créer une image blanche complète
        # ------------------------------------------------------------

        cleaned = np.full_like(
            frame_bgr,
            255,
            dtype=np.uint8,
        )

        cleaned[
            sy1:sy2,
            sx1:sx2,
        ] = segmented

        return cleaned, square_bbox, True

    # ================================================================
    # GrabCut
    # ================================================================

    @staticmethod
    def _grabcut_hand(crop, points, max_side=None, iters=4):
        """
        GrabCut initialized using the MediaPipe hand skeleton.

        Strategy:

        - outside region = certain background
        - around landmarks = probable foreground
        - skeleton itself = certain foreground

        This lets GrabCut recover the real hand pixels instead
        of creating an artificial silhouette.
        """

        if crop is None or crop.size == 0:
            return None

        # Le masque peut etre calcule sur une version reduite du crop : la
        # sortie est de toute facon ramenee a 64x64. On redimensionne crop ET
        # points, tout le code en aval travaille donc a l'echelle reduite.
        full_size = (crop.shape[1], crop.shape[0])
        scale = 1.0

        if max_side:
            scale = min(1.0, float(max_side) / max(crop.shape[:2]))

        if scale < 1.0:
            crop = cv2.resize(
                crop,
                (
                    max(int(crop.shape[1] * scale), 1),
                    max(int(crop.shape[0] * scale), 1),
                ),
                interpolation=cv2.INTER_AREA,
            )

            points = (points.astype(np.float32) * scale).astype(np.int32)

            points[:, 0] = np.clip(points[:, 0], 0, crop.shape[1] - 1)
            points[:, 1] = np.clip(points[:, 1], 0, crop.shape[0] - 1)

        h, w = crop.shape[:2]

        # ------------------------------------------------------------
        # Initial GrabCut mask
        # ------------------------------------------------------------

        mask = np.full(
            (h, w),
            cv2.GC_BGD,
            dtype=np.uint8,
        )

        # ------------------------------------------------------------
        # Probable foreground region
        #
        # Start with the hand landmarks and expand them.
        # ------------------------------------------------------------

        probable_fg = np.zeros(
            (h, w),
            dtype=np.uint8,
        )

        # Hand skeleton
        fingers = [
            [0, 1, 2, 3, 4],
            [0, 5, 6, 7, 8],
            [0, 9, 10, 11, 12],
            [0, 13, 14, 15, 16],
            [0, 17, 18, 19, 20],
        ]

        # Draw skeleton into probable foreground
        for finger in fingers:

            for i in range(len(finger) - 1):

                p1 = tuple(points[finger[i]])
                p2 = tuple(points[finger[i + 1]])

                distance = np.linalg.norm(
                    points[finger[i]]
                    - points[finger[i + 1]]
                )

                thickness = max(
                    10,
                    int(distance * 0.75),
                )

                cv2.line(
                    probable_fg,
                    p1,
                    p2,
                    255,
                    thickness=thickness,
                    lineType=cv2.LINE_AA,
                )

            # Joint circles
            for idx in finger:
                p = tuple(points[idx])

                cv2.circle(
                    probable_fg,
                    p,
                    12,
                    255,
                    -1,
                    lineType=cv2.LINE_AA,
                )

        # ------------------------------------------------------------
        # Expand probable foreground slightly
        # ------------------------------------------------------------

        expansion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (11, 11),
        )

        probable_fg = cv2.dilate(
            probable_fg,
            expansion_kernel,
            iterations=1,
        )

        mask[probable_fg > 0] = cv2.GC_PR_FGD

        # ------------------------------------------------------------
        # Certain foreground
        #
        # A narrower skeleton.
        # ------------------------------------------------------------

        certain_fg = np.zeros(
            (h, w),
            dtype=np.uint8,
        )

        for finger in fingers:

            for i in range(len(finger) - 1):

                p1 = tuple(points[finger[i]])
                p2 = tuple(points[finger[i + 1]])

                distance = np.linalg.norm(
                    points[finger[i]]
                    - points[finger[i + 1]]
                )

                thickness = max(
                    5,
                    int(distance * 0.40),
                )

                cv2.line(
                    certain_fg,
                    p1,
                    p2,
                    255,
                    thickness=thickness,
                    lineType=cv2.LINE_AA,
                )

            for idx in finger:
                p = tuple(points[idx])

                cv2.circle(
                    certain_fg,
                    p,
                    6,
                    255,
                    -1,
                    lineType=cv2.LINE_AA,
                )

        mask[certain_fg > 0] = cv2.GC_FGD

        # ------------------------------------------------------------
        # Wrist / palm
        #
        # Connect the five MCP points with the wrist.
        # ------------------------------------------------------------

        palm_points = np.array(
            [
                points[0],
                points[5],
                points[9],
                points[13],
                points[17],
            ],
            dtype=np.int32,
        )

        palm_hull = cv2.convexHull(palm_points)

        # Fill the palm as probable foreground
        cv2.fillConvexPoly(
            mask,
            palm_hull,
            cv2.GC_PR_FGD,
        )

        # Make the central palm certain foreground
        palm_center = np.mean(
            palm_points,
            axis=0,
        ).astype(np.int32)

        palm_radius = max(
            8,
            int(min(w, h) * 0.07),
        )

        cv2.circle(
            mask,
            tuple(palm_center),
            palm_radius,
            cv2.GC_FGD,
            -1,
        )

        # ------------------------------------------------------------
        # Strong background border
        # ------------------------------------------------------------

        border = max(
            5,
            int(min(w, h) * 0.05),
        )

        mask[:border, :] = cv2.GC_BGD
        mask[-border:, :] = cv2.GC_BGD
        mask[:, :border] = cv2.GC_BGD
        mask[:, -border:] = cv2.GC_BGD

        # ------------------------------------------------------------
        # GrabCut
        # ------------------------------------------------------------

        bgd_model = np.zeros(
            (1, 65),
            dtype=np.float64,
        )

        fgd_model = np.zeros(
            (1, 65),
            dtype=np.float64,
        )

        try:
            cv2.grabCut(
                crop,
                mask,
                None,
                bgd_model,
                fgd_model,
                iters,
                cv2.GC_INIT_WITH_MASK,
            )

        except cv2.error:
            return None

        # ------------------------------------------------------------
        # Final mask
        # ------------------------------------------------------------

        final_mask = np.where(
            (mask == cv2.GC_FGD)
            | (mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)

        # ------------------------------------------------------------
        # Remove tiny isolated components
        # ------------------------------------------------------------

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            final_mask,
            connectivity=8,
        )

        if num_labels > 1:

            # Largest component = presumed hand
            largest_label = 1 + np.argmax(
                stats[1:, cv2.CC_STAT_AREA]
            )

            final_mask = np.where(
                labels == largest_label,
                255,
                0,
            ).astype(np.uint8)

        # ------------------------------------------------------------
        # Small morphological cleanup
        # ------------------------------------------------------------

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        )

        final_mask = cv2.morphologyEx(
            final_mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1,
        )

        final_mask = cv2.morphologyEx(
            final_mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )

        # Slightly soften edge
        final_mask = cv2.GaussianBlur(
            final_mask,
            (3, 3),
            0,
        )

        # ------------------------------------------------------------
        # White background
        # ------------------------------------------------------------

        alpha = (
            final_mask.astype(np.float32)
            / 255.0
        )

        white = np.full_like(
            crop,
            255,
            dtype=np.uint8,
        )

        result = (
            crop.astype(np.float32)
            * alpha[..., None]
            +
            white.astype(np.float32)
            * (1.0 - alpha[..., None])
        )

        result = np.clip(
            result,
            0,
            255,
        ).astype(np.uint8)

        if scale < 1.0:
            result = cv2.resize(
                result,
                full_size,
                interpolation=cv2.INTER_LINEAR,
            )

        return result

    # ================================================================
    # Square bounding box
    # ================================================================

    @staticmethod
    def _square_bbox(bbox, width, height):
        """
        Creates a square bounding box around the hand.
        """

        x1, y1, x2, y2 = bbox

        box_w = x2 - x1
        box_h = y2 - y1

        size = max(
            box_w,
            box_h,
        )

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        sx1 = cx - size // 2
        sy1 = cy - size // 2

        sx2 = sx1 + size
        sy2 = sy1 + size

        # Shift inside image
        if sx1 < 0:
            sx2 -= sx1
            sx1 = 0

        if sy1 < 0:
            sy2 -= sy1
            sy1 = 0

        if sx2 > width:
            shift = sx2 - width
            sx1 -= shift
            sx2 = width

        if sy2 > height:
            shift = sy2 - height
            sy1 -= shift
            sy2 = height

        sx1 = max(0, sx1)
        sy1 = max(0, sy1)

        sx2 = min(width, sx2)
        sy2 = min(height, sy2)

        return (
            int(sx1),
            int(sy1),
            int(sx2),
            int(sy2),
        )

    # ================================================================
    # CNN input
    # ================================================================

    @staticmethod
    def crop_for_model(
        cleaned,
        bbox,
        output_size=64,
    ):
        """
        Extract segmented hand and resize to 64x64.
        """

        if cleaned is None or bbox is None:
            return None

        height, width = cleaned.shape[:2]

        x1, y1, x2, y2 = HandSegmenter._square_bbox(
            bbox,
            width,
            height,
        )

        crop = cleaned[
            y1:y2,
            x1:x2,
        ]

        if crop.size == 0:
            return None

        return cv2.resize(
            crop,
            (output_size, output_size),
            interpolation=cv2.INTER_AREA,
        )

    # ================================================================
    # Close
    # ================================================================

    def close(self):
        if self.landmarker is not None:
            self.landmarker.close()