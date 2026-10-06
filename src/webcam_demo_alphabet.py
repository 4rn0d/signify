"""Ouvre la webcam, detecte la main et affiche la lettre predite en direct.

Deux points a ne pas casser :

1. Meme pretraitement image qu'a l'entrainement. Le CNN a ete entraine sur
   des mains detourees sur fond blanc (HandSegmenter + GrabCut). Lui envoyer
   un crop brut de la webcam fait chuter la precision de 0.985 a 0.384.

2. Meme espace de coordonnees pour les landmarks. MediaPipe normalise x et y
   par la largeur et la hauteur de l'image. Le dataset est carre (200x200),
   donc isotrope ; une webcam 640x480 ne l'est pas et etirerait la geometrie
   de 4:3 (decalage mesure : 0.11, soit ~5% de l'amplitude des coordonnees).
   HandSegmenter.last_landmarks est donc deja en espace pixel.
"""

import json
import os
import time
from collections import deque

import cv2
import numpy as np
import tensorflow as tf

from hand_segmenter import GRABCUT_REALTIME, HandSegmenter
from landmark_extractor import normalize_landmarks


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

CLASSES_PATH = os.path.join(MODELS_DIR, "class_names.json")

# Fusion image + landmarks : meilleur score mesure (0.9431 contre 0.8722
# pour le CNN seul, a echantillons identiques). Utilise si present.
FUSION_MODEL_PATH = os.path.join(MODELS_DIR, "sign_model_fusion.keras")
FUSION_META_PATH = os.path.join(MODELS_DIR, "sign_model_fusion.meta.json")

# Repli CNN seul.
MODEL_PATH = os.path.join(MODELS_DIR, "sign_model.keras")
TFLITE_PATH = os.path.join(MODELS_DIR, "sign_model.tflite")
META_PATH = os.path.join(MODELS_DIR, "sign_model.meta.json")

# doit correspondre a la taille utilisee a l'entrainement
IMG_SIZE = 64

# Le modele n'a pas de classe "nothing" : il choisit toujours une des 28
# lettres, meme sur une main au repos. En dessous de ce seuil on affiche "?"
# plutot qu'une lettre clignotante.
CONFIDENCE_THRESHOLD = 80.0

# Champ de vision horizontal suppose de la webcam. Sert a deduire la focale
# en pixels, donc l'echelle absolue de la distance. NON CALIBRE : une valeur
# fausse decale toutes les distances du meme facteur. Pour calibrer, tiens la
# main a une distance connue et ajuste jusqu'a ce que l'affichage colle.
CAMERA_FOV_DEG = 60.0

# Longueur de paume de reference (poignet -> articulation du majeur), en
# metres. MediaPipe fournit des landmarks "metriques", mais leur ECHELLE est
# instable : mesure sur 9 poses de la meme main, l'empan va de 7.01 a 10.68 cm
# (39% d'ecart). Comme distance = focale x taille_reelle / taille_pixels, une
# taille sous-estimee rapproche la main d'autant — c'est ce qui faisait lire
# 12 cm au lieu de 30 sur un dos de main.
# On ne garde donc de MediaPipe que la FORME, et on impose l'echelle ici.
# Pour calibrer : mesure ton poignet -> jointure du majeur (~9-10 cm adulte).
HAND_LENGTH_M = 0.095

# Plage utilisable : en dessous la main deborde du cadre, au dela elle est
# trop petite pour que le modele ait du detail exploitable.
NEAR_LIMIT_M = 0.15
FAR_LIMIT_M = 1.50

# L'estimation varie de ~50% d'une pose a l'autre (bruit de landmarks). Sans
# lissage, un seuil fixe clignoterait en permanence. On prend la MEDIANE des
# dernieres frames : contrairement a la moyenne, elle ignore les solutions
# aberrantes isolees au lieu de s'y laisser tirer.
DISTANCE_WINDOW = 9

# Marge pour sortir d'une alerte (10%) : evite le clignotement quand la main
# est pile sur le seuil.
DISTANCE_HYSTERESIS = 0.10

# Cadrage : marge au bord du cadre en dessous de laquelle on considere la
# main coupee. MediaPipe extrapole les landmarks hors cadre (mesure : jusqu'a
# x=663 sur une image large de 640), donc un point hors limites est un signe
# franc ; la boite qui touche le bord previent plus tot, avant que la main ne
# soit vraiment tronquee. Au-dela, MediaPipe cesse simplement de detecter.
EDGE_MARGIN_PX = 14

# Nombre de frames consecutives avant d'afficher ou de lever l'alerte cadrage.
FRAMING_PATIENCE = 3

# --------------------------------------------------------------------------
# Saisie de texte
# --------------------------------------------------------------------------
# Le probleme : le modele predit a chaque frame (~34 fps). Ecrire a chaque
# prediction donnerait "AAAAAAAA" des qu'on tient une lettre, et du bruit des
# que la prediction oscille. On exige donc qu'une lettre soit TENUE, puis
# RELACHEE avant d'en accepter une autre.
#
# En secondes et non en frames : le debit varie avec la charge CPU, un seuil
# en frames rendrait la saisie plus lente ou plus nerveuse selon la machine.

# Duree pendant laquelle une lettre doit rester stable pour etre validee.
HOLD_SECONDS = 0.7

# Duree de "repos" (main baissee, confiance faible, ou cadrage invalide)
# avant qu'une nouvelle lettre puisse etre validee. C'est ce qui permet
# d'ecrire deux fois la meme lettre : on relache entre les deux.
RELEASE_SECONDS = 0.3

# Nombre de caracteres affiches dans le champ (on garde la fin du texte).
TEXT_FIELD_CHARS = 32

# Hauteur du bandeau de saisie, en pixels.
TEXT_FIELD_HEIGHT = 58

# Opacite du bandeau : 0 = invisible, 1 = opaque. En dessous de 1 l'image
# reste visible derriere, ce qui evite de masquer une main basse.
TEXT_FIELD_OPACITY = 0.6

# Periode de clignotement du curseur, en secondes (moitie visible).
CURSOR_BLINK_SECONDS = 1.0

# Squelette de la main MediaPipe : 5 doigts partant du poignet (0),
# plus la ligne de paume reliant la base des doigts.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # pouce
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (0, 9), (9, 10), (10, 11), (11, 12),       # majeur
    (0, 13), (13, 14), (14, 15), (15, 16),     # annulaire
    (0, 17), (17, 18), (18, 19), (19, 20),     # auriculaire
    (5, 9), (9, 13), (13, 17),                 # paume
]

# BGR
COLOR_BONE = (150, 220, 140)
COLOR_JOINT = (60, 90, 235)
COLOR_OK = (120, 210, 90)
COLOR_UNSURE = (60, 150, 235)
COLOR_NONE = (70, 70, 220)
COLOR_ALERT = (60, 60, 240)


def read_meta(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_classifier():
    """Retourne (predict, class_names, description).

    predict(crop_bgr, landmarks_px) -> vecteur de probabilites.
    La branche landmarks est ignoree par les modeles image seule, pour que
    la boucle d'affichage n'ait pas a connaitre le type de modele.
    """
    with open(CLASSES_PATH, encoding="utf-8") as handle:
        class_names = json.load(handle)

    if os.path.exists(FUSION_MODEL_PATH):
        model = tf.keras.models.load_model(FUSION_MODEL_PATH)

        @tf.function
        def forward(image, landmarks):
            return model(
                {"image": image, "landmarks": landmarks}, training=False
            )

        def predict(crop_bgr, landmarks_px):
            image = prepare_image(crop_bgr)
            landmarks = prepare_landmarks(landmarks_px)
            return forward(
                tf.constant(image), tf.constant(landmarks)
            ).numpy()[0]

        meta = read_meta(FUSION_META_PATH) or {}
        description = (
            f"FUSION image+landmarks | val_accuracy={meta.get('val_accuracy', '?')} "
            f"| {meta.get('created', '?')}"
        )
        return predict, class_names, description

    # --- repli : CNN seul -------------------------------------------------
    meta = read_meta(META_PATH) or {}
    description = (
        f"CNN seul | split={meta.get('split', '?')} "
        f"| val_accuracy={meta.get('val_accuracy', '?')}"
        "  (sign_model_fusion.keras absent : lance train_fusion.py)"
    )

    if os.path.exists(TFLITE_PATH):
        interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH, num_threads=4)
        interpreter.allocate_tensors()

        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]

        def predict(crop_bgr, landmarks_px):
            interpreter.set_tensor(input_index, prepare_image(crop_bgr))
            interpreter.invoke()
            return interpreter.get_tensor(output_index)[0]

        return predict, class_names, "TFLite " + description

    model = tf.keras.models.load_model(MODEL_PATH)

    @tf.function
    def forward(image):
        return model(image, training=False)

    def predict(crop_bgr, landmarks_px):
        return forward(tf.constant(prepare_image(crop_bgr))).numpy()[0]

    return predict, class_names, "Keras " + description


def prepare_image(crop_bgr):
    """crop_for_model() a deja redimensionne en 64x64 (INTER_AREA), comme a
    la preparation du dataset. Il ne reste que BGR -> RGB et le /255."""
    img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return np.expand_dims(img.astype("float32") / 255.0, axis=0)


def prepare_landmarks(landmarks_px):
    """(21, 3) en pixels -> (1, 63) normalise poignet-origine, echelle 1."""
    if landmarks_px is None:
        return np.zeros((1, 63), dtype=np.float32)

    return normalize_landmarks(landmarks_px).reshape(1, -1).astype("float32")


def estimate_distance(landmarks_px, world_landmarks, frame_shape):
    """Distance camera-main en metres, ou None.

    MediaPipe fournit la main en metres (hand_world_landmarks) ET sa
    projection en pixels. solvePnP retrouve la pose qui relie les deux :
    la norme du vecteur de translation est la distance.

    On resout sur les 21 points plutot que sur la largeur de paume seule :
    une paume vue de biais parait plus etroite, ce qui faisait lire 4.45 m
    sur un "C" a 0.58 m. Mesure sur 8 lettres a distance constante,
    l'ecart passe de 335% (largeur de paume) a 58% (PnP).

    Precision : ordre de grandeur, pas mesure. ~25% d'erreur selon la pose,
    plus l'erreur d'echelle de CAMERA_FOV_DEG.
    """
    if landmarks_px is None or world_landmarks is None:
        return None

    # Rescale sur une taille de main FIXE : on conserve la forme 3D predite
    # par MediaPipe mais on remplace son estimation d'echelle, instable.
    model = world_landmarks.astype(np.float64) - world_landmarks[0]
    palm = np.linalg.norm(model[9] - model[0])   # poignet -> majeur, rigide

    if palm < 1e-6:
        return None

    model = model / palm * HAND_LENGTH_M

    height, width = frame_shape[:2]
    focal = (width / 2.0) / np.tan(np.radians(CAMERA_FOV_DEG) / 2.0)

    camera_matrix = np.array(
        [[focal, 0, width / 2.0],
         [0, focal, height / 2.0],
         [0, 0, 1]],
        dtype=np.float64,
    )

    try:
        ok, _, translation = cv2.solvePnP(
            model,
            landmarks_px[:, :2].astype(np.float64),
            camera_matrix,
            None,
            flags=cv2.SOLVEPNP_SQPNP,
        )
    except cv2.error:
        return None

    if not ok:
        return None

    distance = float(np.linalg.norm(translation))

    # Garde-fou : au-dela de ces bornes c'est une resolution aberrante,
    # pas une main reellement a 5 m dans le champ.
    return distance if 0.05 < distance < 3.0 else None


class TextComposer:
    """Transforme un flux de predictions par frame en texte ecrit.

    Machine a etats volontairement simple :

      repos --(lettre stable HOLD_SECONDS)--> valide --(relachement)--> repos

    Une lettre n'est validee qu'une fois ; il faut relacher (baisser la main,
    changer de signe, ou sortir des conditions valides) avant la suivante.
    C'est ce qui evite a la fois la repetition involontaire et le bruit du
    scintillement : une prediction qui oscille ne reste jamais stable assez
    longtemps pour atteindre HOLD_SECONDS.
    """

    def __init__(self):
        self.text = ""
        self.candidate = None
        self.candidate_since = 0.0
        self.locked = False
        self.idle_since = None

    def update(self, letter, writable, now=None):
        """letter : prediction courante, ou None.

        writable : True seulement si la main est bien cadree, a bonne
        distance, et la prediction au-dessus du seuil de confiance.

        Retourne la progression du maintien (0..1) pour l'affichage.
        """
        now = time.monotonic() if now is None else now

        if not writable or letter is None:
            # Phase de repos : au bout de RELEASE_SECONDS on se rearme.
            if self.idle_since is None:
                self.idle_since = now
            elif now - self.idle_since >= RELEASE_SECONDS:
                self.locked = False
            self.candidate = None
            return 0.0

        self.idle_since = None

        if letter != self.candidate:
            self.candidate = letter
            self.candidate_since = now
            # Changer de lettre compte aussi comme un relachement.
            self.locked = False
            return 0.0

        if self.locked:
            return 1.0

        progress = (now - self.candidate_since) / HOLD_SECONDS

        if progress >= 1.0:
            self._commit(letter)
            self.locked = True
            return 1.0

        return progress

    def _commit(self, letter):
        if letter == "space":
            self.text += " "
        elif letter == "del":
            self.text = self.text[:-1]
        else:
            self.text += letter

    def clear(self):
        self.text = ""
        self.candidate = None
        self.locked = False


class FramingTracker:
    """Dit si la main sort du cadre, et de quel cote.

    L'image est miroir (cv2.flip), donc les directions affichees
    correspondent a ce que l'utilisateur voit bouger a l'ecran.
    """

    DIRECTIONS = {
        "left": "deplace vers la droite",
        "right": "deplace vers la gauche",
        "top": "descends la main",
        "bottom": "monte la main",
    }

    def __init__(self):
        self.pending = None
        self.count = 0
        self.message = None

    def update(self, landmarks_px, frame_shape):
        sides = self._sides(landmarks_px, frame_shape)

        if not sides:
            candidate = None
        elif len(sides) > 1:
            candidate = "CADRE TA MAIN"
        else:
            candidate = f"MAIN COUPEE - {self.DIRECTIONS[sides[0]]}"

        # Il faut FRAMING_PATIENCE frames d'accord avant de changer d'etat.
        if candidate == self.pending:
            self.count += 1
        else:
            self.pending = candidate
            self.count = 1

        if self.count >= FRAMING_PATIENCE:
            self.message = candidate

        return self.message

    @staticmethod
    def _sides(landmarks_px, frame_shape):
        if landmarks_px is None:
            return []

        height, width = frame_shape[:2]
        xs, ys = landmarks_px[:, 0], landmarks_px[:, 1]

        sides = []
        if xs.min() < EDGE_MARGIN_PX:
            sides.append("left")
        if xs.max() > width - EDGE_MARGIN_PX:
            sides.append("right")
        if ys.min() < EDGE_MARGIN_PX:
            sides.append("top")
        if ys.max() > height - EDGE_MARGIN_PX:
            sides.append("bottom")

        return sides


class DistanceTracker:
    """Lisse l'estimation de distance et dit si la main est bien placee.

    Etats : "ok", "near" (trop pres), "far" (trop loin), "unknown".
    L'hysteresis fait qu'une alerte declenchee a NEAR_LIMIT_M ne se leve
    qu'une fois repasse 10% au-dela, sinon l'affichage clignote sur le seuil.
    """

    def __init__(self):
        self.history = deque(maxlen=DISTANCE_WINDOW)
        self.state = "unknown"

    def update(self, distance):
        if distance is None:
            self.history.clear()
            self.state = "unknown"
            return None, self.state

        self.history.append(distance)

        values = sorted(self.history)
        smoothed = values[len(values) // 2]       # mediane

        near = NEAR_LIMIT_M
        far = FAR_LIMIT_M

        # Seuils elargis tant que l'alerte est active.
        if self.state == "near":
            near *= 1.0 + DISTANCE_HYSTERESIS
        elif self.state == "far":
            far *= 1.0 - DISTANCE_HYSTERESIS

        if smoothed < near:
            self.state = "near"
        elif smoothed > far:
            self.state = "far"
        else:
            self.state = "ok"

        return smoothed, self.state


def draw_landmarks(frame, landmarks_px):
    """Dessine le squelette de la main sur l'image."""
    if landmarks_px is None:
        return

    points = landmarks_px[:, :2].astype(np.int32)

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            tuple(points[start]),
            tuple(points[end]),
            COLOR_BONE,
            2,
            cv2.LINE_AA,
        )

    for index, (x, y) in enumerate(points):
        # Poignet un peu plus gros : c'est l'origine de la normalisation.
        radius = 5 if index == 0 else 3
        cv2.circle(frame, (int(x), int(y)), radius, COLOR_JOINT, -1, cv2.LINE_AA)


def draw_alert(frame, text):
    """Bandeau rouge en bas : la main est hors de la plage utilisable."""
    height, width = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)

    x = (width - tw) // 2
    y = height - TEXT_FIELD_HEIGHT - 18   # juste au-dessus du champ de texte

    cv2.rectangle(
        frame, (x - 14, y - th - 12), (x + tw + 14, y + 12), (0, 0, 0), -1
    )
    cv2.rectangle(
        frame, (x - 14, y - th - 12), (x + tw + 14, y + 12), COLOR_ALERT, 2
    )
    cv2.putText(
        frame, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_ALERT, 2, cv2.LINE_AA,
    )


def draw_text_field(frame, text, progress, pending):
    """Bandeau de saisie en bas : texte ecrit + progression du maintien."""
    height, width = frame.shape[:2]
    top = height - TEXT_FIELD_HEIGHT

    # Fond semi-transparent : on melange un aplat sombre avec l'image, plutot
    # que de la recouvrir.
    panel = frame[top:height, 0:width]
    cv2.addWeighted(
        np.full_like(panel, 24),
        TEXT_FIELD_OPACITY,
        panel,
        1.0 - TEXT_FIELD_OPACITY,
        0,
        panel,
    )

    cv2.line(frame, (0, top), (width, top), COLOR_BONE, 1)

    shown = text[-TEXT_FIELD_CHARS:] if len(text) > TEXT_FIELD_CHARS else text

    # Curseur clignotant. On dessine une espace quand il est masque pour que
    # la largeur du texte ne saute pas d'une frame a l'autre.
    half = CURSOR_BLINK_SECONDS / 2.0
    cursor = "_" if int(time.monotonic() / half) % 2 == 0 else " "

    cv2.putText(
        frame, shown + cursor, (14, top + 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2, cv2.LINE_AA,
    )

    # Barre de progression : remplie = la lettre va etre validee.
    bar_w = width - 28
    bar_y = top + 44

    cv2.rectangle(frame, (14, bar_y), (14 + bar_w, bar_y + 6), (60, 60, 60), -1)

    if progress > 0:
        filled = int(bar_w * min(progress, 1.0))
        colour = COLOR_OK if progress >= 1.0 else COLOR_BONE
        cv2.rectangle(frame, (14, bar_y), (14 + filled, bar_y + 6), colour, -1)

    if pending:
        label = {"space": "[espace]", "del": "[effacer]"}.get(pending, pending)
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(
            frame, label, (width - lw - 14, top + 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_BONE, 2, cv2.LINE_AA,
        )


def draw_preview(frame, crop):
    """Apercu de ce que le CNN recoit reellement, en haut a droite.

    Quand une prediction est fausse, cette vignette dit tout de suite si
    c'est la segmentation ou le modele qui a echoue.
    """
    size = 128
    margin = 10
    preview = cv2.resize(crop, (size, size), interpolation=cv2.INTER_NEAREST)

    x1 = frame.shape[1] - size - margin
    y1 = margin

    frame[y1:y1 + size, x1:x1 + size] = preview
    cv2.rectangle(
        frame, (x1 - 1, y1 - 1), (x1 + size, y1 + size), COLOR_BONE, 1
    )


def run_webcam_demo():
    predict, class_names, description = load_classifier()
    print(f"Modele : {description}")

    segmenter = HandSegmenter(num_hands=1, **GRABCUT_REALTIME)
    tracker = DistanceTracker()
    framing = FramingTracker()
    composer = TextComposer()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Impossible d'ouvrir la webcam.")
        segmenter.close()
        return

    print("'q' quitter | 'l' squelette | 'c' effacer le texte")
    print(
        f"Tiens une lettre {HOLD_SECONDS}s pour l'ecrire, puis relache."
    )
    show_landmarks = True

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # effet miroir, plus naturel

            cleaned, bbox, found = segmenter.process(frame)
            landmarks_px = segmenter.last_landmarks
            distance, placement = tracker.update(
                estimate_distance(
                    landmarks_px, segmenter.last_world_landmarks, frame.shape
                )
            )

            # Reinitialises a chaque frame : sans main, rien n'est ecrit.
            pending_letter = None
            writable = False

            if found and bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BONE, 1)

                if show_landmarks:
                    draw_landmarks(frame, landmarks_px)

                crop = segmenter.crop_for_model(
                    cleaned, bbox, output_size=IMG_SIZE
                )

                if crop is not None:
                    predictions = predict(crop, landmarks_px)

                    best = int(np.argmax(predictions))
                    confidence = float(predictions[best]) * 100

                    if confidence >= CONFIDENCE_THRESHOLD:
                        text = f"{class_names[best]} ({confidence:.0f}%)"
                        color = COLOR_OK
                    else:
                        text = f"? ({confidence:.0f}%)"
                        color = COLOR_UNSURE

                    cv2.putText(
                        frame, text, (x1, max(y1 - 12, 28)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA,
                    )

                    # Le cadrage prime : une main coupee fausse de toute
                    # facon l'estimation de distance.
                    framing_message = framing.update(landmarks_px, frame.shape)

                    if distance is not None:
                        label = framing_message or {
                            "near": "TROP PRES - recule",
                            "far": "TROP LOIN - approche",
                        }.get(placement)

                        cv2.putText(
                            frame, f"~{distance:.2f} m",
                            (x1, min(y2 + 26, frame.shape[0] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            COLOR_ALERT if label else COLOR_BONE, 1, cv2.LINE_AA,
                        )

                        if label:
                            draw_alert(frame, label)

                    # On n'ecrit que si TOUT est bon : confiance suffisante,
                    # cadrage correct, distance dans la plage utilisable.
                    # Une main coupee ou mal placee donne des predictions peu
                    # fiables — autant ne rien ecrire que d'ecrire faux.
                    if (
                        confidence >= CONFIDENCE_THRESHOLD
                        and framing_message is None
                        and placement == "ok"
                    ):
                        pending_letter = class_names[best]
                        writable = True

                    draw_preview(frame, crop)
            else:
                cv2.putText(
                    frame, "Aucune main detectee", (24, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_NONE, 2, cv2.LINE_AA,
                )

            progress = composer.update(pending_letter, writable)
            draw_text_field(frame, composer.text, progress, pending_letter)

            cv2.imshow("Sign Language Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("l"):
                show_landmarks = not show_landmarks
            if key == ord("c"):
                composer.clear()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        segmenter.close()


if __name__ == "__main__":
    run_webcam_demo()
