"""Demo webcam pour la reconnaissance de mots, features RTMPose.

Meme principe que webcam_demo_words.py : on enregistre une fenetre de ~2.8 s
a la demande, puis on classe la sequence entiere. On ne classe pas en
continu, car segmenter un flux de signes est un probleme a part — savoir OU
commence et finit un signe est plus dur que de reconnaitre le signe.

Point critique : les features viennent de frame_features() importee de
extract_rtmpose.py, la fonction meme qui a produit l entrainement. Les
reimplementer ici reintroduirait le bug qui, sur l alphabet, faisait tomber
la precision de 0.985 a 0.384. Tout ce qui decide de ce que le modele voit —
seuil de confiance a 0.7, filtre hors-cadre, arbitrage des mains inventees —
est donc applique par cette fonction partagee, pas recopie.

L inference tourne sur l image NON retournee. Sur une image miroir RTMPose
voit une personne inversee et echange ses etiquettes gauche/droite, et
l extraction ne retournait rien : on deduit d abord, on retourne ensuite
l image ET les coordonnees x pour l affichage. webcam_demo_words.py retourne
AVANT d extraire, ce qui lui donne une incoherence entrainement/inference que
seule l augmentation par miroir rend supportable.

Precision attendue : 0.6675 +- 0.0507 en top-1 sur 50 classes (hasard 0.02),
soit 33x le hasard, mesuree en validation croisee par signeur. Ce chiffre
vient de signeurs DU DATASET : sur une personne nouvelle, devant une webcam,
attendez nettement moins — les clips WLASL et une webcam en gros plan ne se
ressemblent pas (docs/WLASL.md section 7.7).
"""

import json
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpu_setup import enable_cuda_dlls

enable_cuda_dlls()

import tensorflow as tf

from extract_rtmpose import (
    FEATURE_DIM,
    LEFT_HAND,
    RIGHT_HAND,
    POSE_KEYPOINTS,
    COCO_LEFT_SHOULDER,
    COCO_RIGHT_SHOULDER,
    frame_features,
    hand_confidences,
)
from train_kfold import (MAX_LEN, Layout, add_handshape, add_rich_features,
                         mirror, pad_sequence)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
META_PATH = os.path.join(PROJECT_ROOT, "models", "wlasl_gru_rtmpose.meta.json")

RECORD_SECONDS = 2.8
TOP_K = 5
HOLD_FRAMES = 6

# Cadence des videos du dataset : mediane 30.0 fps (58% a 29.97, 11% a 23.98,
# min 20, max 30.4). L extraction ne reechantillonne pas, elle prend les
# frames consecutives a la cadence du fichier, donc le modele a appris des
# signes vus a ~30 images par seconde.
#
# La webcam, elle, avance a la vitesse du traitement : ~29 fps quand le GPU
# est libre, bien moins s il est occupe. Or le modele ne connait pas le temps,
# seulement le nombre de frames : a 15 fps un signe de 2.8 s arrive en 42
# frames et se lit comme un signe deux fois plus rapide. L augmentation par
# time_warp ne couvre que 0.8 a 1.25x, soit 24 a 37 fps.
#
# On enregistre donc l HORODATAGE de chaque frame et on ramene la sequence sur
# une grille a 30 fps avant de classer.
TARGET_FPS = 30.0

HAND_LINKS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
    (15, 16), (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]
# Indices DANS la liste POSE_KEYPOINTS = [nez, epG, epD, coudeG, coudeD,
# poignetG, poignetD, hancheG, hancheD].
POSE_LINKS = [(1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 7), (2, 8), (7, 8)]

COLOR_BONE = (150, 220, 140)
COLOR_JOINT = (60, 90, 235)
COLOR_POSE = (210, 180, 90)
COLOR_REC = (60, 60, 240)
COLOR_OK = (120, 210, 90)
COLOR_DIM = (160, 160, 160)


def load_model():
    """Charge l ensemble et rend une fonction de prediction.

    Trois choses doivent correspondre exactement a l entrainement :

    - la disposition des colonnes. Elle est construite a la main d apres la
      fiche, jamais devinee : 106 colonnes se lisent aussi bien comme 9
      points de pose plus 2 confiances que comme 10 points de pose, et le
      mauvais choix ferait negater une confiance au miroir sans erreur.
    - la moyenne sur les 3 modeles de l ensemble.
    - la moyenne avec l image miroir (TTA), appliquee sur la disposition
      brute, avant les features derivees.
    """
    if not os.path.exists(META_PATH):
        raise SystemExit(
            f"Fiche absente : {META_PATH}\n"
            "Lance d abord : python src/wlasl/train_final.py"
        )

    with open(META_PATH, encoding="utf-8") as handle:
        meta = json.load(handle)

    if meta.get("extractor") != "rtmpose":
        raise SystemExit(
            f"Cette fiche a ete produite avec '{meta.get('extractor')}', pas "
            f"rtmpose. Utilise webcam_demo_words.py pour un modele MediaPipe."
        )

    paths = [os.path.join(PROJECT_ROOT, p) for p in meta["models"]]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit("Modeles absents : " + ", ".join(missing))

    models = [tf.keras.models.load_model(p) for p in paths]
    rich = meta.get("rich", False)
    confidence = meta.get("confidence", False)
    handshape = meta.get("handshape", False)

    tail = 2 if confidence else 0
    layout = Layout.from_dim(FEATURE_DIM)
    layout.tail = tail

    expected = FEATURE_DIM + tail + (23 if rich else 0) + (8 if handshape else 0)
    if expected != meta["feature_dim"]:
        raise SystemExit(
            f"La fiche annonce {meta['feature_dim']} dimensions, ce code en "
            f"produirait {expected}. L extraction et la demo ont divergé."
        )

    # Chaque modele est enveloppe dans une tf.function : sans cela chaque
    # appel repasse par le mode eager de Keras, ce qui coutait 1663 ms pour
    # les 6 passes (3 modeles x 2 vues) et gelait l image apres chaque
    # enregistrement. Le graphe est trace une fois, au chargement.
    forwards = []
    for model in models:
        @tf.function(reduce_retracing=True)
        def forward(batch, _model=model):
            return _model(batch, training=False)
        forwards.append(forward)

    def predict(padded):
        """padded : (MAX_LEN, 104 + queue), disposition brute."""
        batch = np.stack([padded, mirror(padded[None, ...], layout)[0]])

        # MEME ORDRE qu a l entrainement : rich puis handshape. La
        # verification de dimension ci-dessus n attraperait pas une
        # inversion, seulement un total faux.
        pieces = []
        if rich:
            pieces.append(add_rich_features(batch, layout)[:, :, -23:])
        if handshape:
            pieces.append(add_handshape(batch, layout)[:, :, -8:])
        if pieces:
            batch = np.concatenate([batch] + pieces, axis=-1)

        tensor = tf.constant(batch, dtype=tf.float32)

        total = None
        for forward in forwards:
            probabilities = forward(tensor).numpy()
            total = probabilities if total is None else total + probabilities

        # somme sur les modeles ET sur les deux vues
        return total.sum(axis=0) / (len(models) * 2)

    # Trace les graphes maintenant, sinon la premiere classification paierait
    # la compilation en pleine utilisation.
    predict(np.zeros((MAX_LEN, FEATURE_DIM + tail), dtype=np.float32))

    return predict, meta, layout, confidence


class Source:
    """Produit les features, exactement comme l extraction.

    Le squelette rendu pour l affichage est en PIXELS et dans le repere de
    l image non retournee ; c est l appelant qui le retourne.
    """

    def __init__(self, mode="balanced", with_confidence=True):
        from rtmlib import Wholebody

        self.model = Wholebody(mode=mode, backend="onnxruntime", device="cuda")
        self.with_confidence = with_confidence
        self.provider = self.model.pose_model.session.get_providers()[0]

    def process(self, frame):
        height, width = frame.shape[:2]
        keypoints, scores = self.model(frame)

        dim = FEATURE_DIM + (2 if self.with_confidence else 0)

        if len(keypoints) == 0:
            return np.zeros(dim, dtype=np.float32), None, False

        points = np.asarray(keypoints)[0]
        confidence = np.asarray(scores)[0]

        vector, found = frame_features(points, confidence, width, height)
        if self.with_confidence:
            vector = np.concatenate(
                [vector, np.asarray(hand_confidences(confidence), dtype=np.float32)])

        # Ce qui est DESSINE suit ce que le modele voit : les drapeaux de
        # presence sont lus dans le vecteur, donc une main ecartee par le
        # seuil, le hors-cadre ou l arbitrage n apparait pas a l ecran.
        flags = vector[FEATURE_DIM - len(POSE_KEYPOINTS) * 2 - 2:
                       FEATURE_DIM - len(POSE_KEYPOINTS) * 2]
        drawable = {
            "pose": points[POSE_KEYPOINTS],
            "left": points[LEFT_HAND] if flags[0] > 0.5 else None,
            "right": points[RIGHT_HAND] if flags[1] > 0.5 else None,
        }
        return vector, drawable, found


class Smoother:
    """Garde la derniere main vue quelques frames, POUR L AFFICHAGE SEUL.

    A seuil 0.7 une main n est retenue que dans ~57% des frames, donc le
    squelette clignote et donne l impression d un suivi rate. Ce lissage ne
    touche jamais les features : une main ecartee doit rester ecartee, sinon
    la demo nourrirait le modele de positions inventees.
    """

    def __init__(self):
        self.last = {"pose": None, "left": None, "right": None}
        self.age = {"pose": 0, "left": 0, "right": 0}

    def update(self, drawable):
        if drawable is None:
            drawable = {"pose": None, "left": None, "right": None}

        shown = {}
        for key, value in drawable.items():
            if value is not None:
                self.last[key] = value
                self.age[key] = 0
                shown[key] = value
            elif self.last[key] is not None and self.age[key] < HOLD_FRAMES:
                self.age[key] += 1
                shown[key] = self.last[key]
            else:
                shown[key] = None
        return shown


def draw_skeleton(frame, shown, width):
    def flip(point):
        return (int(width - point[0]), int(point[1]))

    pose = shown.get("pose")
    if pose is not None:
        for a, b in POSE_LINKS:
            cv2.line(frame, flip(pose[a]), flip(pose[b]), COLOR_POSE, 2, cv2.LINE_AA)

    for key in ("left", "right"):
        hand = shown.get(key)
        if hand is None:
            continue
        points = [flip(p) for p in hand]
        for a, b in HAND_LINKS:
            cv2.line(frame, points[a], points[b], COLOR_BONE, 2, cv2.LINE_AA)
        for index, point in enumerate(points):
            cv2.circle(frame, point, 5 if index == 0 else 3, COLOR_JOINT, -1, cv2.LINE_AA)


def draw_panel(frame, lines, title=None):
    height, width = frame.shape[:2]
    panel_height = 34 + 30 * len(lines) + (28 if title else 0)
    top = height - panel_height

    region = frame[top:height, 0:width]
    cv2.addWeighted(np.full_like(region, 22), 0.65, region, 0.35, 0, region)
    cv2.line(frame, (0, top), (width, top), COLOR_BONE, 1)

    y = top + 26
    if title:
        cv2.putText(frame, title, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, COLOR_DIM, 1, cv2.LINE_AA)
        y += 28
    for text, colour, scale in lines:
        cv2.putText(frame, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, colour, 2, cv2.LINE_AA)
        y += 30


def resample(samples, target_fps=TARGET_FPS):
    """(horodatage, vecteur) -> sequence a cadence fixe.

    Par PLUS PROCHE VOISIN, jamais par interpolation. L extraction
    sous-echantillonne les clips longs en choisissant des frames existantes
    (`wanted[int(i * step)]`), elle n invente aucune position intermediaire ;
    interpoler ici creerait des coordonnees que le modele n a jamais vues.
    Mesure a l appui : combler des trous par interpolation lineaire coutait
    1.4 point (docs/WLASL.md 7.5b).

    Quand la webcam tourne plus lentement que la cible, des frames sont
    repetees : la trajectoire est saccadee mais sa DUREE est juste, ce qui
    est le defaut le moins grave des deux.
    """
    if len(samples) < 2:
        return np.array([v for _, v in samples], dtype=np.float32)

    stamps = np.array([t for t, _ in samples], dtype=np.float64)
    vectors = np.array([v for _, v in samples], dtype=np.float32)

    duration = stamps[-1] - stamps[0]
    if duration <= 0:
        return vectors

    grid = np.arange(0.0, duration, 1.0 / target_fps)
    picked = np.abs(stamps[None, :] - (stamps[0] + grid)[:, None]).argmin(axis=1)
    return vectors[picked]


def classify(predict, samples, classes):
    sequence = resample(samples)
    padded = pad_sequence(np.array(sequence, dtype=np.float32))
    probabilities = predict(padded)
    order = np.argsort(probabilities)[::-1][:TOP_K]
    return [(classes[i], float(probabilities[i])) for i in order]


def run(camera=0, mode="balanced"):
    predict, meta, layout, confidence = load_model()
    classes = meta["classes"]
    expected = meta.get("expected_accuracy") or {}

    source = Source(mode=mode, with_confidence=confidence)
    print(f"RTMPose {mode} | {source.provider}")
    if source.provider != "CUDAExecutionProvider":
        print("  ATTENTION : CPU, environ 20x plus lent")
    print(f"Modele : {meta['num_classes']} signes, {meta['feature_dim']} dimensions, "
          f"precision attendue {expected.get('mean', '?')} "
          f"(+-{expected.get('std', '?')}, validation croisee par signeur)")
    print("ESPACE : enregistrer  |  l : squelette  |  q : quitter")
    print(f"\nSignes connus :\n  {', '.join(classes)}\n")

    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        print("Impossible d ouvrir la webcam.")
        return

    smoother = Smoother()
    times = deque(maxlen=30)
    recording = False
    started_at = 0.0
    sequence = []
    results_top = []
    show_skeleton = True

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            started = time.perf_counter()
            vector, drawable, person = source.process(frame)
            times.append(time.perf_counter() - started)

            height, width = frame.shape[:2]
            frame = cv2.flip(frame, 1)
            shown = smoother.update(drawable)

            if show_skeleton:
                draw_skeleton(frame, shown, width)

            if recording:
                elapsed = time.time() - started_at
                sequence.append((elapsed, vector))
                remaining = max(RECORD_SECONDS - elapsed, 0.0)

                filled = int(width * min(elapsed / RECORD_SECONDS, 1.0))
                cv2.rectangle(frame, (0, 0), (filled, 8), COLOR_REC, -1)
                cv2.putText(frame, f"ENREGISTREMENT  {remaining:.1f}s",
                            (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            COLOR_REC, 2, cv2.LINE_AA)

                if elapsed >= RECORD_SECONDS:
                    recording = False
                    if len(sequence) >= 8:
                        captured = len(sequence) / max(elapsed, 1e-6)
                        resampled = len(resample(sequence))
                        print(f"  {len(sequence)} frames en {elapsed:.2f}s "
                              f"({captured:.1f} fps) -> {resampled} frames "
                              f"a {TARGET_FPS:.0f} fps")
                        if captured < TARGET_FPS * 0.8:
                            print(f"  ATTENTION : cadence basse, des frames "
                                  f"sont repetees")
                        results_top = classify(predict, sequence, classes)
                    else:
                        results_top = []
                    sequence = []

            if results_top:
                lines = []
                for rank, (word, score) in enumerate(results_top):
                    colour = COLOR_OK if rank == 0 else COLOR_DIM
                    scale = 0.85 if rank == 0 else 0.6
                    lines.append((f"{rank + 1}. {word:<14} {score * 100:5.1f}%",
                                  colour, scale))
                draw_panel(frame, lines, "top 5")
            elif not recording:
                hint = ("ESPACE pour enregistrer" if person
                        else "place-toi dans le cadre (buste visible)")
                draw_panel(frame, [(hint, COLOR_OK if person else COLOR_REC, 0.7)])

            fps = 1.0 / max(np.mean(times), 1e-6)
            colour = COLOR_OK if fps >= TARGET_FPS * 0.8 else COLOR_REC
            cv2.putText(frame, f"{fps:4.1f} / {TARGET_FPS:.0f} fps",
                        (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        colour, 1, cv2.LINE_AA)

            cv2.imshow("WLASL50 - RTMPose", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("l"):
                show_skeleton = not show_skeleton
            if key == ord(" ") and not recording and person:
                recording = True
                started_at = time.time()
                sequence = []
                results_top = []
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--mode", default="balanced",
                        choices=("performance", "lightweight", "balanced"))
    args = parser.parse_args()
    run(args.camera, args.mode)
