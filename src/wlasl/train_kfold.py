"""Entraine un GRU sur les sequences WLASL, en validation croisee par signeur.

Pourquoi k-fold plutot qu'un decoupage train/val/test : avec 9 a 16 clips par
glose, un ensemble de validation ferait ~3 clips par classe. Choisir l'epoch
d'arret sur un signal aussi bruite revient a tirer au sort, et cela amputerait
encore l'entrainement. En k-fold, chaque clip est teste exactement une fois et
on obtient un ecart-type — indispensable ici, puisque sur l'alphabet des ecarts
de 0.2 point se sont reveles etre du bruit (+-0.58).

Budget d'epochs FIXE, identique pour tous les folds. Utiliser le fold de test
pour arreter l'entrainement serait une fuite deguisee en rigueur.

Les folds sont construits par SIGNEUR : les decoupages officiels de WLASL
partagent 93% de leurs signeurs entre train et test, ce qui mesure en partie
la reconnaissance de personnes plutot que de signes.
"""

import argparse
import collections
import glob
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "wlasl", "features")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "wlasl", "download_manifest.json")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "models", "wlasl_kfold.json")

# Mediane 69 frames, p75 87. 80 couvre la majorite sans trop de remplissage.
MAX_LEN = 80

HAND_POINTS = 21


class Layout:
    """Disposition du vecteur de features, selon l extracteur utilise.

    MediaPipe donne 3 coordonnees par point (x, y, z) soit 155 colonnes ;
    RTMPose n en donne que 2 (x, y) soit 104. Toutes les fonctions qui
    indexent les mains, les drapeaux ou la pose doivent donc connaitre le
    nombre de coordonnees, sinon elles lisent les mauvaises colonnes — sans
    lever d erreur, en degradant simplement le resultat.
    """

    # Groupes de colonnes de queue a echanger au miroir, en decalage depuis
    # le DEBUT de la queue : (gauche, droite, largeur). Une colonne de queue
    # qui n y figure pas ne suivrait pas sa main quand l image est retournee.
    TAIL_GROUPS = {
        2: [(0, 1, 1)],                      # confiance par main
        12: [(0, 1, 1), (2, 7, 5)],          # + 5 bouts de doigts par main
    }

    def __init__(self, coords, tail=0):
        self.coords = coords
        self.tail = tail                  # colonnes ajoutees en fin de vecteur
        self.hand_block = HAND_POINTS * coords
        self.left_flag = self.hand_block * 2
        self.right_flag = self.left_flag + 1
        self.pose_start = self.left_flag + 2

    @classmethod
    def from_dim(cls, dim):
        """Deduit la disposition du nombre de colonnes.

        NE PEUT PAS deviner un tail : 106 colonnes se lisent aussi bien comme
        "9 points de pose + 2 colonnes de confiance" que comme "10 points de
        pose", et le second ferait negater x sur les confiances au miroir.
        Quand un tail existe, la disposition doit etre passee explicitement.
        """
        for coords in (3, 2):
            pose_points = (dim - HAND_POINTS * coords * 2 - 2) / coords
            if pose_points == int(pose_points) and pose_points > 0:
                return cls(coords)
        raise ValueError(f"disposition inconnue pour {dim} colonnes")

    def pose_points(self, dim):
        return (dim - self.tail - self.pose_start) // self.coords


# Disposition MediaPipe, conservee pour les constantes historiques.
HAND_BLOCK = HAND_POINTS * 3          # 63
LEFT_HAND = slice(0, HAND_BLOCK)
RIGHT_HAND = slice(HAND_BLOCK, HAND_BLOCK * 2)
LEFT_FLAG = HAND_BLOCK * 2
RIGHT_FLAG = HAND_BLOCK * 2 + 1
POSE_START = HAND_BLOCK * 2 + 2

# POSE_KEYPOINTS = [nez, epauleG, epauleD, coudeG, coudeD, poignetG, poignetD,
#                   hancheG, hancheD] -> paires gauche/droite a echanger.
POSE_MIRROR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8)]


def load_sequences(variants="all", features_dir=None, metadata_only=False,
                   with_confidence=False, with_meta=False, with_tips=False):
    """variants : 'all', 'dominant' (variante majoritaire seulement) ou
    'split' (chaque variante devient une classe a part).

    WLASL etiquette plusieurs gestes differents sous une meme glose : 19 des
    50 gloses ont plusieurs variation_id, et 15 d'entre elles n'ont aucune
    variante majoritaire a 75%. Le modele apprend donc des exemples qui se
    contredisent. 'dominant' supprime la contradiction en perdant 14% des
    clips ; 'split' la supprime sans rien perdre, au prix de classes plus
    nombreuses et plus petites.
    """
    with open(MANIFEST_PATH, encoding="utf-8") as handle:
        variation = {
            r["video_id"]: r.get("variation_id", 0)
            for r in json.load(handle)
        }

    records = []
    for path in sorted(glob.glob(os.path.join(features_dir or FEATURES_DIR, "*.npz"))):
        data = np.load(path, allow_pickle=True)
        video_id = str(data["video_id"])
        meta = None
        if metadata_only:
            vector = None
        else:
            vector = data["features"].astype(np.float32)
            if with_tips:
                if "hand_point_scores" not in data.files:
                    raise ValueError(
                        f"{os.path.basename(path)} ne contient pas "
                        f"hand_point_scores. --tip-confidence demande une "
                        f"extraction faite avec --raw."
                    )
                tips = data["hand_point_scores"].astype(np.float32)
                # 21 points par main, bouts aux indices 4, 8, 12, 16, 20
                picked = [t + offset for offset in (0, HAND_POINTS)
                          for t in (4, 8, 12, 16, 20)]
                vector = np.concatenate([vector, tips[:, picked]], axis=1)

            if with_meta:
                for field in ("hand_scores", "hand_inside", "hand_conflict"):
                    if field not in data.files:
                        raise ValueError(
                            f"{os.path.basename(path)} ne contient pas {field}. "
                            f"Le filtrage a l entrainement demande une extraction "
                            f"faite avec --raw."
                        )
                meta = {
                    "hand_scores": data["hand_scores"].astype(np.float32),
                    "hand_inside": data["hand_inside"].astype(np.float32),
                    "hand_conflict": data["hand_conflict"].astype(bool),
                }

            if with_confidence:
                if "hand_scores" not in data.files:
                    raise ValueError(
                        f"{os.path.basename(path)} ne contient pas hand_scores. "
                        f"--confidence demande une extraction faite apres l ajout "
                        f"de ce champ (data/wlasl/features_rtm07 et apres)."
                    )
                # En QUEUE du vecteur : inserees au milieu, elles decaleraient
                # pose_start et tous les indices codes en dur.
                vector = np.concatenate(
                    [vector, data["hand_scores"].astype(np.float32)], axis=1)

        records.append(
            {
                "features": vector,
                "gloss": str(data["gloss"]),
                "signer_id": int(data["signer_id"]),
                "video_id": video_id,
                "variation_id": int(variation.get(video_id, 0)),
                "meta": meta,
            }
        )

    if variants == "all":
        return records

    if variants == "split":
        for record in records:
            record["gloss"] = f"{record['gloss']}_v{record['variation_id']}"
        return records

    # dominant : on garde, par glose, la variante la plus representee —
    # pas forcement variation_id 0 (candy est a 7 contre 11 dans l'autre sens).
    by_gloss = collections.defaultdict(collections.Counter)
    for record in records:
        by_gloss[record["gloss"]][record["variation_id"]] += 1

    keep = {g: counts.most_common(1)[0][0] for g, counts in by_gloss.items()}
    return [r for r in records if r["variation_id"] == keep[r["gloss"]]]


def pad_sequence(sequence, max_len=MAX_LEN):
    """Tronque au centre si trop long, complete par des zeros sinon.

    Tronquer au centre plutot qu'a la fin : un clip contient souvent une
    amorce et un retour au repos, le signe est au milieu.
    """
    length = len(sequence)

    if length >= max_len:
        start = (length - max_len) // 2
        return sequence[start:start + max_len]

    padding = np.zeros((max_len - length, sequence.shape[1]), dtype=np.float32)
    return np.concatenate([sequence, padding], axis=0)


def mirror(batch, layout=None):
    """Miroir horizontal d'un lot de sequences.

    Negater x ne suffit pas : refleter une personne echange aussi sa main
    gauche et sa main droite, ainsi que les points de pose lateraux. Sans
    cet echange, le modele verrait une main droite rangee dans le canal
    gauche — exactement le genre d'incoherence qui avait empoisonne la
    fusion sur l'alphabet.
    """
    layout = layout or Layout.from_dim(batch.shape[-1])
    step = layout.coords

    out = batch.copy()

    left_hand = slice(0, layout.hand_block)
    right_hand = slice(layout.hand_block, layout.hand_block * 2)

    # x est la premiere coordonnee de chaque point.
    for block in (left_hand, right_hand):
        out[..., block.start:block.stop:step] *= -1.0

    # La negation de x s arrete AVANT la queue : un pas de `step` depuis
    # pose_start jusqu a la fin du vecteur tomberait sur les colonnes de
    # confiance et leur changerait le signe, sans erreur ni message.
    pose_end = batch.shape[-1] - layout.tail
    out[..., layout.pose_start:pose_end:step] *= -1.0

    # Echange des deux mains, puis des drapeaux de presence.
    left = out[..., left_hand].copy()
    out[..., left_hand] = out[..., right_hand]
    out[..., right_hand] = left

    flags = out[..., layout.left_flag].copy()
    out[..., layout.left_flag] = out[..., layout.right_flag]
    out[..., layout.right_flag] = flags

    # Echange des points de pose lateraux.
    for a, b in POSE_MIRROR_PAIRS:
        ia = layout.pose_start + a * step
        ib = layout.pose_start + b * step
        tmp = out[..., ia:ia + step].copy()
        out[..., ia:ia + step] = out[..., ib:ib + step]
        out[..., ib:ib + step] = tmp

    # Les colonnes de queue suivent leur main. Les oublier donnerait au modele
    # la confiance de la main droite en face des points de la gauche.
    if layout.tail:
        groups = Layout.TAIL_GROUPS.get(layout.tail)
        if groups is None:
            raise ValueError(
                f"queue de {layout.tail} colonnes sans regle d echange : "
                f"ajoute-la a Layout.TAIL_GROUPS, sinon le miroir laisserait "
                f"ces colonnes sur la mauvaise main sans lever d erreur."
            )
        base = batch.shape[-1] - layout.tail
        for left, right, width in groups:
            a = slice(base + left, base + left + width)
            b = slice(base + right, base + right + width)
            swap = out[..., a].copy()
            out[..., a] = out[..., b]
            out[..., b] = swap

    return out


def time_warp(sequence, rng, low=0.8, high=1.25):
    """Reechantillonne la sequence a une vitesse differente."""
    factor = rng.uniform(low, high)
    length = len(sequence)
    new_len = max(8, min(MAX_LEN, int(length * factor)))

    source = np.linspace(0, length - 1, new_len)
    warped = np.stack(
        [np.interp(source, np.arange(length), sequence[:, c])
         for c in range(sequence.shape[1])],
        axis=1,
    ).astype(np.float32)

    return pad_sequence(warped)


def augment_batch(batch, rng, layout=None, hand_dropout=0.0):
    out = np.empty_like(batch)

    for i, sequence in enumerate(batch):
        item = sequence

        if rng.random() < 0.5:
            item = time_warp(item, rng)

        # Abandon de frames : imite les echecs de detection, frequents ici
        # (la main n'est vue que dans ~60% des frames).
        if rng.random() < 0.3:
            item = item.copy()
            drop = rng.random(len(item)) < 0.1
            item[drop] = 0.0

        # Bruit leger sur les coordonnees.
        if rng.random() < 0.5:
            item = item + rng.normal(0, 0.01, item.shape).astype(np.float32)

        out[i] = item

    flip = rng.random(len(out)) < 0.5
    if flip.any():
        out[flip] = mirror(out[flip], layout)

    if hand_dropout > 0.0:
        out = drop_hand(out, rng, layout or Layout.from_dim(out.shape[-1]),
                        probability=hand_dropout)

    return out


def add_velocity(batch):
    """Concatene les deltas image-a-image aux positions.

    Un signe se definit par le MOUVEMENT : certaines paires ne different que
    par la direction du deplacement, a configuration de main identique. Le
    GRU peut deduire la vitesse des positions successives, mais la lui donner
    explicitement lui evite de l'apprendre.

    Les frames de remplissage restent nulles des deux cotes pour que le
    masquage fonctionne encore, et la vitesse est annulee a la frontiere
    entre vraie frame et remplissage : sinon la transition produirait un
    delta enorme et purement artificiel.
    """
    real = np.abs(batch).sum(axis=2) > 0

    velocity = np.zeros_like(batch)
    velocity[:, 1:] = batch[:, 1:] - batch[:, :-1]

    valid = real[:, 1:] & real[:, :-1]
    velocity[:, 1:][~valid] = 0.0
    velocity[:, 0] = 0.0

    return np.concatenate([batch, velocity], axis=2)


def drop_z_indices(feature_dim=155):
    """Indices a garder quand on retire la profondeur.

    MediaPipe estime z depuis une seule camera, et c'est peu fiable : son
    echelle metrique variait de 39% sur une meme main lors des mesures de
    distance. Un tiers des features est donc du bruit, pour un modele qui
    n'a qu'une dizaine d'exemples par classe.
    """
    # RTMPose ne fournit pas de profondeur : il n y a rien a retirer, et les
    # indices calcules ici (3 coordonnees par point) designeraient les
    # mauvaises colonnes sans lever d erreur.
    if feature_dim != 155:
        raise ValueError(
            f"--no-z ne s applique qu aux features MediaPipe (155 colonnes), "
            f"recu {feature_dim}. Les features RTMPose sont deja en 2D."
        )

    keep = []
    for start in (0, HAND_BLOCK):
        for point in range(HAND_POINTS):
            keep += [start + point * 3, start + point * 3 + 1]

    keep += [LEFT_FLAG, RIGHT_FLAG]          # les drapeaux n'ont pas de z

    for point in range((feature_dim - POSE_START) // 3):
        keep += [POSE_START + point * 3, POSE_START + point * 3 + 1]

    return np.array(keep, dtype=np.int32)


def add_depth_proxy(batch):
    """Ajoute 4 indices de profondeur derives des coordonnees x,y FIABLES.

    MediaPipe estime z depuis une seule camera et c est peu fiable : le
    retirer gagne 7.8 points. Mais la profondeur porte du sens en ASL
    (gestes vers l avant, bras tendu). On la reconstruit donc a partir de
    ce que MediaPipe reussit bien.

    Deux indices classiques, en unites de largeur d epaules donc deja
    invariants a la distance de la personne :

    - largeur de paume (jointures 5 a 17) : une main proche parait plus
      grande. Mesure : la variation A L INTERIEUR d un clip vaut 79% de la
      variation entre clips, elle suit donc bien le mouvement.
    - longueur d avant-bras (coude a poignet) : un bras tendu vers la
      camera parait raccourci.

    4 scalaires au lieu des 42 valeurs z, tous issus de coordonnees que
    MediaPipe estime correctement.
    """
    # Cette fonction code en dur la disposition MediaPipe (3 coordonnees).
    # Sur des features RTMPose (2 coordonnees) elle lirait les mauvaises
    # colonnes sans erreur : on refuse explicitement.
    if batch.shape[-1] != 155:
        raise ValueError(
            f"add_depth_proxy attend la disposition MediaPipe a 155 colonnes, "
            f"recu {batch.shape[-1]}. Utilise add_rich_features, qui gere les "
            f"deux dispositions."
        )

    def segment(a_off, b_off):
        a = batch[..., a_off:a_off + 2]
        b = batch[..., b_off:b_off + 2]
        return np.linalg.norm(a - b, axis=-1)

    left_palm = segment(5 * 3, 17 * 3)
    right_palm = segment(HAND_BLOCK + 5 * 3, HAND_BLOCK + 17 * 3)

    # POSE_KEYPOINTS = [nez, epG, epD, coudeG, coudeD, poignetG, poignetD, ...]
    left_arm = segment(POSE_START + 3 * 3, POSE_START + 5 * 3)
    right_arm = segment(POSE_START + 4 * 3, POSE_START + 6 * 3)

    extra = np.stack([left_palm, right_palm, left_arm, right_arm], axis=-1)
    return np.concatenate([batch, extra.astype(np.float32)], axis=-1)


# Doigts MediaPipe : (base, [articulations], bout)
FINGERS = [
    (1, [1, 2, 3, 4]),      # pouce
    (5, [5, 6, 7, 8]),      # index
    (9, [9, 10, 11, 12]),   # majeur
    (13, [13, 14, 15, 16]), # annulaire
    (17, [17, 18, 19, 20]), # auriculaire
]

# Indices dans POSE_KEYPOINTS = [nez, epG, epD, coudeG, coudeD,
#                                poignetG, poignetD, hancheG, hancheD]
POSE_NOSE = 0


def gate_hands(batch, layout, scores, inside, conflict,
               threshold=0.0, soft=False, inside_threshold=0.0,
               use_conflict=False):
    """Decide quelles mains le modele voit, a partir des features brutes.

    L extraction en mode brut n ecarte rien : toutes les coordonnees sont
    ecrites et les drapeaux valent 1. C est ici que les trois filtres
    s appliquent, ce qui les rend reglables sans reextraire.

    threshold        confiance minimale pour qu une main compte
    soft             si vrai, les mains sous le seuil gardent leurs
                     COORDONNEES mais perdent leur drapeau : le modele sait
                     que la main est douteuse et dispose quand meme de sa
                     position, la confiance precisant a quel point. Si faux,
                     coordonnees ET drapeau sont mis a zero, ce que faisait
                     l extraction.

                     Garder les deux ne serait pas un filtrage doux mais une
                     absence de filtrage, le seuil n ayant alors aucun effet.
    inside_threshold part minimale des points dans l image
    use_conflict     appliquer l arbitrage des mains inventees

    Le filtrage DUR jette de l information ; le filtrage DOUX la garde et
    laisse le modele trancher. La section 7.5b a montre que lui donner une
    information vaut mieux que decider a sa place, donc la question merite
    d etre posee — elle etait jusqu ici hors de portee.
    """
    out = batch.copy()
    hand_block = layout.hand_block

    for side, (flag_index, start) in enumerate(
            ((layout.left_flag, 0), (layout.right_flag, hand_block))):
        reject = np.zeros(out.shape[:2], dtype=bool)

        if threshold > 0.0:
            reject |= scores[..., side] < threshold
        if inside_threshold > 0.0:
            reject |= inside[..., side] < inside_threshold
        if use_conflict:
            # Le conflit ne designe pas un cote : on retire celle des deux
            # qui est la plus loin de SON poignet, ce que resolve a deja
            # tranche a l extraction. Faute de ce detail ici, on ecarte la
            # moins sure des deux.
            weaker = scores[..., side] < scores[..., 1 - side]
            reject |= conflict & weaker

        if soft:
            # On garde les coordonnees ; seul le drapeau tombe, pour que le
            # modele sache que la main est douteuse sans perdre sa position.
            out[..., flag_index] = np.where(reject, 0.0, out[..., flag_index])
        else:
            out[..., start:start + hand_block] = np.where(
                reject[..., None], 0.0, out[..., start:start + hand_block])
            out[..., flag_index] = np.where(reject, 0.0, out[..., flag_index])

    return out


def interpolate_gaps(batch, layout, max_gap=3):
    """Comble les disparitions courtes d une main par interpolation lineaire.

    A seuil 0.7 une main n est annoncee que dans ~60% des frames, donc elle
    clignote au milieu d un signe et le modele recoit des zeros brutaux au
    lieu d une trajectoire. Les coordonnees manquantes ne sont PAS derivables
    par le GRU : il ne peut pas inventer ou se trouvait la main.

    On ne comble que les trous bornes des DEUX cotes par une frame presente,
    ce qui exclut naturellement le remplissage de fin de sequence. Le drapeau
    passe a 1 puisque la position est une estimation raisonnable ; c est la
    colonne de confiance, restee basse, qui dit au modele de s en defier —
    --interpolate a donc tout interet a etre utilise avec --confidence.
    """
    out = batch.copy()
    hand_block = layout.hand_block

    for sequence in range(len(out)):
        for flag_index, start in ((layout.left_flag, 0),
                                  (layout.right_flag, hand_block)):
            present = out[sequence, :, flag_index] > 0.5
            if present.sum() < 2:
                continue

            indices = np.flatnonzero(present)
            for before, after in zip(indices, indices[1:]):
                gap = after - before - 1
                if gap <= 0 or gap > max_gap:
                    continue
                left = out[sequence, before, start:start + hand_block]
                right = out[sequence, after, start:start + hand_block]
                for step in range(1, gap + 1):
                    weight = step / (gap + 1)
                    out[sequence, before + step, start:start + hand_block] = (
                        (1.0 - weight) * left + weight * right)
                    out[sequence, before + step, flag_index] = 1.0

    return out


def smooth_confidence(batch, layout, window=5):
    """Moyenne temporelle ponderee par la confiance.

    La gigue mesuree vaut 0.0359 sur les bonnes frames et 0.1408 sur les
    mauvaises : tirer les frames peu sures vers leurs voisines attaque
    directement cet ecart.

    Un GRU bidirectionnel pourrait en principe moyenner lui-meme, donc un
    lissage AVEUGLE serait une re-derivation et devrait ne rien rapporter
    (section 4). Celui-ci ne l est pas : la ponderation vient de la confiance,
    une information exterieure aux coordonnees. C est le premier cas ou la
    regle est ambigue, et c est une raison de le tester.

    Ne s applique qu aux coordonnees des mains, jamais aux drapeaux, a la pose
    ni aux confiances elles-memes.
    """
    if layout.tail != 2:
        raise ValueError("--smooth a besoin des confiances : ajoute --confidence")

    out = batch.copy()
    half = window // 2
    hand_block = layout.hand_block

    for flag_index, start, conf_index in ((layout.left_flag, 0, -2),
                                          (layout.right_flag, hand_block, -1)):
        weights = np.where(batch[..., flag_index] > 0.5,
                           np.maximum(batch[..., conf_index], 1e-3), 0.0)

        block = batch[..., start:start + hand_block]
        total = np.zeros_like(weights)
        stacked = np.zeros_like(block)

        for shift in range(-half, half + 1):
            rolled_w = np.roll(weights, shift, axis=1)
            rolled_b = np.roll(block, shift, axis=1)
            # np.roll est circulaire : on annule ce qui a fait le tour.
            if shift > 0:
                rolled_w[:, :shift] = 0.0
            elif shift < 0:
                rolled_w[:, shift:] = 0.0
            total += rolled_w
            stacked += rolled_b * rolled_w[..., None]

        safe = total > 1e-6
        smoothed = np.where(safe[..., None], stacked / np.maximum(total, 1e-6)[..., None],
                            block)
        # Une main absente garde ses zeros.
        keep = (batch[..., flag_index] > 0.5)[..., None]
        out[..., start:start + hand_block] = np.where(keep, smoothed, block)

    return out


def drop_hand(batch, rng, layout, probability=0.3, max_span=12):
    """Efface une main sur un intervalle de frames, au hasard.

    En direct les mains disparaissent : couverture ~60% a seuil 0.7, et cinq
    desaccords mesures entre les clips et une vraie webcam (section 7.7). Le
    modele n a jamais ete entraine a cela. C est la seule des quatre pistes
    qui vise l ecart entrainement/direct plutot que la qualite des features.
    """
    out = batch.copy()
    hand_block = layout.hand_block
    length = out.shape[1]

    for sequence in range(len(out)):
        if rng.random() >= probability:
            continue
        side = rng.integers(2)
        flag_index = layout.left_flag if side == 0 else layout.right_flag
        start = 0 if side == 0 else hand_block
        conf_index = -2 if side == 0 else -1

        span = int(rng.integers(2, max_span + 1))
        begin = int(rng.integers(0, max(length - span, 1)))

        out[sequence, begin:begin + span, start:start + hand_block] = 0.0
        out[sequence, begin:begin + span, flag_index] = 0.0
        if layout.tail == 2:
            out[sequence, begin:begin + span, conf_index] = 0.0

    return out


def add_handshape(batch, layout=None, full=False):
    """Decrit la main autrement que par la flexion des doigts.

    Le repliement des 23 features derivees ne mesure QUE la flexion. Il ne
    dit rien de la position du pouce par rapport aux autres doigts, de
    l ecartement, ni de l ouverture globale — or ce sont des parametres de
    forme de main a part entiere en ASL, et c est ce qui separe les paires
    que la matrice de confusion a isolees.

    Pouvoir separateur mesure (AUC entre les deux classes d une paire, sur
    la moyenne par clip) avant d entrainer quoi que ce soit :

                        meilleur curl   meilleur candidat
      apple / candy          0.84            0.91   pouce-index
      bird / orange          0.89            0.98   paume-annul
      bed / mother           0.58            0.87   paume-auric
      thursday / shirt       0.86            0.99   pouce-auric
      apple / deaf           0.86            0.96   pouce-index

    bed / mother est le cas net : le repliement y est a 0.58, soit a peine
    mieux qu une piece lancee en l air.

    Tout est normalise par la largeur de paume, donc invariant a l echelle
    comme le repliement l est a la rotation.

      4 par main   pouce -> bout de chaque autre doigt (opposition)
      full ajoute :
      5 par main   bout -> poignet (main ouverte ou fermee)
      3 par main   ecart entre bouts adjacents
      1 par main   etendue des quatre bouts
    """
    layout = layout or Layout.from_dim(batch.shape[-1])
    step = layout.coords
    hand_block = layout.hand_block

    def point(off, index):
        start = off + index * step
        return batch[..., start:start + 2]

    def dist(a, b):
        return np.linalg.norm(a - b, axis=-1)

    columns = []
    owners = []          # quelle main possede chaque colonne

    for side, off in enumerate((0, hand_block)):
        palm = np.maximum(dist(point(off, 5), point(off, 17)), 1e-6)

        # opposition du pouce : la position du pouce face aux autres doigts
        for tip in (8, 12, 16, 20):
            columns.append(dist(point(off, 4), point(off, tip)) / palm)
            owners.append(side)

        if full:
            for tip in (4, 8, 12, 16, 20):
                columns.append(dist(point(off, tip), point(off, 0)) / palm)
                owners.append(side)
            for a, b in ((8, 12), (12, 16), (16, 20)):
                columns.append(dist(point(off, a), point(off, b)) / palm)
                owners.append(side)
            tips = [point(off, t) for t in (4, 8, 12, 16, 20)]
            stacked = np.stack(tips, axis=-2)
            extent = np.linalg.norm(stacked.max(axis=-2) - stacked.min(axis=-2), axis=-1)
            columns.append(extent / palm)
            owners.append(side)

    extra = np.stack(columns, axis=-1).astype(np.float32)

    # Une main absente ne doit pas produire de valeurs inventees : ses
    # colonnes repassent a zero, les drapeaux disant deja qu il n y a rien.
    present = (batch[..., layout.left_flag] > 0.5,
               batch[..., layout.right_flag] > 0.5)
    for index, side in enumerate(owners):
        extra[..., index] *= present[side]

    return np.concatenate([batch, extra], axis=-1)


def add_symmetry(batch, layout=None):
    """Deux scalaires sur le rapport entre les deux mains.

    Beaucoup de signes bimanuels sont symetriques : les deux mains y prennent
    la meme configuration, en miroir l une de l autre. Le GRU peut en principe
    le deduire des coordonnees, et notre regle dit que ce genre de
    re-derivation ne rapporte rien (les vitesses et le proxy de profondeur
    seuls n ont rien donne, alors que le repliement, invariant a la rotation,
    a rapporte). C est donc un test de cette regle autant qu une feature.

      1  symetrie de position : ecart entre la main gauche et la DROITE
         reflechie par rapport a l axe du corps. L origine etant le milieu
         des epaules, l axe est x = 0.
      1  symetrie de configuration : ecart moyen des repliements doigt a doigt
    """
    layout = layout or Layout.from_dim(batch.shape[-1])
    step = layout.coords
    hand_block = layout.hand_block

    def point(off, index):
        start = off + index * step
        return batch[..., start:start + 2]

    def dist(a, b):
        return np.linalg.norm(a - b, axis=-1)

    left_present = batch[..., layout.left_flag] > 0.5
    right_present = batch[..., layout.right_flag] > 0.5
    both = (left_present & right_present).astype(np.float32)

    # Position : la main droite reflechie doit se superposer a la gauche.
    gaps = []
    for index in range(HAND_POINTS):
        left = point(0, index)
        right = point(hand_block, index)
        reflected = np.stack([-right[..., 0], right[..., 1]], axis=-1)
        gaps.append(dist(left, reflected))
    position = np.mean(np.stack(gaps, axis=-1), axis=-1)

    # Configuration : meme repliement des deux cotes ?
    differences = []
    for base, chain in FINGERS:
        curls = []
        for off in (0, hand_block):
            straight = dist(point(off, chain[0]), point(off, chain[-1]))
            along = sum(dist(point(off, chain[i]), point(off, chain[i + 1]))
                        for i in range(len(chain) - 1))
            curls.append(straight / np.maximum(along, 1e-6))
        differences.append(np.abs(curls[0] - curls[1]))
    handshape = np.mean(np.stack(differences, axis=-1), axis=-1)

    # Les deux grandeurs n ont de sens qu avec DEUX mains.
    extra = np.stack([position * both, handshape * both], axis=-1).astype(np.float32)

    return np.concatenate([batch, extra], axis=-1)


def add_rich_features(batch, layout=None):
    """Rend explicites les parametres linguistiques de l ASL.

    Un signe se decrit par quatre parametres : configuration de la main,
    emplacement, orientation et mouvement. Les coordonnees brutes portent
    bien le mouvement (le GRU le lit dans la sequence) mais laissent les
    trois autres implicites. Avec ~11 clips par classe, le modele n a pas
    de quoi les redecouvrir seul.

    Tout est derive de x,y, que MediaPipe estime correctement — jamais de
    z, dont le retrait vaut +7.8 points.

      4   profondeur  : largeur de paume, longueur d avant-bras
     10   configuration : repliement de chaque doigt
      4   emplacement : distance main-nez et main-buste
      1   structure   : distance entre les deux mains
      4   orientation : angle poignet -> majeur, en sin/cos
     ---
     23 scalaires
    """
    layout = layout or Layout.from_dim(batch.shape[-1])
    step = layout.coords
    hand_block = layout.hand_block
    pose_start = layout.pose_start

    def point(off, index):
        start = off + index * step
        return batch[..., start:start + 2]       # toujours x,y

    def dist(a, b):
        return np.linalg.norm(a - b, axis=-1)

    left_present = batch[..., layout.left_flag] > 0.5
    right_present = batch[..., layout.right_flag] > 0.5

    columns = []

    # --- profondeur : segments rigides, raccourcis par la perspective ----
    for off in (0, hand_block):
        columns.append(dist(point(off, 5), point(off, 17)))      # paume
    columns.append(dist(point(pose_start, 3), point(pose_start, 5)))   # avant-bras G
    columns.append(dist(point(pose_start, 4), point(pose_start, 6)))   # avant-bras D

    # --- configuration : repliement, invariant a la rotation ------------
    # bout-a-base divise par la longueur deployee du doigt : 1 = tendu,
    # ~0.3 = replie. Contrairement aux coordonnees brutes, cette mesure ne
    # change pas quand la main pivote.
    for off in (0, hand_block):
        for base, chain in FINGERS:
            straight = dist(point(off, chain[0]), point(off, chain[-1]))
            along = sum(dist(point(off, chain[i]), point(off, chain[i + 1]))
                        for i in range(len(chain) - 1))
            columns.append(straight / np.maximum(along, 1e-6))

    # --- emplacement : ou la main se trouve par rapport au corps ---------
    # L origine etant le milieu des epaules, |poignet| est deja la distance
    # au buste.
    nose = point(pose_start, POSE_NOSE)
    for off in (0, hand_block):
        wrist = point(off, 0)
        columns.append(dist(wrist, nose))
        columns.append(np.linalg.norm(wrist, axis=-1))

    # --- structure : mains jointes, ecartees, croisees -------------------
    columns.append(dist(point(0, 0), point(hand_block, 0)))

    # --- orientation : direction poignet -> articulation du majeur -------
    # En sin/cos pour eviter la discontinuite a +-180 degres.
    for off in (0, hand_block):
        vector = point(off, 9) - point(off, 0)
        norm = np.maximum(np.linalg.norm(vector, axis=-1), 1e-6)
        columns.append(vector[..., 0] / norm)
        columns.append(vector[..., 1] / norm)

    extra = np.stack(columns, axis=-1).astype(np.float32)

    # Une main absente ne doit pas produire de valeurs inventees : ses
    # colonnes sont remises a zero, les drapeaux de presence disant deja
    # au modele qu il n y a rien a lire.
    #       0-1 paumes, 2-3 avant-bras, 4-13 doigts, 14-17 emplacement,
    #       18 inter-mains, 19-22 orientation
    left_cols = [0, 4, 5, 6, 7, 8, 14, 15, 19, 20]
    right_cols = [1, 9, 10, 11, 12, 13, 16, 17, 21, 22]

    extra[..., left_cols] *= left_present[..., None]
    extra[..., right_cols] *= right_present[..., None]
    extra[..., 18] *= (left_present & right_present)[..., None][..., 0]

    return np.concatenate([batch, extra], axis=-1)


def build_model(num_classes, units=96, feature_dim=155, cell="gru",
                label_smoothing=0.0, project=0):
    """cell : 'gru', 'lstm' ou 'stack' (une couche de chaque).

    GRU et LSTM resolvent le meme probleme (gradients qui s'evanouissent dans
    un RNN) et ne different que par la presence d'un etat de cellule separe.
    Les empiler n'ajoute donc pas d'information : c'est teste ici pour le
    verifier, pas parce qu'on l'attend.
    """
    if cell == "lstm":
        first = layers.LSTM(units, return_sequences=True, dropout=0.3)
        second = layers.LSTM(units, dropout=0.3)
    elif cell == "stack":
        first = layers.GRU(units, return_sequences=True, dropout=0.3)
        second = layers.LSTM(units, dropout=0.3)
    else:
        first = layers.GRU(units, return_sequences=True, dropout=0.3)
        second = layers.GRU(units, dropout=0.3)

    body = [
        layers.Input(shape=(MAX_LEN, feature_dim)),
        # Les frames de remplissage sont entierement nulles ; une vraie frame
        # ne l'est jamais (la pose est detectee dans 97% des frames au moins).
        layers.Masking(mask_value=0.0),
    ]

    if project:
        # Une dimension d entree qui entre DIRECTEMENT dans un GRU
        # bidirectionnel coute 3 x unites x 2 parametres, soit 576 a 96
        # unites, pour 580 clips d entrainement par fold. Avec une projection
        # elle n en coute plus que `project`.
        #
        # Sans activation, volontairement : une projection lineaire isole
        # l effet d economie de parametres, alors qu un ReLU y melerait un
        # gain de non-linearite. Et ce qu on cherche a distinguer, ce sont de
        # petits ecarts de distance entre doigts, qu un ReLU pourrait annuler.
        #
        # Le masque doit traverser : les frames de remplissage valent zero,
        # mais une couche Dense rend son biais pour une entree nulle, donc
        # sans propagation du masque le GRU lirait du remplissage non nul.
        body.append(layers.TimeDistributed(
            layers.Dense(project, activation=None), name="projection"))

    body += [
        layers.Bidirectional(first),
        layers.Bidirectional(second),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]

    model = models.Sequential(body)

    # sparse_categorical_crossentropy ne gere pas le lissage ; avec lissage
    # on passe donc par des labels one-hot.
    loss = (
        tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
        if label_smoothing > 0
        else "sparse_categorical_crossentropy"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=loss,
        metrics=["accuracy"],
    )
    return model


def signer_folds(records, k, seed=0):
    """Repartit les SIGNEURS en k groupes de taille (en clips) comparable."""
    counts = collections.Counter(r["signer_id"] for r in records)

    # Glouton : le signeur le plus gros va au groupe le plus leger.
    folds = [[] for _ in range(k)]
    loads = [0] * k

    for signer, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        target = loads.index(min(loads))
        folds[target].append(signer)
        loads[target] += count

    return [set(f) for f in folds], loads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--units", type=int, default=96)
    parser.add_argument("--project", type=int, default=0, metavar="N",
                        help="projeter l entree sur N dimensions avant le GRU, "
                             "pour qu une feature coute N parametres au lieu de "
                             "3 x unites x 2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--velocity", action="store_true",
                        help="ajoute les deltas image-a-image (155 -> 310)")
    parser.add_argument("--tag", default="")
    parser.add_argument("--cell", default="gru", choices=("gru", "lstm", "stack"))
    parser.add_argument("--features-dir", default=None,
                        help="dossier de sequences (defaut : data/wlasl/features)")
    parser.add_argument("--vocab", type=int, default=50,
                        help="nombre de classes cibles, les mieux fournies "
                             "d'abord. Le dossier features peut en contenir "
                             "davantage (pre-entrainement) : sans ce filtre "
                             "la tache changerait silencieusement de taille.")
    parser.add_argument("--variants", default="all",
                        choices=("all", "dominant", "split"))
    parser.add_argument("--rich", action="store_true",
                        help="23 scalaires derives de x,y (profondeur, "
                             "configuration, emplacement, orientation)")
    parser.add_argument("--depth-proxy", action="store_true",
                        help="4 indices de profondeur derives de x,y")
    parser.add_argument("--no-z", action="store_true",
                        help="retire la profondeur (155 -> 104 dimensions)")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--tta", action="store_true",
                        help="moyenne la prediction avec celle du miroir")
    parser.add_argument("--seeds", type=int, default=1,
                        help="modeles par fold, moyennes (ensemble)")
    parser.add_argument("--pretrain-epochs", type=int, default=60)
    parser.add_argument("--pretrain-probe", default="",
                        help="points de mesure du pre-entrainement, en epochs, "
                             "separes par des virgules : un seul "
                             "pre-entrainement est fait jusqu au plus grand, "
                             "et l affinage repart des poids de chacun")
    parser.add_argument("--pretrain-vocab", type=int, default=0,
                        help="pre-entraine sur les N gloses les mieux fournies "
                             "avant d'affiner sur le sous-ensemble cible")
    parser.add_argument("--confidence", action="store_true",
                        help="donner la confiance de chaque main comme valeur "
                             "continue, en plus du drapeau binaire. Le modele "
                             "ne peut aujourd hui pas distinguer une main a "
                             "0.71 d une a 0.99.")
    parser.add_argument("--gate", type=float, default=0.0, metavar="X",
                        help="seuil de confiance applique A L ENTRAINEMENT, sur "
                             "des features extraites avec --raw")
    parser.add_argument("--gate-soft", action="store_true",
                        help="sous le seuil, garder les coordonnees et ne "
                             "baisser que le drapeau, au lieu de tout mettre a "
                             "zero : le modele garde la position et la confiance "
                             "lui dit de s en defier")
    parser.add_argument("--gate-inside", type=float, default=0.0, metavar="X",
                        help="part minimale des points dans l image")
    parser.add_argument("--gate-conflict", action="store_true",
                        help="appliquer l arbitrage des mains inventees")
    parser.add_argument("--interpolate", type=int, default=0, metavar="N",
                        help="combler les disparitions de main de N frames ou "
                             "moins par interpolation lineaire")
    parser.add_argument("--smooth", type=int, default=0, metavar="W",
                        help="lissage temporel des mains sur W frames, pondere "
                             "par la confiance (exige --confidence)")
    parser.add_argument("--hand-dropout", type=float, default=0.0,
                        help="probabilite d effacer une main sur un intervalle, "
                             "a l entrainement, pour encaisser les pertes de "
                             "detection en direct")
    parser.add_argument("--max-train-signers", type=int, default=0, metavar="N",
                        help="limiter l ENTRAINEMENT a N signeurs, les folds de "
                             "test restant intacts. Mesure ce qu apporte chaque "
                             "signeur supplementaire, independamment du nombre "
                             "de clips.")
    parser.add_argument("--max-per-class", type=int, default=0, metavar="N",
                        help="plafonner a N clips par glose, pour mesurer la "
                             "courbe d apprentissage et donc la valeur "
                             "marginale d un clip supplementaire")
    parser.add_argument("--tip-confidence", action="store_true",
                        help="confiance de chaque bout de doigt (10 colonnes). "
                             "La moyenne par main masque l occlusion : quand la "
                             "main se met de profil et que les doigts passent "
                             "derriere la paume, la confiance des bouts tombe "
                             "de 0.79 a 0.45. Exige --raw a l extraction.")
    parser.add_argument("--handshape", action="store_true",
                        help="8 scalaires d opposition du pouce, que le "
                             "repliement ne porte pas")
    parser.add_argument("--handshape-full", action="store_true",
                        help="26 scalaires : opposition, ouverture, ecartement")
    parser.add_argument("--symmetry", action="store_true",
                        help="deux scalaires de symetrie entre les mains")
    parser.add_argument("--vocab-from", default=None,
                        help="choisir les gloses d apres CE dossier de features. "
                             "Comparer deux detecteurs sans cela refait le "
                             "confondu cheap3 : chacun selectionne son propre "
                             "vocabulaire, et un tiers de l ecart vient de la.")
    parser.add_argument("--clips-from", default=None,
                        help="ne garder que les clips presents dans CE dossier. "
                             "Un meilleur detecteur sauve des clips que l autre "
                             "a rejetes ; sans cela on mesure aussi ce surplus "
                             "de donnees, pas seulement la qualite des features.")
    args = parser.parse_args()

    if args.confidence and args.no_z:
        parser.error("--confidence vient des features RTMPose, --no-z des "
                     "features MediaPipe : les deux ensemble n ont pas de sens")

    gating = (args.gate > 0.0 or args.gate_inside > 0.0 or args.gate_conflict)

    if args.tip_confidence and not args.confidence:
        parser.error("--tip-confidence s ajoute a --confidence : la regle "
                     "d echange au miroir suppose les deux (queue de 12)")

    probes = sorted({int(x) for x in args.pretrain_probe.split(",") if x.strip()})
    if probes:
        if not args.pretrain_vocab:
            parser.error("--pretrain-probe demande --pretrain-vocab")
        if min(probes) < 1:
            parser.error("les sondes doivent etre des epochs >= 1")
        args.pretrain_epochs = max(probes)

    records = load_sequences(args.variants, args.features_dir,
                             with_confidence=args.confidence,
                             with_meta=gating,
                             with_tips=args.tip_confidence)

    if args.clips_from:
        reference = load_sequences(args.variants, args.clips_from, metadata_only=True)
        allowed = {r["video_id"] for r in reference}
        before = len(records)
        records = [r for r in records if r["video_id"] in allowed]
        print(f"clips restreints a {args.clips_from} : {before} -> {len(records)}")

    # Le dossier features contient 100 gloses depuis le pre-entrainement ;
    # on restreint aux N cibles pour que la tache reste comparable.
    if args.vocab_from:
        reference = load_sequences(args.variants, args.vocab_from, metadata_only=True)
        counts = collections.Counter(r["gloss"] for r in reference)
    else:
        counts = collections.Counter(r["gloss"] for r in records)

    target = {g for g, _ in counts.most_common(args.vocab)}
    records = [r for r in records if r["gloss"] in target]

    if args.vocab_from:
        own = collections.Counter(r["gloss"] for r in records)
        missing = sorted(target - set(own))
        print(f"vocabulaire repris de {args.vocab_from}"
              + (f" ; {len(missing)} gloses absentes ici : {missing}" if missing else ""))

    if args.max_per_class:
        # Tirage aleatoire a graine fixe, et non les N premiers : l ordre des
        # fichiers suit le video_id, qui correle avec le signeur, donc
        # tronquer appauvrirait la diversite des signeurs en plus du nombre.
        picker = np.random.default_rng(0)
        grouped = collections.defaultdict(list)
        for record in records:
            grouped[record["gloss"]].append(record)
        capped = []
        for gloss, items in grouped.items():
            if len(items) <= args.max_per_class:
                capped.extend(items)
            else:
                chosen = picker.choice(len(items), args.max_per_class, replace=False)
                capped.extend(items[i] for i in chosen)
        print(f"plafond {args.max_per_class} clips par glose : "
              f"{len(records)} -> {len(capped)} sequences")
        records = capped

    glosses = sorted({r["gloss"] for r in records})
    index = {g: i for i, g in enumerate(glosses)}

    features = np.stack([pad_sequence(r["features"]) for r in records])
    labels = np.array([index[r["gloss"]] for r in records], dtype=np.int32)
    signers = np.array([r["signer_id"] for r in records])

    print(f"{len(records)} sequences, {len(glosses)} gloses, "
          f"{len(set(signers))} signeurs")
    # La profondeur est retiree JUSTE avant le modele, pas ici : mirror() et
    # augment_batch() travaillent sur la disposition complete a 155 colonnes
    # (indices des mains, des drapeaux et de la pose codes en dur). Couper
    # d'abord decalerait tous ces indices.
    keep_columns = drop_z_indices(features.shape[2]) if args.no_z else None

    # Disposition construite a la main des qu il y a une queue : from_dim ne
    # peut pas la deviner sans ambiguite (voir son docstring).
    tail = (2 if args.confidence else 0) + (10 if args.tip_confidence else 0)
    layout = Layout.from_dim(features.shape[2] - tail)
    layout.tail = tail

    # Les deux agissent sur la disposition BRUTE, avant mirror et avant le
    # passage au modele : combler puis lisser une trajectoire n a de sens que
    # sur des coordonnees, pas sur des scalaires derives.
    if gating:
        scores = np.stack([pad_sequence(r["meta"]["hand_scores"]) for r in records])
        inside = np.stack([pad_sequence(r["meta"]["hand_inside"]) for r in records])
        conflict = np.stack([
            pad_sequence(r["meta"]["hand_conflict"].astype(np.float32)[:, None])[:, 0]
            for r in records]).astype(bool)

        before = float((features[..., layout.left_flag] > 0.5).mean()
                       + (features[..., layout.right_flag] > 0.5).mean()) / 2.0
        features = gate_hands(features, layout, scores, inside, conflict,
                              threshold=args.gate, soft=args.gate_soft,
                              inside_threshold=args.gate_inside,
                              use_conflict=args.gate_conflict)
        after = float((features[..., layout.left_flag] > 0.5).mean()
                      + (features[..., layout.right_flag] > 0.5).mean()) / 2.0
        print(f"filtrage a l entrainement : seuil {args.gate}"
              f"{' doux' if args.gate_soft else ' dur'}"
              f"{f', hors-cadre {args.gate_inside}' if args.gate_inside else ''}"
              f"{', conflits' if args.gate_conflict else ''}"
              f"   mains gardees {before:.1%} -> {after:.1%}")

    if args.interpolate:
        filled = int((features[..., layout.left_flag] > 0.5).sum()
                     + (features[..., layout.right_flag] > 0.5).sum())
        features = interpolate_gaps(features, layout, args.interpolate)
        after = int((features[..., layout.left_flag] > 0.5).sum()
                    + (features[..., layout.right_flag] > 0.5).sum())
        # Le denominateur est le nombre de CRENEAUX de main (deux par frame),
        # padding inclus ; diviser par les frames donnerait un taux sur 2.
        slots = features[..., 0].size * 2
        print(f"interpolation (<= {args.interpolate} frames) : "
              f"{after - filled} mains comblees sur {slots} creneaux, "
              f"{filled / slots:.1%} -> {after / slots:.1%}")

    if args.smooth:
        features = smooth_confidence(features, layout, args.smooth)
        print(f"lissage pondere par la confiance sur {args.smooth} frames")

    def to_model(batch):
        # Les indices de profondeur se calculent sur la disposition COMPLETE
        # (positions des mains et de la pose codees en dur), donc avant la
        # selection de colonnes.
        pieces = []
        if args.rich:
            pieces.append(add_rich_features(batch, layout)[:, :, -23:])
        elif args.depth_proxy:
            pieces.append(add_depth_proxy(batch)[:, :, -4:])
        if args.symmetry:
            pieces.append(add_symmetry(batch, layout)[:, :, -2:])
        if args.handshape or args.handshape_full:
            width = 26 if args.handshape_full else 8
            pieces.append(add_handshape(batch, layout,
                                        full=args.handshape_full)[:, :, -width:])

        if keep_columns is not None:
            batch = batch[:, :, keep_columns]

        if pieces:
            batch = np.concatenate([batch] + pieces, axis=-1)

        return batch

    # La velocite double les colonnes BRUTES uniquement : add_velocity est
    # appliquee avant to_model, donc les scalaires derives sont ajoutes APRES
    # le doublement et ne sont pas doubles. Multiplier le total, comme le
    # faisait ce calcul, annoncait 310 colonnes pour 261 reelles — le modele
    # etait construit a la mauvaise taille et levait une erreur de forme au
    # premier batch. Le defaut dormait depuis toujours : --velocity n avait
    # jamais ete combine avec --rich.
    raw_dim = len(keep_columns) if keep_columns is not None else features.shape[2]
    base_dim = raw_dim * (2 if args.velocity else 1)

    if args.symmetry:
        base_dim += 2
    if args.handshape or args.handshape_full:
        base_dim += 26 if args.handshape_full else 8
    if args.rich:
        base_dim += 23
    elif args.depth_proxy:
        base_dim += 4

    feature_dim = base_dim
    print(f"entree : ({features.shape[1]}, {feature_dim}), "
          f"{args.units} unites {args.cell}, budget {args.epochs} epochs\n")

    folds, loads = signer_folds(records, args.folds, args.seed)
    print(f"taille des folds (clips) : {loads}\n")

    rng = np.random.default_rng(args.seed)
    scores = []
    histories = []
    out_of_fold = []
    seed_curves = []
    pretrain_curves = []

    # Sans sonde, une seule "sonde" implicite : le comportement d avant.
    probe_list = probes if probes else [None]
    probe_histories = {probe: [] for probe in probe_list}

    pretrain_x = pretrain_y = None
    if args.pretrain_vocab:
        # Pre-entrainement sur un vocabulaire plus large : les gloses hors
        # cible sont inutiles comme classes (trop peu de clips chacune) mais
        # leurs sequences apprennent quand meme les regularites du mouvement
        # des mains et du corps. On affine ensuite sur les classes visees.
        # Le MEME extracteur et les MEMES colonnes que l affinage : sans
        # --features-dir cet appel relisait le dossier par defaut (MediaPipe,
        # 155 colonnes) pour pre-entrainer un modele affine ensuite sur 129.
        extra = load_sequences(args.variants, args.features_dir,
                               with_confidence=args.confidence)
        counts = collections.Counter(r["gloss"] for r in extra)
        wide = {g for g, _ in counts.most_common(args.pretrain_vocab)}
        extra = [r for r in extra if r["gloss"] in wide]

        wide_index = {g: i for i, g in enumerate(sorted(wide))}
        pretrain_x = np.stack([pad_sequence(r["features"]) for r in extra])
        pretrain_y = np.array([wide_index[r["gloss"]] for r in extra], dtype=np.int32)
        pretrain_signers = np.array([r["signer_id"] for r in extra])
        print(f"pre-entrainement : {len(extra)} sequences, {len(wide)} gloses\n")

    for fold_index, held in enumerate(folds, start=1):
        test_mask = np.isin(signers, list(held))
        train_mask = ~test_mask

        if args.max_train_signers:
            # On sous-echantillonne les signeurs d ENTRAINEMENT en laissant le
            # fold de test tel quel : sinon reduire le vivier changerait aussi
            # la difficulte du test, et l effet mesure serait confondu.
            pool = sorted({int(x) for x in signers[train_mask]})
            picker = np.random.default_rng(1000 + fold_index)
            keep_signers = picker.choice(
                pool, min(args.max_train_signers, len(pool)), replace=False)
            narrowed = train_mask & np.isin(signers, keep_signers)
            print(f"    signeurs d entrainement {len(pool)} -> {len(keep_signers)}, "
                  f"clips {int(train_mask.sum())} -> {int(narrowed.sum())}", flush=True)
            train_mask = narrowed

        train_x, train_y = features[train_mask], labels[train_mask]

        test_x, test_y = features[test_mask], labels[test_mask]

        eval_x = to_model(add_velocity(test_x) if args.velocity else test_x)
        mirror_x = mirror(test_x, layout)
        if args.velocity:
            mirror_x = add_velocity(mirror_x)
        mirror_x = to_model(mirror_x)

        # Un modele par graine ; leurs probabilites sont moyennees. Les
        # graines divergent beaucoup ici (les folds vont de 0.31 a 0.44),
        # et c'est precisement cette variance que l'ensemble convertit en
        # precision.
        curves = []
        probe_curves = {probe: [] for probe in probe_list}
        for seed_index in range(args.seeds):
            tf.keras.utils.set_random_seed(args.seed + fold_index * 100 + seed_index)
            model = build_model(len(glosses), args.units, feature_dim,
                                args.cell, args.label_smoothing, args.project)

            if pretrain_x is not None:
                # Les signeurs du fold de test sont exclus du pre-entrainement
                # aussi, sinon le modele les aurait vus avant l'affinage.
                keep = ~np.isin(pretrain_signers, list(held))
                head = build_model(len(set(pretrain_y)), args.units, feature_dim,
                                   args.cell, args.label_smoothing, args.project)
                snapshots = {}
                # La boucle tournait sans rien rapporter : impossible de dire
                # apres coup si 60 epochs suffisaient, si c etait trop, ou si
                # le pre-entrainement avait seulement commence a converger.
                # On garde donc sa precision, sur sa propre tache.
                pretrain_curve = []
                for index in range(args.pretrain_epochs):
                    order = rng.permutation(int(keep.sum()))
                    px = augment_batch(pretrain_x[keep][order], rng, layout,
                                       args.hand_dropout)
                    if args.velocity:
                        px = add_velocity(px)
                    px = to_model(px)
                    py = pretrain_y[keep][order]
                    if args.label_smoothing > 0:
                        py = tf.keras.utils.to_categorical(py, len(set(pretrain_y)))
                    history = head.fit(px, py, batch_size=args.batch,
                                       epochs=1, verbose=0)
                    pretrain_curve.append(round(
                        float(history.history["accuracy"][0]), 4))

                    # Poids figes aux points demandes. Les sondes partagent
                    # ainsi UNE trajectoire : des runs separes en auraient
                    # chacun une differente, et l ecart entre 120 et 240
                    # melangerait "plus d epochs" et "autre trajectoire".
                    if probes and (index + 1) in probes:
                        snapshots[index + 1] = [
                            layer.get_weights() for layer in head.layers[:-1]]

                if pretrain_curve and seed_index == 0:
                    marks = [0, len(pretrain_curve) // 4,
                             len(pretrain_curve) // 2, len(pretrain_curve) - 1]
                    shown = "  ".join(
                        f"e{m + 1} {pretrain_curve[m]:.3f}"
                        for m in sorted(set(marks)))
                    print(f"    pre-entrainement ({args.pretrain_epochs} epochs,"
                          f" {len(set(pretrain_y))} gloses) : {shown}", flush=True)
                    pretrain_curves.append(pretrain_curve)

            # Un affinage par sonde, chacun repartant des poids figes a son
            # epoch. Sans sonde, une seule passe, depuis la fin du
            # pre-entrainement : exactement le comportement d avant.
            for probe in probe_list:
                if probe is None:
                    fresh = model
                else:
                    tf.keras.utils.set_random_seed(
                        args.seed + fold_index * 100 + seed_index)
                    fresh = build_model(len(glosses), args.units, feature_dim,
                                        args.cell, args.label_smoothing,
                                        args.project)

                if pretrain_x is not None:
                    # Tout sauf la couche de sortie, dont le nombre de classes
                    # differe entre la tache de pre-entrainement et la cible.
                    source = (snapshots[probe] if probe is not None
                              else [l.get_weights() for l in head.layers[:-1]])
                    for target, weights in zip(fresh.layers[:-1], source):
                        if weights:
                            target.set_weights(weights)

                curve = []
                for epoch in range(args.epochs):
                    order = rng.permutation(len(train_x))
                    batch_x = augment_batch(train_x[order], rng, layout,
                                            args.hand_dropout)
                    if args.velocity:
                        batch_x = add_velocity(batch_x)
                    batch_x = to_model(batch_x)

                    fit_y = train_y[order]
                    if args.label_smoothing > 0:
                        fit_y = tf.keras.utils.to_categorical(fit_y, len(glosses))

                    fresh.fit(batch_x, fit_y, batch_size=args.batch,
                              epochs=1, verbose=0)

                    probabilities = fresh.predict(eval_x, verbose=0)
                    if args.tta:
                        # L'entrainement traite deja le miroir comme preservant
                        # le label : moyenner avec lui est coherent.
                        probabilities = probabilities + fresh.predict(
                            mirror_x, verbose=0)
                    curve.append(probabilities)

                probe_curves[probe].append(curve)

            curves = probe_curves[probe_list[-1]]

        # Chaque sonde recoit la meme mesure : precision par epoch, graines
        # moyennees, puis moyenne des 30 dernieres.
        for probe in probe_list:
            group = probe_curves[probe]
            tail = []
            for epoch in range(max(args.epochs - 30, 0), args.epochs):
                summed = sum(c[epoch] for c in group)
                tail.append(float((summed.argmax(axis=1) == test_y).mean()))
            probe_histories[probe].append(round(float(np.mean(tail)), 4))

        if probes:
            shown = "  ".join(
                f"{probe}e {probe_histories[probe][-1]:.4f}"
                for probe in probe_list)
            print(f"    sondes de pre-entrainement : {shown}", flush=True)

        # Precision par epoch, ensemble des graines moyenne.
        curve = []
        for epoch in range(args.epochs):
            summed = sum(c[epoch] for c in curves)
            curve.append(round(float((summed.argmax(axis=1) == test_y).mean()), 4))

        # Predictions HORS FOLD, moyennees sur les 30 dernieres epochs pour
        # correspondre a la metrique rapportee et non a une epoch unique.
        # Sans elles, aucune matrice de confusion n est possible : le modele
        # final a vu toutes les donnees, et l evaluer dessus cacherait
        # precisement les confusions qu on cherche.
        window = range(max(args.epochs - 30, 0), args.epochs)
        tail_probabilities = sum(sum(c[epoch] for c in curves) for epoch in window)
        tail_probabilities = tail_probabilities / (len(curves) * len(window))
        out_of_fold.append({
            "labels": [int(y) for y in test_y],
            "probabilities": np.round(tail_probabilities, 5).tolist(),
        })

        # Precision selon le NOMBRE de graines moyennees, sur la meme fenetre
        # de 30 epochs que la metrique rapportee. Les rendements de
        # l ensemble sont decroissants, et c est la seule facon de savoir ou
        # ils cessent de payer sans lancer un run par valeur.
        by_seed = []
        for count in range(1, len(curves) + 1):
            partial = sum(sum(curves[k][epoch] for k in range(count))
                          for epoch in window)
            by_seed.append(round(float((partial.argmax(axis=1) == test_y).mean()), 4))
        seed_curves.append(by_seed)

        final = curve[-1]
        scores.append(final)
        histories.append(curve)

        # Resultat partiel sur le disque des maintenant. Un run de 2 h qui
        # n ecrit qu a la fin perd tout a la moindre interruption, et le fold
        # est l unite de travail naturelle : ce qui est fini est garde.
        partial = RESULTS_PATH.replace(".json", f"{args.tag}.partial.json")
        tail_so_far = np.array([np.mean(c[-30:]) for c in histories])
        with open(partial, "w", encoding="utf-8") as handle:
            json.dump({
                "folds_done": len(histories),
                "folds_total": args.folds,
                "per_fold": [float(x) for x in scores],
                "per_fold_plateau": [float(x) for x in tail_so_far],
                "plateau_mean": float(tail_so_far.mean()),
                "curves": histories,
                "by_seed_count": seed_curves,
            }, handle, ensure_ascii=False)

        print(f"  fold {fold_index}/{args.folds}  "
              f"train {len(train_x):>4}  test {len(test_x):>4}  "
              f"final {final:.4f}   (meilleur vu {max(curve):.4f})", flush=True)

    scores = np.array(scores)
    # Moyenne de fin de plateau : lire une seule epoch fait dependre le
    # resultat de l'epoch sur laquelle on tombe (ecart de 2 points mesure).
    tail = np.array([np.mean(c[-30:]) for c in histories])
    print(f"\n{'=' * 60}")
    if seed_curves and len(seed_curves[0]) > 1:
        print("  apport de chaque graine (moyenne sur les folds)")
        columns = np.array(seed_curves)
        for count in range(columns.shape[1]):
            gain = (f"  {columns[:, count].mean() - columns[:, count - 1].mean():+.4f}"
                    if count else "")
            print(f"    {count + 1} graine(s) {columns[:, count].mean():.4f}{gain}")

    print(f"  accuracy (budget fixe, {args.epochs} epochs)")
    print(f"    plateau (30 dernieres) {tail.mean():.4f}  "
          f"ecart-type {tail.std():.4f}   <-- metrique retenue")
    print(f"    moyenne {scores.mean():.4f}  ecart-type {scores.std():.4f}")
    print(f"    folds   {', '.join(f'{s:.3f}' for s in scores)}")
    print(f"    hasard  {1 / len(glosses):.4f}")

    out_path = (RESULTS_PATH.replace(".json", args.tag + ".json")
                if args.tag else RESULTS_PATH)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "folds": args.folds,
                "epochs": args.epochs,
                "units": args.units,
                "project": args.project,
                "glosses": len(glosses),
                "sequences": len(records),
                "cell": args.cell,
                "variants": args.variants,
                "velocity": bool(args.velocity),
                "no_z": bool(args.no_z),
                "depth_proxy": bool(args.depth_proxy),
                "rich": bool(args.rich),
                "label_smoothing": args.label_smoothing,
                "tta": bool(args.tta),
                "seeds": args.seeds,
                "pretrain_vocab": args.pretrain_vocab,
                "pretrain_epochs": args.pretrain_epochs,
                "vocab": args.vocab,
                "confidence": args.confidence,
                "symmetry": args.symmetry,
                "handshape": args.handshape,
                "tip_confidence": args.tip_confidence,
                "handshape_full": args.handshape_full,
                "interpolate": args.interpolate,
                "smooth": args.smooth,
                "hand_dropout": args.hand_dropout,
                "gate": args.gate,
                "gate_soft": args.gate_soft,
                "gate_inside": args.gate_inside,
                "gate_conflict": args.gate_conflict,
                "max_per_class": args.max_per_class,
                "max_train_signers": args.max_train_signers,
                "units": args.units,
                "project": args.project,
                "plateau_mean": float(tail.mean()),
                "plateau_std": float(tail.std()),
                # Le plateau PAR FOLD, et pas seulement sa moyenne : sans lui,
                # un test apparie entre deux runs doit se rabattre sur
                # per_fold, qui est la DERNIERE epoque. Les deux metriques
                # peuvent differer de plus d un point et meme s inverser.
                "per_fold_plateau": [float(x) for x in tail],
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                # Derniere epoque, conservee pour l historique : ce n est
                # PAS la metrique de reference, qui est per_fold_plateau.
                "per_fold": [float(s) for s in scores],
                "curves": histories,
                "out_of_fold": out_of_fold,
                "by_seed_count": seed_curves,
                "pretrain_curves": pretrain_curves,
                "pretrain_probes": {str(k): v for k, v in probe_histories.items()
                                    if k is not None},
                "classes": glosses,
            },
            handle, ensure_ascii=False, indent=1,
        )
    print(f"\n  resultats : {out_path}")
    leftover = RESULTS_PATH.replace(".json", f"{args.tag}.partial.json")
    if os.path.exists(leftover):
        os.remove(leftover)


if __name__ == "__main__":
    main()
