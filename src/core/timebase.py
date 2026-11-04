"""Base de temps commune a l entrainement et a la demo.

Le modele lit 80 frames. Ce module decide lesquelles, et il est importe par
les DEUX cotes pour qu ils ne puissent pas diverger — meme regle que
hand_identity.py pour l arbitrage des mains.

1. Grille a 30 fps, par plus proche voisin sur les horodatages reels. Jamais
   d interpolation : elle inventerait des positions que le detecteur n a
   jamais produites (mesure : -1.4 point en comblant les trous, 7.5b). Un
   clip a 15 fps y gagne des doublons, comme une webcam lente dans la demo.

2. Fenetre de 80 frames CENTREE SUR LE SIGNE. Dans ASL Citizen le signe tombe
   surtout dans la premiere moitie du clip (centre median 0.87 s sur 2.67) ;
   dans la demo, il tombe la ou l utilisateur le fait. Centrer les deux cotes
   sur le signe les met d accord sans dependre du moment ou on commence.

Le signe est repere par le MOUVEMENT des poignets et des coudes, pas par leur
hauteur. La hauteur ratait 13% des clips, tous des signes faits devant la
poitrine (WANT, ME, BOOK, OH-I-SEE, YOU) : la main y monte de 0.22 largeur
d epaules contre 0.58 ailleurs, mais elle BOUGE autant (vitesse max 0.78
contre 0.73). Le mouvement est aussi mesure relativement a l immobilite du
clip lui-meme, donc insensible a la posture de repos — mains sur le bureau ou
hors champ, signeur d ASL Citizen ou utilisateur de la demo.
"""

import numpy as np

TARGET_FPS = 30.0
WINDOW = 80

# Disposition brute : 2 x 21 points de main, 2 drapeaux, puis 9 points de
# pose [0, 5, 6, 7, 8, 9, 10, 11, 12]. Coudes (7, 8) et poignets (9, 10) sont
# les 4e a 7e : colonnes pose_start + 6 a 13. Ceux du modele de POSE, pas des
# mains : ils existent meme quand une main est ecartee par le filtrage.
HAND_POINTS = 21
POSE_START = HAND_POINTS * 2 * 2 + 2          # 86
MOTION_POINTS = [(POSE_START + 2 * k, POSE_START + 2 * k + 1) for k in (3, 4, 5, 6)]

SMOOTH = 5            # frames : median puis moyenne glissante
FLOOR_PERCENTILE = 20  # l immobilite du clip, en dessous de laquelle rien ne compte
MASS = 0.90           # part de l energie qui definit l etendue du signe
MIN_MOTION = 0.005    # en dessous, le clip est juge immobile


def resample_indices(stamps, target_fps=TARGET_FPS):
    """Indices des frames retenues sur la grille a target_fps.

    La logique exacte de resample() dans la demo : grille partant du premier
    horodatage, plus proche voisin. Elle vit ici pour que les deux cotes
    l appliquent a l identique.
    """
    stamps = np.asarray(stamps, dtype=np.float64)
    if len(stamps) < 2:
        return np.arange(len(stamps))
    duration = stamps[-1] - stamps[0]
    if duration <= 0:
        return np.arange(len(stamps))
    grid = np.arange(0.0, duration, 1.0 / target_fps)
    return np.abs(stamps[None, :] - (stamps[0] + grid)[:, None]).argmin(axis=1)


def _median_filter(values, width):
    """Filtre median par colonne, bords repliques : efface un saut d une frame."""
    if len(values) < width:
        return values
    half = width // 2
    padded = np.concatenate([np.repeat(values[:1], half, axis=0), values,
                             np.repeat(values[-1:], half, axis=0)])
    stacked = np.stack([padded[i:i + len(values)] for i in range(width)])
    return np.median(stacked, axis=0)


def energy(features):
    """Mouvement par frame, au-dessus de l immobilite propre au clip.

    Somme des vitesses des deux poignets et des deux coudes, en largeurs
    d epaules par frame. Le filtre median d abord : le detecteur fait parfois
    sauter un point d une frame a l autre, et ce saut ressemblerait sinon au
    geste le plus vif du clip.
    """
    n = len(features)
    if n < 2:
        return np.zeros(n)
    columns = [c for pair in MOTION_POINTS for c in pair]
    track = _median_filter(features[:, columns].astype(np.float64), SMOOTH)
    steps = np.diff(track, axis=0).reshape(n - 1, len(MOTION_POINTS), 2)
    speed = np.sqrt((steps ** 2).sum(axis=2)).sum(axis=1)
    speed = np.concatenate([[0.0], speed])
    kernel = np.ones(SMOOTH) / SMOOTH
    speed = np.convolve(speed, kernel, mode="same")
    floor = np.percentile(speed, FLOOR_PERCENTILE)
    return np.maximum(speed - floor, 0.0)


def sign_centre(features):
    """Centre de masse du mouvement, en frames ; None si le clip est immobile.

    Un centre de masse plutot que le milieu entre premiere et derniere frame
    actives : une seule frame aberrante au bord deplacerait ce milieu, alors
    qu elle pese a peine dans une moyenne ponderee.
    """
    e = energy(features)
    if e.max() < MIN_MOTION:
        return None
    t = np.arange(len(e))
    return float((t * e).sum() / e.sum())


def sign_span(features, mass=MASS):
    """(debut, fin) en frames contenant `mass` de l energie ; None si immobile."""
    e = energy(features)
    if e.max() < MIN_MOTION:
        return None
    cumulative = np.cumsum(e) / e.sum()
    tail = (1.0 - mass) / 2.0
    return (int(np.searchsorted(cumulative, tail)),
            int(np.searchsorted(cumulative, 1.0 - tail)))


def window_indices(features, length=WINDOW):
    """Les `length` indices que le modele lira, signe au centre.

    Un clip trop long est recadre, un clip trop court est COMPLETE, dans les
    deux cas autour du signe. Le complement repete la premiere et la derniere
    frame — quelqu un qui reste immobile — et non des zeros : la demo ne
    fournit jamais de frame vide au modele, l entrainement non plus.

    Sans mouvement detecte, on centre sur le milieu du clip.
    """
    n = len(features)
    centre = sign_centre(features)
    if centre is None:
        centre = (n - 1) / 2.0
    start = int(round(centre)) - length // 2
    return np.clip(np.arange(start, start + length), 0, max(n - 1, 0))
