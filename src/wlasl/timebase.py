"""Base de temps commune a l entrainement et a la demo.

Le modele lit 80 frames. Ce module decide lesquelles, et il est importe par
les DEUX cotes pour qu ils ne puissent pas diverger — meme regle que
hand_identity.py pour l arbitrage des mains.

Deux etapes, dans cet ordre :

1. Grille a 30 fps, par plus proche voisin sur les horodatages reels. Jamais
   d interpolation : elle inventerait des positions que le detecteur n a
   jamais produites (mesure : -1.4 point en comblant les trous, 7.5b). Un
   clip a 15 fps y gagne des doublons, exactement comme une webcam lente dans
   la demo — les deux cotes voient donc la meme chose.

2. Fenetre de 80 frames placee sur l ACTIVITE des mains, et non au milieu du
   clip. Les clips d ASL Citizen gardent le temps mort avant et apres le
   signe ; couper au centre du clip, comme pad_sequence le faisait pour
   WLASL dont les clips etaient deja recadres sur le signe, peut tronquer un
   signe qui commence tard ou finit tot.
"""

import numpy as np

TARGET_FPS = 30.0

# Poignets du modele de POSE, pas des mains : ils existent meme quand une
# main est ecartee par le filtrage, ce qui est justement le cas au repos.
# Disposition brute : 2 x 21 points de main, 2 drapeaux, puis 9 points de pose
# dans l ordre [0, 5, 6, 7, 8, 9, 10, 11, 12] ; les poignets 9 et 10 sont les
# 6e et 7e, soit les colonnes pose_start + 10..13.
HAND_POINTS = 21
POSE_START = HAND_POINTS * 2 * 2 + 2          # 86
LEFT_WRIST_Y = POSE_START + 5 * 2 + 1         # 97
RIGHT_WRIST_Y = POSE_START + 6 * 2 + 1        # 99

# Au-dela de cette hauteur au-dessus du repos, en largeurs d epaules, la main
# est consideree en train de signer. Reglable : voir check_timebase.py.
ACTIVE_RISE = 0.3


def resample_indices(stamps, target_fps=TARGET_FPS):
    """Indices des frames retenues sur la grille a target_fps.

    La logique exacte que la demo appliquait dans resample() : grille partant
    du premier horodatage, plus proche voisin. Elle vit ici pour que
    l entrainement l applique a l identique.
    """
    stamps = np.asarray(stamps, dtype=np.float64)
    if len(stamps) < 2:
        return np.arange(len(stamps))
    duration = stamps[-1] - stamps[0]
    if duration <= 0:
        return np.arange(len(stamps))
    grid = np.arange(0.0, duration, 1.0 / target_fps)
    return np.abs(stamps[None, :] - (stamps[0] + grid)[:, None]).argmin(axis=1)


def activity(features):
    """Hauteur de la main la plus haute au-dessus de sa position de repos.

    y croit vers le bas (repere image), normalise a la largeur d epaules. Le
    repos est estime par clip, au 90e percentile de chaque poignet : un
    signeur assis a son bureau et un autre debout n ont pas le meme repos, et
    un seuil absolu raterait l un des deux.
    """
    left = features[:, LEFT_WRIST_Y]
    right = features[:, RIGHT_WRIST_Y]
    rise_left = np.percentile(left, 90) - left
    rise_right = np.percentile(right, 90) - right
    return np.maximum(np.maximum(rise_left, rise_right), 0.0)


def active_span(features, rise=ACTIVE_RISE):
    """(premiere, derniere) frame active, ou None si rien ne se leve."""
    active = np.flatnonzero(activity(features) > rise)
    if len(active) == 0:
        return None
    return int(active[0]), int(active[-1])


def window_start(features, length, rise=ACTIVE_RISE):
    """Debut d une fenetre de `length` frames centree sur l activite.

    Sans activite detectee, on retombe sur le centre du clip, ce que faisait
    pad_sequence : le cas ne devrait concerner que des clips rates.
    """
    n = len(features)
    if n <= length:
        return 0
    span = active_span(features, rise)
    if span is None:
        return (n - length) // 2
    centre = (span[0] + span[1]) // 2
    return int(np.clip(centre - length // 2, 0, n - length))
