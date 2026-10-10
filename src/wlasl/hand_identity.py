"""Qui est la main gauche, qui est la droite, et laquelle n existe pas.

RTMPose est un modele top-down : il regresse toujours les 133 points du corps
entier, y compris pour une main hors champ. Contrairement a MediaPipe
Holistic, qui rendait None pour une main absente, il n offre aucun drapeau
d absence — seules les confiances et la geometrie sont observables. Les points
de la main manquante se posent alors souvent sur celle qui est visible.

Le squelette de pose tranche, parce qu il estime les deux poignets
separement. On demande a chaque jeu de 21 points quel poignet il revendique,
c est-a-dire lequel des deux est le plus proche. Si les deux revendiquent le
MEME, l un est une invention : celui dont le nom ne correspond pas au poignet
revendique.

Formuler le test ainsi, plutot qu en recouvrement des deux jeux, a deux
consequences qui comptent :

  - un recouvrement lache est attrape aussi bien qu un empilement exact, alors
    qu un seuil de distance ne voit que le second (mesure : sur des clips du
    dataset, l empilement exact ne represente que 0.3% des mains) ;
  - deux mains qui se croisent pour de vrai revendiquent des poignets
    DIFFERENTS et ne declenchent donc rien. C est essentiel : la moitie du
    vocabulaire ASL est bimanuel et les mains s y touchent souvent. Un seuil
    de recouvrement assez lache pour attraper les copies supprimerait ces
    vraies mains (8.6% des mains ont leurs deux jeux a moins de 0.25 largeur
    d epaules, presque toutes legitimement).

Le depart est determine et non un arbitrage entre deux distances presque
egales, ce qui evite que l etiquette alterne d une frame a l autre.

Une seule implementation, partagee par l extraction, la correction des
features deja ecrites et les demos : un ecart entre ce que le modele voit a
l entrainement et en direct coute cher (phase alphabet, 0.985 en validation
contre 0.384 sur la vraie camera).
"""

import numpy as np


def resolve_batch(present_left, present_right,
                  left_wrist, right_wrist,
                  pose_left_wrist, pose_right_wrist):
    """Ecarte les mains inventees sur une sequence entiere.

    present_*      (T,) bool   ce que la confiance annonce
    *_wrist        (T, 2)      point 0 de chaque jeu de 21
    pose_*_wrist   (T, 2)      poignets du squelette de pose

    Retourne (garder_gauche, garder_droite, conflit), trois tableaux (T,).
    """
    present_left = np.asarray(present_left, dtype=bool)
    present_right = np.asarray(present_right, dtype=bool)

    def distance(a, b):
        return np.linalg.norm(np.asarray(a, dtype=np.float64)
                              - np.asarray(b, dtype=np.float64), axis=-1)

    # Quel poignet chaque jeu revendique : 0 = gauche, 1 = droit.
    left_claim = np.where(distance(left_wrist, pose_left_wrist)
                          <= distance(left_wrist, pose_right_wrist), 0, 1)
    right_claim = np.where(distance(right_wrist, pose_right_wrist)
                           <= distance(right_wrist, pose_left_wrist), 1, 0)

    conflict = present_left & present_right & (left_claim == right_claim)

    # Les deux revendiquent le meme poignet : on garde celle dont le nom
    # correspond, on efface l autre.
    keep_left = present_left & ~(conflict & (left_claim == 1))
    keep_right = present_right & ~(conflict & (left_claim == 0))

    return keep_left, keep_right, conflict


def resolve(present_left, present_right,
            left_wrist, right_wrist,
            pose_left_wrist, pose_right_wrist):
    """Version une frame. Retourne (garder_gauche, garder_droite, conflit)."""
    keep_left, keep_right, conflict = resolve_batch(
        [present_left], [present_right],
        np.reshape(left_wrist, (1, -1)), np.reshape(right_wrist, (1, -1)),
        np.reshape(pose_left_wrist, (1, -1)), np.reshape(pose_right_wrist, (1, -1)),
    )
    return bool(keep_left[0]), bool(keep_right[0]), bool(conflict[0])


def inside_fraction(points, width, height):
    """Part des points qui tombent dans l image.

    RTMPose place volontiers des points HORS du cadre : une main dont les
    points sortent de l image est soit sortie du champ, soit inventee. C est
    le seul signal d absence directement observable, faute de drapeau.
    """
    points = np.asarray(points)
    x, y = points[..., 0], points[..., 1]
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return inside.mean(axis=-1)
