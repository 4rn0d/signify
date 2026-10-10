"""Ecarte les mains inventees dans les features RTMPose deja extraites.

Le modele corps entier predit toujours 21 points pour CHAQUE main, meme quand
une seule est visible, et les place alors parfois tous les deux sur la meme.
Le critere et son justificatif sont dans hand_identity.py, partage avec
l extraction et les demos pour que le modele voie la meme chose a
l entrainement et en direct.

Reecrit les .npz sur place. Idempotent : une sequence deja corrigee ne
declenche plus rien, les mains effacees n etant plus annoncees presentes.
"""

import argparse
import glob
import os

import numpy as np

from hand_identity import resolve_batch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "wlasl", "features_rtm")

HAND_POINTS = 21
HAND_BLOCK = HAND_POINTS * 2
LEFT_FLAG = HAND_BLOCK * 2
RIGHT_FLAG = LEFT_FLAG + 1
POSE_START = LEFT_FLAG + 2

# POSE_KEYPOINTS = [nez, epG, epD, coudeG, coudeD, poignetG, poignetD, hG, hD]
POSE_LEFT_WRIST = 5
POSE_RIGHT_WRIST = 6


def deduplicate(features):
    """Retourne (features corrigees, nombre de frames corrigees)."""
    out = features.copy()
    count = len(out)

    left = out[:, :HAND_BLOCK].reshape(count, HAND_POINTS, 2)
    right = out[:, HAND_BLOCK:HAND_BLOCK * 2].reshape(count, HAND_POINTS, 2)
    pose = out[:, POSE_START:].reshape(count, -1, 2)

    keep_left, keep_right, conflict = resolve_batch(
        out[:, LEFT_FLAG] > 0.5,
        out[:, RIGHT_FLAG] > 0.5,
        left[:, 0],
        right[:, 0],
        pose[:, POSE_LEFT_WRIST],
        pose[:, POSE_RIGHT_WRIST],
    )

    if not conflict.any():
        return out, 0

    drop_left = conflict & ~keep_left
    drop_right = conflict & ~keep_right

    out[drop_left, :HAND_BLOCK] = 0.0
    out[drop_left, LEFT_FLAG] = 0.0

    out[drop_right, HAND_BLOCK:HAND_BLOCK * 2] = 0.0
    out[drop_right, RIGHT_FLAG] = 0.0

    return out, int(conflict.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", default=FEATURES_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.features_dir, "*.npz")))
    print(f"{len(files)} sequences dans {args.features_dir}")

    total_frames = 0
    total_fixed = 0
    touched = 0
    worst = []

    for path in files:
        data = np.load(path, allow_pickle=True)
        features = data["features"]
        cleaned, fixed = deduplicate(features)

        total_frames += len(features)
        total_fixed += fixed
        if fixed:
            touched += 1
            worst.append((fixed / len(features), os.path.basename(path), fixed, len(features)))

        if fixed and not args.dry_run:
            # Reecrire TOUS les champs, pas la liste de ceux qu on connait :
            # l extracteur en ajoute (hand_scores, hand_threshold) et les
            # enumerer ici les ferait disparaitre silencieusement.
            payload = {key: data[key] for key in data.files}
            payload["features"] = cleaned
            np.savez_compressed(path, **payload)

    verb = "a corriger" if args.dry_run else "corrigees"
    print(f"  frames {verb} : {total_fixed} / {total_frames} "
          f"({total_fixed / max(total_frames, 1):.2%})")
    print(f"  sequences concernees : {touched} / {len(files)}")

    for share, name, fixed, total in sorted(worst, reverse=True)[:5]:
        print(f"    {name:<28} {fixed:>4}/{total:<4} ({share:.0%})")


if __name__ == "__main__":
    main()
