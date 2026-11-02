"""Le reperage du signe par le mouvement est-il fiable ?

Pas de verite terrain pour "ou est le signe", donc quatre tests independants :

1. Taux d echec : clips ou aucun mouvement n est trouve. L ancien reperage,
   par la hauteur des poignets, ratait 13% des clips.
2. Accord avec l ancien reperage, la ou celui-ci fonctionnait : deux
   methodes qui ne partagent rien et tombent au meme endroit se confirment.
3. Equivariance : retarder un signe de k frames (k copies de la premiere
   frame en tete) doit retarder son centre de k frames. Une methode qui ne le
   fait pas ne place pas la fenetre sur le signe mais ailleurs.
4. Position finale : apres window_indices, le centre du signe doit tomber au
   milieu des 80 frames, quel que soit le clip.

    python src/asl_citizen/check_timebase.py [--sheet chemin.png]

--sheet ecrit une planche d images (debut, centre, fin du signe repere) pour
une verification a l oeil. Fichier LOCAL : la licence d ASL Citizen interdit
d en diffuser les images.
"""
import argparse
import collections
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src", "wlasl"))
import timebase as tb  # noqa: E402

LEFT_WRIST_Y, RIGHT_WRIST_Y = tb.POSE_START + 11, tb.POSE_START + 13


def height_span(f, rise=0.3):
    """L ancien reperage, par la hauteur des poignets, pour comparaison."""
    l, r = f[:, LEFT_WRIST_Y], f[:, RIGHT_WRIST_Y]
    a = np.maximum(np.maximum(np.percentile(l, 90) - l, np.percentile(r, 90) - r), 0)
    idx = np.flatnonzero(a > rise)
    return None if len(idx) == 0 else (idx[0], idx[-1])


def load(paths):
    out = []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        grid = tb.resample_indices(d["stamps"])
        out.append((p, str(d["gloss"]), d["features"][grid], grid))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sheet", default="")
    args = parser.parse_args()

    paths = sorted([q for q in glob.glob(os.path.join(ROOT, "data", "asl_citizen", "features", "*.npz"))
     # La cible seulement : l echafaudage s ecrit dans le meme dossier.
     if int(np.load(q, allow_pickle=True)["rank"]) <= 100])
    clips = load(paths)
    signs = collections.Counter(g for _, g, _, _ in clips)
    print(f"{len(clips)} clips, {len(signs)} signes")

    # 1. echecs
    old_miss = [c for c in clips if height_span(c[2]) is None]
    new_miss = [c for c in clips if tb.sign_centre(c[2]) is None]
    print(f"\n1. echecs  hauteur {len(old_miss)} ({len(old_miss)/len(clips):.1%})   "
          f"mouvement {len(new_miss)} ({len(new_miss)/len(clips):.1%})")
    if new_miss:
        print("   restants :", dict(collections.Counter(g for _, g, _, _ in new_miss)))

    # 2. accord la ou les deux fonctionnent
    gaps = []
    for _, _, f, _ in clips:
        hs, c = height_span(f), tb.sign_centre(f)
        if hs is not None and c is not None:
            gaps.append(abs((hs[0] + hs[1]) / 2 - c) / tb.TARGET_FPS)
    gaps = np.array(gaps)
    print(f"\n2. accord avec la hauteur sur {len(gaps)} clips : ecart des centres "
          f"mediane {np.median(gaps)*1000:.0f} ms, p90 {np.percentile(gaps,90)*1000:.0f} ms, "
          f"> 300 ms : {(gaps > 0.3).mean():.1%}")

    # 3. equivariance
    errors = []
    for _, _, f, _ in clips:
        c = tb.sign_centre(f)
        if c is None:
            continue
        for k in (10, 30):
            shifted = np.concatenate([np.repeat(f[:1], k, axis=0), f])
            c2 = tb.sign_centre(shifted)
            errors.append(abs((c2 - c) - k))
    errors = np.array(errors)
    print(f"\n3. retard de 10 ou 30 frames : erreur sur le decalage du centre "
          f"mediane {np.median(errors):.2f} frame, p99 {np.percentile(errors,99):.2f}, "
          f"max {errors.max():.2f}")

    # 4. position finale
    finals = []
    for _, _, f, _ in clips:
        c = tb.sign_centre(f)
        if c is None:
            continue
        idx = tb.window_indices(f)
        window = f[idx]
        finals.append(tb.sign_centre(window))
    finals = np.array([x for x in finals if x is not None]) / tb.TARGET_FPS
    print(f"\n4. centre du signe dans la fenetre finale : mediane {np.median(finals):.2f} s "
          f"(milieu = {tb.WINDOW/2/tb.TARGET_FPS:.2f}), p10 {np.percentile(finals,10):.2f}, "
          f"p90 {np.percentile(finals,90):.2f}")

    if args.sheet:
        write_sheet(args.sheet, old_miss)


def write_sheet(path, misses):
    """Debut, centre et fin du signe repere, pour les clips que la hauteur ratait."""
    import cv2

    rng = np.random.default_rng(0)
    picks = [misses[i] for i in rng.choice(len(misses), min(8, len(misses)), replace=False)]
    rows = []
    for p, gloss, f, grid in picks:
        span, centre = tb.sign_span(f), tb.sign_centre(f)
        if span is None:
            continue
        stem = os.path.splitext(os.path.basename(p))[0]
        video = os.path.join(ROOT, "data", "asl_citizen", "ASL_Citizen", "videos", stem + ".mp4")
        cap = cv2.VideoCapture(video)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        tiles = []
        for label, t in (("avant", max(span[0] - 15, 0)), ("debut", span[0]),
                         ("centre", int(round(centre))), ("fin", span[1]),
                         ("apres", min(span[1] + 15, len(grid) - 1))):
            tile = cv2.resize(frames[min(grid[t], len(frames) - 1)], (192, 144))
            cv2.putText(tile, f"{gloss} {label} {t/30:.2f}s", (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
            tiles.append(tile)
        rows.append(np.hstack(tiles))
    cv2.imwrite(path, np.vstack(rows))
    print(f"\nplanche : {path}")


if __name__ == "__main__":
    main()
