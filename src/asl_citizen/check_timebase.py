"""Combien de signe la fenetre de 80 frames garde-t-elle, selon ou on la pose ?

Deux strategies, sur les clips deja extraits :

  ancienne  80 frames du milieu du clip, a la cadence du fichier
            (ce que faisait pad_sequence)
  nouvelle  grille a 30 fps, puis 80 frames centrees sur l activite des mains
            (timebase.py)

Mesures : duree de la partie ACTIVE de chaque clip, part des frames actives
que chaque strategie conserve, et part des clips dont le signe ne tient pas
dans 80 frames quoi qu on fasse. Le seuil d activite est balaye, puisque
c est un reglage et pas une donnee.
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src", "wlasl"))
import timebase  # noqa: E402

WINDOW = 80
paths = sorted(glob.glob(os.path.join(ROOT, "data", "asl_citizen", "features", "*.npz")))
print(f"{len(paths)} clips extraits")

clips = []
for path in paths:
    d = np.load(path, allow_pickle=True)
    clips.append((d["features"], d["stamps"]))

# Valeurs brutes des poignets : le repos doit etre nettement plus bas (y plus
# grand) que la hauteur atteinte en signant, sinon les colonnes sont fausses.
rest, peak = [], []
for features, _ in clips:
    for col in (timebase.LEFT_WRIST_Y, timebase.RIGHT_WRIST_Y):
        rest.append(np.percentile(features[:, col], 90))
        peak.append(features[:, col].min())
print(f"poignets, en largeurs d epaules sous les epaules : repos mediane "
      f"{np.median(rest):+.2f}, point le plus haut mediane {np.median(peak):+.2f}")
print()

for rise in (0.2, 0.3, 0.5):
    spans, kept_old, kept_new, none = [], [], [], 0
    for features, stamps in clips:
        picked = timebase.resample_indices(stamps)
        grid = features[picked]
        span = timebase.active_span(grid, rise)
        if span is None:
            none += 1
            continue
        first, last = span
        spans.append((last - first + 1) / timebase.TARGET_FPS)
        active = np.zeros(len(grid), dtype=bool)
        active[first:last + 1] = True

        start = timebase.window_start(grid, WINDOW, rise)
        kept_new.append(active[start:start + WINDOW].sum() / active.sum())

        # Ancienne strategie : milieu du clip BRUT, a sa cadence native. On
        # ramene l activite brute sur les memes frames pour comparer.
        raw_span = timebase.active_span(features, rise)
        raw_active = np.zeros(len(features), dtype=bool)
        if raw_span:
            raw_active[raw_span[0]:raw_span[1] + 1] = True
        n = len(features)
        old_start = max((n - WINDOW) // 2, 0)
        kept_old.append(raw_active[old_start:old_start + WINDOW].sum()
                        / max(raw_active.sum(), 1))

    spans, kept_old, kept_new = map(np.array, (spans, kept_old, kept_new))
    print(f"seuil {rise:.1f} : {len(spans)} clips actifs, {none} sans activite")
    print(f"  duree active : mediane {np.median(spans):.2f} s, "
          f"p90 {np.percentile(spans, 90):.2f}, p99 {np.percentile(spans, 99):.2f}")
    print(f"  signe plus long que 80 frames (2.67 s) : {(spans > WINDOW / 30).mean():.1%}")
    print(f"  activite conservee   ancienne : moyenne {kept_old.mean():.1%}, "
          f"clips amputes {(kept_old < 0.999).mean():.1%}")
    print(f"                       nouvelle : moyenne {kept_new.mean():.1%}, "
          f"clips amputes {(kept_new < 0.999).mean():.1%}")
    print()
