"""Avec une cadence REELLE, qui gigue, que vaut un plafond a 30 fps ?

Mon test precedent espacait les horodatages parfaitement, ce qui donnait
0.0 ms d erreur a 30 fps exactement et rendait le plafond seduisant. Une
webcam ne fait pas ca : 30 fps est une MOYENNE, et les intervalles
individuels varient.

Des qu un intervalle depasse 33.3 ms, le point de grille suivant n a plus
d echantillon frais et resample en repete un. A 30 fps de moyenne, la moitie
des intervalles sont donc au-dessus du seuil. A 45, meme un mauvais
intervalle reste en dessous.

On simule une gigue de +-20% sur l intervalle, ce qui est modeste pour une
webcam USB.
"""
import sys
import numpy as np
sys.path.insert(0, "src/wlasl")
from webcam_demo_rtm import RECORD_SECONDS, TARGET_FPS, resample

rng = np.random.default_rng(0)
TRIALS = 200

print(f"  gigue +-20% sur l intervalle, {TRIALS} enregistrements simules de "
      f"{RECORD_SECONDS} s")
print()
print(f"  {'cadence moyenne':>16s} {'doublons moyens':>16s} {'pire cas':>9s} "
      f"{'% avec doublons':>16s}")
for fps in (30, 36, 45, 60):
    counts = []
    for _ in range(TRIALS):
        intervals = (1.0 / fps) * (1.0 + rng.uniform(-0.2, 0.2,
                                                     int(RECORD_SECONDS * fps)))
        stamps = np.cumsum(intervals) - intervals[0]
        stamps = stamps[stamps <= RECORD_SECONDS]
        vectors = [np.full(106, i, dtype=np.float32) for i in range(len(stamps))]
        out = resample(list(zip(stamps, vectors)), target_fps=TARGET_FPS)
        picked = [int(v[0]) for v in out]
        counts.append(len(picked) - len(set(picked)))
    counts = np.array(counts)
    print(f"  {fps:14d}   {counts.mean():16.1f} {counts.max():9d} "
          f"{(counts > 0).mean():15.0%}")
print()
print("  Un doublon = une paire de frames consecutives ou le modele voit")
print("  un mouvement nul qui n a pas eu lieu.")
