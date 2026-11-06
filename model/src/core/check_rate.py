"""A 45 fps, que fait resample ? Et combien d erreur de calage reste-t-il ?

resample projette les echantillons sur une grille a 30 fps par plus proche
voisin. Plus il y a d echantillons reels, plus le point de grille trouve un
voisin proche : sur-echantillonner ne peut donc pas nuire, il reduit l erreur
de calage. Sous-echantillonner, lui, duplique.
"""
import sys
import numpy as np
sys.path.insert(0, "src/core")
from webcam_demo_rtm import RECORD_SECONDS, TARGET_FPS, resample, MAX_LEN
from train_kfold import pad_sequence

print(f"  fenetre {RECORD_SECONDS} s, cible {TARGET_FPS} fps, MAX_LEN {MAX_LEN}")
print()
print(f"  {'cadence':>9s} {'frames':>7s} {'apres resample':>15s} "
       f"{'doublons':>9s} {'erreur de calage':>18s} {'apres pad':>10s}")
for fps in (16, 24, 30, 36, 45, 60):
    n = int(RECORD_SECONDS * fps)
    stamps = np.arange(n) / fps
    # Un vecteur par frame, identifiable : l index, pour reperer les doublons.
    vectors = [np.full(106, i, dtype=np.float32) for i in range(n)]
    out = resample(list(zip(stamps, vectors)), target_fps=TARGET_FPS)
    picked = [int(v[0]) for v in out]
    duplicates = len(picked) - len(set(picked))
    # Erreur de calage : ecart entre le point de grille et l echantillon pris.
    grid = np.arange(0.0, stamps[-1] - stamps[0], 1.0 / TARGET_FPS)
    error = np.abs(stamps[picked] - (stamps[0] + grid))
    padded = pad_sequence(np.asarray(out, dtype=np.float32))
    kept = len(set(int(v[0]) for v in padded if v.any()))
    print(f"  {fps:7d}   {n:7d} {len(out):15d} {duplicates:9d} "
          f"{error.max()*1000:14.1f} ms {kept:10d}")
print()
print("  doublons > 0 : la trajectoire est en escalier, le modele voit du")
print("  mouvement nul la ou il y en avait. C est ce que 16 fps produisait.")
