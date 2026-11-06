"""Les fenetres de 80 frames d un meme signe se ressemblent-elles ?

Le centrage sur le signe n a de valeur que s il rend comparables les clips
d un meme signe. On le mesure, pour chaque methode de fenetrage :

  nouvelle  grille a 30 fps, fenetre centree sur le mouvement (timebase)
  ancienne  80 frames du milieu, a la cadence du fichier, completees de zeros
            (ce que pad_sequence faisait pour WLASL)

Trois mesures, sur la trajectoire des bras (poignets et coudes de pose, 8
colonnes, toujours presentes) :

1. Alignement residuel : entre deux clips du meme signe, le decalage qui
   superpose le mieux leurs profils de mouvement. 0 = parfaitement alignes.
2. Tempo : variation de la duree du signe d un signeur a l autre, que
   l alignement ne peut pas corriger.
3. Separabilite : en plus proche voisin sur la trajectoire, un clip
   trouve-t-il un clip du MEME signe ? Les voisins du meme signeur sont
   exclus, sinon un signeur qui a enregistre deux fois le meme signe se
   trouverait lui-meme. Le hasard vaut 1 / nombre de signes.
"""
import collections
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src", "core"))
import timebase as tb  # noqa: E402

ARM = [c for pair in tb.MOTION_POINTS for c in pair]
MIN_CLIPS = 20
MAX_LAG = 30


def old_window(raw):
    n = len(raw)
    if n >= tb.WINDOW:
        s = (n - tb.WINDOW) // 2
        return raw[s:s + tb.WINDOW]
    return np.concatenate([raw, np.zeros((tb.WINDOW - n, raw.shape[1]), raw.dtype)])


def best_lag(a, b):
    """Decalage (frames) qui maximise la correlation de deux profils."""
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    scores = [np.mean(a[max(0, k):len(a) + min(0, k)] * b[max(0, -k):len(b) - max(0, k)])
              for k in range(-MAX_LAG, MAX_LAG + 1)]
    return int(np.argmax(scores)) - MAX_LAG


paths = sorted([q for q in glob.glob(os.path.join(ROOT, "data", "asl_citizen", "features", "*.npz"))
     # La cible seulement : l echafaudage s ecrit dans le meme dossier.
     if int(np.load(q, allow_pickle=True)["rank"]) <= 100])
clips = []
for p in paths:
    d = np.load(p, allow_pickle=True)
    raw = d["features"].astype(np.float32)
    grid = raw[tb.resample_indices(d["stamps"])]
    clips.append({"gloss": str(d["gloss"]), "signer": int(d["signer_id"]),
                  "new": grid[tb.window_indices(grid)], "old": old_window(raw),
                  "grid": grid})
counts = collections.Counter(c["gloss"] for c in clips)
clips = [c for c in clips if counts[c["gloss"]] >= MIN_CLIPS]
signs = sorted({c["gloss"] for c in clips})
print(f"{len(clips)} clips, {len(signs)} signes d au moins {MIN_CLIPS} clips "
      f"(hasard en plus proche voisin : {1/len(signs):.1%})")

rng = np.random.default_rng(0)

# 1. alignement residuel
print("\n1. alignement residuel entre clips du meme signe (|decalage|, frames a 30 fps)")
for method in ("old", "new"):
    lags = []
    for g in signs:
        group = [c for c in clips if c["gloss"] == g]
        for _ in range(60):
            i, j = rng.choice(len(group), 2, replace=False)
            lags.append(abs(best_lag(tb.energy(group[i][method]), tb.energy(group[j][method]))))
    lags = np.array(lags)
    print(f"   {'ancienne' if method == 'old' else 'nouvelle':9s} mediane {np.median(lags):4.1f}  "
          f"p90 {np.percentile(lags, 90):4.1f}  ({np.median(lags)/30*1000:.0f} ms / "
          f"{np.percentile(lags, 90)/30*1000:.0f} ms)")

# 2. tempo
print("\n2. duree du signe (90% du mouvement), d un signeur a l autre")
rows = []
for g in signs:
    spans = [tb.sign_span(c["grid"]) for c in clips if c["gloss"] == g]
    d = np.array([(b - a + 1) / 30 for a, b in spans if a is not None and b is not None])
    rows.append((g, np.median(d), d.std() / d.mean(), np.percentile(d, 10), np.percentile(d, 90)))
cv = np.array([r[2] for r in rows])
print(f"   coefficient de variation par signe : mediane {np.median(cv):.0%}, "
      f"min {cv.min():.0%}, max {cv.max():.0%}")
for g, med, c, lo, hi in sorted(rows, key=lambda r: -r[2])[:4]:
    print(f"   le plus variable : {g:10s} mediane {med:.2f} s, de {lo:.2f} a {hi:.2f} s (p10-p90)")
for g, med, c, lo, hi in sorted(rows, key=lambda r: r[2])[:2]:
    print(f"   le plus regulier : {g:10s} mediane {med:.2f} s, de {lo:.2f} a {hi:.2f} s")

# 3. separabilite
print("\n3. plus proche voisin sur la trajectoire des bras (autre signeur seulement)")
labels = np.array([c["gloss"] for c in clips])
signers = np.array([c["signer"] for c in clips])
for method in ("old", "new"):
    X = np.stack([c[method][:, ARM].reshape(-1) for c in clips]).astype(np.float64)
    sq = (X ** 2).sum(axis=1)
    D = sq[:, None] + sq[None, :] - 2 * X @ X.T
    D[signers[:, None] == signers[None, :]] = np.inf
    nn = D.argmin(axis=1)
    hit = (labels[nn] == labels).mean()
    within = np.median([D[i][(labels == labels[i]) & np.isfinite(D[i])].mean() for i in range(len(clips))])
    between = np.median([D[i][(labels != labels[i]) & np.isfinite(D[i])].mean() for i in range(len(clips))])
    print(f"   {'ancienne' if method == 'old' else 'nouvelle':9s} bon voisin {hit:.1%}   "
          f"distance moyenne meme signe / autres signes : {within/between:.2f}")
