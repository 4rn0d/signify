"""Le filtrage par frame de la demo donne-t-il la MEME entree que celui du
k-fold, applique a un tableau deja rembourre ?

La question n est pas rhetorique : le k-fold appelle gate_hands sur un
tableau (N, 80, 106) deja rembourre, la demo sur une frame (1, 1, 106) avant
rembourrage. Si les deux divergent d un seul bit, le modele expedie recoit a
l inference une entree qu il n a jamais vue a l entrainement — la panne qui a
fait tomber la phase alphabet de 0.985 a 0.384.

On compare sur de vrais clips, pas sur du bruit : les frames en conflit sont
rares (~5%) et c est precisement la que les deux chemins pourraient differer.
"""
import glob
import sys

import numpy as np

sys.path.insert(0, "src/core")
from train_kfold import Layout, gate_hands, pad_sequence

GATE = dict(threshold=0.7, soft=False, inside_threshold=0.5, use_conflict=True)

paths = sorted(glob.glob("data/wlasl/features_rtm_all/*.npz"))[:300]
assert paths, "aucun npz"

layout = Layout.from_dim(104)
layout.tail = 2

worst = 0.0
conflict_frames = 0
total_frames = 0
gated_away = 0

for path in paths:
    data = np.load(path, allow_pickle=True)
    raw = data["features"].astype(np.float32)
    scores = data["hand_scores"].astype(np.float32)
    inside = data["hand_inside"].astype(np.float32)
    conflict = data["hand_conflict"].astype(bool)

    # Le dossier stocke 104 colonnes ; la queue des deux confiances est
    # ajoutee par load_sequences. On la reconstitue a l identique.
    full = np.concatenate([raw, scores], axis=-1)

    # --- chemin k-fold : rembourrer puis filtrer, en lot ---
    batch = pad_sequence(full)[None, ...]
    pad_scores = pad_sequence(scores)[None, ...]
    pad_inside = pad_sequence(inside)[None, ...]
    pad_conflict = pad_sequence(
        conflict.astype(np.float32)[:, None])[None, :, 0].astype(bool)
    kfold = gate_hands(batch, layout, pad_scores, pad_inside, pad_conflict,
                       **GATE)[0]

    # --- chemin demo : filtrer frame par frame, puis rembourrer ---
    frames = []
    for index in range(len(full)):
        one = gate_hands(full[index][None, None, :], layout,
                         scores[index][None, None, :],
                         inside[index][None, None, :],
                         np.array([[conflict[index]]], dtype=bool),
                         **GATE)[0, 0]
        frames.append(one)
    demo = pad_sequence(np.array(frames, dtype=np.float32))

    worst = max(worst, float(np.abs(kfold - demo).max()))
    conflict_frames += int(conflict.sum())
    total_frames += len(full)
    before = (full[..., layout.left_flag] > 0.5).sum() + \
             (full[..., layout.right_flag] > 0.5).sum()
    after = (demo[:len(full), layout.left_flag] > 0.5).sum() + \
            (demo[:len(full), layout.right_flag] > 0.5).sum()
    gated_away += int(before - after)

print(f"{len(paths)} clips, {total_frames} frames, "
      f"{conflict_frames} en conflit ({conflict_frames / total_frames:.2%})")
print(f"mains ecartees par le filtrage : {gated_away}")
print(f"ecart maximal entre les deux chemins : {worst:.2e}")
print()
if worst == 0.0 and conflict_frames > 0:
    print("IDENTIQUES au bit, conflits inclus")
elif conflict_frames == 0:
    print("ATTENTION : aucune frame en conflit dans cet echantillon, "
          "le cas interessant n est pas couvert")
else:
    print(f"DIVERGENCE de {worst:.2e} : ne pas expedier")
    sys.exit(1)
