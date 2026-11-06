"""La fenetre centree sur le signe est-elle la MEME a l entrainement et en demo ?

Deux chemins, sur de vrais clips :

  demo         Source.process frame par frame (TensorRT, filtrage de la
               fiche), horodatages du decodeur, puis model_input(centred=True)
  entrainement load_sequences (grille a 30 fps et fenetre via timebase), puis
               gate_hands comme dans train_kfold

Les deux doivent rendre les memes 80 x 106 valeurs au bit pres : sinon le
modele verrait en demo une fenetre placee ailleurs que pendant son
entrainement, ce que le centrage devait precisement empecher.
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import extract as ex  # noqa: E402

from train_kfold import Layout, gate_hands, load_sequences  # noqa: E402
import webcam_demo_rtm as demo  # noqa: E402

GATE = {"raw_features": True, "gate": 0.7, "gate_inside": 0.5,
        "gate_conflict": True, "gate_soft": False, "extraction_threshold": 0.7}
WANTED = ("WANT1", "ME", "OHISEE", "WATER", "BOOK")

layout = Layout.from_dim(ex.FEATURE_DIM)
layout.tail = 2

records = load_sequences("all", ex.FEATURES_DIR, with_confidence=True, with_meta=True)
chosen, seen = [], set()
for r in records:
    if r["gloss"] in WANTED and r["gloss"] not in seen:
        chosen.append(r)
        seen.add(r["gloss"])
print(f"{len(records)} clips charges, {len(chosen)} compares")

source = demo.Source(mode="balanced", with_confidence=True, layout=layout,
                     gate=GATE, tensorrt=True)
print(f"demo : {source.model.det_model.session.get_providers()[0]}")

worst = 0.0
for r in chosen:
    m = r["meta"]
    trained = gate_hands(r["features"][None], layout, m["hand_scores"][None],
                         m["hand_inside"][None], m["hand_conflict"][None],
                         threshold=0.7, inside_threshold=0.5, use_conflict=True)[0]

    capture = cv2.VideoCapture(os.path.join(ex.CITIZEN, "videos", r["video_id"] + ".mp4"))
    samples, index = [], 0
    fps = capture.get(cv2.CAP_PROP_FPS)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        stamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if stamp <= 0.0 and index > 0 and fps > 0:
            stamp = index / fps
        vector, _, _ = source.process(frame)
        samples.append((stamp, vector))
        index += 1
    capture.release()
    live = demo.model_input(samples, centred=True)

    same_shape = live.shape == trained.shape
    difference = float(np.abs(live - trained).max()) if same_shape else float("inf")
    worst = max(worst, difference)
    print(f"  {r['gloss']:8s} {len(samples):4d} frames -> demo {live.shape}, "
          f"entrainement {trained.shape}, ecart {difference:.2e}")

print()
print("IDENTIQUE au bit" if worst == 0.0 else f"DIVERGENCE {worst:.2e}")
