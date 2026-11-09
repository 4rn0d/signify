"""L extraction d ASL Citizen et la demo donnent-elles la MEME entree au modele ?

On prend de vrais clips, dont un a basse cadence et un en 960x540, et on
compare frame par frame :
  - ce que la demo calcule (Source.process, TensorRT, filtrage de la fiche)
  - ce que l extraction a stocke, filtre ensuite par gate_hands comme a
    l entrainement.
Un seul bit d ecart et le modele verrait en demo une entree jamais vue.

On verifie aussi les horodatages : monotones, et coherents avec la cadence
annoncee, ou sinon la preuve d une cadence variable.
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import extract as ex  # noqa: E402

from train_kfold import Layout, gate_hands  # noqa: E402
import webcam_demo_rtm as demo  # noqa: E402

audit = json.load(open(os.path.join(ex.PROJECT_ROOT, "data", "asl_citizen", "audit.json")))
ranked = json.load(open(ex.RANKED))
top = {s["gloss"] for s in ranked[:100]}
pool = [a for a in audit if a["gloss"] in top]
picks = ([a for a in pool if a["fps"] < 20][:1]
         + [a for a in pool if a["w"] == 960][:1]
         + [a for a in pool if 29 < a["fps"] < 31 and a["w"] == 640][:2])

GATE = {"raw_features": True, "gate": 0.7, "gate_inside": 0.5,
        "gate_conflict": True, "gate_soft": False, "extraction_threshold": 0.7}
layout = Layout.from_dim(ex.FEATURE_DIM)
layout.tail = 2

model, detector = ex.build_model(tensorrt=True)
source = demo.Source(mode="balanced", with_confidence=True, layout=layout,
                     gate=GATE, tensorrt=True)
print(f"extraction : {detector}   demo : "
      f"{source.model.det_model.session.get_providers()[0]}")
print()

worst = 0.0
for a in picks:
    clip = {"file": a["file"], "gloss": a["gloss"], "code": "",
            "signer_id": int(a["signer"].lstrip("Pp")), "split": a["split"],
            "rank": 0}
    stem = os.path.splitext(a["file"])[0]
    out = os.path.join(ex.FEATURES_DIR, f"{stem}.npz")
    if os.path.exists(out):
        os.remove(out)
    status, n = ex.process(model, clip, detector)
    data = np.load(out, allow_pickle=True)

    raw = np.concatenate([data["features"], data["hand_scores"]], axis=1)
    gated = gate_hands(raw[None], layout, data["hand_scores"][None],
                       data["hand_inside"][None], data["hand_conflict"][None],
                       threshold=0.7, inside_threshold=0.5,
                       use_conflict=True)[0]

    capture = cv2.VideoCapture(os.path.join(ex.CITIZEN, "videos", a["file"]))
    live = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        vector, _, _ = source.process(frame)
        live.append(vector)
    capture.release()
    live = np.array(live, dtype=np.float32)

    difference = float(np.abs(live - gated).max()) if len(live) == len(gated) else float("inf")
    worst = max(worst, difference)
    stamps = data["stamps"]
    steps = np.diff(stamps)
    print(f"  {a['file'][:34]:34s} {a['w']}x{a['h']} {a['fps']:5.1f} fps  "
          f"{status} {n} frames")
    print(f"      horodatages : monotones {bool((steps > 0).all())}, "
          f"pas median {np.median(steps)*1000:.1f} ms "
          f"(attendu {1000/a['fps']:.1f}), min {steps.min()*1000:.1f}, "
          f"max {steps.max()*1000:.1f}, duree {stamps[-1]-stamps[0]:.2f} s")
    print(f"      ecart demo / extraction filtree : {difference:.2e}")

print()
print("IDENTIQUE au bit" if worst == 0.0 else f"DIVERGENCE {worst:.2e}")
