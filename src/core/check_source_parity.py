"""Le chemin d entree du modele dans la demo est-il celui de l entrainement ?

load_model() ne suffit pas a le dire : c est Source.__init__ qui active le mode
brut d apres la fiche, et Source a besoin de rtmlib et d une camera. On remplace
donc le detecteur par un faux qui rend des points fixes, ce qui laisse intact
tout ce qui nous interesse : frame_features, le mode brut, et gate_hands avec
les parametres de la fiche.

On compare a ce que l entrainement aurait fait de la MEME frame : extraction
brute, puis gate_hands. Si les deux divergent, le modele recoit a l inference
une entree qu il n a jamais vue.
"""
import json
import sys
import types

import numpy as np

sys.path.insert(0, "src/core")

# --- faux detecteur, installe AVANT que Source ne l importe ---
fake = types.ModuleType("rtmlib")


class Wholebody:
    def __init__(self, *a, **k):
        session = types.SimpleNamespace(
            get_providers=lambda: ["CUDAExecutionProvider"])
        self.pose_model = types.SimpleNamespace(session=session)

    def __call__(self, frame):
        return FRAMES["keypoints"], FRAMES["scores"]


fake.Wholebody = Wholebody
sys.modules["rtmlib"] = fake

import extract_rtmpose
from extract_rtmpose import frame_features, hand_confidences
import webcam_demo_rtm as demo
from train_kfold import Layout, gate_hands

rng = np.random.default_rng(0)
WIDTH, HEIGHT = 640, 480

# Points plausibles : epaules sures, une main sure, l autre douteuse, de quoi
# declencher le seuil et le hors-cadre.
points = rng.uniform(100, 500, size=(133, 2)).astype(np.float32)
points[extract_rtmpose.COCO_LEFT_SHOULDER] = (250, 200)
points[extract_rtmpose.COCO_RIGHT_SHOULDER] = (390, 200)
scores = rng.uniform(0.75, 0.95, size=133).astype(np.float32)
scores[extract_rtmpose.RIGHT_HAND] = rng.uniform(0.1, 0.4, size=21)  # douteuse

FRAMES = {"keypoints": [points], "scores": [scores]}

meta = json.load(open("models/wlasl/wlasl_gru_rtmpose.meta.json", encoding="utf-8"))
gate = {k: meta.get(k) for k in ("raw_features", "gate", "gate_inside",
                                 "gate_conflict", "gate_soft",
                                 "extraction_threshold")}
layout = Layout.from_dim(extract_rtmpose.FEATURE_DIM)
layout.tail = 2

source = demo.Source(mode="balanced", with_confidence=True, layout=layout,
                     gate=gate)

print(f"  fiche : raw_features={gate['raw_features']} gate={gate['gate']} "
      f"inside={gate['gate_inside']} conflict={gate['gate_conflict']}")
print(f"  Source a-t-il active le mode brut ? {extract_rtmpose.RAW_MODE}")
print(f"  seuil d extraction : {extract_rtmpose.HAND_SCORE_THRESHOLD}")
assert extract_rtmpose.RAW_MODE is True, "le mode brut n a PAS ete active"

vector, drawable, found = source.process(np.zeros((HEIGHT, WIDTH, 3), np.uint8))

# --- ce que l entrainement aurait produit de cette frame ---
raw, _ = frame_features(points, scores, WIDTH, HEIGHT)
conf = np.asarray(hand_confidences(scores), dtype=np.float32)
full = np.concatenate([raw, conf])
expected = gate_hands(
    full[None, None, :], layout,
    conf[None, None, :],
    np.asarray(frame_features.last_inside, dtype=np.float32)[None, None, :],
    np.array([[frame_features.last_conflict]], dtype=bool),
    threshold=gate["gate"], soft=gate["gate_soft"],
    inside_threshold=gate["gate_inside"], use_conflict=gate["gate_conflict"],
)[0, 0]

difference = float(np.abs(vector - expected).max())
flags = vector[extract_rtmpose.FEATURE_DIM - len(extract_rtmpose.POSE_KEYPOINTS) * 2 - 2:
               extract_rtmpose.FEATURE_DIM - len(extract_rtmpose.POSE_KEYPOINTS) * 2]
print(f"  colonnes rendues : {vector.shape[0]}  (attendu 106)")
print(f"  drapeaux de presence apres filtrage : {flags}")
print(f"  main droite douteuse ecartee : {flags[1] < 0.5}")
print(f"  ce qui est DESSINE suit : droite={drawable['right'] is None and 'absente' or 'presente'}")
print(f"  ecart avec le chemin d entrainement : {difference:.2e}")
print()
if difference == 0.0 and flags[1] < 0.5:
    print("L ENTREE DU MODELE EST IDENTIQUE A CELLE DE L ENTRAINEMENT")
else:
    print("DIVERGENCE : la demo ne nourrit pas le modele comme l entrainement")
    raise SystemExit(1)
