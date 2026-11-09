# models/

Every model file has a `.meta.json` sidecar (a "card") beside it recording how
it was trained and what it scored. Two `.keras` files are otherwise
indistinguishable on disk, so **read the card, not the filename**.

A model is its `.keras` **plus** its card, and `.tflite` where present. Move or
delete them as a set. A card lists its weight files by path: moving the
weights without updating the card breaks the demo.

`.gitignore` excludes `*.keras`, `*.tflite`, `*.h5` and `*.task` anywhere in the
tree: only the cards and the result JSONs are committed.

---

## Top level — ASL Citizen, the current phase

| file | role |
|---|---|
| `aslc_kfold_<tag>.json` | k-fold results, written by `src/core/train_kfold.py` (`.partial.json` while a run is going) |
| `aslc_gru_rtmpose*.keras` + `aslc_gru_rtmpose.meta.json` | the shipped model, once `src/core/train_final.py` has built it. The demo loads it first |

Pretrainings kept by `--pretrain-cache` are not here: they live in
`data/asl_citizen/pretrain_cache/`, since they are derived from ASL Citizen
data and fall under its license.

---

## wlasl/ — word phase on WLASL, closed

| file | role |
|---|---|
| `wlasl_gru_rtmpose_s{0-4}_k{0-4}.keras` + `wlasl_gru_rtmpose.meta.json` | the shipped WLASL model: 5 seeds x 5 snapshots. The demo falls back to it while no ASL Citizen card exists |
| `wlasl_kfold*.json` | every WLASL k-fold experiment, cited throughout `docs/WLASL.md` |
| `wlasl_gru.meta.json` | card of the MediaPipe-era model; its weights are gone |
| `holistic_landmarker.task` | MediaPipe asset for `src/wlasl/extract_landmarks.py` and `webcam_demo_words.py` |
| `archive/before_best/` | the WLASL model shipped before the best configuration |

---

## alphabet/ — fingerspelling phase, closed

All accuracies use the temporal split (see `docs/FINDINGS.md`).

| file | role | val_accuracy |
|---|---|---|
| `sign_model_fusion.keras` + `.meta.json` | what `src/alphabet/webcam_demo_alphabet.py` runs: image + landmarks, mirror-augmented so both hands work | 0.9277 |
| `sign_model.keras` + `.tflite` + `.meta.json` | CNN only, fallback when the fusion model is absent | 0.8357 |
| `class_names.json` | the 28 class names, in output order, shared by both models. Order is positional: a mismatched file predicts confident nonsense rather than failing |
| `hand_landmarker.task` | MediaPipe asset, downloaded automatically |
| `comparison.json` | the three-way comparison, written by `train_fusion.py` |

`archive/` keeps two superseded models, each documenting a mistake:

| file | why it was replaced |
|---|---|
| `sign_model_fusion_noflip.*` | Fusion before mirror augmentation: 0.9373 on the trained hand but **0.0872 on the other** (chance 0.036) |
| `sign_model_randomsplit.*` | Random train/val split, which leaks because consecutive video frames are near-identical: reports 0.9938, honest score 0.8277. Its card carries `val_accuracy_is_inflated: true` |

Regenerate the alphabet models with:

```bash
python src/alphabet/train.py                      # CNN -> sign_model.keras + .tflite
python src/alphabet/train_fusion.py               # all three, writes comparison.json
python src/alphabet/train_fusion.py --only fusion # just the fusion model
```
