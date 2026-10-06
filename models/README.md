# models/

Every model file has a `.meta.json` sidecar beside it recording how it was
trained and what it scored. Two `.keras` files are otherwise indistinguishable
on disk, so **read the sidecar, not the filename**.

All accuracies below use the temporal split (see `docs/FINDINGS.md`). Numbers
from a random split are not comparable and are marked as such.

---

## Active — loaded by the code

| file | role | val_accuracy |
|---|---|---|
| `sign_model_fusion.keras` + `.meta.json` | **what `webcam_demo.py` runs**: image + landmarks, mirror-augmented so both hands work | 0.9277 |
| `sign_model.keras` + `.tflite` + `.meta.json` | CNN only. Fallback used by `webcam_demo.py` when the fusion model is absent | 0.8357 |

The demo prints which one it loaded at startup, so you can always tell:

```
Modele : FUSION image+landmarks | val_accuracy=0.9277 | 2026-10-01T20:06:48-04:00
```

### Which files belong together

A model is its `.keras` **plus** its `.meta.json`, and `.tflite` where present.
Move or delete them as a set — a `.keras` without its sidecar becomes an
unidentifiable blob, which is what happened to the archived no-flip model
before it was documented retroactively.

Both models share `class_names.json`. Class order is positional, so a model
and a mismatched `class_names.json` will predict confident nonsense rather
than fail — never mix them across models trained on different class sets.

---

## Shared

| file | role |
|---|---|
| `class_names.json` | the 28 class names, in model output order. Shared by every model here |
| `hand_landmarker.task` | MediaPipe asset, downloaded automatically. Used by `hand_segmenter.py` and `landmark_extractor.py` |
| `comparison.json` | scores from the three-way comparison, written by `train_fusion.py` |

---

## archive/ — superseded, kept for reference

Not loaded by any code. Kept because each one documents a specific mistake
worth not repeating.

| file | why it was replaced |
|---|---|
| `sign_model_fusion_noflip.*` | Fusion before mirror augmentation. Scores 0.9373 on the trained hand but **0.0872 on the other** — chance is 0.036. Shows what a single-handed dataset produces |
| `sign_model_randomsplit.*` | Trained with a random train/val split, which leaks on this dataset because consecutive video frames are near-identical. Reports 0.9938; honest score is 0.8277. Its sidecar carries `val_accuracy_is_inflated: true` |

---

## Not in git

`.gitignore` excludes `*.keras`, `*.tflite`, `*.h5` anywhere in the tree, so
no model binaries are committed — only the `.meta.json` sidecars and
`comparison.json`, which are small and record what produced what.

Regenerate the models with:

```bash
python src/train.py                      # CNN -> sign_model.keras + .tflite
python src/train_fusion.py               # all three, writes comparison.json
python src/train_fusion.py --only fusion # just the fusion model
```
