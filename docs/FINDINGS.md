# Signify — findings

Record of what was broken, what was measured, and what the numbers mean.
Every figure here was measured on this machine against this dataset.

Dataset: ASL Alphabet, 28 classes, 71,930 segmented images.
Environment: TensorFlow 2.21, Keras 3.15.1, Python 3.11, CPU only
(TF has no GPU support on native Windows since 2.11).

---

## 1. Training was broken

Three separate bugs in `src/data_loader.py`. The first stopped training
outright; the other two would have wasted the run silently.

### 1.1 Mismatched label formats — hard crash

`train_ds` used `label_mode="int"`, `val_ds` used `label_mode="categorical"`,
and the model compiles with `sparse_categorical_crossentropy`. Epoch 1
trained fine and died at the first validation step:

```
ValueError: Argument `output` must have rank (ndim) `target.ndim - 1`.
Received: target.shape=(None, 28), output.shape=(None, 28)
```

Both splits now come from one shared kwargs dict so they cannot drift apart.

### 1.2 RandomBrightness destroyed every training image

The pipeline normalises to `[0,1]` before augmenting, but `RandomBrightness`
defaults to `value_range=(0, 255)` and scales its shift by that range:

```
input  range: 0.000 .. 1.000
RandomBrightness(0.15) -> 0.000 .. 39.164  (mean 15.956)   <- destroyed
RandomContrast(0.15)   -> 0.000 .. 1.050   (mean 0.500)    <- fine
```

Training images were flattened to near-uniform values while validation
images stayed clean. Fixed with `value_range=(0.0, 1.0)`.

### 1.3 Cache exceeded available RAM and froze the shuffle

`.cache()` sat after the float32 rescale: 2.83 GB (train) + 0.71 GB (val)
against 2.9 GB free. The cache now holds `uint8` ahead of normalisation
(0.88 GB for both splits).

`.cache()` also freezes whatever order it receives, so every epoch replayed
an identical batch sequence:

```
shuffle -> cache: [3, 8, 14, 19, ...] / [3, 8, 14, 19, ...]   frozen
cache -> shuffle: [17, 1, 3, 16, ...] / [10, 1, 3, 17, ...]   reshuffles
```

A `.shuffle()` after the cache restores per-epoch reshuffling.

---

## 2. The webcam demo contradicted its own training data

The training set is hand-on-**white-background**, produced by
`HandSegmenter.process()` (GrabCut). Measured on the files: corner pixels
average 254.7/255, and 43-75% of every image is near-white.

`webcam_demo.py` never called `HandSegmenter`. It cropped the raw frame and
padded with black, so the model received real background where it was
trained to see white.

Same 336 images, same model, only preprocessing differs:

```
SEGMENTED (white bg)  -> 331/336 = 0.985   <- what training saw
RAW       (real bg)   -> 129/336 = 0.384   <- what the demo fed it
```

The demo now runs the same chain as `prepare_segmented_dataset.py`.
After the fix, raw frames score **0.947**.

A second bug: `MODEL_PATH = "../models/sign_model.keras"` only resolved when
launched from inside `src/`. Paths now derive from `__file__`.

---

## 3. Real-time performance: 11 fps to 54 fps

### 3.1 `model.predict()` costs 36 ms on a single frame

`predict()` is built for large batches and carries heavy per-call overhead:

| call | time | speedup |
|---|---|---|
| `model.predict()` | 36.51 ms | 1x |
| `model(x, training=False)` | 5.38 ms | 6.8x |
| `tf.function` wrapped | 1.11 ms | 33x |
| **TFLite (4 threads)** | **0.38 ms** | **96x** |

TFLite output is numerically identical — max absolute difference **4.9e-11**.
Model size drops 13.8 MB to 4.6 MB.

### 3.2 GrabCut ran at pointless resolution

At 640x480, GrabCut was **39.7 ms of 49 ms** segmentation time (81%);
MediaPipe and everything else was 9.4 ms. It ran 4 iterations at full crop
resolution to produce a mask that is downscaled to 64x64 anyway.

Computing the mask at 128 px with 2 iterations, on 280 real images:

```
OLD code, default args                 169/180 = 0.939
NEW code, default args (dataset prep)  169/180 = 0.939
NEW code, realtime args (demo)         169/180 = 0.939
```

Identical accuracy, roughly half the time. `HandSegmenter` defaults are
unchanged so `prepare_segmented_dataset.py` stays reproducible; the demo
opts in via `GRABCUT_REALTIME`.

### 3.3 Result

```
BEFORE  full GrabCut + model.predict()    53.1 ms   18.8 fps
        full GrabCut + TFLite             32.2 ms   31.1 fps
AFTER   realtime GrabCut + TFLite         18.4 ms   54.4 fps
```

Measured on synthetic 640x480 frames, where GrabCut converges faster than on
real photos — treat **2.9x** as the reliable figure rather than absolute fps.

### 3.4 GrabCut is nondeterministic

Same code, same inputs, run twice:

```
identical bbox   : 38/38
identical pixels :  9/38
```

OpenCV seeds GrabCut's GMM k-means randomly. Byte-equality is therefore the
wrong regression test — use statistical equivalence (section 3.2). Some live
prediction flicker is inherent and cannot be fixed by threshold or model.

---

## 4. The 99.38% was not real

### 4.1 The dataset is video, not photographs

Files are named `{class}1.jpg` through `{class}3000.jpg` — sequential frames
from a continuous capture. Neighbouring frames are near-duplicates:

```
mean abs pixel diff, consecutive frames (A_n vs A_n+1):   4.73
mean abs pixel diff, random frames same class         :  61.04
ratio: random is 12.9x more different than consecutive
```

A random 80/20 split therefore puts each validation frame's immediate
neighbours into training.

### 4.2 Quantified

Same model, same epochs, same augmentation — only the split changes:

| split | val_accuracy |
|---|---|
| random (original) | 0.9938 |
| temporal holdout | **0.8277** |

The ~17 point gap is leakage.

### 4.3 The leak also hid genuine overfitting

Under the fair split, validation accuracy reaches 0.81 in epoch 1 and then
stalls for fourteen epochs while train loss keeps falling and val_loss
climbs 0.61 to 1.49. The random split showed val_loss falling monotonically
to 0.0216 and concealed this completely.

### 4.4 Official test set

The bundled test set is 28 images (one per class). Of 27 usable:

```
TEST ACCURACY: 21/27 = 0.778   (hand not found in 5)
```

Five failures were `NO HAND DETECTED` — the classifier was never asked. Only
one genuine misclassification (`Z -> D` at 41% confidence, below the 80%
display threshold). Separating the failure modes:

- classifier, on images that segmented: **21/22 = 0.955**
- detection failure rate: **5/27 = 18.5%**

These images are unusually dark; detection on a live camera is a different
regime.

---

## 5. Callbacks recovered 4.9 points immediately

`EarlyStopping` and `ModelCheckpoint(save_best_only=True)` were added, both
monitoring `val_accuracy` rather than `val_loss` — under the fair split
val_loss rises from epoch 2 while val_accuracy keeps improving in bursts, so
patience on val_loss stops too early.

First run with the fixed split:

```
ep 1  0.8015   ep 5  0.8206   ep  9  0.8140
ep 2  0.8140   ep 6  0.8357 <- best
ep 3  0.7705   ep 7  0.8339   ep 10  0.7859
ep 4  0.7824   ep 8  0.8133   ep 11  0.7872 <- last, stopped here
```

```
SAVED model on the temporal val set: accuracy=0.8357
  best epoch (6) : 0.8357
  last epoch (11): 0.7872
```

Under the old code `model.save()` ran after the final epoch and would have
shipped **0.7872**.

---

## 6. Landmarks beat the CNN

MediaPipe landmarks were extracted for the whole dataset
(`src/landmark_extractor.py`, 8 worker processes):

```
63,068 landmark sets from 71,930 images = 87.7%
```

Coordinates are normalised to wrist-origin at unit scale, without which the
model learns where the hand sits in frame rather than its shape.

Three models, **identical samples** (53,563 train / 9,505 val — those having
both an image and landmarks), identical temporal split, identical callbacks:

| model | val_accuracy | best epoch | params |
|---|---|---|---|
| CNN only | 0.8722 | 3 | 1,145,564 |
| Landmarks only | **0.9354** | 8 | **60,892** |
| Fusion (image + landmarks) | **0.9431** | 9 | 1,229,340 |

Restricting to shared samples matters: comparing fusion on 85% of the data
against a CNN on 100% would confound two changes at once.

### What this means

Landmarks beat pixels by **6.3 points with 19x fewer parameters**. Fusion
then adds only **0.8** on top. The headline is not "fusion helps" — it is
that the CNN was the weak component.

This is consistent with everything above. Pixel models on this dataset lean
on session conditions — lighting, hand placement, camera auto-exposure —
that do not survive into held-out frames. Landmark geometry has nothing to
memorise.

### Cost to run live

| | accuracy | params | needs GrabCut |
|---|---|---|---|
| CNN | 0.8722 | 1.15 M | yes (~20 ms/frame) |
| Landmarks | 0.9354 | 0.06 M | **no** |
| Fusion | 0.9431 | 1.23 M | yes |

The landmark model gives up 0.8 points and drops the entire segmentation
stage — the slowest part of the pipeline, the nondeterministic one, and the
source of the bug in section 2.

---

## 7. Model provenance

Two `.keras` files on disk are indistinguishable. `train.py` now writes
`models/sign_model.meta.json` recording split, score, best epoch, sample
counts and git commit, and `webcam_demo.py` prints it at startup:

```
Modele : split=temporal (grouped by frame index) | val_accuracy=0.8357 | epoch 6/11 | 2026-09-24T20:05:03-04:00
```

The superseded random-split model is kept in `models/archive/` with its own
sidecar recording `val_accuracy: 0.9938`, `val_accuracy_is_inflated: true`,
`honest_estimate: 0.8277`.

---

## 8. Housekeeping

- Three byte-identical copies of `hand_landmarker.task` (md5 `15318430ea38...`,
  7.5 MB each); only `models/` is referenced. Removed two.
- Removed an empty `notebooks/exploration.ipynb` (0 bytes, invalid JSON).
- Added `.gitignore` — `data/` alone is 1.5 GB and nothing was ignored.
- **28 MB** freed; tracked set is 20 files, largest 20 KB.

---

## Open items

**Never tested on a real webcam.** Every number here comes from still images.
Watch three things: the `cv2.flip` mirror versus the handedness of the
training data; the 80% confidence threshold, tuned blind; and segmentation
quality against a cluttered background. The preview tile in the corner shows
whether a bad prediction is segmentation's fault or the model's.

**Subject leakage remains.** The temporal split removes frame-level leakage
but not subject or setup leakage — all numbers come from one person, one
camera, one session. Expect all three models to drop on a new hand, and the
landmark model to drop least.

**No `nothing` class.** The current model has 28 classes; the archived one
had 29 including `nothing`. With a hand detected but not forming a letter,
the model must still pick one of 28. The 80% threshold is the mitigation.

**`evaluate.py` is still a stub.** A confusion matrix would show which
letters fail — M/N/S/T and A/E/S are the classic ASL look-alikes — which is
more actionable than a single accuracy figure.

**Push is blocked.** `4rn0d/signify` grants `pull` but not `push` to
`ulrichdubjob`. Commit `875f08c` sits on local branch `ulrich_POC`.
