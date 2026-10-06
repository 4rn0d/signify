# Signify — alphabet phase findings

Complete record of the ASL **alphabet** (fingerspelling) work: what was
broken, what was measured, what helped, and what didn't. This phase is
closed — word-level recognition is a different problem and starts fresh.

Every figure was measured on this machine against this dataset. Numbers that
are not comparable to each other are marked as such.

Dataset: ASL Alphabet, 28 classes, 84,000 raw images.
Environment: TensorFlow 2.21, Keras 3.15.1, Python 3.11, CPU only
(TF has no GPU support on native Windows since 2.11).

**Final state:** fusion model (image + landmarks, mirror-augmented),
val_accuracy **0.9277**, works with either hand, runs at ~34 fps.

---

## 1. Training was broken

Three bugs in `src/data_loader.py`. The first stopped training outright; the
other two would have wasted the run silently.

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

Training images were flattened to near-uniform values while validation images
stayed clean. Fixed with `value_range=(0.0, 1.0)`.

### 1.3 Cache exceeded available RAM and froze the shuffle

`.cache()` sat after the float32 rescale: 2.83 GB (train) + 0.71 GB (val)
against 2.9 GB free. It now caches `uint8` ahead of normalisation (0.88 GB).

`.cache()` also freezes whatever order it receives, so every epoch replayed an
identical batch sequence:

```
shuffle -> cache: [3, 8, 14, 19, ...] / [3, 8, 14, 19, ...]   frozen
cache -> shuffle: [17, 1, 3, 16, ...] / [10, 1, 3, 17, ...]   reshuffles
```

---

## 2. The demo contradicted its own training data

Training images are hands on a **white background**, produced by
`HandSegmenter.process()` (GrabCut). Measured on the files: corner pixels
average 254.7/255, and 43-75% of every image is near-white.

`webcam_demo.py` never called `HandSegmenter`. It cropped the raw frame and
padded with black, so the model received real background where it was trained
to see white.

Same 336 images, same model, only preprocessing differs:

```
SEGMENTED (white bg)  -> 331/336 = 0.985   <- what training saw
RAW       (real bg)   -> 129/336 = 0.384   <- what the demo fed it
```

The demo now runs the same chain as `prepare_segmented_dataset.py`. After the
fix, raw frames score **0.947**.

A second bug: `MODEL_PATH = "../models/sign_model.keras"` only resolved when
launched from inside `src/`. Paths now derive from `__file__`.

---

## 3. Real-time performance: 11 fps to 54 fps

### 3.1 `model.predict()` costs 36 ms on a single frame

| call | time | speedup |
|---|---|---|
| `model.predict()` | 36.51 ms | 1x |
| `model(x, training=False)` | 5.38 ms | 6.8x |
| `tf.function` wrapped | 1.11 ms | 33x |
| **TFLite (4 threads)** | **0.38 ms** | **96x** |

TFLite output is numerically identical — max absolute difference **4.9e-11**.
Model size drops 13.8 MB -> 4.6 MB.

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
unchanged so dataset prep stays reproducible; the demo opts in via
`GRABCUT_REALTIME`.

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
wrong regression test — use statistical equivalence (3.2). Some live
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

The ~17 point gap is leakage. `data_loader.py` now splits by frame index
(first 80% of each class train, last 20% val).

### 4.3 The leak also hid genuine overfitting

Under the fair split, validation accuracy reaches 0.81 in epoch 1 and then
stalls for fourteen epochs while train loss keeps falling and val_loss climbs
0.61 -> 1.49. The random split showed val_loss falling monotonically to
0.0216 and concealed this completely.

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

`EarlyStopping` and `ModelCheckpoint(save_best_only=True)` both monitor
`val_accuracy` rather than `val_loss` — under the fair split val_loss rises
from epoch 2 while val_accuracy keeps improving in bursts, so patience on
val_loss stops too early.

```
ep 1  0.8015   ep 5  0.8206   ep  9  0.8140
ep 2  0.8140   ep 6  0.8357 <- best, saved
ep 3  0.7705   ep 7  0.8339   ep 10  0.7859
ep 4  0.7824   ep 8  0.8133   ep 11  0.7872 <- last, stopped here
```

Under the old code `model.save()` ran after the final epoch and would have
shipped **0.7872** instead of **0.8357**.

---

## 6. Landmarks beat the CNN

MediaPipe already computes 21 hand landmarks to find the crop box, then
discards them. `src/landmark_extractor.py` extracts them for the dataset
(8 worker processes):

```
63,068 landmark sets from 71,930 images = 87.7%
```

Coordinates are normalised to wrist-origin at unit scale, without which the
model learns where the hand sits in frame rather than its shape.

Three models, **identical samples** (53,563 train / 9,505 val — those having
both an image and landmarks), identical temporal split and callbacks:

| model | val_accuracy | params |
|---|---|---|
| CNN only | 0.8722 | 1,145,564 |
| Landmarks only | **0.9354** | **60,892** |
| Fusion (image + landmarks) | 0.9373 | 1,229,340 |

Landmarks beat pixels by **6.3 points with 19x fewer parameters**. Fusion adds
only 0.8 on top — and a second fusion run scored 0.9431, so the **run-to-run
spread is ~0.58 points** and fusion's edge over landmarks alone is not
established.

Pixel models on this dataset lean on session conditions — lighting, hand
placement, camera auto-exposure — that do not survive into held-out frames.
Landmark geometry has nothing to memorise.

### Cost to run live

| | accuracy | params | needs GrabCut |
|---|---|---|---|
| CNN | 0.8722 | 1.15 M | yes (~20 ms/frame) |
| Landmarks | 0.9354 | 0.06 M | **no** |
| Fusion | 0.9373 | 1.23 M | yes |

The landmark model gives up ~0.8 points and drops the entire segmentation
stage — the slowest part of the pipeline, the nondeterministic one, and the
source of the bug in section 2. It remains the better choice if simplicity or
speed ever matters more than the last point.

---

## 7. What did not help

Four separate attempts to strengthen the image branch, all null. Recorded
because each one is a direction not worth retrying.

| lever | tried | result |
|---|---|---|
| architecture | 93k scratch CNN -> 2.3M MobileNetV2 (frozen, ImageNet) | 0.8722 -> 0.8712 |
| fine-tuning | unfroze top third, 26 min at LR 1e-4 | 0.9455 -> 0.9406 (worse) |
| resolution | regenerated dataset at 128px | 0.8722 -> 0.8741 |
| — | all deltas | within the ±0.6 noise floor |

### Transfer learning

Frozen MobileNetV2 matched the scratch CNN exactly (0.8712 vs 0.8722) despite
24x the parameters. Fine-tuning made it slightly worse. Both numbers sit
inside the noise band.

One genuine mistake on the way: the first fusion attempt concatenated a
1280-d image embedding with 63 landmark values. With 95% of the input width
being image, the landmark signal was swamped and the result (0.8739) fell
*below* landmarks alone. Summarising each branch to comparable width first
(image 128, landmarks 64) fixed it — 0.9455. **Fusion needs balanced
branches; naive concatenation is not fusion.**

### Resolution

The 128px dataset changed nothing:

```
                                        64px      128px     delta
  Landmarks only (control)            0.9354     0.9417     +0.63
  CNN from scratch                    0.8722     0.8741     +0.19
  MobileNetV2 frozen + landmarks      0.9455     0.9493     +0.38
```

The landmarks-only control takes **identical input** in both runs, so its
+0.63 drift is pure noise — and every other delta is smaller than that. The
ceiling is in the data, not the model.

### But the regeneration did find a real bug

`prepare_segmented_dataset.py` sorted filenames **lexicographically**
(`L1, L10, L100, L1000`), which scrambles temporal order. Since
`HandSegmenter` runs in `RunningMode.VIDEO`, it was carrying tracking state
between frames that are not adjacent in time. Sorting by frame index:

```
71,930 images (85.6% success) -> 78,911 images (93.9%)   +9.7%
```

The script is also parallel now — **by class, not by image**, so each worker
walks one class in frame order and the video tracking continuity is
preserved. Splitting by image would have produced a *worse* dataset.

---

## 8. One hand only — and the fix

The dataset is filmed almost entirely with one hand:

```
MediaPipe handedness across raw training images:  Right 61, Left 3
```

The demo mirrors the camera (`cv2.flip`), so a user's **left** hand appears
right-handed to the model — which is why only that hand worked.

Measured on validation, with mirrored images and mirrored landmarks:

| | as trained | mirrored (other hand) |
|---|---|---|
| before | 0.9712 | **0.0872** (chance is 0.036) |
| after mirror augmentation | 0.9736 | **0.9544** |

In ASL, a left-handed signer's letter *is* the mirror of a right-handed
signer's, so a flipped image is still a valid image of the same letter. The
augmentation flips **image and landmarks on a single shared random draw** —
independent draws would show opposite hands to the two branches and quietly
poison the fusion. Verified before training: 400/400 agreement.

Cost: the headline val_accuracy went 0.9373 -> **0.9277**, about a point,
since the model now covers two orientations with the same capacity. Trading
one point for 87 is worth it, but it is a trade.

> **Caveat:** 0.9544 comes from mirrored *validation images* — a synthetic
> left hand. A real left hand differs in lighting, thumb angle and
> proportion, so treat it as an upper bound.

---

## 9. The live demo

`src/webcam_demo.py` runs the fusion model at ~34 fps (segmentation 26.5 ms,
inference 2.8 ms, drawing 0.2 ms).

### Distance estimation

`solvePnP` over all 21 landmarks, using MediaPipe's metric
`hand_world_landmarks` for shape but **imposing a fixed hand size**.

Two failed approaches on the way, both informative:

- **Palm width alone** foreshortens when the hand tilts: spread 335% across
  poses at constant distance (read a "C" as 4.45 m while others read 0.6 m).
  solvePnP, which solves rotation and distance together, gives 58%.
- **Trusting MediaPipe's metric scale** — its wrist-to-knuckle estimate
  varies **7.01 to 10.68 cm** on the same hand (39%). Since
  `distance = focal x real_size / pixel_size`, an underestimated hand reads
  as closer. This was the cause of "shows 12 cm when I turn my hand over".
  Keeping only the *shape* and imposing `HAND_LENGTH_M` removes it.

Accuracy remains ~±25% — an indicator, not a measurement. `CAMERA_FOV_DEG`
and `HAND_LENGTH_M` set the absolute scale and are uncalibrated by default.

### Placement warnings

- **Too close / too far** (<0.30 m, >1.00 m): median over 9 frames, plus 10%
  hysteresis. Tested — a signal wobbling across the threshold twelve times
  produces **one** state change, and a single bad frame is absorbed entirely.
- **Hand leaving frame**: MediaPipe extrapolates landmarks past the frame
  edge (x=663 on a 640-wide frame), so out-of-bounds points are a real
  signal. Gives a direction ("deplace vers la gauche"), needs 3 consecutive
  frames, and overrides the distance warning since a cut-off hand makes
  distance meaningless. Past ~100px visible MediaPipe stops detecting
  entirely, so this covers the *approach* to the edge.

### Text input

Dwell-and-release state machine: a letter must be stable for **0.7 s**, then
cannot repeat until a **0.3 s** release. Thresholds are in seconds, not
frames, so typing feel does not change with CPU load.

```
held A for 3s             -> 'A'      (not 'AAAAAAA')
A/B alternating for 4s    -> ''       (flicker never commits)
hold L, release, hold L   -> 'LL'     (deliberate doubles work)
confident but mis-framed  -> ''       (warnings gate writing)
H, I, space, A, del       -> 'HI '
```

Writing requires confidence >= 80% **and** good framing **and** distance in
range — a badly placed hand writes nothing rather than writing wrong.

---

## 10. Model provenance

Two `.keras` files are indistinguishable on disk. `train.py` and
`train_fusion.py` write a `.meta.json` sidecar recording split, score, best
epoch, sample counts and git commit; `webcam_demo.py` prints it at startup:

```
Modele : FUSION image+landmarks | val_accuracy=0.9277 | 2026-10-01T20:06:48-04:00
```

See `models/README.md` for the full map. Superseded models live in
`models/archive/` with sidecars explaining what each one got wrong.

> **Note on `models/comparison.json`:** the `fusion` entry (0.9277) was
> overwritten by the mirror-augmented rerun, while `cnn` and `landmark` are
> from the original non-augmented comparison. The clean three-way comparison
> is the table in section 6 (fusion 0.9373). Do not read those three JSON
> values as a like-for-like comparison.

---

## 11. Housekeeping

- Three byte-identical copies of `hand_landmarker.task` (7.5 MB each); only
  `models/` is referenced. Removed two.
- Added `.gitignore`. Patterns are unanchored (`*.keras`, not
  `models/*.keras`) — the anchored form does not match subdirectories and
  would have committed 18 MB from `models/archive/`.
- `data/` is 1.5 GB (plus 490 MB for the 128px set) and is excluded.

---

## Closing state

**Works:** fusion model at 0.9277, either hand, ~34 fps, with framing and
distance guidance and dwell-based text entry.

**Known limits, not addressed:**

- **Single subject, single camera, single session.** The temporal split
  removes frame-level leakage but not signer or setup leakage. Expect all
  numbers to drop on a new person; expect the landmark model to drop least.
- **No `nothing` class.** With a hand detected but not forming a letter, the
  model must still pick one of 28. The 80% threshold is a mitigation, not a
  fix. Retraining with a `nothing` class is the real answer.
- **J and Z are motion letters.** The model only gets them because the
  dataset's frames share a pose — it is not recognising the movement.
- **No confusion matrix.** `evaluate.py` is still a stub. Knowing *which*
  letters fail (M/N/S/T and A/E/S are the classic look-alikes) would be more
  actionable than a single accuracy figure.

**The lesson worth carrying forward:** the single most valuable thing in this
phase was checking whether the evaluation was honest. The 0.9938 was not a
bug in the model — it was the model answering an easier question than the one
that mattered. Every architecture decision made before that check was made on
bad information.

Word-level recognition (WLASL) starts a separate document.
