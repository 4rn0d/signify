# Signify — word-level phase (WLASL)

Recognising whole signed words from motion, rather than static letters. A
separate problem from the alphabet phase (`FINDINGS.md`) and a separate
pipeline: a sign is defined by movement, often uses two hands, and carries
meaning in *where* it happens relative to the body — none of which a
single-frame image classifier represents.

**Current state:** 0.6254 ± 0.0450 on 50 signs, chance 0.0200 — **31× chance**,
on a signer-disjoint split.

---

## 1. Getting the data was most of the work

WLASL ships URLs, not videos, and the links have rotted.

### 1.1 Five dead hosts

```
signingsavvy   225 clips   403, blocks automated access
handspeak      174         404
aslpro         162         404  (Flash .swf regardless)
aslsignbank    108         DNS failure
aslsearch      114         parked domain: 200 + HTML lander
--------------------------------------------------------
               783 clips, 38% of WLASL100
```

`aslsearch` is the instructive one. It answers **HTTP 200** with a redirect
page, so a reachability probe scored it healthy. Only a minimum-size check
caught it:

```
b'<!DOCTYPE html><html><head><script>window.onload=function(){...
```

**A status code describes the HTTP transaction, not whether you got what you
asked for.**

### 1.2 YouTube blocks the default client

A third of WLASL100 is YouTube, and the default yt-dlp client gets
`Sign in to confirm you're not a bot` on roughly half of requests. The
`android` player client passes without authentication — tested 6/6 against
0/6 for web, ios and tv.

### 1.3 The mirror doubles the usable set

Direct download tops out at **880 clips (43%)**. The Hugging Face mirror
(`Voxel51/WLASL`, public and ungated) holds video files archived when more
links were alive. It is itself incomplete (11,880 of 21,083), but its
survivors differ from ours:

```
direct download only   880 clips   50 signers
+ mirror              1321 clips   73 signers
```

The signer count matters more than the clip count — see below. The mirror
fetch succeeded 433/433.

---

## 2. The official splits are not signer-independent

```
signers in train AND val  : 64  (93% of val signers)
signers in train AND test : 52  (93% of test signers)
```

A score on those splits partly measures recognising *people*, not signs —
the same failure as the alphabet phase's 0.9938, and worse here, because a
signer is easier to memorise than a letter.

Every number in this document uses a **signer-disjoint** split built by
holding out whole signers.

That is what made the mirror essential: with 50 signers, no vocabulary size
supported a clean split covering every gloss. With 73, WLASL50 does.

```
 vocab  clips  min/gloss  signers  glosses with no test clip
    30    516         15       57        0
    50    812         14       64        0    <- chosen
    75   1114         10       70        1    blocked
   100   1321          6       73        8    blocked
```

---

## 3. Three data bugs

### 3.1 Frame ranges are indexed against the original video

88 clips are long (up to 460 s) and contain several signs; all carry an
explicit frame range. But **39% of them were re-encoded to 30 fps while the
metadata says 25**, so using the raw indices lands elsewhere entirely:

```
gloss      meta fps  real fps   correct t   naive t    error
book             25      30.0       86.0s     71.7s     14.3s
drink            25      30.0      228.4s    190.5s     37.9s
offset error: median 28.9s, max 52.4s   (a sign lasts ~2.5s)
```

Converting through **seconds** fixes it, verified to within 0.1 s on all 34
affected clips. Without it, 34 clips would carry a label from a sign a
minute away.

### 3.2 Landmarker instances cannot be reused across clips

VIDEO mode requires rising timestamps, so a second clip starting at zero
raises `Input timestamp must be monotonically increasing`. Worse, tracking
state would bleed between unrelated signers. Each clip gets a fresh
landmarker — the same trap as the alphabet phase's frame-ordering bug.

### 3.3 Normalisation is anchored to the body, not the wrist

For letters, hand position in frame was noise. Here it is meaning: the same
handshape at the forehead or the chin are different signs. Origin is the
shoulder midpoint, scale is shoulder width — verified exact (shoulder
separation 1.000, std 0.0000).

Missing hands get zeros **plus a presence flag**, so the model can tell
"no hand" from "hand at origin". That flag carries real signal: 11 of 25
sampled clips never show a left hand, because they are one-handed signs.

---

## 4. Results

155 features per frame (2 hands × 21 × xyz, 2 presence flags, 9 upper-body
pose points), 80-frame windows, 2-layer bidirectional GRU, 5-fold
signer-disjoint cross-validation, fixed 150-epoch budget.

### What worked

```
configuration                        plateau      std     step
all clips, old vocabulary             0.3702   0.0435
dominant variant only                 0.4258   0.0598  +0.0557
  (vocabulary selection changed)      0.4564   0.0371  +0.0305
+ drop-z, label smoothing, TTA        0.5720   0.0586  +0.1157
+ ensemble x3, pretrain on 100        0.6254   0.0450  +0.0534
```

Every step won **5/5 folds**.

### What did not

```
LSTM instead of GRU        0.2776   -0.093   0/5 folds
GRU -> LSTM stack          0.2768   -0.093   0/5
velocity features (deltas) 0.3563   -0.014   2/5
capacity 96 -> 160 units   0.3579   -0.012   2/5
variants as separate cls   0.3636   (71 classes, not comparable)
more epochs (80 -> 250)    +0.003   plateau reached at ~150
```

**Nine experiments. Every model-side change was neutral or harmful; every
data-side change helped.** LSTM's extra gate costs 30% more parameters, and
with ~11 clips per class the smallest adequate model wins.

### The two findings worth keeping

**Sign variants.** WLASL labels several distinct gestures under one gloss —
19 of 50 have multiple `variation_id`, and 15 have no variant holding 75% of
their clips (`computer` splits 7/9/5). Keeping only each gloss's dominant
variant **discards 14% of the clips and gains 5.6 points**: the contradictory
labels cost more than the data was worth.

**Dropping the z coordinate.** MediaPipe estimates depth from a single
camera, and we had already measured its unreliability during the alphabet
phase (39% scale variation on one hand). Removing it cuts the input from 155
to 104 dimensions and is the largest single contributor to the +11.6 step.
**Accuracy improved by deleting a third of the input.**

---

## 5. Methodology notes

**k-fold with a fixed epoch budget, not early stopping.** With 9–16 clips
per class, a validation split would be ~3 clips per gloss — too noisy to
choose a stopping epoch — and stopping on the test fold is leakage.

**Reported metric is the mean over the last 30 epochs**, not the final
epoch. Single-epoch readings swing ±2 points purely on where training stops;
the tail mean removes that without touching the fixed budget.

**Pretraining excludes the fold's held-out signers**, or they would be seen
before fine-tuning and the split would be fiction.

### A confound that nearly became a result

The `cheap3` run initially appeared to gain **+14.6 points** over `dominant`.
It did not. The two runs selected their 50 glosses differently:

```
dominant : top 50 by total clip count, THEN variant filtering
cheap3   : variant filtering, THEN top 50 by what remains
```

Only 37 of 50 glosses overlapped. The second ordering drops exactly the
glosses that lost most to filtering (`computer` 21→9, `cool` 18→9) and
replaces them with cleaner-labelled ones, making the vocabulary
systematically easier.

A control run (same code path, same vocabulary, improvements off) decomposed
it: **+11.6 from the improvements, +3.0 from the easier vocabulary.** The
improvements were real, but one third of the apparent gain was not.

---

## 6. Pipeline

```
audit_metadata.py    splits, signer overlap, host distribution  (no download)
download_videos.py   direct + yt-dlp, skips dead hosts, resumable
fetch_mirror.py      fills gaps from the Hugging Face mirror
organise_videos.py   videos/<gloss>/<gloss>_s<signer>_<id>.mp4
audit_downloaded.py  what actually survived, per gloss and signer
extract_landmarks.py MediaPipe Holistic -> one .npz per clip
train_kfold.py       cross-validation, all experiment flags
train_final.py       trains the shipped ensemble on all data
webcam_demo_words.py live demo, SPACE records 2.8 s, shows top-5
```

Videos (1.0 GB), features (17 MB) and model weights are gitignored; all are
regenerable from these scripts. `download_manifest.json` is the source of
truth, carrying gloss, signer, split, fps, frame range and variation per clip.

---

## 7. Open items

**Not tested on a real signer yet.** 0.6254 comes from held-out *dataset*
signers. A new person, camera and lighting is harder; expect lower.

**Top-5 matters at this accuracy.** The demo shows five candidates; a correct
answer anywhere in them means the model found real signal.

**~11 training clips per class** remains the binding constraint. Nine
experiments confirm the model is not the limit. The ways forward are all
data-side: a third mirror, recording your own clips, or a smaller vocabulary.

**Recording your own clips** is the highest-value next step for a demo that
works on *you* — the capture window and feature extraction already exist, so
10 takes each of 10 signs would give more examples of you than the dataset
has of anyone.

**No confusion matrix.** Knowing which signs collide would guide vocabulary
choices; WLASL has regional variants that may genuinely overlap.
