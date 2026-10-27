# Signify — word-level phase (WLASL)

Recognising whole signed words from motion, rather than static letters. A
separate problem from the alphabet phase (`FINDINGS.md`) and a separate
pipeline: a sign is defined by movement, often uses two hands, and carries
meaning in *where* it happens relative to the body — none of which a
single-frame image classifier represents.

**Current state:** 0.7413 ± 0.0380 on 50 signs, chance 0.0200 — **37× chance**,
on a signer-disjoint split: RTMPose features at hand threshold 0.7, derived
scalars, per-hand confidence, the full handshape block, a linear input
projection, and the full training stack.

The landmark extractor was then changed to RTMPose (§7), which finds hands in
84% of frames against MediaPipe's 47%. Compared like for like, without the
training stack:

```
MediaPipe + derived features          0.5867
RTMPose @0.7 + derived + confidence   0.6005
```

With the full training stack on both sides:

```
MediaPipe + stack                     0.6762
RTMPose @0.7 + confidence + stack     0.6675
```

**A tie.** The gap of 0.9 points is smaller than the disagreement between two
reasonable ways of aggregating the same runs (see §7.6), and is not
significant. RTMPose is kept for its hand coverage, not for accuracy.

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
+ drop z                              0.5347   0.0415  +0.0783
+ 23 derived features                 0.5937   0.0591  +0.0590
+ ensemble x3, pretrain, smoothing,
  TTA                                 0.6762   0.0619  +0.0824
```

Every step won **5/5 folds**. Per fold at the end:
0.731, 0.575, 0.656, 0.669, 0.750 — fold 2, consistently the hardest draw
of held-out signers, went from 0.310 to 0.575 over the project.

### What did not

```
LSTM instead of GRU        0.2776   -0.093   0/5 folds
GRU -> LSTM stack          0.2768   -0.093   0/5
velocity features (deltas) 0.3563   -0.014   2/5
capacity 96 -> 160 units   0.3579   -0.012   2/5
depth proxy alone          0.5300   -0.005   1/5
variants as separate cls   0.3636   (71 classes, not comparable)
more epochs (80 -> 250)    +0.003   plateau reached at ~150
```

**Twelve experiments. Every model-side change was neutral or harmful; the
gains came entirely from what the model is shown, never from the model.** LSTM's extra gate costs 30% more parameters, and
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
to 104 dimensions and is worth **+7.8 points on its own** (measured by
ablation). **Accuracy improved by deleting a third of the input.**

**23 features derived from the x,y that MediaPipe gets right.** ASL is
described by four parameters — handshape, location, orientation, movement.
Raw landmark sequences carry movement well but leave the other three
implicit, and with ~11 clips per class the model cannot rediscover them.

```
 4  depth        palm width, forearm length (rigid segments foreshorten)
10  handshape    per-finger curl: tip-to-base over summed bone length
 4  location     hand-to-nose, hand-to-chest
 1  structure    distance between the two hands
 4  orientation  wrist->knuckle direction, as sin/cos
```

Worth **+5.9 points**, 5/5 folds, at 127 dimensions — still fewer than the
155 we started with.

The distinction that matters: **velocity features and the depth proxy alone
both returned nothing, because the GRU could already derive them from the
coordinates.** Finger curl is different — it is *rotation-invariant*, which
raw coordinates cannot express, so it is genuinely new information rather
than a shortcut. Tested alone the depth proxy was -0.5; inside the group it
contributes.

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
gpu_setup.py         puts the CUDA DLLs on PATH; import BEFORE onnxruntime
extract_rtmpose.py   RTMPose whole-body on GPU -> one .npz per clip
hand_identity.py     which hand is which, and which one does not exist
dedup_hands.py       applies that rule to already-extracted features
train_kfold.py       cross-validation, all experiment flags
train_final.py       trains the shipped ensemble on all data
webcam_demo_words.py live demo, SPACE records 2.8 s, shows top-5
webcam_rtm_viewer.py diagnostic viewer: 133 points, per-region confidence,
                     live handshape jitter, presence threshold on `t`
```

`gpu_setup.py` is not optional. onnxruntime announces CUDA as available and
then falls back to CPU silently if the `nvidia-*-cu12` DLL directories are not
on `PATH` — 780 ms per frame instead of 34, with no error. `os.add_dll_directory`
is not enough, because the OS loader resolves the transitive dependencies.

Videos (1.0 GB), features (17 MB) and model weights are gitignored; all are
regenerable from these scripts. `download_manifest.json` is the source of
truth, carrying gloss, signer, split, fps, frame range and variation per clip.

---

## 7. Changing detector: RTMPose

MediaPipe loses hands whenever they rotate away from the camera, which in ASL
is constantly. RTMPose (`rtmlib`, RTMW whole-body, 133 keypoints, ONNX on GPU)
finds them far more often. Migrating to it **cost 14.9 points**, and the reason
turned out to be the opposite of what it looked like.

### 7.1 The controlled comparison

```
RTMPose + rich    plateau 0.4443  std 0.0581
MediaPipe + rich  plateau 0.5937  std 0.0591
                          -14.9, losing on 5/5 folds
```

Identical clips (1321 both sides, 726 after vocabulary and variant filtering),
vocabulary pinned with `--vocab-from`, same 127 features. Only the detector
differs. Pinning matters: `--vocab` picks the top 50 glosses from whatever is
in the features directory, so without it each detector selects its own
vocabulary and part of any gap is an easier word list — the §5 confound again.

### 7.2 The cause was the presence threshold, not the detector (confirmed)

The first explanation was that RTMPose's hands are noisier, since it regresses
them from a 34x34 pixel patch while MediaPipe re-crops the hand for a dedicated
model. Finger-curl jitter appeared to confirm it: 0.0857 against 0.0443.

**That measurement was wrong.** It scored each detector on *its own* detected
frames, and MediaPipe only reports a hand in 47% of them against RTMPose's 84%,
so MediaPipe was being judged on its easiest half. On identical frames:

```
                        jitter    S/N
MediaPipe               0.0444   2.59
RTMPose                 0.0359   2.60     equal, and lower jitter
```

The hands are equally good. The damage is in the 37 points of extra coverage:

```
frames MediaPipe also sees   jitter 0.0359   S/N 2.60
frames MediaPipe REJECTS     jitter 0.1408   S/N 1.42     3.9x noisier
```

**MediaPipe's low coverage was not a weakness being fixed — it was an implicit
quality filter, and the migration discarded it.** Extraction ran at a hand
confidence threshold of 0.3, keeping 91% of hands including unusable ones.
Confidence separates them cleanly and monotonically:

```
threshold   hands kept   curl jitter
0.3              91.0%      0.0688
0.5              75.7%      0.0565
0.7              60.2%      0.0395   <- chosen, beats MediaPipe on both axes
0.8              48.6%      0.0273
MediaPipe        47.0%      0.0444
```

At 0.7 we keep more hands than MediaPipe *and* cleaner ones. Confirmed
independently on a live webcam before adopting. Per-hand confidences are now
stored in each `.npz`, so future thresholds can be swept without re-extracting.

Re-extracting at 0.7 and re-running the same five folds:

```
MediaPipe        0.5937  std 0.0591
RTMPose @ 0.3    0.4443  std 0.0581
RTMPose @ 0.7    0.5736  std 0.0542
```

Paired per fold, which is the right test since a fold is a fixed set of
held-out signers:

```
threshold 0.7 vs 0.3     +0.1185   t=+5.29   5/5 folds   significant
RTMPose@0.7 vs MediaPipe -0.0151   t=-1.05   2/5 folds   not significant
                                   95% interval -0.055 to +0.025
```

Adding confidence as a feature (§7.3b) takes it to **0.5892**, against
MediaPipe's 0.5937 — still a tie, with MediaPipe marginally ahead on the
metric of record. The honest claim is parity on accuracy with much better hand
coverage, never a win.

**The threshold recovered 11.9 of the 14.9 points, and the detector choice is a
wash.** The two are statistically indistinguishable on accuracy. The case for
RTMPose is therefore not accuracy but coverage: it matches MediaPipe on the
benchmark while finding hands 84% of the time against 47%, which was the
original complaint. The robustness comes for free rather than as a trade.

A side effect worth noting: at 0.7 the phantom-hand arbitration finds **zero**
conflicts in 1321 sequences, against 5.48% of frames at 0.3. The duplicated
hands were the low-confidence hands — one root cause behind two symptoms chased
separately. `hand_identity.py` is now a no-op on the training features; it is
retained for the live demo, where confidence varies more and where it fixed
observable misbehaviour.

### 7.3 Phantom hands

The whole-body model always emits 21 points per hand, even for a hand that is
absent, and often stacks them on the visible one — **5.48% of frames**, 310 of
1321 sequences, up to 93% of frames in one-handed signs like `full`, `fine`,
`like`.

The pose skeleton arbitrates, because it estimates both wrists separately. Each
hand is asked which wrist it *claims* (the nearer of the two); if both claim the
same one, the hand whose name does not match is the invention.

Formulating it as a claim rather than an overlap matters twice. An overlap
threshold only catches exact stacking — in the affected frames the two hand sets
sit 0.10 to 0.12 shoulder-widths apart, and a threshold of 0.08 caught almost
none of them. And genuinely crossed hands claim *different* wrists, so nothing
fires: 8.6% of hands sit within 0.25 of each other, almost all legitimately, and
a threshold loose enough to catch the copies would delete real hands. A unit test
covers that case.

The arbitration is deterministic, which also fixed a second symptom: comparing
"left hand to left wrist" against "right hand to right wrist" compares distances
to *different* wrists, is near-tied when both sets sit on one hand, and made the
label alternate every frame. Shared by extraction, correction and demo via
`hand_identity.py` — three copies of this rule is how the alphabet phase got
0.985 in validation and 0.384 on camera.

RTMPose has no absence flag at all, unlike MediaPipe which returns `None`. The
only directly observable absence signal is that it places keypoints **outside
the image**; a hand with over half its points out of frame is treated as absent.
That changes the decision on ~3.2% of frames.

### 7.3b Confidence as a feature, worth +1.6

With the threshold at 0.7 the model is told *whether* a hand is present but
not how much to trust it — a hand at 0.71 and one at 0.99 look identical. The
per-hand confidence is stored in each `.npz`, so feeding it as two continuous
columns costs nothing but a flag. Tested against the same five folds:

```
                              plateau    last epoch, paired
confidence alone              +0.0156    +0.0289  t +3.01  5/5 folds
symmetry, added to confidence -0.0155    -0.0139  t -0.75  1/5 folds
```

**Confidence is worth +1.6 points on the metric of record** (it was first
reported as +2.9, which came from the last-epoch figures — see §7.6). It won
every fold on the last-epoch measure, with the largest gain
on fold 2 — the hardest draw of held-out signers — which is what one expects if
the benefit is the model learning to discount unreliable hands. It must swap
with its hand under mirroring, like the presence flags.

Two hand-symmetry scalars (positional mirror symmetry, and handshape
difference between the hands) returned **nothing**, and being slightly negative
they cancel confidence's gain when both are enabled (0.6005 -> 0.5866). Not
adopted.

That is the re-derivation rule from §4 holding for the third time:

```
re-derivable from coordinates    velocity deltas, depth proxy, symmetry   0
not expressible in coordinates   finger curl (+5.9), confidence (+2.9)    win
```

**The test that predicts a feature's value is whether it expresses something
the raw input cannot** — not whether it sounds informative. Symmetry sounds
highly relevant to a bimanual language, and a GRU reading both hands'
coordinates can already compute it.

### 7.4 What did not work

```
dedicated hand model (RTMPose-m hand5, 256x256)   -28%   rejected
  also rejected on a live webcam
bbox padding toward full-body proportions          coverage +15, precision -10..17%
RTMW3D depth (Wholebody3d)                         unusable at hand scale
performance mode (288x384 vs 192x256)              +43% S/N on clips, rejected live
```

**`performance` mode** is the sharpest clip-versus-camera disagreement of the
phase. On clips it is unambiguously better: same `rtmw-dw-x-l` family, 1.5x the
input, jitter 0.0617 -> 0.0564 and S/N 2.80 -> 4.01, with spread rising while
jitter fell, which is how genuine discrimination is distinguished from noise.
On a live webcam it was rejected immediately: keypoints visibly unstable,
"lines flying around". Not a frame-rate artefact — it stayed at 27 fps.

The likely cause is that the two differ by more than input size (they are
separate checkpoints, trained at 256x192 and 384x288), and that a webcam feed
carries sensor noise, compression artefacts and motion blur that a 192x256
downscale smooths away and a 288x384 input preserves. Higher resolution
amplifies high-frequency noise when the source is noisy; WLASL clips are
cleaner than a close-range webcam.

`balanced` is used everywhere, including extraction.

**The dedicated hand model** should have worked: the whole-body model gives each
hand 34x34 pixels of its input, a dedicated model gives it 256x256. But the
binding constraint is the *source*: the hand is only ~40 px in the video, so
256x256 is 6.4x empty magnification, and the model expects real detail at that
size. Its own hand detector does no better (confidence 0.303 vs 0.236), so this
is not a bounding-box problem. **Input resolution cannot exceed what the camera
captured.**

**Padding the person box** confirmed a real framing effect — RTMPose is top-down
and trained on whole bodies filling the box, so a webcam torso violates that.
Stretching the box raises hand coverage from 76.8% to 91.7% on identical images,
with the padding filled by black pixels, so the model is responding purely to
the scale relationship. But precision falls as the hand shrinks in the input.
Not adopted; the threshold addresses the same problem without the trade.

**RTMW3D** does produce depth, so "RTMPose is 2D" is false as a general claim —
it is a property of the model chosen. Its body depth is sound (shoulder-to-
shoulder within 0.035 m facing the camera) but hand depth is not: depth spread
*inside* a hand reaches 1.74x the hand's own size, which is impossible, with up
to 0.139 m of frame-to-frame jitter. Same defect that made dropping MediaPipe's
z worth +7.8 points. Pose-only z remains untested.

### 7.5 Survivorship bias, three times

Every wrong conclusion in this phase came from the same error: comparing two
methods on the frames *each one chose for itself*.

```
dedicated hand model, own frames        +74%      (kept the easiest 37%)
dedicated hand model, same frames       -28%
RTMPose vs MediaPipe, own frames        -22% S/N
RTMPose vs MediaPipe, same frames         +0%
```

A selective method looks good because it discards its hard cases. **Any
comparison of two detectors, thresholds or models must hold the evaluated set
fixed** — the same discipline that decomposed the §5 vocabulary confound, applied
to frames instead of glosses.

A smaller lesson: `ast.parse` validates grammar but skips the symbol-table pass,
so it accepts a `global` declaration placed after a read of the same name.
`py_compile.compile(..., doraise=True)` catches it. Several files "verified" with
`ast.parse` during this phase had only their grammar checked.

### 7.5b Four more ideas, all rejected

Measured against rich + confidence (0.5892), paired on the plateau metric:

```
interpolating hand gaps <=3 frames   0.5756   -0.0136  t -2.54  0/5  noise
confidence-weighted smoothing        0.5672   -0.0220  t -3.07  1/5  HARMFUL
random hand erasure, p=0.3           0.5843   -0.0049  t -0.47  1/5  noise
```

**Interpolation** was proposed on the premise that hands flicker mid-sign.
They do not: of 13265 absent hand-slots only 489 (3.7%) sit in a gap bounded
on both sides by a present frame, because the absences are long — those are
one-handed signs. Where it does fire it appears to hurt, which is coherent:
a zero is honestly flagged absent, whereas a filled frame asserts a position
the model then trusts.

**Confidence-weighted smoothing is the instructive failure.** The argument for
it was that it escapes the re-derivation rule, since the weighting comes from
confidence, which is information outside the coordinates. It does not escape.
A bidirectional GRU reading 80 frames can already average over time, and
handing it pre-averaged coordinates *removes* detail it was using. Jitter was
measured and assumed to be noise; some of it is signal.

That sharpens why confidence itself works at +1.6: it helps as an **input the
model conditions on**, not as a correction applied on the model's behalf.
Telling the model what to distrust beats distrusting for it. The rule, restated:

```
give the model information it cannot derive          -> can win
do the deriving for it, however well motivated       -> loses
```

**Random hand erasure** is the only one of the four that leaves the features
alone and changes the training distribution instead, so the rule does not
apply to it. It costs nothing measurable (-0.5, t=-0.47) and aims at
robustness to dropout, which a held-out-dataset benchmark barely exercises —
its value, if any, would show on a live camera. Unproven, so not adopted.

A fifth change was infrastructural rather than an improvement: `--raw`
extraction keeps every coordinate and stores `hand_scores`,
`hand_point_scores` (42 per frame), `hand_inside` and `hand_conflict`, so the
confidence threshold, hard versus soft gating, per-finger weighting, the
off-frame cutoff and whether to apply the phantom-hand arbitration at all all
become training-time flags. Each of those previously cost an 80-minute
re-extraction. Note that `--hand-score 0.0` alone is not sufficient: the
off-frame filter and the arbitration delete hands irrespective of threshold.

### Hard versus soft gating: the rule has a boundary

The raw extraction made one question askable for the first time. Hard gating
*deletes* a rejected hand's coordinates; soft gating keeps them and drops only
the presence flag, leaving the model to judge. Section 7.5b's lesson predicted
soft should win.

```
filtering at extraction time        0.5892
hard filtering at training time     0.5858   -0.0034   equivalent
soft filtering                      0.4915   -0.0976   t -10.48  0/5
```

**Soft gating is catastrophic** — a larger effect than the threshold fix that
started this section, in the opposite direction to the prediction.

It leaves 42 unreliable coordinate columns for the model to learn to ignore,
with ~11 clips per class to learn it from. Hard gating zeroes them, so the
flag, the zeros and the `Masking` layer agree that nothing is there. The
derived features are zeroed by the presence flag either way, so soft gating's
only effect is to inject noise.

So the rule needs a boundary. "Give the model information rather than deciding
for it" holds when the information is **reliable**. The discriminator is signal
per added dimension, not who does the deciding:

```
+2 reliable columns (confidence)          +1.6
+42 doubtful columns (kept coordinates)   -9.4
```

This also settles what the +12.9 threshold gain was: removing bad data,
entirely. No good data was being discarded alongside it.

Hard gating at training time reproducing extraction-time gating to within 0.34
points validates the raw pipeline. The residual is a known difference in the
conflict rule — at training time the lower-confidence hand is dropped, while
`resolve` drops the one whose name does not match the claimed wrist.

Running total: 24 experiments, 5 of which gained anything.

### 7.5c Frame rate: the demo needs a fixed time base

The dataset's videos run at a median of **30.0 fps** (58% at 29.97, 11% at
23.98, min 20, max 30.4) and extraction does not resample — it takes
consecutive frames at the file's own rate. So the model learned signs seen at
roughly 30 images per second.

A webcam advances at whatever the pipeline achieves: ~29 fps with a free GPU,
far less with a busy one. The model has no notion of time, only of frame
count, so at 15 fps a 2.8 s sign arrives as 42 frames and reads as a sign
performed twice as fast. `time_warp` augmentation covers only 0.8-1.25x.

Measured by subsampling stored sequences to simulate a slower camera, then
classifying (training clips, so the 92% ceiling is inflated — the pattern is
what transfers):

```
camera rate   uncorrected   resampled to 30 fps
   30.0 fps       92.2%           91.9%
   24.0 fps       91.9%           91.9%
   20.0 fps       91.7%           91.7%
   15.0 fps       91.2%           91.9%
   10.0 fps       84.9%           91.4%
```

**The effect only bites below ~15 fps**, where the uncorrected path loses 7.3
points and resampling recovers 6.5. At the ~29 fps the demo achieves it was
never costing anything, and resampling costs 0.3 points there — noise. Cheap
insurance against a failure we have already seen: sharing the GPU with an
extraction made the demo crawl.

The demo records each frame's timestamp and resamples onto a 30 fps grid by
**nearest neighbour**, never interpolation — extraction subsamples long clips
by picking existing frames (`wanted[int(i * step)]`) and invents no
intermediate positions, and 7.5b measured that fabricating coordinates costs
1.4 points. Below the target rate frames are repeated: the trajectory is jerky
but its duration is right, which is the lesser fault.

### 7.5d Confusion matrix, and thumb opposition (+1.1)

Out-of-fold predictions, averaged over the same last-30-epoch window as the
reported metric, make a confusion matrix possible — the §8 open item. The
shipped model has seen every clip, so evaluating it on them would hide exactly
the confusions being looked for.

```
726 clips out of fold    top-1 61.6%   top-5 91.2%

lowest recall
  thursday  17%   -> shirt, bowling
  candy     18%   -> apple x6   (apple -> candy x1 only)
  eat       18%   -> wrong, color
  hearing   40%   -> apple x5
  deaf      35%   -> apple x3
```

**Every confused pair differs mainly in handshape at the same body location**
with similar gross motion: `dance`/`study` (hand over palm, different finger
action), `bird`/`orange` (pinch or squeeze at the mouth), `apple`/`candy`
(twist at the cheek). `apple` is an attractor, predicted 27 times for 15 real
clips, absorbing `candy`, `hearing` and `deaf`.

Three findings that narrow the cause:

**No class collapses.** The least-predicted class still gets 7 predictions
across 726 clips, so nothing is entirely absent.

**Class imbalance is not the cause.** Correlation between clip count and
recall is +0.116; recall is 57.8% for classes with <=12 clips against 62.5%
for the rest. Class weighting would not address this.

**Errors are confident, not narrow.** When top-1 is wrong the probability gap
to the true label is a median 0.433, and only 6% of errors fall under 0.05. So
the discriminating information is not reaching the model, rather than being
present and mis-ranked.

Curl measures finger FLEXION only — nothing about thumb position relative to
the other fingers, which is a primary ASL handshape parameter. Screening
candidate features by pairwise AUC before training:

```
                   best curl   best candidate
apple / candy          0.84         0.91   thumb-index
bird / orange          0.89         0.98   palm-ring
bed / mother           0.58         0.87   palm-little
thursday / shirt       0.86         0.99   thumb-little
```

Adding 8 thumb-opposition scalars (thumb tip to each other fingertip, per
hand, normalised by palm width) is worth **+1.1 points, 5/5 folds, t=+4.06**.
The full 26-scalar version (opposition, openness, spread) returns nothing
(-0.5, 2/5) — dimensions are not free at ~11 clips per class, the same lesson
as soft gating.

**But it does not fix the confusions it was designed for.** Per-class recall
moved only 61.2% -> 61.6%, with 18 classes better and 21 worse, and `candy`
fell from 18% to 9%. `bed`/`mother`, the pair with the largest AUC gain, did
improve as predicted (8 confusions -> 3), but `candy`/`apple` stayed at 7.

**Pairwise AUC over-promises.** It measures two classes in isolation, while
the model works in a 50-way setting where sharpening one boundary displaces
errors elsewhere. The screen predicted one pair out of six. Keep it as a
filter for hopeless ideas, not as a predictor of gains.

What remains for `candy` is 11 clips of a sign differing from `apple` by a
bent versus straight index finger, on ~40 px hands. That is §8's binding
constraint seen up close, and the ways out are dataset-side: merge the pair,
or record more clips.

### 7.5e Parameter economy: the axis nobody had tested

Asked what limits the number of inputs, the answer turned out to be arithmetic
rather than opinion. A dimension entering a bidirectional GRU directly costs
`3 gates x units x 2 directions` parameters — **576 at 96 units** — against
**580 training clips per fold**. So each added feature is roughly one new
parameter per training example, which is why small additions pay and large
ones drown:

```
+2  columns (per-hand confidence)        ~1 200 params   +1.6
+8  columns (thumb opposition)           ~4 600          +1.1
+10 columns (fingertip confidence)       ~5 800          -2.3
+26 columns (full handshape)            ~15 000         -0.5
+42 columns (kept coordinates)          ~24 000         -9.4
```

**It is not insufficient capacity.** 96 -> 160 units scored worse (0.3579),
and 96 -> 48 units also scored worse (-0.0452, t=-4.48, 0/5). Bracketed on
both sides, 96 is near-optimal, and a model that degrades when grown is
data-limited rather than capacity-limited.

But the 576 figure is an artifact of feeding features straight into the GRU,
not a law. A linear projection before it changes the economics:

```
                                     params    per dim
96 units, no projection              333 554       576
96 units + projection to 48          288 914        48    12x cheaper
```

The model gets *smaller* while each feature gets ~12x cheaper. Measured
against the same features and gating:

```
projection 48                        0.6288   +0.0299  t +2.10  4/5
projection 48 + full handshape (26)  0.6462   +0.0472  t +2.42  4/5
```

**0.6462 is the best bare configuration in the project** — no ensemble, no
pretraining, no smoothing, no TTA — and 2.1 points below the full-stack
0.6675 that needs three seeds to reach. Neither clears t=2.78, so both are
suggestive rather than established, and confirmation on fresh fold seeds is
owed before either is treated as real.

Note what this does to a standing conclusion. "Every model-side change was
neutral or harmful" came from sixteen experiments that all made the model
BIGGER or DIFFERENT — never cheaper per input. The conclusion was drawn from a
biased sample of the space.

### 7.5f Re-testing the failures once they became affordable

If features failed because they were unaffordable, making them 12x cheaper
should rescue them. Tested against the 0.6462 base:

```
feature               before     after       t      verdict
soft gating          -0.0942   -0.0769   -2.38   still harmful
fingertip confidence -0.0233   -0.0010   -0.06   neutral
velocity             -0.0139   +0.0031   +0.36   neutral
symmetry (control)   -0.0139     ~-0.02           still negative
```

**The hypothesis was right for one of four.** Only fingertip confidence failed
because of cost: the projection moved it from significantly harmful to exactly
zero, which separates two facts the original -2.3 had conflated — the columns
were expensive AND the information is worthless. Cheapening removed the harm
and revealed no benefit.

**Soft gating is misleading, not expensive.** 12x cheaper recovered 1.7 of the
9.4 points it costs. A rejected hand's coordinates are a confident-looking
guess and the model trusts them; hard zeros are honest.

**Velocity is redundant at any price.** 106 columns, 12x cheaper, still zero.
This is the cleanest confirmation that the re-derivation rule is about
*information* rather than cost: a bidirectional GRU can already difference its
own input, so handing it the differences adds nothing however little they cost.

**Symmetry served as the negative control** and earned its place: had it
improved under projection, the gains would have been an artifact of the setup
rather than of affordability. It went negative in two folds and was stopped
there, which is all a control needs to deliver.

A latent bug surfaced here. `feature_dim` multiplied the *total* by two for
`--velocity`, but `add_velocity` runs before `to_model`, so only the raw
columns double. It announced 310 columns for 261 real ones and died on the
first batch. Wrong since velocity was written, and never hit because
`--velocity` had never been combined with `--rich` — the original velocity
test predates the derived features.

### 7.5g Occlusion is visible in the confidences

Fingers hidden behind the palm cannot be measured in 2D, and RTMPose does not
report that — it emits 21 points regardless, inferring the hidden ones. But it
does know it is guessing:

```
palm width / hand length        mean fingertip confidence
[0.00, 0.35]  edge-on                   0.448
[0.63, 0.81]  face-on                   0.788
```

Confidence collapses by 43% when the hand turns edge-on. Fingertips are also
the least confident points overall (0.655 against 0.687 for the whole hand).

A methodological catch worth keeping: the linear correlation is **-0.016**,
essentially zero, because the relationship is an inverted U. A correlation
coefficient cannot see it; banding revealed what the correlation hid.

Exploiting it as 10 fingertip-confidence columns does not work (above). What
genuinely resolves occlusion is a second camera or a depth sensor. RTMW3D does
not, as §7.4 measures: monocular depth for a hidden part is also inference.
The existing mitigation is temporal — a sign is a motion, the hand rotates
through it, and the bidirectional GRU sees the frames where the fingers were
visible.

Running total: 34 experiments, 7 of which gained anything.

### 7.5h The projection plus the full stack: 0.7413

The confirmation run the projection owed, with the full training stack on top:

```
projection 48 + full handshape + STACK   0.7413  std 0.0380   37x chance
   folds  0.777  0.692  0.719  0.725  0.794

MediaPipe + stack                        0.6762   +0.0651
RTMPose + conf + stack                   0.6675   +0.0737
projection + handshape, bare             0.6462   +0.0951
```

**The stack's contribution on top of the projection is +9.5 points, 5/5 folds,
t=+4.75 — significant**, and the two gains stack cleanly: the stack was worth
~+8 in every other configuration and the projection ~+4.7 bare, so they are
additive rather than overlapping.

Three things make this firmer than the earlier suggestive results:

**Variance collapsed.** Standard deviation 0.0380, against 0.0591-0.0619 for
every previous configuration. The weakest fold is 0.692 where folds previously
ran as low as 0.5639, and fold 2 — which sat at 0.51-0.62 all day — reaches
0.692.

**It retroactively supports the projection**, whose own run was only t=2.42
and was recorded as unproven. The 3-seed ensemble reduces variance, and under
that reduced noise the combination lands 6.5 points clear.

**Snapshot averaging adds only +0.5 here**, against +2.5 on single-seed runs —
the correlation predicted in §7.8, since a 3-seed ensemble already captures
most of that variance reduction. Snapshot ensembling is therefore worth much
less on top of a seed ensemble than it is alone.

Still owed: these are the same five folds that ~35 experiments have run
against. A 6.5-point margin is far harder to attribute to chance than 4.7 was,
but confirming on a fresh signer partition (`--seed`) remains item 5 of §7.8.

Operational note: results are now written to `<tag>.partial.json` after every
fold. A 2 h run that wrote nothing until the end lost everything to an
interruption once, after two folds that held the project's two best scores at
the time.

### 7.5i Pretraining: duration stops mattering early, breadth may not

`--pretrain-epochs 60` was an unexamined default: never varied, not recorded
in the results, unjustified in these docs, and its loop ran blind. All three
are now fixed, and the sweep says the duration barely matters.

```
100 gloses / 1127 clips, 5 folds
   0 epochs   0.6838  std 0.0538
  30 epochs   0.6916  std 0.0504   t +0.65  3/5
  60 epochs   0.7054  std 0.0321   t +1.26  3/5
 120 epochs   0.7009  std 0.0370   t +1.03  4/5
```

Not monotonic, and **none of it significant** — the whole range spans 2.2
points. On fold 1 alone the trend is *negative* (0.7498 at 0 epochs down to
0.7162 at 120), which shows how little signal the mean was carrying.

The observability added here explains why. The pretraining task keeps
improving long after the transfer stops: ~0.50 accuracy at epoch 60, 0.737 at
120, and 0.834 at 360 on the 300-gloss version. With ~10 clips per gloss that
is **memorisation**, so the later epochs learn gloss-specific detail — and
gloss-specific detail is precisely what does not transfer to a 50-class task.

A correction worth recording: on seeing the pretraining curve still climbing at
60, this document briefly claimed the "60 is already past where that is
learned" guess was wrong. It was not. A rising *pretraining* accuracy is not
evidence of better *transfer*, and conflating the two was the error.

**Breadth is a different lever from duration.** Expanding the scaffold from 100
to 300 glosses (1127 -> 3032 clips after variant filtering) on fold 1:

```
100 gloses, best over all durations   0.7498
300 gloses   120e 0.7758   240e 0.7824   360e 0.7753
                                      +0.0326 on this fold
```

All three 300-gloss probes beat every 100-gloss result on that fold, and the
probes are flat among themselves — consistent with more *glosses* pushing
toward generic motion while more *epochs* push toward gloss-specific detail.
One fold only, and fold 1 is the easiest draw, so this is unconfirmed.

Methodology: `--pretrain-probe 120,240,360` pretrains once to the largest
probe, freezes weights at each, and fine-tunes from each. It saves 30% at
three probes and 42% at six since fine-tuning dominates, but the better reason
is that the probes share one pretraining trajectory — separate runs each draw
their own, so a 120-vs-240 comparison would mix "more epochs" with "different
trajectory".

### 7.6 Two metrics, and a reporting error

Every paired t-test in this phase was run on `per_fold`, which holds the
**last epoch** of each fold — while the reported figure is `plateau_mean`, the
mean over the last 30 epochs. §5 adopted the tail mean precisely because
single-epoch readings swing about two points on where training happens to
stop. So the significance tests used the noisy metric and were quoted against
the clean one.

The two disagree, and can invert:

```
                             plateau   last epoch
threshold 0.3 -> 0.7         +0.1293     +0.1185
confidence                   +0.0156     +0.0289
RTMPose+conf vs MediaPipe    -0.0045     +0.0138
full stack vs MediaPipe      -0.0087     +0.0082
```

The threshold result is unaffected — it is far too large to care. But
confidence was first reported as +2.9 when it is +1.6, and RTMPose was
described as having moved *ahead* of MediaPipe when on the metric of record it
stays marginally behind in every comparable configuration.

`per_fold_plateau` is now written to each result file so paired tests can use
the right metric. It cannot be recovered for the runs above, so their paired
statistics remain last-epoch and should be read as directional only.

**A difference smaller than the gap between two defensible aggregations of the
same data is not a result.** Both full-stack configurations sit within one
point of each other; they are tied.

### 7.7 Dataset framing does not match a webcam

Lower-body confidence never exceeds 0.22 in any WLASL clip, so the dataset cannot
answer any question about wide framing — the stratification attempted for that
split 0.186 against 0.186 and measured noise. Every clip is torso-framed, the
same regime as a webcam, which is why the padding effect had to be manufactured
by moving the box rather than found in the data.

More generally the dataset and a live camera disagreed five times in this phase:
swapped hand labels, the duplicate threshold, the dedicated hand model,
`performance` mode, and the 0.7 presence threshold (the only one where the
camera *agreed* with the clips) — each measured favourably on clips and
rejected on camera, or the reverse.

This is not a curiosity, it is a risk to the whole premise. The model is
trained on WLASL features and will be served webcam features, and the feature
statistics evidently differ enough to flip conclusions. §8 lists "not tested on
a real signer yet" as an open item; these five disagreements are evidence that
it is the binding one. Source hand size is the likely reason: ~40 px median in the clips
(p90 216), far larger in a webcam close-up. **The camera should be the tiebreaker.**

---

## 7.8 Agreed but not yet built: snapshots and more seeds

Two cheap ensembling axes, decided on measurement rather than taste, to be
applied to the shipped model in `train_final.py` and the demo.

**Snapshot ensembling, worth +2.3 points.** Every run already averages its
out-of-fold predictions over the last 30 epochs, and that average scores
better than the reported plateau on **12 of 12 runs**, median +0.0254:

```
run            plateau   predictions moyennees
proj48full      0.6462          0.6694
p_vel           0.6493          0.6763
tipbase         0.5989          0.6295
```

Three different numbers must not be confused, and were at first:

```
derniere epoch     ce que le modele expedie EST        proj48full 0.6390
plateau 30 epochs  ce qu on rapporte                              0.6462
predictions moy.   un AUTRE modele, non construit                 0.6694
```

So the reported figures do **not** understate the shipped model; if anything
the plateau slightly overstates it, since the artefact we save is the final
epoch. The 0.6694 is a model we never build — it needs the weights saved at
several points in the plateau.

Free in *training* (those epochs already happen), not free in storage: ~5
checkpoints per seed instead of 1. Inference cost is irrelevant here because
the demo classifies **once per recording**, so 25 forward passes at ~3 ms are
invisible after a 2.8 s capture.

Snapshots must be chosen **by position, not by score** — evenly spread across
the plateau. Picking the "best" epochs would be selection on the test fold,
and in `train_final.py` there is no held-out set at all, so the only
rankable signal is training accuracy, which would favour the most overfit
snapshots.

**More seeds, worth about +1 point.** Measured costs and the 1/n variance
curve:

```
MESURE, octobre 2026, projection 48 + echafaudage 300 gloses :

  epoch d affinage k-fold   2.92 s  (580 clips + evaluation TTA)
  epoch d affinage expedie  ~3.2 s  (733 clips, pas d evaluation)
  epoch de pre-entrainement  ~9 s   (3428 clips, 300 gloses)
  chargement des features    95 s   une fois par run

entrainement final    26 min par graine    3 -> 1 h 20, 5 -> 2 h 10, 8 -> 3 h 30
k-fold (mesure)      3 graines ~5 h 30,  5 graines ~9 h

Les 26 minutes sont CHRONOMETREES sur le vrai run (graine 1 finie a 20:07,
graine 2 a 20:33). L epoch de pre-entrainement etait le seul chiffre estime
plutot que mesure, et il l etait de 67% en trop : ~9 s et non ~15.

Les 13 min par graine citees avant mesure valaient pour l echafaudage a 100
gloses et 60 epochs. Le pre-entrainement est desormais 4/5 du cout : 30 des
38 minutes par graine. C est le seul levier de vitesse qui reste, et c est
celui que --shared-pretrain aurait actionne.

3 -> 5   environ +1.0 point
3 -> 8   environ +1.6
3 -> 12  environ +1.9
```

Returns flatten hard. **5 is the sweet spot**; past 8 it is hours for tenths.

**Measure with 3, ship with 5.** The ensemble gain is variance reduction and
so expected rather than hoped for, which means shipping with more seeds than
were measured makes the real model slightly better than the quoted figure — a
conservative error. The k-fold stays at 3 seeds to keep the measurement at 2 h.

Plan: **5 seeds x 5 snapshots = 25 models** for the shipped model, 63 minutes
of training, k-fold quoted as the floor. The two axes decorrelate differently
(random initialisation versus stopping point) so they should stack better than
doubling either. Caveat: snapshot members are consecutive epochs of one run
and therefore correlated, so part of the +2.3 may already be captured by the
3-seed average.

### The shipped model, 27 October

Built at last, on the configuration this document has been accumulating:

```
25 modeles       5 graines x 5 instantanes (epochs 121, 128, 136, 143, 150)
155 dimensions   106 brutes + 23 derivees + 26 configuration de main
projection 48    289 778 parametres par modele
filtrage         seuil 0.7, hors-cadre 0.5, arbitrage des conflits
echafaudage      300 gloses, 3428 clips, 120 epochs, NON filtre
733 clips        50 gloses, 63 signeurs
2 h 35           33 min par graine
```

Card: **0.7623 +- 0.0372, 38x chance**, cited from `wlasl_kfold_best_4folds`
-- the 4-fold measurement of this exact pretraining setting, not the older
0.7413 at 100 glosses / 60 epochs. Two things make the real model likely a
little better than that figure: the missing fold 5 was the easiest of the five
in the reference run (0.794 against 0.728 across the other four), and the
measurement used 3 seeds while this ships 5, the ensemble reducing variance in
the favourable direction.

Verified before trusting it, all three worth keeping as scripts:

- the demo loads all 25 models in 17 s and classifies in 208 ms, which is
  invisible after a 2.8 s capture;
- `check_gate_parity.py` -- per-frame gating equals batched gating, 300 clips,
  21739 frames, 59 conflicts, max difference 0.00e+00;
- `check_source_parity.py` -- new: stubs out rtmlib so the demo's *model input*
  can be checked without a camera. Confirms the card switches the extractor to
  raw mode, the extraction threshold is replayed at 0.7, a low-confidence hand
  is gated away, the drawn skeleton follows what the model sees, and the vector
  is bit-identical to the training path.

**What this number still does not answer:** every figure here comes from
held-out WLASL signers. §7.7 lists five occasions where a conclusion measured
on clips reversed on a live webcam. The binding risk is unchanged and now
unavoidable -- the model is built, so the next real information comes from
pointing a camera at it.

### Resuming an interrupted run, and the shared RNG it exposed

Two runs have now died mid-flight: one after two folds, which at the time held
the project's two best scores, and one at the fifth fold of five, killed by a
Windows restart (Kernel-Power logged a shutdown transition at 02:10 on
15 October). The `.partial.json` written after every fold saved the results both
times, but nothing could re-run only the missing folds -- finishing 65 minutes
of work meant redoing 5 h 30.

`--only-fold 4,5` now runs just those folds, and `merge_kfold.py` reassembles
the pieces. The skip happens *inside* the fold loop rather than by filtering the
fold list, because `enumerate` must keep its index: the training seeds are
`seed + fold_index * 100 + seed_index`, so a filtered list would renumber fold 5
as fold 1 and train different models than the ones it is meant to complete.

**That was not enough, and the first version of the flag was silently wrong.**
Verified by running fold 2 both ways:

```
run complet     [0.034,  0.0136, 0.0136]
--only-fold 2   [0.0136, 0.0272, 0.0136]
```

Same partition, same TensorFlow seeds, different models. The cause:
`rng = np.random.default_rng(args.seed)` was created **once, before the fold
loop**, so its state ran through the whole run. Fold 2 of a full run inherited a
stream already advanced by fold 1's thousands of shuffle and augmentation draws,
while an isolated fold 2 started fresh. The last epoch agreeing at 0.0136 was
coincidence -- exactly the kind of near-miss that makes a broken thing look
fine.

The generator is now built per (fold, seed) on the same convention as the
TensorFlow seed. Fold 2, 4 and 5 then reproduce bit-for-bit, and
`merge_kfold.py` rebuilds a result identical to the full run's.
`train_final.py` had always done this correctly, per seed.

**What this changes about every earlier run:** their folds were never
independent draws -- fold N's augmentation stream depended on how much work
folds 1..N-1 had done before it. The paired comparisons in this document are
unaffected, because they pair on *which signers are held out*, and that comes
from `signer_folds(records, folds, args.seed)`, which never touched this
generator. A future run is still comparable fold-by-fold with an archived one;
it simply draws different augmentations, which is noise of the same kind as any
re-run.

The merge refuses more than it accepts, which is the point: any differing
configuration key, any fold present in two files, any missing fold. Averaging
folds from two configurations would produce a number describing no model at all,
and nothing in the output file would say so.

### The scaffold was being filtered in one script and not the other

Spotted by comparing the two pretraining curves while both were running:

```
train_final (echafaudage filtre)    e40 0.413   e80 0.648   e120 0.745
train_kfold (non filtre)            e31 0.171   e61 0.337   e120 0.527
```

`train_kfold.py` has never gated the pretraining set -- the scaffold sees the
raw features with both hands always present. When `train_final.py` was written
to express the measured config, gating was added to its scaffold too, on the
reasoning that it should see the same distribution as the fine-tuning. The
reasoning is defensible and was never measured, and it made the shipped model a
different configuration from the one the k-fold reports.

Gating evidently makes the scaffold's own task far more learnable, +0.22 on its
120th epoch. That says nothing about the fine-tuning that follows, which is the
only thing that matters, and the scaffold's own accuracy has never correlated
with anything -- §7.5i found its *duration* stops mattering by epoch 60 even
though its accuracy keeps climbing.

Now `--gate-pretrain` in both scripts, **defaulting to off in both**, so the
shipped model is the measured model. Worth testing, as one flag, alongside item
3: a scaffold trained on the distribution the fine-tuning actually sees is a
reasonable thing to want, and the measurement is cheap.

The general lesson, third time in this project: a change that sounds right and
costs nothing to make still has to be measured before it ships, and the
dangerous ones are those that touch only one of two scripts that are supposed
to agree.

### Heavy pretraining makes the last fine-tuning epochs worthless, not harmful

Asked while the shipped model was training, from the right observation: if
pretraining reaches 0.745 on its own task, is the 150-epoch fine-tuning budget
now excessive? The pretrain sweep answers it without a new run.

What the last 30 fine-tuning epochs earn, by how much pretraining came first:

```
                  band 121-150 minus 91-120     per fold
pre0  (none)             +0.0176           +.020 +.021 +.015 +.015 +.017
pre30                    +0.0121           +.009 +.003 +.027 +.011 +.011
pre60 (what 0.7413 used) +0.0165           +.012 +.020 +.015 +.018 +.017
pre120                   +0.0009           -.000 +.003 -.006 -.002 +.009
```

**At 120 pretraining epochs those 30 epochs contribute nothing** -- +0.09 of a
point against +1.76, three of five folds slightly negative. The smoothed curve
reaches 99% of its maximum by epoch ~101 with pre120 against ~126 with none.

But flat is not falling: the final point sits 0.006 below the smoothed peak,
which is the plateau wobble §7.8 already documented, not decay. So the budget
is **wasteful, not wrong**, and the waste is small -- fine-tuning is 8 of each
seed's 38 minutes, so 150 -> 115 would save ~8 minutes across 5 seeds.

Left at 150 deliberately. It is the budget the 0.7413 was measured at, and
changing it would make the card cite a number for a configuration never
trained.

Two traps avoided here, both worth naming:

- **training accuracy is not the signal.** The figure printed each epoch comes
  from the augmented batch inside `fit`, with a fresh random mirror and jitter,
  so its 0.985 -> 0.958 wobble is noise. Overfitting means training accuracy
  rising while held-out accuracy falls, which needs a held-out set
  train_final.py deliberately does not have.
- **not moving the snapshot window.** The 5 snapshots sit at 121-150, chosen
  when that was the best band. Under pre120 the best region is nearer 105-135
  -- but the two bands measure 0.7000 and 0.7009, indistinguishable, so moving
  the window would optimise noise. Worse, choosing it from held-out curves is
  exactly the score-based selection snapshots are supposed to avoid.

Where this does cash out is the k-fold: at pretrain 300/120, `--epochs 120` is
defensible, and there fine-tuning is 1.8 h of the 9, so it saves ~20 real
minutes. Worth folding into item 3.

### Sharing the pretraining across seeds: stopped early, decided on judgment

The pretraining dominates the cost. At 5 seeds each fold pretrains five times,
which is 2.7x the total run time, and all five runs learn the same task on the
same clips. Sharing one pretraining across the seeds would make the planned
cumulative run drop from ~10 h to ~3.5 h and put 8 seeds within reach.

The objection is that an ensemble only pays if its members disagree, and
per-seed pretraining is plausibly where most of that disagreement comes from.
Shared, the seeds start from *identical* weights and diverge only through
shuffle order and fine-tuning dropout.

`--shared-pretrain` was added to measure instead of assume. It never shares
across *folds* — pretraining excludes the held-out fold's signers, so carrying
weights between folds would leak test signers — only across seeds within a
fold. Arm A is the existing `proj_everything` run (0.7413), which already is
per-seed pretraining at the default 60 epochs; arm B repeats it with
`--pretrain-epochs 60 --shared-pretrain` so sharing is the only difference.

Stopped at 4 of 5 folds, plateau metric:

```
          A per-seed    B shared      diff
fold 1      0.7770       0.7815      +0.0046
fold 2      0.6917       0.6246      -0.0671
fold 3      0.7193       0.7218      +0.0025
fold 4      0.7248       0.6963      -0.0285
mean        0.7282       0.7061      -0.0221
```

**No fold gains, two lose, two are flat.** Folds 1 and 3 sit within noise;
folds 2 and 4 drop clearly, and fold 2 is also where A scored lowest. The sign
is consistent across every fold that moved at all, but at n=4 the test gives
t = -1.32 against a threshold of 3.18, so **-2.2 is a direction, not a
measurement**, and should not be quoted as one. A's 5-fold mean is 0.7413; on
these four common folds it is 0.7282, so fold 5 is A's easiest.

Decision taken without finishing it, as a judgment call rather than a result:
**keep per-seed pretraining.** The reading that fits the shape of the data is
that seed diversity is worth little when the fold is easy and worth real points
when it is hard — and the hard case is the
one that matters, since the live demo on new signers is harder than any fold.
Sharing would buy speed exactly where accuracy is already fine and cost it
where it is not. Partial result kept at
`models/wlasl_kfold_shared60_3folds.json`.

So the cumulative run costs its full price: ~9 h at 5 seeds, ~5 h 30 at 3. The
`--shared-pretrain` flag stays in the harness; if the budget ever binds, the
test to finish is this one, and only fold 5 is missing.

### A train/serve divergence found while wiring the shipped model

`train_final.py` and the demo were both built for `features_rtm07`, where the
extractor applies the 0.7 threshold, the off-frame filter and the wrist-claim
arbitration at extraction time. The measured 0.7413 configuration uses
`features_rtm_all`, which is extracted in **raw mode**: both hands are always
written, flags at 1, and `gate_hands()` decides afterwards. That makes the
gate parameters part of the *definition of the model's input*, not a setting
of the extractor.

Left alone, the demo would have filtered with the extractor's rule while the
model had been trained on `gate_hands`' rule. The two agree on the threshold
and the off-frame test and **disagree on conflicts**: the extractor drops the
hand that loses the wrist claim, `gate_hands` drops the one with the lower
confidence. Same intent, different hand on some frames.

Measured on 300 clips: conflicts are **0.27% of dataset frames** — but 5.48%
on a close-range webcam (§7.3). So the disagreement would have been invisible
in every number in this document and present in every demo session. That is
the shape of the bug that took the alphabet phase from 0.985 to 0.384.

The demo now builds model input through the *same* `gate_hands` call, with the
parameters read from the model card. Verified bit-exact: per-frame gating
before padding against batched gating after padding, 300 clips, 21739 frames,
59 of them conflicts, **maximum difference 0.00e+00**.

Two details that look like noise and are not:

- the card carries `extraction_threshold` separately from `gate`. The conflict
  flag stored in the npz files was computed at the *extraction* threshold, so
  the demo must replay conflicts at that value, not at the gate's. Both are
  0.7 today, which is exactly why they needed separating — nothing would have
  reported them drifting apart.
- the demo reads `last_inside`/`last_conflict` with the same `getattr`
  defaults the extraction loop uses. `frame_features` returns early when the
  shoulders are missing, before setting them, so the features directory
  carries that same approximation. The vector is all zeros in that case, so
  it changes nothing — but the path stays identical rather than merely
  equivalent.

The display still follows what the model sees: presence flags are read from
the *gated* vector, so a hand dropped by threshold, framing or arbitration
does not appear on screen.

### The 150 epochs are fine; the 60 pretraining epochs are unexamined

Early stopping was asked about and the measurement says there is nothing to
stop early from. Mean accuracy per epoch band:

```
                 0-50    50-80   80-110  110-140  140-150
proj48full      0.2705   0.5627   0.6202   0.6431   0.6475
rtm07_everything 0.5773  0.6503   0.6633   0.6658   0.6721
```

**The last band is the highest in every run.** The curve rises and flattens,
never turning down, so the per-fold "peak" around epoch 113-127 is plateau
noise rather than a maximum followed by decay. Combined with §5's reason — a
validation split would leave ~3 clips per gloss, too noisy to choose a
stopping epoch, and stopping on the test fold is leakage — the fixed budget
stands. Snapshot averaging handles the plateau wobble better than choosing a
point could.

`--pretrain-epochs 60` is a different matter. It has **never been varied in
any experiment, is not recorded in the k-fold results** ("non enregistre"),
and is unjustified in these docs. The pretraining loop also runs blind, with
no evaluation inside it, so there is no curve to inspect after the fact. Unlike
the 150, which has evidence behind it (80 -> 250 moved +0.003), 60 is a default
nobody has questioned.

Queued, by value rather than by cost:

```
1  THE CUMULATIVE RUN, every confirmed gain at once    the agreed next step
   @0.7 + rich + confidence + handshape-full
   + projection 48 + PILE + gate
   + --pretrain-vocab 300 --pretrain-epochs 120
   + 5 seeds + 5 snapshots per seed
   Two pieces easy to forget: drop --clips-from (726
   -> 733 target clips) and widen the SCAFFOLD with
   --pretrain-variants all, 3032 -> 3428 clips (same
   300 glosses, measured). Per-seed pretraining, not
   shared (§ above). ~9 h at 5 seeds, ~5 h 30 at 3,
   on the timed per-seed rate. Command below.

2  SHIPPED MODEL REBUILT, 27 October                  DONE
   25 models (5 seeds x 5 snapshots at epochs 121,
   128, 136, 143, 150), 155 dims, projection 48,
   pretraining 300 glosses / 120 epochs ungated.
   Card cites 0.7623 +- 0.0372, 38x chance, from the
   4-fold measurement of this exact pretraining.
   Took 2 h 35 at 33 min/seed. Demo verified: loads
   all 25 in 17 s, 208 ms per classification, and
   check_source_parity.py confirms its model input is
   bit-identical to the training path. train_final.py now takes
   --project, --handshape-full, --gate*, --vocab-from,
   --pretrain-variants and --snapshots; the demo
   replays gate_hands from the card and loads all 25
   models. Dry-run clean, 733 clips, 155 dims,
   filtering 80.9% -> 43.2%, scaffold 3428.
   ~2 h 10 at 5 seeds (26 min/graine, chronometre).
   The demo still runs ~0.67.
   Does NOT depend on item 1: the card cites 0.7413
   with the caveat that 300 glosses, 120 epochs and
   5 seeds were never measured together.

3  --pretrain-vocab 300 CONFIRMED on 4 of 5 folds     +3.4, t=3.20
   Folds 1/2/4 gain 4.1/4.4/4.9, fold 3 is flat
   (+0.25). t=3.20 against a 3.18 threshold at n=4,
   so it clears significance by a hair. Fold 5 was
   killed by a Windows restart and deliberately not
   re-run -- four folds answer the question, and
   --only-fold 5 is there if the fifth is ever wanted.
   Mean on those folds 0.7623 against 0.7282.
   Try --epochs 120 there too: at pretrain 120 the
   last 30 fine-tuning epochs earn +0.0009 (§ above),
   and cutting them saves ~20 min of the k-fold.
   And --gate-pretrain, one flag, never measured:
   filtering the scaffold lifts its own accuracy by
   0.22 but may or may not survive fine-tuning.

4  ship with 5 seeds, keep measuring with 3            DONE (shipped at 5)

5  confirm on FRESH fold seeds (`--seed`)              validity, not points
   ~35 experiments have run against the same 5 folds.
   A 6.5-point margin is far harder to attribute to
   chance than 4.7 was, but the check is still owed.

6  snapshot ensembling, 5 per seed, by position        DONE, +0.5 expected
   DOWNGRADED: measured +0.0053 on top of a 3-seed
   ensemble (§7.5h) against +0.0254 on single-seed runs.
   The seeds already capture that variance reduction.
   Still free, no longer the headline.

   CLOSED: pretrain_epochs is now recorded in the
   results; the 30/60/120 sweep ran (§7.5i: duration
   stops mattering by ~60); sharing the pretraining
   across seeds was tested and rejected (§ above).
```

Item 1 as a command, every flag deliberate and none inherited by accident:

```
python src/wlasl/train_kfold.py --folds 5 --epochs 150 --vocab 50 \
  --variants dominant --features-dir data/wlasl/features_rtm_all \
  --rich --confidence --handshape-full \
  --gate 0.7 --gate-inside 0.5 --gate-conflict \
  --project 48 --label-smoothing 0.1 --tta --seeds 3 \
  --pretrain-vocab 300 --pretrain-epochs 120 --pretrain-variants all \
  --vocab-from data/wlasl/features --tag _best
```

`--vocab-from data/wlasl/features` keeps the 50 target glosses identical to
every earlier experiment, so the number stays comparable; `--clips-from` is
deliberately absent, which is what returns those 7 clips. `--seeds 3` measures;
the shipped model uses 5 (item 4). Pair the result against
`models/wlasl_kfold_proj_everything.json`'s `per_fold_plateau`, never against
`per_fold` — §7.6 is the record of getting that wrong.

Item 2 as a command -- every default already matches the measured config,
so the bare call is the right one:

```
python src/wlasl/train_final.py
```

It writes 25 models (5 seeds x 5 snapshots at epochs 121, 128, 136, 143, 150)
plus `models/wlasl_gru_rtmpose.meta.json`. The previous shipped model is kept
at `models/archive/before_best/`, because `*.keras` is gitignored and
overwriting it would be permanent -- which has happened once.

Guess on record for item 3, so it can be wrong: pretraining *breadth*
should matter more than its length, since what transfers is generic hand
and body motion rather than gloss-specific knowledge. The 30/60/120 sweep
already bore out the second half of that (duration stops mattering by ~60);
item 3 tests the first half.

---

### 7.10 The demo ran at 16 fps, and the detector was the whole reason

Reported from a live session: 46 frames captured in 2.83 s, 16.2 fps, and
`resample` warning that frames were being repeated -- 46 real frames becoming
84 on the 30 fps grid, so nearly half the sequence was duplicates. The model
was trained on ~30 fps clips and the time-warp augmentation only covers
0.8-1.25x (24-37 fps), so 16 fps sits outside what it was taught to tolerate.

Where the 62 ms per frame went:

```
boucle complete        ~62 ms   (16.2 fps)
  detecteur Wholebody    40 ms   dont 93% d inference GPU dans YOLOX-m
  lecture camera         ~9 ms   recouvrable
  dessin / affichage     ~13 ms
25 modeles TF charges   +3.6 ms  negligeable
filtrage par frame       0.015 ms negligeable
```

Neither the 25-model ensemble nor the per-frame gating was the problem,
despite both being new that day. **The person detector was 76% of the cost**,
re-running YOLOX-m on every frame to locate someone who had barely moved.

**Four fixes rejected by measurement, not by taste.** Each was tested the same
way: feature differences per frame on six real clips, since what matters is
what the model sees, in shoulder-width units.

```
                     gain            derive par rapport a l entrainement
YOLOX-tiny           52 fps          28.4% des frames > 0.1, 10.1% > 0.5
reutiliser la boite  ~30 ms/frame    max 2.07 largeurs d epaules
YOLOX-m @416         -               impossible, entree ONNX figee a 640
MJPG / 640x480       aucun           la camera negocie deja ca a 30 fps
```

The box-reuse result is the instructive one: the *mean* difference was 1% of a
shoulder width, perfectly acceptable, while the maximum was 2.07 -- occasional
frames where a stale box sends the keypoints somewhere else entirely. A mean
would have green-lit it.

**Two fixes kept.**

*Threaded camera read.* `cap.read()` blocks until the next frame, so capture
and detection added up instead of overlapping. Worth **+3 fps** -- not the +9 a
first benchmark implied, because `read()` only blocks for a full frame interval
when called back to back, and after 40 ms of detection the camera already has
one buffered. Zero effect on features. Sample timestamps now come from the
capture instant rather than from after processing, which also removes a skew
`resample` was inheriting.

*TensorRT, behind `--trt`, off by default.* Same graph, compiled: the detector
drops from 24.9 to 11.0 ms (**2.27x**), and Wholebody as a whole from 37.3 to
24.8 (**1.50x**), which should put the loop near 23 fps. Two obstacles, both
worth recording:

- TensorRT cannot build YOLOX's TopK ("K exceeds the maximum value allowed
  (3840)"). Excluded via `trt_op_types_to_exclude`, it stays on CUDA and costs
  nothing, since the time is in the convolutions.
- **TF32.** TensorRT enables it by default, which shifts features on 4.44% of
  frames (max 2.20): box scores move slightly and, where two boxes are close,
  numpy's NMS stops picking the same one. Disabling it makes the detector
  faithful (0.25% of frames) -- but `NVIDIA_TF32_OVERRIDE` is **global** and
  then shifts the pose model, which ran with TF32 during extraction: 4.69% of
  frames. And building with it off while running with it on is refused by
  TensorRT itself ("Inconsistent setting of NVIDIA_TF32_OVERRIDE env var at
  build 0 and at execution -1").

So both routes cost ~4.5% of frames moving more than 0.1 shoulder width
relative to the training pipeline, mean 0.035, **with no gating decision
changed across 810 hands**. Six times more faithful than YOLOX-tiny, and still
not free -- hence a flag that is off until the trade is accepted deliberately.

The clean way out, if the drift matters: re-extract the training features
through this same TensorRT pipeline (~1.4 h) and retrain the shipped model
(~2.6 h). Both sides then become identical *and* fast, at the cost of the card
citing a k-fold run measured on the older features.

Installing it is not small: `tensorrt-cu12` is **3.4 GB** on disk, and
`gpu_setup.enable_tensorrt_dlls()` was needed because the wheels land in
`site-packages/tensorrt_libs/`, outside the `nvidia/*/bin` sweep -- the same
silent-fallback trap already documented for CUDA.

## 8. Open items

**Not tested on a real signer yet, and this is now the binding risk.** Every
number here comes from held-out *dataset* signers. §7.7 documents five
occasions where a conclusion measured on clips was reversed on a live webcam —
swapped hand labels, the duplicate threshold, the dedicated hand model,
`performance` mode, and (the one agreement) the 0.7 presence threshold. The
feature statistics of a close-range webcam evidently differ enough from WLASL
clips to flip decisions about the extractor itself, so they will differ for the
classifier too. Expect lower, and measure rather than assume.

**Top-5 matters at this accuracy.** The demo shows five candidates; a correct
answer anywhere in them means the model found real signal.

**~11 training clips per class** remains the binding constraint. Nineteen
experiments confirm the model is not the limit. The ways forward are all
data-side: a third mirror, recording your own clips, or a smaller vocabulary.

**Recording your own clips is ready to start.** `docs/RECORDING.md` is the
brief and `make_reference.py` produces two reference clips per sign of the
variant the model actually learned — which matters because 9 of the 50 words
have two variants in the data and `candy` uses variant 1, so a signer doing the
version they know would record what the model treats as a different class.

Protocol: 4 signers, 6 takes for the 22 words the confusion matrix flagged and
3 for the rest, 216 takes per signer (~54 min), 864 clips. Vary **distance,
angle and speed** — not the room: the features are shoulder-normalised joint
coordinates, so the image never reaches the model and environment only
perturbs the detector upstream. Distance matters most, since hand size in
pixels is the measured bottleneck on the finger distinctions those pairs need
(~40 px in the dataset against ~200 px on a close webcam).

Record the signer ID. It is what makes leave-one-signer-out possible, and that
is the only way to get an honest number for "does this work on a new person at
our setup" — which 0.7413 does not answer, since it measures held-out WLASL
signers.

One measured caveat: at 7 clips per class accuracy is 0.3014 against 0.5998 at
~14.5, so halving the data halves the accuracy. Doubling it is worth far more
than any feature work in this document — but 864 clips from 4 signers would be
58% of the training set from four people, so expect the demo to improve on your
signers while the WLASL benchmark may not move.

**The confusion matrix exists** (§7.5d). Every badly confused pair differs
mainly in handshape at the same body location — `apple`/`candy`,
`bird`/`orange`, `dance`/`study`, `bed`/`mother` — and `apple` is an attractor,
predicted 27 times for 15 real clips. No class collapses entirely, and class
imbalance is not the cause (clip-count/recall correlation +0.116). What remains
is that errors are *confident* rather than narrow: when top-1 is wrong, the gap
to the true label is a median 0.433. The discriminating information is not
reaching the model, which is why thumb opposition helped the average (+1.1) but
did not repair the pairs it targeted.
