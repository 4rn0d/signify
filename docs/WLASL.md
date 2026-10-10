# Signify — word-level phase (WLASL)

Recognising whole signed words from motion, rather than static letters. A
separate problem from the alphabet phase (`FINDINGS.md`) and a separate
pipeline: a sign is defined by movement, often uses two hands, and carries
meaning in *where* it happens relative to the body — none of which a
single-frame image classifier represents.

**Current state:** 0.6762 ± 0.0619 on 50 signs, chance 0.0200 — **34× chance**,
on a signer-disjoint split, with MediaPipe features and the full training stack.

The landmark extractor was then changed to RTMPose (§7), which finds hands in
84% of frames against MediaPipe's 47%. Compared like for like, without the
training stack:

```
MediaPipe + derived features          0.5867
RTMPose @0.7 + derived + confidence   0.6005
```

Parity on accuracy (+1.4, t=+0.80, not significant) with far better hand
coverage. The same stack has not yet been re-measured on RTMPose features.

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

Adding confidence as a feature (§7.3b) takes it to **0.6005**, which is ahead
of MediaPipe's 0.5867 by +1.4 — but at t=+0.80 on 3/5 folds that is not a
significant win, so the honest claim is parity on accuracy with much better
hand coverage.

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

### 7.3b Confidence as a feature, worth +2.9

With the threshold at 0.7 the model is told *whether* a hand is present but
not how much to trust it — a hand at 0.71 and one at 0.99 look identical. The
per-hand confidence is stored in each `.npz`, so feeding it as two continuous
columns costs nothing but a flag. Tested against the same five folds:

```
confidence alone                +0.0289   t +3.01   5/5 folds   significant
symmetry, added to confidence    -0.0139   t -0.75   1/5 folds   noise
```

**Confidence is worth +2.9 points, winning every fold**, with the largest gain
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

### 7.6 Dataset framing does not match a webcam

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

## 8. Open items

**Not tested on a real signer yet, and this is now the binding risk.** Every
number here comes from held-out *dataset* signers. §7.6 documents five
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

**Recording your own clips** is the highest-value next step for a demo that
works on *you* — the capture window and feature extraction already exist, so
10 takes each of 10 signs would give more examples of you than the dataset
has of anyone.

**No confusion matrix.** Knowing which signs collide would guide vocabulary
choices; WLASL has regional variants that may genuinely overlap.
