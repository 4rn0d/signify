# Signify — ASL Citizen phase

Same task as the WLASL phase (`WLASL.md`), recognising whole signed words from
a webcam, rebuilt on a different dataset. The WLASL notebook stays the record
of how the pipeline was built and why; this one records what changed when the
data did.

**Current state (16 November):** a TCN recipe reaches **0.8864 ± 0.0102**
on 100 signs, signer-disjoint 5-fold, **+2.4 points over the GRU baseline on
every fold** (§9). A 5-fold now takes 31 minutes instead of ~7 hours (§10).
The demo and the app's backend still run a 1-seed GRU model for live testing;
shipping the TCN recipe is the next step (§12).

```
                                       signes   precision, signeurs jamais vus
WLASL, meilleure recette (octobre)        50     0.762
ASL Citizen, GRU de reference (3 nov.)   100     0.8610 ± 0.0073   (0.8626 sur les epochs 11-40)
ASL Citizen, recette TCN (16 nov.)       100     0.8864 ± 0.0102   +2.4, 5/5 folds, p = 0.002
```

Where to read what: why ASL Citizen and its license (§1), the data (§2), the
vocabulary (§3), extraction (§4), placing the sign in time (§5-6), the models
(§7), the GRU baseline and what it took (§8), the TCN recipe (§9), training on
the GPU and its speed (§10), how every number here is measured (§11), and
what is still open (§12).
---

## 1. Why switch

Every WLASL experiment pointed at the same wall: **~14.7 clips per word**, and
nineteen model-side experiments confirming that data, not the model, was the
limit. Halving the data halved the accuracy (0.3014 at 7 clips per class
against 0.5998 at ~14.5).

ASL Citizen (Microsoft Research, NeurIPS 2023) removes most of that wall:

```
                       WLASL (50 mots cibles)     ASL Citizen
clips par signe              ~14.7                30.5 (ecart-type 1.7)
complet                      non, liens YouTube   oui, une archive Microsoft
                             morts, recupere via
                             des miroirs
equilibre                    longue traine        quasi parfait
conditions                   videos YouTube       webcams, cadrage haut du corps
signes                       2000 (dont ~300      2731
                             recuperables)
signeurs                     63 sur nos 50 mots   52
```

The recording platform showed participants the least-recorded signs first,
which is why the counts are so even. And the footage comes from webcams with an
outline overlay framing the upper body: the demo's own condition. §7.7 of the
WLASL notebook recorded five occasions where a conclusion measured on dataset
clips reversed on a live webcam; training on webcam footage attacks that gap
directly.

A published reference point exists: 63% top-1 over all 2,731 signs and 91%
recall@10 on held-out signers (I3D baseline), which no WLASL-scale number could
be compared against.

**Decision: ASL Citizen alone, WLASL discarded.** Mixing the two would have
required matching every WLASL variant to an ASL Citizen sign by eye. Using only
one dataset makes the question disappear. WLASL stays on disk, unused.

### What carries over and what does not

The **code** carries over unchanged: raw extraction with train-time gating
(`gate_hands`), the 155 features, the linear projection, ensembling and
snapshots, the k-fold harness with `--only-fold` and `merge_kfold.py`, the
parity checks, the demo with TensorRT and the threaded camera.

The **numbers** do not. The 0.7623, the 0.7 hand threshold, which features
helped: all were measured at ~14 clips per class on WLASL. The strongest lesson
of that phase — every model-side change made things worse — was a symptom of
data starvation, and may stop holding at twice the data per sign. The few big
decisions get re-measured; the 35 experiments do not get re-run.

### License, and what it constrains

Microsoft Research License: **non-commercial, non-revenue research use only.**
No redistribution of the videos or of modifications, no hosted service built on
them, and the data is to be destroyed when the research ends. Commercial use
requires contacting `ASL_Citizen@microsoft.com`. The terms do not mention
trained models, which is not the same as permitting their commercial use.
ASL-LEX, used below to rank signs, is CC BY-NC 4.0.

Consequences already applied:

- `data/asl_citizen/` is gitignored. Before that rule, the 46 GB archive was a
  single `git add -A` away from the repository, which would also have been
  redistribution.
- Every page or image built from the videos is written locally and never
  published (the variant review page, the contact sheets used to check the
  sign detector).

---

## 2. The data

`ASL_Citizen.zip`, **45.9 GB**, one Microsoft Download Center archive,
downloaded in 17 minutes and extracted with every entry's CRC verified
(Python's `zipfile` checks it on read). 83,399 videos and three split files,
exactly the paper's counts. The zip was then deleted; it re-downloads in under
20 minutes.

```
splits officielles   train 40 154 (35 signeurs)   val 10 304 (6)   test 32 941 (11)
                     disjointes par signeur

resolution           640x480 : 80 184    960x540 : 3 211    480x640 : 4
codec                h264 : 79 873       FMP4 : 3 526 (illisible dans un navigateur)

cadence              ~30 fps     92.8%
                     24-28 fps    4.5%   dans la marge de l augmentation temporelle
                     < 24 fps     2.6%   2 132 clips, 14 signeurs, dont 1 205 pour P49 seul
                                         cible : 88 clips (2.8%)   echafaudage : 241 (2.6%)

duree                mediane 2.57 s, p90 4.33 s, p99 6.93 s, max 22.6 s
                     45.2% depassent 2.67 s, soit les 80 frames du modele
                     clips NON recadres : le temps mort avant et apres le signe reste
```

Each video carries an **ASL-LEX code** identifying the sign itself rather than
the English word. That turned out to matter more than anything else in the
metadata (§3).

---

## 3. Vocabulary

### Variants are labelled, and each one is a full class

Where one English word covers several signs, ASL Citizen splits it into
**numbered glosses** — `CANDY1`, `CANDY2` — each with its own ASL-LEX code.
Across all 2,731 glosses, none carries two codes. The problem that cost the
WLASL phase most (contradictory variants hidden under one label, worth +5.6
points once filtered) is resolved in the labels themselves.

And every variant has enough clips to stand alone:

```
736 signes numerotes dans ASL Citizen
clips par variante : min 21, mediane 31, max 45
WLASL, pour memoire : ~14.7 par glose, variantes a 5-14 clips
```

The lopsided pairs (`EAT1` 39 against `EAT2` 24, `DEAF1` 39 against `DEAF2` 23)
come from signers recording the same sign twice, not from cross-labelling:
`EAT1`'s 39 clips come from 29 signers.

So **each numbered variant is its own class.**

### Choosing the signs: frequency of use, not convenience

With ~30 clips per sign everywhere, "the N signs with the most clips" — how
WLASL's vocabulary was chosen — no longer means anything. The signs are ranked
instead by **how often Deaf signers report using them**: ASL-LEX 2.0's
`SignFrequency(M)`, a 1–7 rating, joined to ASL Citizen through the ASL-LEX
code (2,725 of 2,731 signs matched).

```
top  50   frequence >= 6.35    1 533 clips
top 100   frequence >= 6.08    3 089 clips   <- cible
top 150   frequence >= 5.93    4 611 clips
top 200   frequence >= 5.79    6 146 clips
top 300   frequence >= 5.56    9 199 clips
```

**Target: ranks 1–100.** **Pretraining scaffold: ranks 101–400** (300 signs,
9,181 clips), keeping the one WLASL lesson that clearly held — breadth in
pretraining is what transfers (+3.4 at 300 glosses, §7.5i of the WLASL
notebook).

The top 100 reads like everyday ASL: `WATER`, `ILOVEYOU`, `YOU`, `YES`, `EAT1`,
`WANT1`, `DEAF1`, `ME`, `WORK`, `THANKYOU`, `HELLO`, `BATHROOM`... 19 of the
old 50 WLASL words made it on frequency alone. The full ranking is in
`data/asl_citizen/ranked_signs.json`.

Two things flagged in the list, without changing it:

- **Numbered twins inside the target**: `EAT1`/`EAT2`, `BED1`/`BED2`,
  `FINE1`/`FINE2`. Separate classes for training; the demo should display them
  merged ("EAT"). Several other target signs have their twin in the scaffold
  (`WANT2`, `MONEY1`, `NIGHT1`, `NOTHING2`, `WALK2`, `WOMAN1`), so pretraining
  learns to tell them apart.
- **Seven pointing signs**: `YOU`, `ME`, `HE`, `MY`, `YOUR`, `THIS/IT`,
  `POINTINDEX`. Several differ mainly in the *direction* of a point or palm,
  and pointing at the camera versus at one's own chest is a depth difference
  that 2D keypoints cannot see. A prediction, not a measurement: they are kept,
  being among the most useful signs, and are the first thing to inspect in the
  confusion matrix.

### In the training harness

When clips carry a rank, "the top N" means the N best-ranked (`top_glosses`);
`--pretrain-vocab 400` gives ranks 1–400, target included, as on WLASL. A
folder mixing ranked and unranked clips is refused. Wiring it exposed three
traps, now closed: `train_final` inferred the extractor from the folder name
(`asl_citizen` contains no "rtm", so the model would have been labelled
MediaPipe and refused by the demo); its output files had the same names as
the shipped WLASL model, which a first ASL Citizen run would have overwritten
permanently (now prefixed `aslc_`); and the card always cited the WLASL
k-fold, so an ASL Citizen model would have shipped claiming 0.7623. The card
now cites only a k-fold measured on the same features (`--kfold`), or says
the accuracy is not yet measured. The demo loads the ASL Citizen card as soon
as one exists, `--card` to override.

---

## 4. Extraction: the demo's pipeline, every frame

`src/asl_citizen/extract.py`. Three choices, each taken from an error already
paid for on WLASL.

**The detector is the demo's, TensorRT included.** `use_tensorrt` is imported
from the demo as is. On WLASL, training features came from CUDA and the demo
could run TensorRT: 4.4% of frames diverged, and TF32 made the two impossible to
reconcile (§7.10 of the WLASL notebook). Here both sides run the same engine, so
that drift is gone by construction. `check_parity.py`: on real clips —
including one at 16.7 fps and one at 960x540 — the demo's gated input and the
extracted features gated by `gate_hands` are **bit-identical**.

**Raw mode.** Both hands always written, with their confidences, out-of-frame
fraction and conflict flag. Gating happens at training time, tunable without
re-extracting.

**Every frame, with its real timestamp, and no cropping.** The timestamp comes
from the decoder (`CAP_PROP_POS_MSEC`), not `index / fps`, since browser-made
recordings can have a variable frame rate. On the clips checked, timestamps are
monotonic and regular: the 16.7 fps clip steps a constant 60 ms, the rest 33 ms.
Resampling and window placement happen at load time (§5).

Measured rate: **2.09 s per clip** — about 1.8 h for the target, about 5.3 h
for the scaffold at the same rate. The extraction resumes where it stopped and
writes the highest-ranked signs first.

### The GPU had been silently disabled for three days

On 30 October, `pip install pymupdf` pulled in the CPU-only `onnxruntime`
1.31.1 as a dependency (`pymupdf` and `rtmlib` both declare it). It shares a
Python module with `onnxruntime-gpu`, overwrote its files, and from then on
onnxruntime offered only `AzureExecutionProvider` and `CPUExecutionProvider`.
Neither pip nor onnxruntime said a word; the demo had been running at roughly
20x the per-frame cost since.

Repaired with:

```
pip uninstall -y onnxruntime
pip install --force-reinstall --no-deps onnxruntime-gpu==1.22.0
```

and verified with a real session, not just the provider list. `pymupdf` still
imports, using the same module from the GPU package. **The next `pip install`
of anything depending on `onnxruntime` will do this again**, so the extractor no
longer trusts its environment: it refuses to start if the pose model is not on
CUDA, and if TensorRT does not take the detector — extracting on CUDA while the
demo runs TensorRT would reintroduce the 4.4% drift without any message.

---

## 5. Time: where in the window is the sign?

### A correction first

During the switch I stated that long clips were being sped up — that a 4.3 s
clip would play at 1.6x in training. **That was wrong.** `pad_sequence` keeps
the **middle 80 frames**; it does not compress. Only clips over 150 frames were
compressed, at extraction (`MAX_FRAMES`). The real problems were two others:

1. **80 frames are not 80 frames of time.** At 15 fps they span 5.3 s, at
   24 fps 3.3 s: the sign plays faster than at 30.
2. **The crop is centred on the clip, not on the sign.** WLASL clips were
   already trimmed to the sign; ASL Citizen's are not.

### The sign sits in the first half; in the demo it sits anywhere

On the first 197 clips (7 signs), inside the window the model would read:

```
centre du signe : p10 0.43 s   mediane 0.87 s   p90 1.33 s   (fenetre de 2.67 s)
```

Participants start signing right after pressing record and the clip runs on.
In the demo, a user who starts a little late puts the sign at 1.5–2.5 s, a
position training almost never showed.

Teaching the model to recognise a sign anywhere in the window was considered
and dropped: **if both sides centre the window on the sign, they agree by
construction**, and position invariance is no longer needed. What remains of it
is at most a small jitter sized to the detector's error, deferred (§6).

### Finding the sign: movement, not height

The first detector used wrist height above the clip's resting position. It
missed **12–13% of clips**, and the misses were not random:

```
rates par signe : WANT1 11/32, ME 11/31, OHISEE 10/30, BOOK 8/31,
                  ILOVEYOU 6/30, YOU 4/31, WATER 4/30
                                   detectes    rates
hauteur gagnee par le poignet        0.58       0.22
mains presentes                      100%       100%
vitesse max des poignets             0.73       0.78
```

Signs made in front of the chest barely raise the hand but **move it just as
much**. Hand presence is useless here: in this framing the hands stay visible
even at rest. The contact sheets later showed why height was weak and movement
strong — **resting hands are out of shot**, below the frame, and a sign is the
hand coming into view.

`timebase.py` now locates the sign by the **speed of the pose wrists and
elbows** (always present, untouched by hand gating), median-filtered so a
one-frame detection jump does not look like the clip's sharpest gesture,
measured relative to the clip's own stillness (20th percentile), and centred on
the **centre of mass** of that movement rather than between a first and last
active frame, which a single aberrant edge frame would drag.

No ground truth exists for "where the sign is", so four independent checks
(`check_timebase.py`, 510 clips, 17 signs):

```
1. echecs                    hauteur 12.2%  ->  mouvement 0.0%
2. accord avec la hauteur    ecart des centres mediane 74 ms, p90 288 ms
   la ou elle marchait       (> 300 ms : 8.7%)
3. signe retarde de k frames centre decale de k, erreur mediane 0.49 frame,
                             p99 3.67, max 5.98
4. centre dans la fenetre    mediane 1.34 s (milieu 1.33), p10 1.30, p90 1.40
```

and a contact sheet of eight former misses, where the frame chosen as centre
shows the hand mid-gesture every time.

Re-run on the **full target set** (3,089 clips, 100 signs) once extracted, with
the same verdict: misses 12.1% by height against 0.0% by movement; centres
within 78 ms of the height method at the median (p90 289 ms); a delayed sign
moves its centre by the delay to within 0.48 frame at the median (max 9.6);
final centre at 1.35 s for a window midpoint of 1.33 (p10 1.30, p90 1.40).

### The window

`window_indices` returns the 80 indices the model reads, sign at the centre.
Long clips are cropped around the sign; short ones — the median clip is 77
frames, just under the window — are **completed by repeating their first and
last frames**, someone holding still, never zeros: the demo never feeds the
model an empty frame. The same indices apply to every array of the clip, so
confidences and conflict flags stay in step with coordinates.

Both sides import it. `load_sequences` applies it to any timestamped clip; the
demo applies it to its recording. `check_window_parity.py`, on five real clips
including a 56-frame one (padded), a 152-frame one (cropped) and three former
misses: **the demo's path and the training loader give the same 80 x 106 input
to the bit.**

### What changes in the demo

The model card now records, read from the clips themselves by `train_final`:
whether the window is centred (`timebase`), the recording length
(`record_seconds`) and the detector the features came from. The demo applies
them. A model trained on ASL Citizen will make the demo **record 4 s instead of
2.8** — more room to start — centre the window on the sign wherever it falls,
and switch to TensorRT on its own. `--trt` and `--no-trt` override.

The current WLASL card carries none of these fields, and the demo gives it
exactly the input it gave before (checked).

---

## 6. How alike are windows of the same sign?

Centring only has value if it makes clips of the same sign comparable.
`check_consistency.py`, on 707 clips over 23 signs, old centre crop against the
new placement:

```
                                      ancienne   nouvelle
decalage residuel, mediane             500 ms     233 ms
                   p90                 967 ms    1000 ms    (borne de recherche)
bon voisin, bras seuls, autre signeur   20.5%      27.0%    (hasard 4.3%)
distance meme signe / autres signes      0.95       0.89
```

Centring halves typical misalignment and brings same-sign clips measurably
closer. The arm trajectory alone separates signs poorly — expected, since many
signs differ only by handshape at the same location, which this test ignores
and the model's handshape features exist for.

**What centring cannot fix is tempo.** The same sign spans roughly 1 to 3 s
depending on the signer:

```
coefficient de variation par signe : mediane 50%, de 36% a 140%
YES     0.93 a 2.85 s (p10-p90)      CAR       1.13 a 2.33 s
SLEEP   1.02 a 3.38 s                ILOVEYOU  1.00 a 2.06 s
YOU     0.83 a 2.93 s
NIGHT2  0.90 a 3.29 s
```

That span includes bringing the hands into frame and lowering them, so it
overstates the sign itself somewhat, but it is what the model sees. The
training's time-warp augmentation covers only **0.8–1.25x**, against a real
spread of about 3x. About 10% of pairs also stay at the 1 s search limit,
probably repeated-movement signs (`YES` nods) where several alignments fit
equally well; not inspected.

On the **full target set** (3,089 clips, 100 signs) the picture holds:

```
                                      ancienne   nouvelle
decalage residuel, mediane             467 ms     233 ms
bon voisin, bras seuls                  7.1%      11.9%     (hasard 1.0%)
distance meme signe / autres            0.96       0.90

coefficient de variation de la duree par signe : mediane 47%, de 27% a 140%
BED1  1.07 a 3.73 s    DOG1  0.97 a 3.39 s    YES  0.93 a 2.85 s
le plus regulier : HOLD2  1.10 a 1.80 s,  SCHOOL  1.20 a 2.00 s
```

The tempo spread is real across all 100 signs: 2x to 3x between signers,
against a 0.8-1.25x augmentation. What these statistics cannot say is whether
it costs accuracy — ~20 signers per sign already show the model much of that
variety. That is a k-fold question: widened time-warp (about 0.6-1.7x) against
the current one, as one of the first comparisons after the baseline.

**Decision: tempo is left as is for now**, re-examined once the full target
set was extracted (above), and handed to the k-fold. If needed, the first candidate is widening time-warp to about
0.6–1.7x, measurable in the k-fold and invisible to the demo. Normalising tempo
on both sides would also work but erases speed, which is sometimes meaningful
in ASL.

**Correction (2 November): the current time-warp only ever speeds signs up.**
`time_warp` resamples the window it is handed, and every window is now exactly
80 frames. A factor above 1 asks for more than 80 frames, is capped at 80, and
returns the window unchanged. A factor below 1 squeezes the 80 frames into
fewer and pads the end with zeros, which shifts the sign off-centre and shows
the model trailing zeros the demo never produces. The real range is therefore
0.8–1.0x, applied to ~22% of clips (half are warped, 44% of those draw a factor
below 1), and widening the upper bound would change nothing. The same held on
WLASL, so every WLASL number was measured with speed-ups only.

**Fixed before the baseline.** `time_warp` now stretches or squeezes the
window around its centre and keeps 80 frames: slowing pushes the edges out of
the window, speeding up repeats the first and last frames, as `timebase` and
the demo do when a window overruns the clip. Real frames beyond the window
would be better, but that changes the shape of every array in the harness, and
at 0.8x the repeats are at most 10 frames at each end. Checked on the 3,089
target clips: the median motion lasts 36 / 43 / 50 frames at 0.8 / 1.0 /
1.25x, centred at frame 40 in all three, and five rounds of augmentation
produce no zero frame. The range stays 0.8–1.25x; widening it is the later
test.

---

## 7. Models

The WLASL verdicts (LSTM -9.3, 160 units worse) were reached at ~12 training
clips per class; there are now ~24, and 3.5x the pretraining data. Candidates,
by relevance to the identified bottleneck (handshape at the same location):
**ST-GCN**, a graph over the skeleton that can learn hand structure instead of
relying on hand-built features — and the pose baseline of the ASL Citizen
paper itself (59.5% top-1 over all 2,731 signs, against 63.1% for video; a
later pose transformer reports 75.1%); a **TCN**, likely much faster to train
on CPU; a transformer only if capacity is shown to pay. LSTM and state-space
models are ruled out (LSTM measured worse; 80 frames is not a long sequence).

All three are built (`--cell gru|tcn|stgcn`, in `train_kfold` and
`train_final`) and read exactly the same input columns, so gating, mirror, the
centred window and TTA are untouched.

- *GRU*: projection 48, two bidirectional GRU layers of 96 units. Its readout
  had to change on ASL Citizen: reading the final state never left chance;
  `--readout attention` weighs the frames (§8).
- *TCN*: projection 48, five residual blocks of two convolutions (kernel 3,
  96 channels, dilations 1-16), **non-causal** — the literature's TCN only
  sees the past because it is built for streaming; we classify a whole window,
  like the bidirectional GRU. Receptive field 125 frames. Its options
  (attention readout, BatchNorm, spatial dropout, kernel, width) are in §9;
  each is off by default, so the TCN above is unchanged, weight for weight.
- *ST-GCN*: the raw columns are regrouped into 51 joints x (x, y, presence,
  confidence) — 21 per hand in COCO-WholeBody order, 9 pose points, the pose
  wrists linked to the hand wrists. Five blocks of one graph step (each joint
  takes in its neighbours along the bones, with a learnable weight per bone)
  then one temporal step (kernel 9, per joint), residual, time downsampled
  80 -> 40 -> 20. One neighbourhood partition instead of the original's three.
  The derived columns (rich, finger curls) go through a small side stream, so
  running with and without them tells whether the graph learns hand shape by
  itself. **Never trained yet.**

`bench_cells.py`, on the 3,089 target clips, one epoch over 80% of them, on
the CPU (measured while the pretraining extraction ran, so inflated, but alike
for all three):

```
           parametres    s / epoch
gru          296 228        10.8
tcn          301 124         4.2     2.6x plus rapide
stgcn        391 473        82.9     7.7x plus lent
```

All three save and reload with identical predictions (the ST-GCN's custom
layers are registered with Keras, and the demo imports them), and the
pretraining transfer lines up layer for layer.

Why a "parallel" architecture is slower: parallel does not mean less work.
Counted layer by layer on the real models, multiplications per clip are 21 M
for the GRU, 22 M for the TCN and **739 M for the ST-GCN (36x)**. Both
convolutional models do get through their work faster than the GRU — 2.8x for
the TCN, 4.7x for the ST-GCN — because no frame waits for the previous one.
But the ST-GCN carries 96 numbers for each of 51 joints (4,896 per frame,
against the GRU's 192) and reapplies each temporal weight at every joint and
every frame: 36x the work at 4.7x the rate is 7.7x slower. Parameters measure
the size of the recipe, multiplications how often it is applied. On the GPU
(§10) the ST-GCN becomes affordable.

---

## 8. The GRU baseline: 0.8610 (3 November)

Signer-disjoint 5-fold over 52 signers with the existing harness, on the
12,270 extracted clips (3,089 target, 9,181 scaffold, all re-read, no
failures). WLASL's best recipe plus the fixes below:

```
python src/core/train_kfold.py --folds 5 --epochs 150 --vocab 100 \
  --variants all --features-dir data/asl_citizen/features \
  --rich --confidence --handshape-full \
  --gate 0.7 --gate-inside 0.5 --gate-conflict \
  --project 48 --label-smoothing 0.1 --tta --seeds 1 --readout attention \
  --pretrain-vocab 400 --pretrain-epochs 60 --pretrain-variants all \
  --gate-pretrain --pretrain-cache data/asl_citizen/pretrain_cache/baseline \
  --deterministic --tag _baseline
```

(through `scripts/wsl_gpu.sh`). One seed: it is a reference to compare
against, not the shipped model.

**Result: 0.8610 ± 0.0073** (`models/aslc_kfold_baseline.json`, the reference
for every comparison):

```
fold        1       2       3       4       5
plateau   0.8539  0.8730  0.8588  0.8651  0.8541
```

For scale, WLASL's best was 0.762 on 50 signs. Scored on epochs 11-40, the
window later runs use (§11), the same run gives 0.8626.

### What it took: four fixes

**1. Hand-shape ratios reached 900,000.** The first launch learned nothing:
pretraining accuracy stayed at chance for 400 signs (e1 0.002, e60 0.003),
where WLASL's reached 0.5-0.75. `add_handshape` divides every hand-shape
distance by the on-screen palm width (index base to little-finger base),
floored at 1e-6. That width collapses when the palm turns sideways, and is
exactly 0 when RTMPose folds an unseen hand onto a single point. Measured on
ASL Citizen:

```
                      frames > 10   clips touches   max
brut (echafaudage)       27.6%          96%       2 879 243
filtre (cible, demo)      1.9%          25%         944 661
apres correction, filtre  0.001%                         19.6
```

So the target and the live demo were affected too, not only the scaffold.
Fix: the divisor is at least half the wrist-to-middle-finger-base length,
which stays visible when the palm turns, and at least 0.02 (10% of a median
palm). A frontal palm is ~0.7 times that length, so frontal hands keep exactly
their values; the 23% of filtered hands that change are those turned past
~45°. The demo imports the same function, so it follows. The WLASL model it
can still load was trained on the old values, which differ only on those
turned hands.

**2. The scaffold is filtered like the target, and that crashed.** Unfiltered,
the scaffold still has 6% of frames above 10: folded hands with junk
coordinates that the filter would remove. The scaffold is therefore filtered
like the target (`--gate-pretrain`). On WLASL this was off "because every
measurement was made that way"; there is no ASL Citizen measurement to stay
comparable with, the target and demo are filtered, and on WLASL filtering made
the scaffold task far more learnable (e120 0.745 against 0.527). The flag had
never run in `train_kfold`: the scaffold was loaded without its hand
confidences and the filter crashed on them. Fixed.

**3. The GRU's last-state readout never left chance.** With both fixes,
pretraining stayed at 0.003 (e60) and fold 1 fine-tuned to 0.0178, the
frequency of one class: a constant prediction. The weights were finite. The
GRU answered from its final state in each direction, i.e. at the window's
edges, after ~18 frames of the signer at rest (the window is centred on a
~43-frame sign and padded by repeating edge frames). At initialisation a GRU
keeps about half its memory per frame, so almost nothing of the sign survives
to the readout. On WLASL the clips were trimmed to the sign and zero padding
was masked, so the last state sat right after the sign. Diagnosis on a random
80/20 split (not signer-disjoint, so inflated), same seeds:

```
                                   100 signes          400 signes
                                   e5     e10    e20    e5     e15
centroide le plus proche           0.165 (sans entrainement)
TCN (moyenne sur le temps)         0.328  0.589  0.718
GRU, etat final                    0.008  0.008  0.380  0.004  0.004
GRU, moyenne des frames            0.322  0.571  0.714  0.309  0.603
GRU, attention                     0.513  0.681  0.790  0.360  0.618
```

`--readout attention` (`AttentionPool`: a small network scores each frame, a
softmax over time turns the scores into weights) lets the model weigh the sign
above the rest frames. +12k parameters (308,644 against 296,228), and it saves
and reloads with identical predictions. The default stays `last` so WLASL runs
reproduce.

**4. Time-warp only ever sped signs up**, and frame dropout made zero frames
that broke the GPU's fast GRU kernel. Both fixed before the baseline (§6,
§10).

### Fold 5 collapsed and was redone

At seed 0 (500 for that fold) its pretraining never left chance (e60 0.004,
against 0.72-0.76 elsewhere), and fine-tuning from it predicted one class for
all 617 clips (0.0081). Traced layer by layer on the cached weights: the
projection's output was 2.5x larger than fold 1's, the first GRU's spread
across clips fell to 0.105 (fold 1: 0.49), the second's to 0.005, 62% of its
outputs at ±1, the attention stayed uniform and 0% of the head's ReLUs were
active. The GRUs had settled where they ignore their input. The inputs are
clean (8 clips of 12,270 exceed 10) but not on a common scale: column spreads
run from 0.008 to 1.25. Gradient clipping (`--clipnorm 1.0`) does not prevent
it, tested on that seed: with Adam the step size barely depends on the
gradient's size. Fold 5 was rerun with the same recipe at `--seed 1`
(partition unchanged, it does not depend on the seed): 0.8541. Merged with
`merge_kfold.py --replace-fold 5`, which records the swap in the file; the
collapsed run is kept as `aslc_kfold_baseline_run1.json`. A collapse is now
caught by `train_final`'s guard at pretraining epoch 10. The TCN (§9) never
collapsed, at that seed included.

### LayerNorm: fixes the collapse, costs accuracy — rejected

`--layer-norm` puts a LayerNormalization between the projection and the GRU,
rescaling its input at every frame. On the seed that collapsed, 10
pretraining epochs:

```
                          e1      e3      e6      e10
fold 5, sans LayerNorm   0.003   0.004   0.004   0.004
fold 5, avec LayerNorm   0.003   0.054   0.178   0.333
fold 1, avec LayerNorm   0.004   0.061   0.198   0.365
```

One k-fold then tested it together with phonology heads in fine-tuning only
(`--layer-norm --phonology 0.3`, 40 epochs), to save a run (5 November):

```
epochs 11-40     1        2        3        4        5      mean
baseline       0.8668   0.8713   0.8578   0.8624   0.8546   0.8626
LN + phonology 0.8668   0.8606   0.8433   0.8592   0.8399   0.8539
gap           -0.0001  -0.0107  -0.0145  -0.0031  -0.0147  -0.0086
```

0/5 folds up, paired t = -2.88, p = 0.045. The pretraining differed only by
LayerNorm, and it ended lower on four folds (e60: -0.009 to -0.031). The
fine-tuning gap follows it fold by fold (correlation 0.98), and fold 1, where
both pretrainings reached 0.723, shows no gap at all. So the loss is
LayerNorm's, through a slower pretraining; phonology looked neutral there,
used in fine-tuning only. LayerNorm stays off, in `train_final.py` and in the
live model. Both pretrainings were still rising at e60, by 2 to 3 points
between epochs 41-50 and 51-60, which pointed to longer pretraining (§9).

---

## 9. The TCN recipe: 0.8864 (16 November)

The path from the baseline, fold 1 for screening, 5 folds for verdicts. All
scores are the mean over fine-tuning epochs 11-40 (§11).

```
                                                     fold 1     5 folds
GRU baseline                                         0.8668     0.8626
GRU + phonologie (2 phases) + 150 epochs pre.        0.8805
TCN, recette de base                                 0.8628     0.8627
TCN + phonologie + 150 epochs                        0.8771
TCN combine (+ attention, BatchNorm, spatial
  dropout, 128 canaux), phonologie, 150 epochs       0.9000
  meme recette, lot 128 + XLA (§10)                  0.9051     0.8864
```

### Phonology in both phases, and 150 pretraining epochs

Besides the sign, the model predicts its six ASL-LEX components (§12 for the
source): handshape (44 values across target and scaffold), selected fingers
(9), flexion (7), major location (5), movement (8), sign type (5). Extra
output heads, weight 0.3 in the loss, masked where ASL-LEX has no value, used
in training only. The components are shared across signs (`open_b` alone
covers 13 of the 100), so each head learns from many more examples than any
single sign. `--phonology-pretrain` trains the heads during pretraining too
and carries them into fine-tuning; in fine-tuning only (§8) they did nothing.

On the GRU, fold 1, both phases and 150 pretraining epochs: **0.8805 against
0.8668**. Pretraining reached 0.746 at e60 against the baseline's 0.723, and
flattened near 150 (0.805; last gains +0.5, +0.5, +0.3 per 10 epochs).
Fine-tuning starts higher (0.835 at epoch 2 against 0.767) and is flat from
epoch ~10. On the combined TCN, 200 pretraining epochs instead of 150 added
2.6 points of pretraining accuracy and 0.1 of test: 150 stays.

### TCN against GRU: a tie; their ensemble: dropped

Base recipe, 5 folds: **0.8627 against 0.8626** (p = 0.98). The TCN's
pretraining was far higher (0.823 at e60 against 0.723) for nothing: see §11.
Averaging GRU and TCN predictions gained +1.2 on all 5 folds (each is right
alone on ~4% of clips), but once both had phonology and 150 epochs, fold 1's
GRU alone (0.901 on averaged predictions) matched the pair. Dropped.

### TCN variants, one at a time

Fold 1, 60 pretraining epochs, no phonology:

```
                                  e11-40   pre-entr. e60
TCN                               0.8628       0.823
--readout attention               0.8717       0.839     ~36 des 80 frames sont du repos
--tcn-norm batch                  0.8640       0.633     apprend lentement, monte encore
--tcn-spatial-dropout             0.8790       0.731     des canaux entiers retires
--tcn-kernel 5                    0.8630       0.859
--units 128                       0.8680       0.821     plus vite, pas plus haut
```

The lower pretraining of spatial dropout and BatchNorm is regularisation, not
a defect: spatial dropout has the lowest pretraining of the winners and the
best test score.

**Combined** (attention + BatchNorm + spatial dropout + 128 channels,
phonology in both phases, 150 epochs), fold 1: **0.9000**. Dropout placed
before a BatchNorm makes it learn the noisier training scale ("variance
shift", Li et al., CVPR 2019). `--bn-recalibration` scores each epoch a second
time with BatchNorm re-measured without dropout: +0.1, negligible, so the
alternative (`--tcn-dropout-end`, a single dropout after the last block) was
not run.

### Result: 0.8864 ± 0.0102 on 5 folds, +2.4 on every fold

```
epochs 11-40     1        2        3        4        5      mean
baseline       0.8668   0.8713   0.8578   0.8624   0.8546   0.8626
TCN recipe     0.9028   0.8902   0.8806   0.8859   0.8723   0.8864
gap           +0.0360  +0.0189  +0.0228  +0.0235  +0.0177  +0.0238
```

Paired t = 7.34, p = 0.002; 31 minutes for the five folds. Errors on unseen
signers drop from 13.7% to 11.4%. Fine-tuning is flat at the end (+0.001
within the window). Results: `models/aslc_kfold_tcn_xla_5fold.json`.

```
python src/core/train_kfold.py --folds 5 --epochs 40 --vocab 100 \
  --variants all --features-dir data/asl_citizen/features \
  --rich --confidence --handshape-full \
  --gate 0.7 --gate-inside 0.5 --gate-conflict \
  --project 48 --label-smoothing 0.1 --tta --seeds 1 \
  --cell tcn --readout attention --tcn-norm batch --tcn-spatial-dropout \
  --units 128 --phonology 0.3 --phonology-pretrain \
  --pretrain-vocab 400 --pretrain-epochs 150 --pretrain-variants all \
  --gate-pretrain --pretrain-batch 128 --pretrain-scaling adam --fast-prep \
  --pretrain-cache data/asl_citizen/pretrain_cache/tcn_xla_b128 --tag _tcn_xla_5fold
```

(no `--deterministic`: it would turn XLA off, §10).

### Each piece, checked on 5 folds (17 November)

On fold 1, BatchNorm (+0.1), 128 channels (+0.5) and kernel 5 (+0.0) were all
within one fold's noise (§11). Rerun on 5 folds, each against the recipe:

```
epochs 11-40, 5 folds     moyenne    ecart    folds en hausse     p
recette                   0.8864
sans BatchNorm            0.8811    -0.0053        0/5           0.11
96 canaux au lieu de 128  0.8802    -0.0061        0/5           0.06
+ noyau 5                 0.8792    -0.0072        2/5           0.20
```

Removing BatchNorm or the 128 channels loses on every fold, by about half a
point each: too little for 5 folds to prove alone, but every fold agrees. Both
stay. Kernel 5 loses and costs 58% more parameters: rejected. The recipe
stands as above.

---

## 10. Training on the GPU, and making it fast

A fold of the final recipe takes ~7 minutes; the same work took ~86 minutes on
11 November and would have taken days on the CPU.

### The GPU through WSL2

TensorFlow has no GPU support on native Windows since 2.11; it does under
WSL2. `scripts/wsl_gpu.sh` runs any script of the project there (RTX 4060).
Only training goes through it; extraction (onnxruntime + TensorRT) and the
demo stay on Windows and read the saved models as is (same TensorFlow and
Keras versions on both sides). `bench_cells.py` on the GPU:

```
                  s / epoch   --deterministic    CPU (§7)
gru (masque)        108.5          98.2              10.8
gru sans masque       4.0           4.2              —
tcn                  17.3           3.9               4.2
stgcn                35.9          14.4              82.9
```

**The GRU's mask made it ~10x slower on the GPU than on the CPU — fixed.** Its
`Masking` layer tells it to skip all-zero frames (WLASL's padding). On ASL
Citizen real clips contain none: 0 of 981,600 windowed frames, target and
scaffold, since the window pads by repeating edge frames and the detector
found a person in every frame. Zero frames came only from augmentation: frame
dropout (10% of frames in 30% of clips) and time-warp's trailing padding (§6).
cuDNN, the GPU's fast RNN kernel, runs the whole recurrence in one call but
only accepts masks that hide the *end* of a sequence, so the first GPU run
crashed. Forcing the generic kernel (`use_cudnn=False`) runs the 80 frames as
80 small steps, each waiting on the previous one: 98 s/epoch. Fix:
augmentation no longer makes zero frames. Time-warp was repaired (§6), and a
dropped frame now repeats the previous one — what `timebase` does to a webcam
that skips images — instead of turning to zero. The mask stays (WLASL needs
it) but hides nothing, so cuDNN accepts it: back to ~4 s/epoch. A zero frame
in mid-clip would now crash the GPU run instead of passing silently, which
serves as a check.

### Caches

**Pretraining cache** (`--pretrain-cache`): keeps each fold's pretraining
(weights, phonology heads, and the random generator's state at its end) and
reloads it in later runs, so an experiment that only changes fine-tuning skips
pretraining and starts from the same weights. Pretraining cannot be shared
*across folds*: every signer is a test signer in one fold, and signs 1-100 are
part of the scaffold. A cache made with different options, different
pretraining code (augmentation, features, models) or another device is
refused, naming what differs; options that act only after pretraining
(`--epochs`, `--tta`, `--seeds`...) reuse it.

**Load cache** (`cached_sequences`): the 12,270 files were read twice per run
(~2.5 min, 22 s of it just reading through WSL's bridge to the Windows disk).
They are now loaded once and kept as one 435 MB file in
`data/asl_citizen/load_cache/` (git-ignored), keyed by the file names, sizes
and dates, the options and the loading code: **4 s** afterwards
(`--no-load-cache` to bypass).

### Where the time went, and what fixed it

The GPU was 38% busy and the training process used one CPU core of 28: each
step is many small GPU operations, and the time went into the gaps between
them. Synthetic timing of one pretraining epoch (combined TCN, 9,800 clips):

```
                                      lot 32    lot 128
deterministe (avant)                  25.8 s     6.8 s
non deterministe, sans XLA            22.7 s     6.1 s
non deterministe, XLA                  2.3 s     1.5 s
```

- **XLA** merges those small operations. Keras refuses it whenever op
  determinism is on, so it comes only without `--deterministic`; runs are
  then not bit-reproducible, and two identical runs differ by up to ~1 point
  on one fold (§11). Mixed precision and `steps_per_execution` gained nothing:
  the GPU's arithmetic was never the bottleneck.
- **Batch 128 for pretraining** (`--pretrain-batch 128`), with Adam adapted
  by the square-root rule of Malladi et al. (NeurIPS 2022, def. 5.2):
  learning rate x sqrt(k), 1-beta1 and 1-beta2 x k, epsilon / sqrt(k), with
  k = 128/32, i.e. 2e-3, 0.6, 0.996, 5e-8 (`--pretrain-scaling adam`).
  Scaling only the learning rate did worse (0.8855 against 0.8925 on fold 1,
  deterministic). The linear rule of Goyal et al. is for SGD. Batch 256 would
  put beta1 at 0.2, near the rule's limit: not tried. Fine-tuning stays at
  batch 32.
- **Data preparation** (`--fast-prep`): each epoch is split into 16 chunks
  prepared on several threads, with an augmentation without the per-clip
  Python loop (`augment_batch_fast`, checked equal to `time_warp` and to the
  frame dropout on fixed draws; same laws, other random numbers). 3.8 s ->
  1.0 s per epoch, hidden behind the GPU (0.0 s of waiting). The threads still
  cost training ~0.5 s per epoch through Python's lock (1.88 s against 1.36 s
  without them); worker processes would recover it but fork a TensorFlow
  process: not done.
- **Prefetch**: each epoch is prepared while the GPU trains the previous one
  (`--no-prefetch` restores the old order; checked bit-identical).
- `--bn-recalibration` (§9) is opt-in: it nearly doubled fine-tuning.

`train_kfold` prints pretraining accuracy every 10 epochs, then the time per
epoch, the time spent waiting for data and the training time alone.

```
fold, recette finale            11 novembre     16 novembre
chargement                        ~2.5 min          4 s
pre-entrainement (150 epochs)     ~75 min         ~5.5 min
affinage (40 epochs)              ~9 min          ~1.5 min
```

**Memory**: a run takes ~6.6 GB in WSL. WSL's limit was raised to 12 GB
(`%USERPROFILE%\.wslconfig`, `memory=12GB`), enough for one run with room to
spare. Two at once fitted earlier (5.6 GB each) and no longer do.

---

## 11. How experiments are measured

**Signer-disjoint 5-fold.** The 52 signers are split into 5 groups of similar
clip counts; each fold tests on signers never seen in training, pretraining
included. The partition does not depend on the seed.

**The score is a plateau, not a best epoch.** Mean test accuracy over
fine-tuning epochs 11-40 of a 40-epoch budget. Fine-tuning reaches its plateau
within ~20 epochs, since pretraining (ranks 1-400) already covers the target
signs: the baseline scores 0.8626 on epochs 11-40 and 0.8628 on 61-90. The
best single epoch runs 2-4 points above the plateau: the maximum of noisy
epochs (epoch-to-epoch spread ~0.9 point), chosen by looking at the test fold,
so not a level the model holds.

**Comparisons are paired, fold by fold** (`compare_kfold.py reference
experiment`): each fold has its own test signers, and the spread between folds
(2 points) is larger than most effects. The reference is scored on the
experiment's own window, read from its saved curves, and the tool warns when a
curve is still rising at the end of that window.

**One fold decides nothing under ~1 point.** The same recipe scored 0.8925 and
0.9051 on fold 1, changing only XLA and the augmentation's random draws. Fold 1
is also the friendliest fold (0.903 against a 5-fold mean of 0.886). Fold 1
screens; 5 folds decide.

**Pretraining accuracy does not compare across architectures.** It measures
fit to the 400 scaffold signs: the TCN pretrains 10 points higher than the GRU
and ties it on the test, kernel 5 pretrains highest of the variants and gains
nothing. Within one architecture it does track fine-tuning: LayerNorm's
pretraining deficit predicted its test deficit fold by fold (correlation
0.98, §8).

**Bookkeeping.** `--only-fold` reruns some folds, `merge_kfold.py` joins the
pieces and refuses to merge different configurations (`--replace-fold N`
swaps a failed fold and records it). Result files keep the per-epoch curves,
the pretraining curves and every option; only the final recipe's file keeps
the per-clip predictions in git (95% of the size), full copies of the others
stay in `models/full/`.

---

## 12. Open items

In order of priority:

1. **Ship the TCN recipe.** `train_final.py` defaults to the GRU baseline's
   recipe (`--readout attention`, `--gate-pretrain`, 60 pretraining epochs),
   cites `models/aslc_kfold_baseline.json`, refuses to start when its options
   differ from the recipe the cited k-fold measured, and stops after 10
   pretraining epochs if accuracy is still under 5x chance. It must learn the
   TCN options, the phonology heads, the pretraining batch and Adam scaling and
   the fast preparation, then cite `models/aslc_kfold_tcn_xla_5fold.json`. The
   demo and the backend run a 1-seed GRU model for live testing (3 November);
   they load the saved `.keras` files, which carry their architecture (save
   and reload checked for the TCN, §7), but have not run a TCN yet. For
   testing, `src/asl_citizen/make_sign_sheet.py` writes a local page of the
   target signs, two ASL Citizen clips each, never to be shared (license).
2. **Live testing remains the binding risk**, as in the WLASL phase. Every
   number here comes from held-out ASL Citizen signers.
3. **The confusion matrix**, now possible from the final recipe's per-clip
   predictions: the seven pointing signs predicted in §3, and whether numbered
   twins (`EAT1`/`EAT2`) should be **displayed merged** ("EAT"; not built).
4. **Data questions left to a k-fold**:
   - the hand threshold: gating at 0.7 keeps 30% of hand slots here against
     43% on WLASL. Plausible — resting hands are out of frame and most signs
     are one-handed — but 0.7 was tuned on WLASL;
   - clips under 24 fps (2.6%, P49 alone 1,205 of them) are resampled with
     duplicate frames, as a slow webcam would be in the demo: with and without
     them;
   - tempo: time-warp covers 0.8-1.25x against a real 2-3x spread between
     signers; widening to about 0.6-1.7x (§6).
5. **The signers' brief (`RECORDING.md`) is obsolete**: it describes the 50
   WLASL words and WLASL variants. It needs the new vocabulary and new
   reference clips — and the license forbids distributing ASL Citizen videos,
   so sending signers clips from the dataset is not an option. ASL-LEX
   publishes its own video for each sign under CC BY-NC, which may serve; not
   yet checked.
6. **Later, if accuracy needs more**:
   - the ST-GCN (§7), affordable now that a fold takes minutes; then learned
     graph connections (an adaptive graph), since handshapes such as F or O
     hinge on the thumb tip touching the index tip, which the skeleton does
     not connect;
   - skeleton streams (one model on joints, one on bones, one on motion,
     votes averaged), the standard skeleton recipe — not the WLASL velocity
     test (-1.4), which added columns to a single model;
   - pretraining on all of ASL Citizen (2,731 signs, 83,399 clips, against 400
     signs): breadth is the one lever that clearly paid on WLASL (+3.4); about
     a day and a half of extraction with two processes on the GPU;
   - the official splits, to compare with the published baseline, which
     covers all 2,731 signs, not 100.

   Not pursued: self-supervised masked-pose pretraining (meant for scarce
   labels; there are 83,000 labelled clips), and what was already tested or
   set aside — face, 3D, a dedicated hand model. The phonology source cited
   in planning (Kezar, Thomason & Sehyr, *Improving Sign Recognition with
   Phonology*, 2023) was quoted from memory and not checked.
7. **Environment fragility**: any `pip install` depending on `onnxruntime`
   reinstalls the CPU build over the GPU one. The repair is in §4; the
   extractor refuses to run in that state, the demo only prints a warning.
