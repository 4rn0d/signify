# Signify — ASL Citizen phase

Same task as the WLASL phase (`WLASL.md`), recognising whole signed words from
a webcam, rebuilt on a different dataset. The WLASL notebook stays the record
of how the pipeline was built and why; this one records what changed when the
data did.

**Current state (3 November):** the GRU baseline is measured, **0.8610 ±
0.0073** on 100 signs, signer-disjoint 5-fold (§7 item 3b), against 0.762 on
50 signs for WLASL's best. All 12,270 clips are extracted (3,089 target, 9,181
pretraining). Getting there took five fixes, all in §7 3b: time-warp only ever
sped signs up, hand-shape ratios reached 900,000 on turned or folded hands,
`--gate-pretrain` crashed, the GRU's last-state readout never left chance (now
`--readout attention`), and one fold collapsed (redone at another seed;
`--layer-norm` prevents it on that seed). Next: a k-fold with `--layer-norm`
against the baseline, then the four queued improvements (§7 3a).
The shipped model is still the WLASL one (0.7623 expected on 50 signs), and
the demo behaves exactly as before until it is replaced.

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

## 7. Open items

In rough order of when they block something:

1. ~~Re-run the checks at scale.~~ Done on all 3,089 target clips: 0% misses
   confirmed; the tempo spread is confirmed too, and whether it matters goes to
   the k-fold (§6).
2. ~~Select the vocabulary by ASL-LEX rank in the training harness.~~ Done:
   when clips carry a rank, "the top N" means the N best-ranked (`top_glosses`);
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
   as one exists, `--card` to override. Smoke-tested end to end on the first 60
   signs: rank selection, 80 x 155 input, pretraining, card, demo loading.
2b. **Re-measure the hand threshold on ASL Citizen.** Gating at 0.7 keeps 30% of
   hand slots here against 43% on WLASL. Plausible — resting hands are out of
   frame and most signs are one-handed — but 0.7 was tuned on WLASL.
3. **Architecture.** The WLASL verdicts (LSTM -9.3, 160 units worse) were
   reached at ~12 training clips per class; there are now ~24, and 3.5x the
   pretraining data. Candidates, by relevance to the identified bottleneck
   (handshape at the same location): **ST-GCN**, a graph over the skeleton
   that can learn hand structure instead of relying on hand-built features —
   and the pose baseline of the ASL Citizen paper itself (59.5% top-1 over all
   2,731 signs, against 63.1% for video; a later pose transformer reports
   75.1%); a **TCN**, likely much faster to train on CPU; a transformer only if
   capacity is shown to pay. LSTM and state-space models are ruled out (LSTM
   measured worse; 80 frames is not a long sequence).

   **Built** as `--cell tcn` and `--cell stgcn`, in `train_kfold` and
   `train_final`. Both read exactly the GRU's input columns, so gating, mirror,
   the centred window and TTA are untouched.
   - *TCN*: projection 48, five residual blocks of two convolutions (kernel 3,
     96 channels, dilations 1-16), **non-causal** — the literature's TCN only
     sees the past because it is built for streaming; we classify a whole
     window, like the bidirectional GRU. Receptive field 125 frames.
   - *ST-GCN*: the raw columns are regrouped into 51 joints x (x, y, presence,
     confidence) — 21 per hand in COCO-WholeBody order, 9 pose points, the pose
     wrists linked to the hand wrists. Five blocks of one graph step (each joint
     takes in its neighbours along the bones, with a learnable weight per bone)
     then one temporal step (kernel 9, per joint), residual, time downsampled
     80 -> 40 -> 20. One neighbourhood partition instead of the original's
     three. The derived columns (rich, finger curls) go through a small side
     stream, so running with and without them tells whether the graph learns
     hand shape by itself.

   `bench_cells.py`, on the 3,089 target clips, one epoch over 80% of them
   (measured while the pretraining extraction ran, so inflated, but alike for
   all three):

   ```
              parametres    s / epoch
   gru          296 228        10.8
   tcn          301 124         4.2     2.6x plus rapide
   stgcn        391 473        82.9     7.7x plus lent
   ```

   All three save and reload with identical predictions (the ST-GCN's custom
   layers are registered with Keras, and the demo imports them), and the
   pretraining transfer lines up layer for layer. The TCN is cheaper to
   measure than the GRU baseline. The ST-GCN is not affordable as is: about
   14 h per fold and seed by extrapolation, ~3 days for a 5-fold on CPU.

   Why a "parallel" architecture is slower: parallel does not mean less work.
   Counted layer by layer on the real models, multiplications per clip are
   21 M for the GRU, 22 M for the TCN and **739 M for the ST-GCN (36x)**. Both
   convolutional models do get through their work faster than the GRU — 2.8x
   for the TCN, 4.7x for the ST-GCN — because no frame waits for the previous
   one. But the ST-GCN carries 96 numbers for each of 51 joints (4,896 per
   frame, against the GRU's 192) and reapplies each temporal weight at every
   joint and every frame: 36x the work at 4.7x the rate is 7.7x slower.
   Parameters measure the size of the recipe, multiplications how often it is
   applied. A dozen CPU cores turn parallelism into 4.7x; a GPU's thousands
   would turn it into far more.
   Options: slim it further and pretrain 60 epochs; a first signal on one or
   two folds without pretraining; or train on the GPU through WSL2, since
   TensorFlow has no GPU support on native Windows — which would change the
   cost of every architecture.

   **On the GPU** (WSL2, RTX 4060, `scripts/wsl_gpu.sh`), same benchmark,
   extraction finished:

   ```
                     s / epoch   --deterministic    CPU (ci-dessus)
   gru (masque)        108.5          98.2              10.8
   gru sans masque       4.0           4.2              —
   tcn                  17.3           3.9               4.2
   stgcn                35.9          14.4              82.9
   ```

   The deterministic column is the one that counts (`--deterministic` keeps
   `--only-fold` exact). The ST-GCN becomes affordable: 5.8x faster than on
   CPU, so the 14 h per fold and seed above scale to about 2.5 h. The TCN gains
   nothing. The non-deterministic TCN figure (17.3) is not explained.

   **The GRU's mask made it ~10x slower on the GPU than on the CPU — fixed.**
   Its `Masking` layer tells it to skip all-zero frames (WLASL's padding). On
   ASL Citizen real clips contain none: 0 of 981,600 windowed frames, target
   and scaffold, since the window pads by repeating edge frames and the
   detector found a person in every frame. Zero frames came only from
   augmentation: frame dropout (10% of frames in 30% of clips) and time-warp's
   trailing padding (§6). cuDNN, the GPU's fast RNN kernel, runs the whole
   recurrence in one call but only accepts masks that hide the *end* of a
   sequence, so the first GPU run crashed. Forcing the generic kernel
   (`use_cudnn=False`) runs the 80 frames as 80 small steps, each waiting on
   the previous one: 98 s/epoch.

   Fix: augmentation no longer makes zero frames. Time-warp was repaired (§6),
   and a dropped frame now repeats the previous one — what `timebase` does to a
   webcam that skips images — instead of turning to zero. The mask stays (WLASL
   needs it) but hides nothing, so cuDNN accepts it: back to ~4 s/epoch, with
   no change to what the GRU sees in real clips. A zero frame in mid-clip would
   now crash the GPU run instead of passing silently, which serves as a check.
   `bench_cells.py` checks reloading on real clips, which is what the demo
   feeds the model.
3a. **Queued until the baseline exists — decided, not to start before.** The
   baseline now exists (3b); the `--layer-norm` k-fold comes first, since it
   decides the model the four below are built on. Four
   improvements, in the order they should be tried, each to be measured against
   the GRU baseline on the same folds:
   1. **Phonology as auxiliary targets.** Besides the sign, the model predicts
      its components from ASL-LEX — handshape (30 values among the 100 target
      signs), selected fingers (9), flexion (7), major location (5), movement
      (5), one or two hands (5) — all filled for 100/100 target signs (flexion
      94/100) and available for the whole scaffold. Extra output heads used in
      training only. It targets the measured weakness directly (confused pairs
      differ by handshape at the same location), and the components are shared
      across signs: `open_b` alone covers 13 of the 100, so its head learns from
      13x the examples of any single sign. No new data; works with any
      architecture. The ASL-LEX team reported gains on WLASL with this
      (Kezar, Thomason & Sehyr, *Improving Sign Recognition with Phonology*,
      2023) — cited from memory, to check before relying on it.
   2. **Ensemble across architectures and input streams.** Seeds of one model
      make similar mistakes, which is why 3 -> 5 seeds bought only ~+1 point.
      GRU, TCN and ST-GCN see the data differently. The standard skeleton
      recipe adds streams: one model on joints, one on bones (the vector between
      two connected joints), one on motion, votes averaged. Not the same as the
      WLASL velocity test (-1.4), which added columns to a single model.
   3. **Learned graph connections for the ST-GCN.** Bones only link neighbours,
      but handshapes such as F or O hinge on the thumb tip touching the index
      tip, which the skeleton does not connect. An adaptive graph learns such
      edges. A refinement for after the first ST-GCN result.
   4. **Pretraining on all of ASL Citizen** (2,731 signs, 83,399 clips, against
      400 signs). Breadth is the one lever that clearly paid (+3.4). About a day
      and a half of extraction with two processes on the GPU.

   Not pursued: self-supervised masked-pose pretraining (meant for scarce
   labels; there are 83,000 labelled clips), and what was already tested or set
   aside — face, 3D, a dedicated hand model.

3b. **First k-fold on ASL Citizen.** Signer-disjoint over 52 signers with the
   existing harness. Extraction is complete: 12,270 clips (3,089 target, 9,181
   scaffold), all re-read, no failures.

   **First launch (3 November), stopped after one fold's pretraining: it
   learned nothing.** Pretraining accuracy stayed at chance for 400 signs
   (e1 0.002, e60 0.003), where WLASL's reached 0.5-0.75. Cause:
   `add_handshape` divides every hand-shape distance by the on-screen palm
   width (index base to little-finger base), floored at 1e-6. That width
   collapses when the palm turns sideways, and is exactly 0 when RTMPose folds
   an unseen hand onto a single point. Measured on ASL Citizen:

   ```
                         frames > 10   clips touches   max
   brut (echafaudage)       27.6%          96%       2 879 243
   filtre (cible, demo)      1.9%          25%         944 661
   apres correction, filtre  0.001%                         19.6
   ```

   So the target and the live demo were affected too, not only the scaffold.
   Fix: the divisor is at least half the wrist-to-middle-finger-base length,
   which stays visible when the palm turns, and at least 0.02 (10% of a median
   palm). A frontal palm is ~0.7 times that length, so frontal hands keep
   exactly their values; the 23% of filtered hands that change are those
   turned past ~45°. The demo imports the same function, so it follows. The
   WLASL model it can still load was trained on the old values, which differ
   only on those turned hands.

   Unfiltered, the scaffold still has 6% of frames above 10: folded hands with
   junk coordinates that the filter would remove. The scaffold is therefore
   filtered like the target (`--gate-pretrain`). On WLASL this was off "because
   every measurement was made that way"; there is no ASL Citizen measurement to
   stay comparable with, the target and demo are filtered, and on WLASL
   filtering made the scaffold task far more learnable (e120 0.745 against
   0.527). The flag had never run in `train_kfold`: the scaffold was loaded
   without its hand confidences and the filter crashed on them. Fixed.

   **Second launch: still at chance — the GRU's readout.** With both fixes,
   pretraining stayed at 0.003 (e60) and fold 1 fine-tuned to 0.0178, the
   frequency of one class: a constant prediction. The weights were finite.
   The GRU answered from its final state in each direction, i.e. at the
   window's edges, after ~18 frames of the signer at rest (the window is
   centred on a ~43-frame sign and padded by repeating edge frames). At
   initialisation a GRU keeps about half its memory per frame, so almost
   nothing of the sign survives to the readout. On WLASL the clips were
   trimmed to the sign and zero padding was masked, so the last state sat
   right after the sign. Diagnosis on a random 80/20 split (not
   signer-disjoint, so inflated), same seeds:

   ```
                                      100 signes          400 signes
                                      e5     e10    e20    e5     e15
   centroide le plus proche           0.165 (sans entrainement)
   TCN (moyenne sur le temps)         0.328  0.589  0.718
   GRU, etat final                    0.008  0.008  0.380  0.004  0.004
   GRU, moyenne des frames            0.322  0.571  0.714  0.309  0.603
   GRU, attention                     0.513  0.681  0.790  0.360  0.618
   ```

   `--readout attention` (`AttentionPool`: a small network scores each frame,
   a softmax over time turns the scores into weights) lets the model weigh the
   sign above the rest frames. +12k parameters (308,644 against 296,228), and
   it saves and reloads with identical predictions. The default stays `last`
   so WLASL runs reproduce.

   **Relaunched** on the GPU with WLASL's best recipe plus those changes:

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

   **`--pretrain-cache`** keeps each fold's pretraining (weights, plus the
   random generator's state at its end) and reloads it in later runs. An
   experiment that only changes fine-tuning then costs ~1h15 instead of ~3 h,
   and starts from the same weights as the baseline, which makes the paired
   comparison tighter. Pretraining cannot be shared *across folds*: every
   signer is a test signer in one fold, and signs 1-100 are part of the
   scaffold, so one shared pretraining would have trained on each fold's test
   clips. Checked: a run that reloads gives exactly the curves and
   probabilities of the run that pretrained. A cache made with different
   options, different pretraining code (augmentation, features, models) or
   another device is refused, naming what differs. Options that act only after
   pretraining (`--epochs`, `--tta`, `--seeds`, `--interpolate`, `--smooth`,
   `--max-train-signers`; the gate options while `--gate-pretrain` is off)
   reuse it. With the flag, TensorFlow is reseeded before fine-tuning so both
   paths start from the same state; without it nothing changes.

   **Result (3 November): 0.8610 ± 0.0073**, signer-disjoint, 100 signs
   (`models/aslc_kfold_baseline.json`, the reference for every comparison):

   ```
   fold        1       2       3       4       5
   plateau   0.8539  0.8730  0.8588  0.8651  0.8541
   ```

   For scale, WLASL's best was 0.762 on 50 signs. The best single epoch runs
   2-4 points above the plateau on every fold: the maximum of 150 noisy epochs
   (epoch-to-epoch spread ~0.9 point), chosen by looking at the test fold, so
   not a level the model holds. Every fold is within 1% of its plateau by
   epoch 12 and stays flat to 150: no harm from training long, but most of
   the budget only confirms the plateau. The curves are saved per epoch and
   the run is bit-reproducible, so any shorter budget (say 90 epochs, plateau
   = epochs 61-90) can be scored for the baseline without rerunning it.

   **Fold 5 collapsed and was redone.** At seed 0 (500 for that fold) its
   pretraining never left chance (e60 0.004, against 0.72-0.76 elsewhere), and
   fine-tuning from it predicted one class for all 617 clips (0.0081). Traced
   layer by layer on the cached weights: the projection's output was 2.5x
   larger than fold 1's, the first GRU's spread across clips fell to 0.105
   (fold 1: 0.49), the second's to 0.005, 62% of its outputs at ±1, the
   attention stayed uniform and 0% of the head's ReLUs were active. The
   GRUs had settled where they ignore their input. The inputs are clean (8
   clips of 12,270 exceed 10) but not on a common scale: column spreads run
   from 0.008 to 1.25. Gradient clipping (`--clipnorm 1.0`) does not prevent
   it, tested on that seed: with Adam the step size barely depends on the
   gradient's size. Fold 5 was rerun with the same recipe at `--seed 1`
   (partition unchanged, it does not depend on the seed): 0.8541. Merged with
   `merge_kfold.py --replace-fold 5`, which records the swap in the file; the
   collapsed run is kept as `aslc_kfold_baseline_run1.json`.

   One start in five collapsing makes every later run fragile. Candidate fix,
   `--layer-norm`: a LayerNormalization between the projection and the GRU,
   rescaling its input at every frame, inside the model so the demo follows.
   Tested on the seed that collapsed, 10 pretraining epochs:

   ```
                             e1      e3      e6      e10
   fold 5, sans LayerNorm   0.003   0.004   0.004   0.004
   fold 5, avec LayerNorm   0.003   0.054   0.178   0.333
   fold 1, avec LayerNorm   0.004   0.061   0.198   0.365
   ```

   It fixes that seed; one seed is evidence, not a guarantee. Whether it
   changes accuracy is the next k-fold, paired against the baseline, at 90
   epochs (the baseline is scored at 90 from its saved curves). Adding these
   options changed
   `build_model`, so the current code refuses the baseline's cached
   pretrainings (as designed: the fingerprint covers the model code). A run
   with `--layer-norm` needs its own pretrainings anyway.

   **Prefetch.** Each epoch is now prepared on the CPU while the GPU trains
   the previous one (`--no-prefetch` restores the old order). Checked
   bit-identical (curves, probabilities, pretraining) on fold 5. Gain: 17% on
   pretraining, 14% on fine-tuning, less than the third the GPU's idle time
   suggested, since the two halves still slow each other a little.

   The official splits also exist and would allow comparison with the
   published baseline, which covers all 2,731 signs, not 100.
4. **Clips under 24 fps** (2.6%, P49 alone 1,205 of them) are resampled with
   duplicate frames, as a slow webcam would be in the demo. Whether they hurt is
   a k-fold question: with and without them.
5. **Pointing signs** in the confusion matrix, as predicted in §3.
6. **Display merged twins** (`EAT1`/`EAT2` shown as "EAT"). Not built.
7. **Rebuild the shipped model** on ASL Citizen. `train_final.py` now defaults
   to the baseline's recipe (`--readout attention`, `--gate-pretrain`, 60
   pretraining epochs), cites `models/aslc_kfold_baseline.json`, and refuses
   to start when its options differ from the recipe the cited k-fold measured
   (so a mismatch costs seconds, not an hour). It stops after 10 pretraining
   epochs if accuracy is still under 5x chance: a collapsed GRU, as in fold 5.
   A first 1-seed model is for live testing (3 November); the full ensemble
   is `--seeds 5`. For testing, `src/asl_citizen/make_sign_sheet.py` writes a
   local page of the target signs, two ASL Citizen clips each, never to be
   shared (license). The demo will then follow the
   new card on its own.
8. **The signers' brief (`RECORDING.md`) is obsolete**: it describes the 50
   WLASL words and WLASL variants. It needs the new vocabulary and new reference
   clips — and the license forbids distributing ASL Citizen videos, so sending
   signers clips from the dataset is not an option. ASL-LEX publishes its own
   video for each sign under CC BY-NC, which may serve; not yet checked.
9. **Environment fragility**: any `pip install` depending on `onnxruntime`
   reinstalls the CPU build over the GPU one. The repair is in §4; the
   extractor refuses to run in that state, the demo only prints a warning.
10. **Live testing remains the binding risk**, as in the WLASL phase. Every
    number here will come from held-out ASL Citizen signers.
