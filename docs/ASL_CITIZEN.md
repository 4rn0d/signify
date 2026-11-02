# Signify — ASL Citizen phase

Same task as the WLASL phase (`WLASL.md`), recognising whole signed words from
a webcam, rebuilt on a different dataset. The WLASL notebook stays the record
of how the pipeline was built and why; this one records what changed when the
data did.

**Current state:** no model trained on ASL Citizen yet. The 100 target signs are
extracted (3,089 clips, no failure, 2.03 s per clip); the 300 pretraining signs
are being extracted with the demo's exact pipeline.
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
   Options: slim it further and pretrain 60 epochs; a first signal on one or
   two folds without pretraining; or train on the GPU through WSL2, since
   TensorFlow has no GPU support on native Windows — which would change the
   cost of every architecture.
3b. **First k-fold on ASL Citizen.** Signer-disjoint over 52 signers with the
   existing harness. The official splits also exist and would allow comparison
   with the published baseline, which covers all 2,731 signs, not 100.
4. **Clips under 24 fps** (2.6%, P49 alone 1,205 of them) are resampled with
   duplicate frames, as a slow webcam would be in the demo. Whether they hurt is
   a k-fold question: with and without them.
5. **Pointing signs** in the confusion matrix, as predicted in §3.
6. **Display merged twins** (`EAT1`/`EAT2` shown as "EAT"). Not built.
7. **Rebuild the shipped model** on ASL Citizen. The demo will then follow the
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
