# Sign Language CV

ASL recognition from a webcam: first fingerspelled letters, then whole signed
words from body and hand keypoints.

## Structure

- data/raw/ — ASL Alphabet dataset (alphabet phase)
- data/processed/ — resized/normalized images
- data/landmarks/ — extracted MediaPipe landmark CSVs
- data/wlasl/ — WLASL videos and features (word phase, no longer used)
- data/asl_citizen/ — ASL Citizen videos and features (current phase).
  Research-only license: never commit, share or publish anything from it.
- data/asl_lex/ — ASL-LEX 2.0 sign data, used to rank signs by frequency
- notebooks/ — exploration and visualization
- src/ — alphabet pipeline
- src/wlasl/ — word pipeline: extraction, features, training, demo
- src/asl_citizen/ — ASL Citizen extraction and checks
- models/ — saved model weights and model cards

All of `data/` except `.gitkeep` files is gitignored.

## Setup

```
pip install -r requirements.txt

# rtmlib pulls in the CPU onnxruntime, which silently overwrites the GPU build.
# Run this after any pip install, or detection drops to ~20x slower:
pip uninstall -y onnxruntime
pip install --force-reinstall --no-deps onnxruntime-gpu==1.22.0

# Check that CUDA really runs, not just that it is listed:
python src/wlasl/gpu_setup.py
```

TensorFlow trains on CPU (no GPU support on native Windows since 2.11); only
pose detection uses the GPU.

ASL Citizen (45.9 GB, Microsoft Research License):

```
curl -L -C - -o data/asl_citizen/ASL_Citizen.zip \
  https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip
```

## Build order

1. CNN pipeline on the ASL Alphabet dataset (image -> label) — done
2. MediaPipe landmark extraction, landmark-based model, comparison — done
3. Live webcam demo for both — done
4. Word-level recognition on WLASL with temporal models — done, closed
5. Word-level recognition on ASL Citizen, 100 most-used signs — in progress

## Documentation

- [docs/FINDINGS.md](docs/FINDINGS.md) — alphabet phase (fingerspelling, 28 classes), closed
- [docs/WLASL.md](docs/WLASL.md) — word phase on WLASL (50 signs), closed
- [docs/ASL_CITIZEN.md](docs/ASL_CITIZEN.md) — word phase on ASL Citizen, current
- [docs/RECORDING.md](docs/RECORDING.md) — signers' brief, obsolete until rewritten for the new vocabulary

## Running

```
python src/webcam_demo_alphabet.py      # letters, fusion model
python src/wlasl/webcam_demo_rtm.py     # words, RTMPose + GRU ensemble
```

The word demo follows the model card: recording length, window placement and
detector (TensorRT or CUDA) are those the model was trained with. `--trt` and
`--no-trt` override the detector.

ASL Citizen pipeline:

```
python src/asl_citizen/extract.py --ranks 1-100     # 100 target signs
python src/asl_citizen/extract.py --ranks 101-400   # pretraining signs
python src/asl_citizen/check_parity.py              # extraction == demo input
python src/asl_citizen/check_window_parity.py       # same 80-frame window both sides
```
