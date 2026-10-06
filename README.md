# Sign Language CV

CNN and landmark-based sign language recognition.

## Structure

- data/raw/ — original downloaded dataset
- data/processed/ — resized/normalized images
- data/landmarks/ — extracted MediaPipe landmark CSVs
- notebooks/ — exploration and visualization
- src/ — pipeline code (data loading, preprocessing, models, training, evaluation, webcam demo)
- models/ — saved model weights

## Setup

pip install -r requirements.txt

## Build order

1. Get CNN pipeline working on ASL Alphabet dataset (image -> label)
2. Add MediaPipe landmark extraction, train landmark-based model, compare
3. Add live webcam_demo.py to test both models in real time
4. (Advanced) Move to WLASL dataset + temporal modeling (CNN+LSTM or 3D CNN) for full word/sentence signs
