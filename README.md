# Hebbian-Decay ANN with Reinforcement and Punishment Learning

This repository implements a hybrid learning system combining:

- A CNN for feature extraction from grayscale images.
- A custom biologically-inspired ANN trained using:
  - Hebbian Learning
  - Weight Decay
  - Reinforcement and Punishment
  - Sleep/Residual Cleaning Mechanisms

The system learns to classify two classes with limited data and emphasizes low-magnitude feature accumulation and corrective learning over multiple passes.

---

## Key Features

- **CNN Frontend**: Lightweight convolutional network to extract 64-dimensional feature vectors.
- **ANN Backend**:
  - 64 cycles of sequential processing per image.
  - Hebbian updates during cycles.
  - Reinforcement or punishment after one full pass based on prediction correctness.
  - Sleep phase to clean residual noise.
- **Per-image repeated learning**: Images are repeatedly shown until correctly classified or reaching a maximum epoch limit.
- **Manual alternation**: Images from two classes are alternated to reduce bias and overfitting.

---

## Project Structure

- `cnn_model`: Extracts features from 64x64 grayscale images.
- `custom_ann_update`: Handles 64-cycle updates using Hebbian, reinforcement, punishment, and sleep mechanisms.
- `alternating_train`: Preloads and orders training images for balanced learning.
- `training loop`: Trains on each image until classification is correct.
- `evaluation loop`: Measures test set performance without weight updates.

---

## Training Details

- **Training Strategy**: 
  - Train until the image is correctly classified or `MAX_EPOCHS_PER_IMAGE` is reached.
  - Small weight adjustments accumulate to gradually cross firing thresholds.
- **Lossless Bucket Reset**: After a neuron fires, its bucket resets to zero.
- **Decayed Accumulation**: Every cycle applies a small decay to prevent runaway accumulation.

---

## Parameters

| Parameter                   | Value | Description                                  |
| ---------------------------- | ----- | --------------------------------------------|
| `NUM_CYCLES`                 | 64    | Number of sequential cycles per image       |
| `INPUT_DIM`, `HIDDEN_DIM`    | 64    | Dimensionality of feature space and neurons |
| `OUTPUT_DIM`                 | 2     | Number of output classes                    |
| `MAX_EPOCHS_PER_IMAGE`       | 50    | Max tries to correctly classify an image    |
| `INITIAL_BUCKET_THRESHOLD`   | 1.5   | Initial firing threshold per neuron         |
| `DECAY_RATE`                 | 0.02  | Per-cycle decay factor                      |
| `WEIGHT_HEBBIAN`             | 0.3   | Strength of immediate Hebbian update        |
| `WEIGHT_DECAY`               | 0.2   | Decay influence on weights                  |
| `WEIGHT_REINFORCEMENT`       | 0.5   | Strength when reinforcing a firing          |
| `WEIGHT_PUNISHMENT`          | 0.2   | Strength when punishing a firing            |

---

## How to Run

1. **Prepare Dataset**

2. dataset_directory/ training_set/ class0/ class1/ test_set/ class0/ class1/

markdown
Copy
Edit
- Grayscale images recommended at 64x64 pixels.

2. **Install Dependencies**
```bash
pip install tensorflow numpy
Train the Model

bash
Copy
Edit
python your_script_name.py
View Results

Training progress printed per image.

Final test accuracy shown after evaluation.
