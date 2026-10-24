# Arabic Sign Language Letter Classification

A CNN built with PyTorch that recognizes Arabic sign language letters from hand images.

## Problem

Given a picture of a hand sign, predict which of the 32 Arabic letters it represents.
This is a 32-class image classification problem.

## Dataset

[ArASL](https://www.kaggle.com/datasets/muhammadalbrham/rgb-arabic-alphabets-sign-language-dataset)
(Arabic Alphabet Sign Language), around 54,000 images in 32 folders, one per letter.

- Images are converted to grayscale and resized to 64x64
- Pixel values are normalized to [-1, 1]
- Random split: 70% train, 15% validation, 15% test (seed = 21)

## Model

| Layer | Details |
|---|---|
| Conv1 + ReLU + MaxPool | 1 -> 32 channels, 3x3 |
| Conv2 + ReLU + MaxPool | 32 -> 64 channels, 3x3 |
| Conv3 + ReLU + MaxPool | 64 -> 128 channels, 3x3 |
| AdaptiveAvgPool | output 4x4 |
| Fully connected | 2048 -> 256, ReLU, Dropout 0.3 |
| Output | 256 -> 32 |

Training settings: Adam, learning rate 0.001, batch size 32, 15 epochs, cross entropy loss.
The weights with the lowest validation loss are saved to `arasl_model.pth`.

## Results

| Metric | Value |
|---|---|
| Test accuracy | 97.35% |
| Final train loss | 0.074 |
| Final validation loss | 0.119 |
| Best validation loss | 0.1165 (epoch 11) |

The weakest letters were `zay` (about 89%), `dha` (94%), `jeem` (94%) and `fa` (95%).
Most other letters are above 96%.

The validation loss stops improving after around epoch 7 while the training loss keeps
dropping, so there is a small amount of overfitting but nothing serious.

### Limitations

The split is random, and the dataset has many very similar images (same signers, same
backgrounds). Some near-duplicates probably ended up in both train and test, so the real
accuracy on new photos is likely lower than 97%.

## How to run

1. Install the requirements:
   ```
   pip install -r requirements.txt
   ```
2. Download the ArASL dataset and extract it so the folders are in
   `ArASL/ArASL_Database_54K_Final/`. If you use a different path, change it in the
   notebook where `ImageFolder` is called.
3. Open `notebook.ipynb` and run the cells in order to train and evaluate the model.
4. To test the saved model on one image:
   ```
   python predict.py path/to/image.jpg
   ```
   It prints the top 3 predicted letters with their probabilities.

## Files

- `notebook.ipynb`: training, evaluation, loss plot, per-letter accuracy
- `predict.py`: loads the saved model and predicts a single image
- `arasl_model.pth`: saved model weights
- `requirements.txt`: Python packages
