# Cat vs Dog Image Classifier 🐱🐶

A Convolutional Neural Network (CNN) built with TensorFlow/Keras that classifies images as either a **cat** or a **dog**. Trained on the [Dogs vs. Cats dataset](https://www.kaggle.com/datasets/salader/dogsvscats) from Kaggle.

## Project Structure

```
image-classification/
│
├── image_classification.ipynb   # Main notebook (data download, training, evaluation, inference)
├── dog.jpg                      # Sample test image
├── cat.jpg                      # Sample test image
├── train/                       # Training images (downloaded via Kaggle API, not committed)
├── test/                        # Validation/test images (downloaded via Kaggle API, not committed)
└── README.md
```

> **Note:** The `train/` and `test/` folders, the dataset zip, and `kaggle.json` are **not** pushed to GitHub. They're either downloaded automatically by the notebook or contain private credentials. Make sure they're listed in `.gitignore`.

## What It Does

The notebook walks through a complete, beginner-friendly image classification pipeline:

1. **Downloads the dataset** directly from Kaggle using the Kaggle API.
2. **Loads images** into TensorFlow datasets using `image_dataset_from_directory`, resizing everything to 256×256.
3. **Normalizes pixel values** (0–255 → 0–1) for stable training.
4. **Builds a CNN from scratch** — 3 convolutional blocks (Conv2D → BatchNormalization → MaxPooling) followed by fully connected layers with Dropout, ending in a single sigmoid output for binary classification (cat = 0, dog = 1).
5. **Trains the model** for 10 epochs using the Adam optimizer and binary cross-entropy loss.
6. **Visualizes performance** with accuracy/loss curves for both training and validation sets.
7. **Runs inference** on new images (`dog.jpg`, `cat.jpg`), including preprocessing (resize, reshape, normalize) and prints a human-readable prediction.

## Who Uses It

This project is aimed at:
- **Beginners/students in deep learning** who want a hands-on, end-to-end example of image classification — from raw images to a working prediction — without relying on a pre-trained model.
- **Anyone learning CNN fundamentals**: how convolution, pooling, batch norm, and dropout fit together in practice.
- **Developers prototyping a binary image classifier** who want a simple template to adapt to their own two-class problem (e.g., defective vs. non-defective parts, healthy vs. diseased leaf, etc.) by simply swapping the dataset.

## Real-World Impact

While "cat vs. dog" itself is a toy problem, the underlying technique generalizes directly to real, valuable applications:
- **Content moderation & tagging** — auto-tagging pet photos on social/e-commerce platforms.
- **Veterinary & animal shelter tech** — sorting incoming photos by species for shelter management systems.
- **Template for binary image classification** — the same architecture, with minimal changes, can power quality-control checks in manufacturing, medical pre-screening (e.g., presence/absence of a feature in scans), or agricultural defect detection.
- **Educational value** — it's a low-stakes way to internalize a CNN pipeline that scales conceptually to harder, higher-impact problems.

## Issues Faced & How I Resolved Them

| Issue | Cause | Resolution |
|---|---|---|
| **Overfitting** — training accuracy climbed to ~97% by epoch 9–10, but validation accuracy stayed around 73–79% and bounced around a lot (even dropping as low as 60% in one epoch) | The model memorized the training set faster than it learned generalizable features; no data augmentation was used, and the network is fairly deep for the amount of regularization applied | Added `BatchNormalization` after each conv layer and `Dropout` in the dense layers to reduce this — though as the results show, it only partially solved the problem. **Next step:** add data augmentation and/or stronger regularization (see Future Improvements) |
| **Validation loss increasing while validation accuracy fluctuated** | A classic overfitting symptom — the model became increasingly confident (lower training loss) but those confident predictions were sometimes wrong on unseen data | Tracked both accuracy *and* loss curves (not accuracy alone) to catch this rather than stopping after looking at accuracy alone |
| **BGR vs. RGB color mismatch during inference** | OpenCV (`cv2.imread`) loads images in BGR format by default, but `matplotlib` and the model were trained on RGB data from Keras' image loader, causing images to display with incorrect colors | Used `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` before visualizing/feeding images into the model |
| **Forgetting to normalize test images the same way as training images** | Training images were scaled to [0, 1] via the `process()` function, but early inference cells fed in raw [0, 255] pixel values, which silently produces wrong predictions | Added `test_input = test_input / 255.0` to the inference code so the test-time preprocessing matches training-time preprocessing exactly |
| **Kaggle API key permission warning** | Kaggle's CLI warned that the API key file was readable by other users on the system, which is a security risk | This is solved by running `chmod 600 ~/.kaggle/kaggle.json` to restrict file permissions to the current user only |

## Future Improvements

- **Data augmentation** (random flips, rotations, zoom, brightness shifts) to reduce overfitting and improve generalization on unseen images.
- **Transfer learning** — replace the from-scratch CNN with a pre-trained backbone (e.g., MobileNetV2, ResNet50, EfficientNet) and fine-tune it; this typically yields much higher accuracy with far less training data and time.
- **Early stopping & learning rate scheduling** to stop training automatically once validation loss stops improving, instead of running a fixed 10 epochs.
- **K-fold cross-validation** for a more reliable estimate of true model performance.
- **Confusion matrix & precision/recall metrics**, not just accuracy, to better understand where the model is making mistakes.
- **Model export & deployment** — save the trained model (`model.save()`) and wrap it in a simple API (Flask/FastAPI) or Streamlit app so it can classify user-uploaded images in real time.
- **Larger/more diverse test set** for inference demos, instead of just one cat and one dog image.

## Tech Stack

- Python, TensorFlow/Keras
- OpenCV (`cv2`) for image loading/preprocessing during inference
- Matplotlib for visualization
- Kaggle API for dataset access

## Setup

```bash
pip install tensorflow opencv-python matplotlib kaggle

# Place your Kaggle API token at ~/.kaggle/kaggle.json (get it from kaggle.com/account)
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download -d salader/dogsvscats
```

Then run the cells in `image_classification.ipynb` in order.

---

*If you found this useful, feel free to ⭐ the repo!*
