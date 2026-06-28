# 🐱🐶 Cat vs Dog Image Classifier (CNN)

A Convolutional Neural Network built from scratch with TensorFlow/Keras to classify images as either a **cat** or a **dog**, trained on the Kaggle "Dogs vs Cats" dataset.

---

## 🖼️ Demo

| Input: Cat | Input: Dog |
|---|---|
| <img src="samples/cat.jpg" width="300" /> | <img src="samples/dog.jpg" width="300" /> |
| Predicted: **CAT** ✅ | Predicted: **DOG** ✅ |

---

## 📁 Project Structure

```
Cat-vs-Dog-Classifier/
│
├── image_classification.ipynb   # Full pipeline: data loading, CNN model, training, inference
├── samples/
│   ├── cat.jpg                  # Sample test image
│   └── dog.jpg                  # Sample test image
│
└── README.md
```

> **Note:** `kaggle.json` (your Kaggle API credentials) should **never** be committed to GitHub. See [Setup](#-setup) below for how to use it safely, and add it to `.gitignore`.

---

## ⚙️ How It Works

### 1. Data Acquisition
- Dataset downloaded via the Kaggle API: [`salader/dogsvscats`](https://www.kaggle.com/datasets/salader/dogsvscats)
- Extracted into `train/` and `test/` folders, each containing `cats/` and `dogs/` subfolders

### 2. Data Loading & Preprocessing
- Images loaded with `keras.utils.image_dataset_from_directory`
- Resized to **256x256**, batched in groups of 32
- Pixel values normalized to the `[0, 1]` range (`image / 255.`)

### 3. CNN Architecture

| Layer | Details |
|---|---|
| Conv2D (32 filters) | 3x3 kernel, ReLU → BatchNorm → MaxPooling |
| Conv2D (64 filters) | 3x3 kernel, ReLU → BatchNorm → MaxPooling |
| Conv2D (128 filters) | 3x3 kernel, ReLU → BatchNorm → MaxPooling |
| Flatten | — |
| Dense (128) + Dropout (0.1) | ReLU |
| Dense (64) + Dropout (0.1) | ReLU |
| Dense (1) | Sigmoid (binary output: 0 = cat, 1 = dog) |

- **Optimizer:** Adam
- **Loss:** Binary Crossentropy
- **Epochs:** 10

### 4. Inference
- A new image is read with OpenCV, converted from BGR → RGB, resized to 256x256, normalized, and reshaped to `(1, 256, 256, 3)` before being passed to `model.predict()`
- Output < 0.5 → **Cat**, output ≥ 0.5 → **Dog**

---

## 🚀 Setup

### Prerequisites

```bash
pip install tensorflow opencv-python matplotlib kaggle
```

### Configure Kaggle API

1. Place your `kaggle.json` in `~/.kaggle/kaggle.json` (not in the repo!)
2. Run:
```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d salader/dogsvscats
```

### Run the Notebook
Open `image_classification.ipynb` in Jupyter, Google Colab, or VS Code and run all cells in order.

---

## 🌍 Real-World Impact

Binary image classification is a foundational computer vision task with applications far beyond cats and dogs:

- **Animal shelters & rescue apps** — auto-tagging intake photos by species/breed for faster cataloguing
- **Veterinary & pet-tech apps** — pre-sorting uploaded pet photos
- **E-commerce** — automatically categorizing product images (e.g., pet supply listings)
- **Education** — a clean, minimal example for students learning CNNs, commonly used as a "Hello World" of computer vision

**Who uses it?**
- Students and beginners learning CNN architecture and Keras
- Developers prototyping an image classification pipeline before adapting it to a real use case
- Researchers benchmarking simple custom CNNs against pretrained/transfer-learning approaches

---

## 🐛 Issues Faced & How They Were Fixed

### 1. Forgot to Normalize Test Images at Inference Time
**Problem:** Early prediction cells resized the test image and reshaped it to `(1, 256, 256, 3)` but never divided by 255 — meaning pixel values (0–255) were fed directly into a model trained on normalized (0–1) inputs. This caused unreliable/incorrect predictions.

**Fix:** Added `test_input = test_input / 255.0` before calling `model.predict()`, matching the normalization applied during training.

---

### 2. BGR vs RGB Color Channel Mismatch
**Problem:** OpenCV (`cv2.imread`) loads images in **BGR** format by default, but `matplotlib` and the model's training pipeline (via `image_dataset_from_directory`) expect **RGB**. Displaying or feeding BGR images caused color distortion in visualizations.

**Fix:** Added `cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)` before plotting with `matplotlib`.

---

### 3. Stale Variable Reused Across Predictions
**Problem:** In early test cells, `test_imput` (note the typo) was defined once for the dog image and then reused for the cat image prediction — so the cat image was resized but never actually fed into the model; the dog tensor was predicted again instead.

**Fix:** Refactored into a single reusable block (cells 24–25) that reads, converts, resizes, normalizes, reshapes, and predicts on the image path explicitly each time — eliminating stale variable reuse.

---

### 4. No File-Existence Check
**Problem:** If an image path was wrong (e.g., wrong working directory in Colab vs local), `cv2.imread` would silently return `None`, causing a cryptic crash later in `cv2.resize`.

**Fix:** Added `os.path.exists(image_path)` checks with a clear error message before attempting to read/process the image.

---

### 5. Kaggle API Credentials Exposure Risk
**Problem:** `kaggle.json` contains a username and a secret API key. If accidentally committed to a public GitHub repo, this allows anyone to download datasets and consume your Kaggle API quota under your account.

**Fix:** `kaggle.json` is excluded via `.gitignore`, and the README documents that it must be placed manually in `~/.kaggle/` rather than stored in the project.

---

## 🔮 Future Improvements

- **Transfer learning** — swap the custom CNN for a pretrained backbone (e.g., MobileNetV2, ResNet50, EfficientNet) via fine-tuning for higher accuracy with less training data
- **Data augmentation** — add random flips, rotations, zoom, and brightness shifts (`tf.keras.layers.RandomFlip`, etc.) to improve generalization and reduce overfitting
- **Larger/more diverse dataset** — extend beyond cats/dogs to a multi-class pet/animal classifier
- **Model evaluation** — add a confusion matrix, precision/recall, and ROC-AUC instead of relying on accuracy/loss curves alone
- **Hyperparameter tuning** — experiment with learning rate schedules, different optimizers, and deeper/shallower architectures
- **Deployment** — wrap the trained model in a Streamlit or Flask app with image upload support for real-time predictions
- **Model export** — save in `.h5` or `SavedModel` format with a documented inference script, instead of requiring the full notebook to run predictions
- **Batch inference & API** — expose a REST endpoint (FastAPI/Flask) so other apps/services can classify images programmatically
- **Explainability** — add Grad-CAM visualizations to show which regions of the image influenced the prediction

---

## 📊 Dataset

- **Source:** [Dogs vs Cats Dataset on Kaggle](https://www.kaggle.com/datasets/salader/dogsvscats)
- **Classes:** `cat` (0), `dog` (1)
- **Image size used:** 256x256 (resized from originals)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| Deep Learning | TensorFlow / Keras |
| Image Processing | OpenCV |
| Visualization | Matplotlib |
| Data Source | Kaggle API |
| Development | Jupyter Notebook / Google Colab |

---

## 📄 License

This project is for educational purposes. Feel free to fork, extend, and adapt it.

---

## 🙌 Acknowledgements

- [Dogs vs Cats Dataset](https://www.kaggle.com/datasets/salader/dogsvscats) by salader on Kaggle
- [TensorFlow/Keras](https://www.tensorflow.org/) for the deep learning framework
- [OpenCV](https://opencv.org/) for image preprocessing
