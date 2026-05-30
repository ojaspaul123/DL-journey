# 🔢 Handwritten Digit Classification using Artificial Neural Networks (ANN)

A deep learning project that classifies handwritten digits (0–9) using a Neural Network trained on the MNIST dataset with TensorFlow and Keras.

---

## 📁 Project Structure

```
Handwritten_Digit_Classification/
│
├── Handwritten_Digit_Classification.ipynb   # Main Jupyter Notebook (model building, training & evaluation)
├── README.md                                # Project documentation
```

---

## 🧠 What Does It Do?

This project builds a **Multi-Layer Perceptron (MLP)** neural network that learns to recognize handwritten digits from grayscale 28×28 pixel images.

### Workflow:
1. **Load Data** — Uses the built-in MNIST dataset (60,000 training + 10,000 test images)
2. **Visualize** — Plots sample digits with `matplotlib`
3. **Preprocess** — Normalizes pixel values from `[0, 255]` → `[0, 1]`
4. **Build Model** — A Sequential neural network:
   - `Flatten` layer → converts 28×28 image to a 784-element vector
   - `Dense(128, relu)` → first hidden layer
   - `Dense(32, relu)` → second hidden layer
   - `Dense(10, softmax)` → output layer (one node per digit class)
5. **Train** — 25 epochs with 20% validation split using Adam optimizer
6. **Evaluate** — Accuracy score via `sklearn`, loss/accuracy curves plotted
7. **Predict** — Runs inference on individual test images

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `TensorFlow / Keras` | Model building & training |
| `NumPy` | Array operations |
| `Pandas` | Data handling |
| `Matplotlib` | Visualization |
| `Scikit-learn` | Accuracy evaluation |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

### Run
Open the notebook in Jupyter or Google Colab:
```bash
jupyter notebook Handwritten_Digit_Classification.ipynb
```

Or open directly in [Google Colab](https://colab.research.google.com/) by uploading the `.ipynb` file.

---

## 📊 Results

- **Training Accuracy:** ~98%+ after 25 epochs
- **Test Accuracy:** ~97–98% (via `accuracy_score`)
- Loss and accuracy curves are plotted to visualize model convergence

---

## 👥 Who Uses This?

| User | Use Case |
|---|---|
| **ML Beginners** | A perfect first deep learning project to understand image classification |
| **Students** | Understanding neural network layers, activation functions, and training loops |
| **Educators** | Teaching supervised learning and CNNs with a well-known benchmark dataset |
| **Researchers** | Baseline model before experimenting with CNNs, dropout, or regularization |

---

## 🌍 Real-World Impact

While MNIST is a benchmark dataset, the underlying technology powers:

- **🏦 Banking & Finance** — Cheque amount recognition, form processing
- **📮 Postal Services** — Automatic ZIP/PIN code reading on envelopes
- **🏥 Healthcare** — Digitizing handwritten prescriptions and medical forms
- **📝 Education** — Auto-grading handwritten exams and assignments
- **📦 Logistics** — Reading handwritten labels and delivery slips

This project is a foundational stepping stone toward **Optical Character Recognition (OCR)** systems used in real-world document automation.

---

## 🐛 Problems Faced & How I Fixed Them

### 1. `tensorflow` Import Errors
**Problem:** `ModuleNotFoundError: No module named 'tensorflow'`  
**Fix:** Installed the correct version:
```bash
pip install tensorflow
```
For older systems or CPU-only environments:
```bash
pip install tensorflow-cpu
```

### 2. MNIST Dataset Download Failure
**Problem:** `keras.datasets.mnist.load_data()` failed due to network issues or SSL errors.  
**Fix:** Downloaded the dataset manually and loaded it from disk, or used:
```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

### 3. Overfitting During Training
**Problem:** Training accuracy was high but validation accuracy lagged — signs of overfitting over 25 epochs.  
**Fix:** Monitored `val_loss` curves; considered adding `Dropout` layers or reducing epochs based on where validation loss stopped improving.

### 4. Pixel Values Not Normalized
**Problem:** Model trained slowly and gave poor results with raw pixel values `[0–255]`.  
**Fix:** Normalized inputs to `[0–1]` range:
```python
X_train = X_train / 255
X_test = X_test / 255
```

### 5. Wrong Prediction Shape
**Problem:** `model.predict(X_test[1])` threw a shape error — model expected a batch dimension.  
**Fix:** Reshaped the single image before predicting:
```python
model.predict(X_test[1].reshape(1, 28, 28)).argmax(axis=1)
```

---

## 📌 Future Improvements

- [ ] Replace MLP with a **CNN** (Convolutional Neural Network) for better accuracy
- [ ] Add **Dropout** layers to reduce overfitting
- [ ] Deploy as a **web app** using Flask or Streamlit with a canvas to draw digits
- [ ] Export and save the trained model using `model.save()`

---

## 📄 License

This project is open-source and free to use for educational purposes.

---

## 🙋 Author

**[OJAS PAUL]**  
Feel free to connect on [LinkedIn](#https://www.linkedin.com/in/ojas-paul-324aa5337/).
