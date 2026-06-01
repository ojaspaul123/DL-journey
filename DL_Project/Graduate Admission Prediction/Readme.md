# 🎓 Graduate Admission Prediction

A deep learning regression project that predicts the probability of a student's admission into graduate school based on academic and profile parameters, using an Artificial Neural Network (ANN) built with TensorFlow and Keras.

---

## 📁 Project Structure

```
Graduate Admission Prediction/
│
├── Graduate_Admission_Prediction.ipynb   # Main Jupyter Notebook (EDA, model, evaluation)
├── Admission_Predict.csv                 # Dataset (500 student records)
└── README.md                             # Project documentation
```

---

## 🧠 What Does It Do?

This project trains an **ANN regression model** to predict a student's **chance of admission** (a value between 0 and 1) based on 7 key academic features.

### Input Features:
| Feature | Description |
|---|---|
| GRE Score | Graduate Record Examination score (out of 340) |
| TOEFL Score | English proficiency score (out of 120) |
| University Rating | University prestige (1–5) |
| SOP | Statement of Purpose strength (1–5) |
| LOR | Letter of Recommendation strength (1–5) |
| CGPA | Undergraduate GPA (out of 10) |
| Research | Research experience (0 or 1) |

### Output:
- **Chance of Admit** — a probability score between 0 and 1

### Model Workflow:
1. **Load & Explore** — Load CSV, check shape, columns, and data types
2. **Split** — 80/20 train-test split using `train_test_split`
3. **Scale** — Normalize features using `MinMaxScaler`
4. **Build Model** — ANN with Dropout for regularization:
   - `Dense(128, relu)` → `Dropout(0.2)`
   - `Dense(64, relu)` → `Dropout(0.2)`
   - `Dense(32, relu)`
   - `Dense(1, linear)` — regression output
5. **Train** — Up to 500 epochs with `EarlyStopping` (patience=20)
6. **Evaluate** — R² Score via `sklearn`
7. **Visualize** — Smooth loss/val_loss convergence curve

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `TensorFlow / Keras` | ANN model building & training |
| `Scikit-learn` | Train-test split, scaling, R² evaluation |
| `Pandas / NumPy` | Data handling and manipulation |
| `Matplotlib` | Loss curve visualization |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
```

### Run
```bash
jupyter notebook Graduate_Admission_Prediction.ipynb
```

Or open in [Google Colab](https://colab.research.google.com/) by uploading the `.ipynb` file.

---

## 📊 Results

| Metric | Value |
|---|---|
| Final Training Loss | ~0.0025 |
| Final Validation Loss | ~0.0030 |
| **R² Score** | **~0.85+** |

The loss curve shows both training and validation loss converging smoothly — indicating a well-generalizing model with no overfitting.

---

## 👥 Who Uses This?

| User | Use Case |
|---|---|
| **Students** | Self-assess admission chances before applying |
| **Education Consultants** | Guide students toward realistic university choices |
| **Universities** | Automate initial screening of applicants |
| **ML Learners** | Practice regression with ANNs on a real-world dataset |
| **EdTech Platforms** | Integrate as a feature in career counseling tools |

---

## 🌍 Real-World Impact

- **🎓 Student Guidance** — Helps students make data-driven decisions about which universities to target, saving time and application fees
- **📉 Reduced Rejection Rates** — Consultants can better match students to universities where they have a realistic chance
- **🏫 University Admissions** — Assists admissions officers in shortlisting candidates efficiently
- **🌐 EdTech Integration** — Platforms like Shiksha, Yocket, and Leverage Edu use similar models to recommend universities
- **💡 Awareness** — Highlights which factors (CGPA, GRE, Research) matter most for admission, helping students plan their profiles early

---

## 🐛 Problems Faced & How I Resolved Them

### 1. `ValueError` — Input Shape Mismatch
**Problem:**
```
expected axis -1 of input shape to have value 7, but received input with shape (32, 8)
```
**Cause:** The dataset had 8 features (including the Serial No. column) but the model was built with `input_dim=7`.

**Fix:** Either drop the Serial No. column before training or set `input_dim=8`:
```python
# Option 1 — Drop serial number
df = df.drop('Serial No.', axis=1)

# Option 2 — Fix input_dim
model.add(Dense(128, activation='relu', input_dim=8))
```

---

### 2. Low R² Score (~0.31)
**Problem:** Model only explained 31% of variance after 10 epochs.

**Cause:** Too few epochs — model hadn't converged yet.

**Fix:** Increased epochs from 10 → 500 with EarlyStopping:
```python
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = model.fit(..., epochs=500, callbacks=[early_stop])
```

---

### 3. Noisy Validation Loss Curve
**Problem:** Val loss bounced around erratically after epoch 40 — sign of overfitting.

**Cause:** No regularization in the model.

**Fix:** Added `Dropout(0.2)` layers after hidden layers:
```python
model.add(Dense(128, activation='relu', input_dim=8))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
```

---

### 4. Cluttered Training Output (500 epoch logs)
**Problem:** Training printed 500 lines of epoch logs making the notebook messy.

**Fix:** Set `verbose=0` in `model.fit()`:
```python
history = model.fit(..., epochs=500, verbose=0)
print("Training complete!")
```

---

### 5. Duplicate Plot Cell
**Problem:** Two separate plot cells created overlapping graphs without labels.

**Fix:** Merged into one clean cell with proper labels, legend, and title:
```python
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss - Graduate Admission Prediction')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

---

## 📌 Future Improvements

- [ ] Try **Random Forest** or **XGBoost** regression for comparison
- [ ] Add **feature importance** visualization (which factor impacts admission most)
- [ ] Deploy as a **web app** using Streamlit with sliders for each input feature
- [ ] Add **cross-validation** for more robust evaluation
- [ ] Extend dataset with more diverse student records

---

## 📄 License

This project is open-source and free to use for educational purposes.

---

## 🙋 Author

**ojaspaul123**  
Part of the **DL-journey** repository — a collection of deep learning projects.  
Feel free to connect on [GitHub](https://github.com/ojaspaul123)
