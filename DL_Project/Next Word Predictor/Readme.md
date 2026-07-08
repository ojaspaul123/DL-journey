# 🤖 LSTM Next-Word Predictor / FAQ Chatbot

A deep learning project that trains a **stacked LSTM (Long Short-Term Memory)** model on a custom FAQ dataset to predict the next word in a sequence — enabling an auto-complete or chatbot-style response system, trained entirely on domain-specific course FAQ text.

---

## 📁 Project Structure

```
LSTM-Next-Word-Predictor/
│
├── lstm_project.ipynb        # Full pipeline: data prep, tokenization, model training & inference
│
└── README.md
```

> The FAQ text corpus is embedded directly in the notebook (no external file needed). The trained model and tokenizer are in-memory during the session — see [Future Improvements](#-future-improvements) for saving/loading guidance.

---

## ⚙️ How It Works

### 1. Corpus / Training Data
- A real-world FAQ document from **CampusX Data Science Mentorship Program (DSMP 2023)** is used as the text corpus
- Covers topics like course fee, syllabus, payment, certificates, and placement assistance

### 2. Tokenization
- `Tokenizer` from Keras builds a vocabulary of **283 unique words** from the FAQ text
- Each word is assigned a unique integer index via `fit_on_texts()`

### 3. N-Gram Sequence Generation
- For every sentence in the corpus, all possible **prefix sub-sequences** are generated
- e.g. `"what is the fee"` → `["what is", "what is the", "what is the fee"]`
- This creates the full training set of input-output word pairs

### 4. Padding
- All sequences are padded to a **uniform length of 56** (the longest sequence) using pre-padding with zeros via `pad_sequences()`

### 5. Feature / Label Split
- `X` = all tokens except the last one (context window)
- `y` = the last token (next word to predict), one-hot encoded across 283 classes with `to_categorical()`

### 6. Model Architecture

| Layer | Details |
|---|---|
| Embedding | Vocab size 283 → dense vectors of size 100, input length 56 |
| LSTM (150 units) | First LSTM layer — learns sequential patterns, returns sequences |
| LSTM (150 units) | Second LSTM layer — deeper temporal reasoning |
| Dense (283, softmax) | Output layer — probability distribution over all 283 words |

- **Loss:** Categorical Crossentropy
- **Optimizer:** Adam
- **Epochs:** 100

### 7. Inference / Text Generation
- A seed phrase (e.g. `"what is the fee"`) is tokenized and padded
- The model predicts the next word (highest probability via `np.argmax`)
- The predicted word is appended to the input and the process repeats for **10 words**, generating a full response sentence

---

## 💡 Example Output

```
Seed: "what is the fee"

what is the fee for data science mentorship program (dsmp 2023)
```

The model learns to auto-complete questions and answers based purely on the FAQ text it was trained on.

---

## 🌍 Real-World Impact

LSTM-based next-word prediction and text generation is the foundation of many production NLP systems:

- **Customer support bots** — train on your own product FAQ / documentation to build a domain-specific auto-responder without using large external APIs
- **EdTech platforms** — auto-answer student queries about courses, fees, schedules, and eligibility
- **Smart keyboards** — next-word suggestion engines (similar to what runs on mobile keyboards) trained on domain-specific or personal text
- **Content generation tools** — assist writers by suggesting contextually relevant next words or phrases
- **Search auto-complete** — predict query completions based on a corpus of past searches

**Who uses it?**
- NLP students learning sequence modeling and Keras
- Developers building lightweight domain-specific chatbots without large LLMs
- EdTech startups wanting instant FAQ bots trained on their own content
- Researchers exploring character/word-level language models from scratch

---

## 🐛 Issues Faced & How They Were Fixed

### 1. `np` Used Without Import
**Problem:** `np.argmax()` is called in the inference cell (cell 22) but `import numpy as np` was never added to the notebook. This causes a `NameError: name 'np' is not defined` at inference time.

**Fix:** Add the following import at the top of the notebook alongside other imports:
```python
import numpy as np
```

---

### 2. Short / Single-Word Sentences Skipped Silently
**Problem:** When the FAQ text is split by `\n`, blank lines and single-word lines produce a tokenized sequence of length 0 or 1. The inner `for i in range(1, len(...))` loop simply doesn't execute — so those lines contribute nothing to training without any warning.

**Fix:** Add an explicit length check before building sequences:
```python
for sentence in faqs.split('\n'):
    tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]
    if len(tokenized_sentence) < 2:
        continue  # skip too-short sequences
    for i in range(1, len(tokenized_sentence)):
        input_sequences.append(tokenized_sentence[:i+1])
```

---

### 3. Hardcoded Vocabulary Size and Sequence Length
**Problem:** The `Embedding` layer uses `283` (vocab size) and `56` (max sequence length) hardcoded as magic numbers. If the corpus changes even slightly, these numbers go stale and the model silently produces wrong-shaped inputs.

**Fix:** Derive these values dynamically:
```python
vocab_size = len(tokenizer.word_index) + 1   # +1 for padding token 0
max_len = max([len(x) for x in input_sequences])

model.add(Embedding(vocab_size, 100, input_length=max_len - 1))
```

---

### 4. Model and Tokenizer Not Saved
**Problem:** After training for 100 epochs, neither the model weights nor the tokenizer are saved to disk. Every time the notebook is re-run, training restarts from scratch — wasting compute time and making deployment impossible.

**Fix:** Save both after training:
```python
import pickle

# Save model
model.save("lstm_faq_model.h5")

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
```

---

### 5. Inference Loop Has No Stopping Condition
**Problem:** The text generation loop runs a fixed 10 iterations regardless of output quality — it can repeat words, run off-topic, or never form a complete sentence.

**Fix:** Add a stop token (e.g. a period `.`) and break on it, or check for repeated bigrams:
```python
for i in range(20):
    token_text = tokenizer.texts_to_sequences([text])[0]
    padded = pad_sequences([token_text], maxlen=max_len - 1, padding='pre')
    pos = np.argmax(model.predict(padded))
    next_word = [w for w, idx in tokenizer.word_index.items() if idx == pos]
    if not next_word:
        break
    text += " " + next_word[0]
    if next_word[0] in [".", "?"]:
        break
```

---

## 🔮 Future Improvements

- **Larger & more diverse corpus** — add multiple FAQs, product docs, or scraped web content to improve fluency and vocabulary coverage
- **Bidirectional LSTM** — use `Bidirectional(LSTM(150))` so the model considers context from both directions, improving coherence
- **Beam search decoding** — replace greedy `argmax` with beam search for better, less repetitive text generation
- **Temperature sampling** — introduce randomness in prediction (`softmax temperature`) for more creative and varied outputs vs. always picking the single most probable word
- **Save & load pipeline** — serialize model (`.h5` / `SavedModel`) and tokenizer (`.pkl`) so it can be deployed without retraining
- **Streamlit / Flask deployment** — wrap the generator in a web app where users type a seed question and get a live auto-completed FAQ answer
- **Transfer learning** — fine-tune a pretrained model (GPT-2, DistilBERT) on the FAQ corpus instead of training an LSTM from scratch, for dramatically better output quality
- **Evaluation metrics** — add BLEU score or perplexity to objectively measure generation quality beyond just training accuracy
- **Multi-domain FAQ support** — extend the corpus to cover multiple products/courses and train the model to distinguish between domains

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| Deep Learning | TensorFlow / Keras |
| Model Type | Stacked LSTM (sequence-to-one) |
| Text Processing | Keras Tokenizer, pad_sequences |
| Numerical Computing | NumPy |
| Development | Jupyter Notebook / Google Colab |

---

## 📄 License

This project is for educational purposes. Feel free to fork, extend, and adapt it.

---

## 🙌 Acknowledgements

- [TensorFlow / Keras](https://www.tensorflow.org/) for the LSTM implementation
