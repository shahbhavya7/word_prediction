# 📚 Next Word Prediction using LSTM (Hamlet Dataset)

This project implements a **Next Word Prediction** system using a **Long Short-Term Memory (LSTM)** model trained on Shakespeare's *Hamlet* text data. It leverages **deep learning**, **NLP preprocessing**, and a **Streamlit** web app for interactive predictions.

---

## 🚀 Features
- **LSTM model** trained for sequential next-word prediction.
- **Tokenizer** built to map words to integers and handle unseen input dynamically.
- **Streamlit interface** for real-time prediction.
- **Early stopping** used during training to prevent overfitting (see `experiments.ipynb`).
- **Hamlet Dataset** sourced from Kaggle. 
- **Pre-trained model** (`next_word_lstm.h5`) and tokenizer (`tokenizer.pickle`) loading mechanism.

---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/word-prediction-lstm.git
   cd word-prediction-lstm

---

## 📦 Requirements

`tensorflow==2.15.0`, `pandas`, `numpy`, `scikit-learn`, `tensorboard`, `matplotlib`, `streamlit`, `scikeras`

---

## 📈 Model Details

- **Architecture**:
  - Embedding layer
  - LSTM layers with dropout regularization
  - Dense output layer with softmax activation
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **EarlyStopping**: Monitored validation loss to stop training early if no improvement
- **Max Sequence Length**: Dynamically inferred at runtime based on model input shape
- **Training Metrics**: Accuracy, Validation Accuracy

---

## 🧪 Technical Details (from `app.py` and `experiments.ipynb`)

- **Model Loading**:
  - Model is loaded from `next_word_lstm.h5` using TensorFlow Keras
  - Tokenizer is loaded from `tokenizer.pickle`
- **Prediction Workflow**:
  - Input text is tokenized and padded to match expected sequence length
  - Model predicts the next word's probability distribution
  - Word with the highest probability (`argmax`) is selected
- **Web Interface**:
  - Built with Streamlit
  - Real-time text input and output
  - Dynamic `max_sequence_len` adjustment
- **Experiments**:
  - Explored different LSTM sizes (100, 200 units)
  - Tuned batch size (32, 64)
  - Used EarlyStopping to prevent overfitting after observing loss plateaus
  - Finalized training around 20-30 epochs based on validation accuracy

---



