# 🧠 Text-to-SQL Language Model  
*A Seq2Seq Model for Natural Language to SQL Translation*  
University of Sharjah — Jan 2025 to May 2025

This project implements a deep learning system that translates natural language questions into SQL queries using a custom-built **Sequence-to-Sequence (Seq2Seq)** architecture with an **attention mechanism**. Built from scratch in PyTorch and trained on a large-scale dataset of 425,000 question-query pairs, this model aims to make databases more accessible for non-technical users.

---

## ✨ Features

- ✅ **Encoder-Decoder Architecture**  
  - Bidirectional GRU Encoder  
  - Attention mechanism  
  - Unidirectional GRU Decoder  

- 🧠 **Custom Training Pipeline**  
  - Teacher forcing  
  - Adam optimizer  
  - CrossEntropy loss (ignoring padding)  
  - Gradient clipping for stability  
  - Epoch-wise model checkpointing

- 🧪 **SQL Generation**  
  - Greedy decoding during inference  
  - Evaluated on diverse question patterns  
  - Sample test questions provided

---

## 📊 Dataset

- Total samples: **425,000+**
- Format: Natural language question → Corresponding SQL query
- Tokenized using Hugging Face's **T5-small tokenizer**
- Cleaned to remove null entries and unnecessary schema columns

---

## 🏋️‍♀️ Training

To start training the model, simply run the training script:

```python
python text_to_sql.py
