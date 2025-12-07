# SentimentScope: Movie Sentiment Analysis using Transformers

## 📌 Project Overview
This project focuses on building and training a **Transformer-based model** from scratch to perform **Sentiment Analysis** on the IMDB movie reviews dataset. The goal is to classify reviews as either **Positive (1)** or **Negative (0)** to enhance recommendation systems.

Unlike using a pre-trained BERT model directly, this project demonstrates a deep understanding of NLP by constructing the Transformer architecture components (Attention Head, Multi-Head Attention, FeedForward, and Blocks) and customizing them for a binary classification task using **PyTorch**.

## 🛠️ Technologies & Tools
* **Language:** Python 3.x
* **Deep Learning Framework:** PyTorch
* **NLP Tools:** Hugging Face Transformers (for `bert-base-uncased` Tokenizer)
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib

## 📂 Project Structure
The repository contains the main notebook which handles the end-to-end pipeline:

* **Data Loading & Preprocessing:** Handling the IMDB Dataset and creating a custom `IMDBDataset` class for the PyTorch DataLoader.
* **Model Architecture:** A custom `DemoGPT` class built from scratch, featuring a modified Classification Head.
* **Training Loop:** Implemented using CrossEntropyLoss and AdamW Optimizer.
* **Evaluation:** Testing the model's accuracy on unseen data.

## 🧠 Model Architecture
The model is a custom Transformer Decoder-based architecture designed specifically for classification:
1.  **Tokenizer:** Utilizes `bert-base-uncased` tokenizer for efficient subword splitting.
2.  **Embeddings:** Combines Learned Token Embeddings with Positional Embeddings.
3.  **Transformer Blocks:** Stacked layers of Multi-Head Attention and FeedForward networks to capture contextual relationships.
4.  **Pooling Mechanism:** Applies **Mean Pooling** across the time dimension to create a single vector representation for the entire sequence.
5.  **Classifier Head:** A final Linear layer mapping the pooled output to 2 classes (Positive/Negative).

## 📊 Results
* **Validation Accuracy:** Achieved ~79% after 3 epochs.
* **Test Accuracy:** Achieved **76.17%**, successfully exceeding the project benchmark of 75%.

---

## 🔗 Author
* **Name:** Omar Ibrahim Al-Ali
* **Date:** 2025-11-24
