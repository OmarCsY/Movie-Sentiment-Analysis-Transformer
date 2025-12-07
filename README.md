# SentimentScope: Movie Sentiment Analysis using Transformers

## 📌 Project Overview
This project focuses on building and training a **Transformer-based model** from scratch to perform **Sentiment Analysis** on the IMDB movie reviews dataset. The goal is to classify reviews as either **Positive (1)** or **Negative (0)**.

Unlike using a pre-trained BERT model directly, this project involves constructing the Transformer architecture (Attention Head, Multi-Head Attention, FeedForward, and Blocks) and customizing it for a binary classification task using **PyTorch**.

## 🛠️ Technologies & Tools
* **Language:** Python 3.x
* **Deep Learning Framework:** PyTorch
* **NLP Tools:** Hugging Face Transformers (for `bert-base-uncased` Tokenizer)
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib

## 📂 Project Structure
The repository contains the main notebook and data handling logic:

* `SentimentScope_starter.ipynb`: The main Jupyter Notebook containing:
    * Data Loading & Preprocessing (IMDB Dataset).
    * Custom `IMDBDataset` class for PyTorch DataLoader.
    * **Model Architecture:** Custom `DemoGPT` class with modified Classification Head.
    * Training Loop (CrossEntropyLoss, AdamW Optimizer).
    * Evaluation & Testing.

## 🧠 Model Architecture
The model is a custom Transformer Decoder-based architecture designed for classification:
1.  **Tokenizer:** Uses `bert-base-uncased` tokenizer for subword splitting.
2.  **Embeddings:** Learned Token Embeddings + Positional Embeddings.
3.  **Transformer Blocks:** Stacked layers of Multi-Head Attention and FeedForward networks.
4.  **Pooling:** Applies **Mean Pooling** across the time dimension to summarize the sequence.
5.  **Classifier Head:** A final Linear layer mapping the pooled output to 2 classes (Positive/Negative).

## 🚀 How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/OmarCsY/Movie-Sentiment-Analysis-Transformer.git](https://github.com/OmarCsY/Movie-Sentiment-Analysis-Transformer.git)
    ```
2.  **Install dependencies:**
    Ensure you have PyTorch and Transformers installed:
    ```bash
    pip install torch transformers pandas matplotlib
    ```
3.  **Open the Notebook:**
    Launch Jupyter Notebook or JupyterLab and open `SentimentScope_starter.ipynb`.
4.  **Run the Cells:**
    Execute the cells sequentially to load data, train the model, and evaluate performance.

> **Note:** The project requires a GPU for efficient training. If running locally, ensure CUDA is set up, or use Google Colab.

## 📊 Results
* **Validation Accuracy:** Achieved ~79% after 3 epochs.
* **Test Accuracy:** Achieved **76.17%**, exceeding the project requirement of 75%.

---

## 🔗 Author
* **Name:** Omar Ibrahim Al-Ali
* **Date:** 2025-11-24
