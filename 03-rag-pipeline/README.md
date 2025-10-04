# Retrieval-Augmented Generation (RAG) Pipeline

Welcome to the RAG pipeline demo! This project shows how to build a simple Retrieval-Augmented Generation (RAG) system using Python, ChromaDB, and Gemini LLM. It is designed for beginners and provides step-by-step instructions.

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is an approach that combines information retrieval with generative AI models. Instead of relying only on the model's internal knowledge, RAG first retrieves relevant documents from a database and then uses those documents as context for the language model to generate accurate, grounded answers.

**Why use RAG?**
- Answers are based on real data, not just the model's training.
- Reduces hallucinations and improves reliability.
- Useful for question answering, summarization, and more.

## Project Overview

This folder contains two main files:

- `rag_setup.py`: Prepares the vector database (ChromaDB) and embeds news headlines using a sentence transformer model.
- `rag_app.py`: Runs the full RAG pipeline: retrieves context for a user query, builds a prompt, and gets an answer from Gemini LLM.

### Data
- Uses a subset of news headlines from `../data/News_Category_Dataset_v3.json` (limited to 1000 headlines for demo speed).

## Step-by-Step Implementation

### 1. Install Requirements

Create a `requirements.txt` file in this folder with the following contents:

```
pandas
chromadb
sentence-transformers
```

Install dependencies:
```powershell
pip install -r requirements.txt
```

### 2. Prepare the Vector Database

`rag_setup.py` loads the news headlines, generates embeddings, and stores them in a ChromaDB collection.

- **Collection name:** `news_test_collection`
- **Embeddings model:** `all-MiniLM-L6-v2`
- **Max records:** 1000

To run setup (optional, as `rag_app.py` will do this automatically):
```powershell
uv run .\rag_setup.py
```

Expected output:
```
--- Setting up collection 'news_test_collection'...
    Total documents to embed: 1000
    Processed batch 1. Documents added: 1000
--- ✅ Setup Complete. Total documents indexed: 1000 ---
```

### 3. Run the RAG Pipeline

`rag_app.py` orchestrates the full process:


```

--- Retrieved Context ---

Retrieved Context:

--- Grounded Answer from Gemini ---
The American Airlines flyer who punched a flight attendant was charged and banned for life from the airline.
-----------------------------------------------------
Headline 1: Tatum's Layup At Buzzer Gives Celtics 115-114 Win Over Nets        
Headline 2: Vanessa Bryant Says Kobe Bryant Crash Photos Turned Grief To Horror

--- Grounded Answer from Gemini (Expected to be 'Unanswerable') ---
I cannot find the answer in the provided news headlines.

```

## How It Works
4. **Generation**: Gemini LLM answers the query using only the provided context.
## File Explanations

### `rag_setup.py`
- Loads headlines from the dataset.
- Embeds them using SentenceTransformer.
- Stores them in ChromaDB for fast similarity search.
- Provides functions to retrieve context for a query.

- Imports setup and retrieval functions.
- Defines a system instruction to keep answers grounded in the context.
- Builds the prompt and calls Gemini LLM API.
- Handles two test queries: one answerable, one unanswerable.

## Troubleshooting
- Make sure `../data/News_Category_Dataset_v3.json` exists and contains headlines.
- If you see import errors, install missing packages with `pip install -r requirements.txt`.
- For API errors, check your Gemini API credentials (handled automatically in Canvas).

## License
This project is for educational purposes.
