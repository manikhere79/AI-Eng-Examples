# 📖 Beginner's Guide to Semantic Search (RAG) with ChromaDB and Sentence Transformers

This project demonstrates a fundamental **Retrieval-Augmented Generation (RAG)** pipeline component: **Semantic Search**. We use a **Sentence Transformer** model to create powerful numerical representations (**embeddings**) of text and store them in **ChromaDB**, an open-source vector database.

When a user asks a question, the program finds the most **semantically similar** news headlines from the database, regardless of exact keyword matches.

-----

## 🎯 Project Goal

To create a simple, single-file Python program that:

1.  Loads a dataset of text documents (**news headlines**).
2.  Uses a pre-trained **Sentence Transformer** model to convert the text into **vector embeddings**.
3.  Stores these embeddings in a **ChromaDB** collection.
4.  Performs a **similarity search** based on user input.

-----

## 🛠️ Prerequisites

Before running the program, ensure you have **Python 3.8+** and the **`uv`** package installed. `uv` is a modern, fast Python package installer and resolver.

### 1\. Create `requirements.txt`

In your project directory, create a file named `requirements.txt` and add the following contents:

```txt
chromadb
sentence-transformers
pandas
```

### 2\. Dataset and Folder Setup

The program is configured to use the `News_Category_Dataset_v3.json` file.

1.  **Data File Name:** The required data file is **`News_Category_Dataset_v3.json`**. You can find this dataset on Kaggle.
2.  **Create a Folder**: In the same directory where you save your Python file, create a new folder named `data`.
3.  **Place the File**: Put the downloaded dataset file inside the `data` folder.

Your project structure should look like this:

```
/01-semantic-search
├── semantic-search.py 
├── requirements.txt
└── /data
    └── News_Category_Dataset_v3.json 
```

### 3\. Data Sample (JSON Lines Format)

The `News_Category_Dataset_v3.json` file is in [JSON Lines format](http://jsonlines.org/). The program processes the `headline` field for embedding.

```json
{"link": "https://www.huffpost.com/entry/covid-boosters-uptake-us_n_632d719ee4b087fae6feaac9", "headline": "Over 4 Million Americans Roll Up Sleeves For Omicron-Targeted COVID Boosters", "category": "U.S. NEWS", "short_description": "Health experts said it is too early to predict whether demand would match up with the 171 million doses of the new boosters the U.S. ordered for the fall.", "authors": "Carla K. Johnson, AP", "date": "2022-09-23"}
{"link": "https://www.huffpost.com/entry/american-airlines-passenger-banned-flight-attendant-punch-justice-department_n_632e25d3e4b0e247890329fe", "headline": "American Airlines Flyer Charged, Banned For Life After Punching Flight Attendant On Video", "category": "U.S. NEWS", "short_description": "He was subdued by passengers and crew when he fled to the back of the aircraft after the confrontation, according to the U.S. attorney's office in Los Angeles.", "authors": "Mary Papenfuss", "date": "2022-09-23"}
{"link": "https://www.huffpost.com/entry/funniest-tweets-cats-dogs-september-17-23_n_632de332e4b0695c1d81dc02", "headline": "23 Of The Funniest Tweets About Cats And Dogs This Week (Sept. 17-23)", "category": "COMEDY", "short_description": "\"Until you have a dog you don't understand what could be eaten.\"", "authors": "Elyse Wanshel", "date": "2022-09-23"}
```

-----

## 🚀 How to Run the Project using `uv`

1.  **Create Virtual Environment**: Create a virtual environment to isolate the project's dependencies (good practice\!).

    ```bash
    uv venv
    ```

2.  **Activate Environment**: Activate the environment to ensure packages are installed locally.

      * **macOS/Linux:**
        ```bash
        source .venv/bin/activate
        ```
      * **Windows (Command Prompt):**
        ```bash
        .venv\Scripts\activate
        ```

3.  **Install Dependencies**: Use `uv` with the requirements file to quickly install the necessary packages.

    ```bash
    uv pip install -r requirements.txt
    ```

4.  **Execute the Script**: Run the main Python file.

    ```bash
    uv run semantic-search.py
    ```

## ⚙️ Core Concepts Explained

### Vector Embeddings

An **embedding** is a numerical list (a **vector**) that represents a piece of text (like a word or sentence) in a high-dimensional space. The key idea is: **text with similar meaning will have vectors that are numerically closer together** in this space. This is what enables semantic search.

### Sentence Transformers

This is a Python framework that simplifies the process of creating these high-quality embeddings. The model we use, **`all-MiniLM-L6-v2`**, is a compact, fast model specifically trained to produce semantically meaningful sentence embeddings.

### ChromaDB (Vector Database)

A **vector database** is specialized storage designed to efficiently handle and search through massive numbers of these high-dimensional vectors.

  * **Collection**: The primary container in ChromaDB, which holds your documents, their embeddings, and any associated metadata.
  * **Vectorization**: The process of converting your text into an embedding vector.
  * **Similarity Search**: When you query the database, your query text is also turned into an embedding. ChromaDB then quickly calculates the **distance** between your query vector and all stored vectors to find the closest matches.
  * **Distance (Similarity Metric)**: In the final search results, you see a **Distance** score. **The lower the distance, the more similar in meaning** the two pieces of text are.

-----

## 🎯 Sample Execution and Output

After running `uv run semantic-search.py`, you will see the following setup process:

```
--- Initializing Sentence Transformer model: all-MiniLM-L6-v2...
    ✅ Model loaded successfully.
--- 1. Loading collection 'news_headlines_collection'...
    ℹ️ No existing collection to clean up (first run or delete failed).
--- 2. Reading dataset from data/News_Category_Dataset_v3.json...
    ✅ Loaded 1000 records from the dataset.
--- 3. Vectorizing and saving to ChromaDB...
    ✅ Successfully added 1000 documents to ChromaDB.

====================================================================================================
✨ RAG Search Ready! Enter a question below to find relevant headlines.
====================================================================================================
```

When you enter a search query, the output will look like this:

```
Enter your search query (e.g., 'cricket game news' or 'quit'): **major sports matches**

Searching for: 'major sports matches'...

--- Top 3 Most Relevant Headlines (by Vector Similarity) ---
[1] Similarity (Distance): 0.2831 | Headline: 'What We Learned From Sunday's NFL Games'
[2] Similarity (Distance): 0.3590 | Headline: 'NFL Star Richard Sherman Arrested On Suspicion Of DUI'
[3] Similarity (Distance): 0.3802 | Headline: 'FIFA's Bidding Battle for the 2026 World Cup'
```

*In this example, the search for "major sports matches" correctly retrieved headlines about the NFL and the World Cup, even though it didn't use the exact keywords "major" or "matches." This demonstrates the power of **semantic search** using embeddings.*