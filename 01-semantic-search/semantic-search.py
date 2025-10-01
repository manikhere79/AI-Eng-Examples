import json
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

# 4. Constant for data file path and SentenceTransformer embedding.
DATA_FILE_PATH = "data/News_Category_Dataset_v3.json"
CHROMA_COLLECTION_NAME = "news_headlines_collection"
# We use a compact, high-quality model suitable for beginners
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' 

def setup_chroma_and_embed_data(
    client: chromadb.Client, 
    model: SentenceTransformer, 
    file_path: str
):
    """
    3. Loads the dataset, vectorizes it, and saves it to a ChromaDB collection.
    Every time cleanup the collection in chroma db.
    """
    print(f"--- 1. Loading collection '{CHROMA_COLLECTION_NAME}'...")
         
    # Create the collection with the appropriate embedding function
    # Note: ChromaDB will automatically use the Sentence Transformer model
    # if you install it and pass the name to the embedding_function argument.
    # We pass the model name here for clarity and setup, though the model 
    # itself is used manually below for a common beginner workflow.
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        # ChromaDB automatically handles the embedding for the IDs/Metadatas/Documents 
        # using the provided embedding function. We'll use the model manually 
        # for a clearer step-by-step example.
    )
    
    print(f"--- 2. Reading dataset from {file_path}...")
    
    # 2. Add the dataset to a folder in project (Data loading)
    try:
        data = []
        with open(file_path, 'r') as f:
            # The dataset is a JSON Lines file
            for line in f:
                data.append(json.loads(line))
        
        # Convert to a DataFrame for easy handling
        df = pd.DataFrame(data)
        # We only need the headline text for a simple example
        documents = df['headline'].tolist()
        
        # Limit to a smaller number for faster demonstration
        MAX_RECORDS = 1000
        documents = documents[:MAX_RECORDS]
        
        if not documents:
            raise ValueError("Dataset loaded is empty.")
        
        print(f"    ✅ Loaded {len(documents)} records from the dataset.")

    except FileNotFoundError:
        print(f"\nFATAL ERROR: Data file not found at '{file_path}'.")
        print("Please ensure you have created the 'data' folder and placed the 'News_Category_Dataset_v6.json' file inside it.")
        return None
    except Exception as e:
        print(f"\nFATAL ERROR during data loading: {e}")
        return None

    print("--- 3. Vectorizing and saving to ChromaDB...")

    # Generate embeddings using the Sentence Transformer model
    # This is the 'simple embedding' step to vectorize the text
    embeddings = model.encode(documents).tolist()

    # Prepare data for ChromaDB: ids, embeddings, and documents (text)
    ids = [f"doc_{i}" for i in range(len(documents))]
    
    # Add the documents and embeddings to the collection
    collection.add(
        embeddings=embeddings,
        documents=documents,
        ids=ids
    )

    print(f"    ✅ Successfully added {collection.count()} documents to ChromaDB.")
    return collection

def main():
    """
    Main function to run the RAG demonstration.
    """
    # Initialize ChromaDB client (in-memory for simplicity)
    # Use chromadb.PersistentClient(path="./chroma_data") for persistent storage
    chroma_client = chromadb.Client()
    
    # Initialize the Sentence Transformer model
    try:
        print(f"--- Initializing Sentence Transformer model: {EMBEDDING_MODEL_NAME}...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("    ✅ Model loaded successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Could not load Sentence Transformer model. Check your internet connection and 'pip install sentence-transformers'. Error: {e}")
        return

    # 3. Main function of the python file should invoke a method which will 
    # load the dataset, then use simple embedding to vectorize and save it.
    collection = setup_chroma_and_embed_data(
        client=chroma_client, 
        model=embedding_model, 
        file_path=DATA_FILE_PATH
    )

    if collection is None:
        return
    
    # 4. Once embedding completed, give user with prompt on the console to enter a question.
    print("\n" + "="*100)
    print("✨ RAG Search Ready! Enter a question below to find relevant headlines.")
    print("="*100)

    while True:
        question = input("\nEnter your search query (e.g., 'cricket game news' or 'quit'): ").strip()
        
        if question.lower() == 'quit':
            print("Exiting RAG search. Goodbye!")
            break

        if not question:
            continue
            
        print(f"Searching for: '{question}'...")
        
        # Convert the user's question into an embedding
        query_embedding = embedding_model.encode([question]).tolist()
        
        # Perform a similarity search (Query the collection)
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=3, # Retrieve the top 3 most similar results
            include=['documents', 'distances']
        )
        
        print("\n--- Top 3 Most Relevant Headlines (by Vector Similarity) ---")
        
        # Check if we have results
        if results and results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                document = results['documents'][0][i]
                distance = results['distances'][0][i]
                # Lower distance means higher similarity (better match)
                print(f"[{i+1}] Similarity (Distance): {distance:.4f} | Headline: '{document}'")
        else:
            print("No relevant headlines found.")
            
# Check if the script is being run directly
if __name__ == "__main__":
    main()