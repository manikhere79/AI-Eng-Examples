import json
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
import asyncio

# --- Configuration ---
DATA_FILE_PATH = "../data/News_Category_Dataset_v3.json" 
CHROMA_COLLECTION_NAME = "news_test_collection"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' 
K_CONTEXT = 3 # Number of relevant documents to retrieve for the LLM
BATCH_SIZE = 5000 # Max documents to process per batch

def setup_chroma_and_embed_data(client: chromadb.Client, model: SentenceTransformer, file_path: str):
    """Loads the subset data and populates a new ChromaDB collection, using batching to avoid errors."""
    print(f"--- Setting up collection '{CHROMA_COLLECTION_NAME}'...")

    # Ensure we start fresh by deleting the collection if it exists
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME) 
    except Exception:
        pass # Ignore error if collection does not exist

    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Load the JSON Lines file containing the filtered test data
            df = pd.DataFrame([json.loads(line) for line in f])
            documents = df['headline'].tolist()

            # Limit to a smaller number for faster demonstration
            MAX_RECORDS = 10000
            documents = documents[:MAX_RECORDS]
            
            # Prepare data for ChromaDB: ids, embeddings, and documents (text)
            ids = [f"doc_{i}" for i in range(len(documents))]
            total_docs = len(documents)
            
            print(f"    Total documents to embed: {total_docs}")

            # Implement batching to avoid exceeding internal limits (e.g., 5461)
            for i in range(0, total_docs, BATCH_SIZE):
                batch_documents = documents[i:i + BATCH_SIZE]
                batch_ids = ids[i:i + BATCH_SIZE]
                
                # Encode the headlines to create vector embeddings
                batch_embeddings = model.encode(batch_documents).tolist()
                
                # Add the batch to ChromaDB
                collection.add(
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    ids=batch_ids
                )
                print(f"    Processed batch {i // BATCH_SIZE + 1}. Documents added: {len(batch_documents)}")
                
            print(f"--- ✅ Setup Complete. Total documents indexed: {collection.count()} ---")
            return collection

    except FileNotFoundError:
        print(f"FATAL ERROR: Data file not found at {file_path}. Please run 'generate-test-subset.py' first.")
        exit()
    except Exception as e:
        print(f"FATAL ERROR during data loading/embedding: {e}")
        exit()

def format_context(documents: list) -> str:
    """Formats the retrieved documents into a single, clean string for the LLM."""
    context_str = "\n".join([f"Headline {i+1}: {doc}" for i, doc in enumerate(documents)])
    return f"Retrieved Context:\n---\n{context_str}\n---"

def get_context_for_rag(client: chromadb.Client, model: SentenceTransformer, query: str) -> str:
    """
    Performs the vector search and returns a formatted string of the top K results.
    This is the core RAG retrieval step.
    """
    collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    
    # 1. Embed the user query
    query_embedding = model.encode([query]).tolist()
    
    # 2. Retrieve the top K documents
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=K_CONTEXT, 
        include=['documents', 'distances'] 
    )
    
    # 3. Extract the document text
    retrieved_documents = results['documents'][0]

    if not retrieved_documents:
        print("Warning: No documents retrieved from ChromaDB.")
        return "No relevant context found."

    # 4. Format the documents into a single string
    return format_context(retrieved_documents)

if __name__ == "__main__":
    # This block ensures the vector database is initialized when the file is run standalone
    print("Initializing ChromaDB client and Sentence Transformer...")
    try:
        chroma_client = chromadb.Client()
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"Initialization Error: {e}")
        print("Please ensure 'chromadb' and 'sentence-transformers' libraries are installed.")
        exit()
        
    # Setup the collection with data
    collection = setup_chroma_and_embed_data(chroma_client, embedding_model, DATA_FILE_PATH)
    
    # Example usage after setup
    # Run a test query to confirm context retrieval is working
    test_query = "What happened with the passenger who hit a flight attendant?"
    context = get_context_for_rag(chroma_client, embedding_model, test_query)
    
    print("\n--- Test Retrieval (from rag_setup.py) ---")
    print(f"Query: {test_query}")
    print(context)