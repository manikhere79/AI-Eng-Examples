import json
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
import os

# --- Configuration ---
DATA_FILE_PATH = "data/test_data_subset.json"
CHROMA_COLLECTION_NAME = "news_test_collection"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' 
# Heuristic Threshold: Tune this value based on your actual score distribution.
# A lower L2 score means closer/better match. We reject scores above this threshold.
L2_DISTANCE_THRESHOLD = 0.75 
N_RESULTS = 5 
BATCH_SIZE = 1000 

def setup_chroma_and_embed_data(client: chromadb.Client, model: SentenceTransformer, file_path: str):
    """Loads the subset data and populates a new ChromaDB collection, using batching to avoid errors."""
    print(f"--- Setting up collection '{CHROMA_COLLECTION_NAME}'...")

    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            df = pd.DataFrame([json.loads(line) for line in f])
            documents = df['headline'].tolist()
            ids = df['id'].tolist()
            total_docs = len(documents)
            
            print(f"    Total documents to embed: {total_docs}")

            # Generate ALL embeddings first
            print("    Generating all document embeddings...")
            embeddings = model.encode(documents).tolist()
            print("    ✅ Embeddings generated.")

            # --- IMPLEMENT BATCHING FOR ADDING TO CHROMA ---
            for i in range(0, total_docs, BATCH_SIZE):
                end_index = min(i + BATCH_SIZE, total_docs)
                
                batch_documents = documents[i:end_index]
                batch_embeddings = embeddings[i:end_index]
                batch_ids = ids[i:end_index]
                
                print(f"    Adding batch {i//BATCH_SIZE + 1} ({len(batch_documents)} items)...")
                
                # Add the current batch of documents and embeddings to the collection
                collection.add(
                    embeddings=batch_embeddings, 
                    documents=batch_documents, 
                    ids=batch_ids
                )
            
            # --- END BATCHING ---

            print(f"    ✅ Successfully loaded {collection.count()} documents into ChromaDB.")
            return collection
    except FileNotFoundError:
        print(f"FATAL ERROR: Data file not found at '{file_path}'. Please ensure 'data/test_data_subset.json' exists.")
        return None
    except Exception as e:
        print(f"FATAL ERROR during data processing or ChromaDB insert: {e}")
        return None
    
def analyze_threshold(collection: chromadb.Collection, model: SentenceTransformer):
    """Demonstrates filtering results based on a distance threshold."""
    
    # Test Queries: One good expected match, and one potentially bad/ambiguous match
    test_queries = [
        # 1. Good Match Expected (Query: COVID boosters)
        ("New update on the latest COVID booster shots", "doc_0"), 
        # 2. Ambiguous/Bad Match Expected (Query: Very vague and general)
        ("Something interesting that happened this week", "None"),
        # 3. Mid-range Match (Query: About a celebrity/movie)
        ("Action movie trailers released this year", "doc_8413"),
    ]
    
    print(f"\n--- Running Distance Threshold Analysis (Threshold: < {L2_DISTANCE_THRESHOLD:.2f}) ---")
    
    for i, (query, expected_id) in enumerate(test_queries):
        query_embedding = model.encode([query]).tolist()
        
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=N_RESULTS,
            include=['documents', 'distances']
        )
        
        print(f"\n[{i+1}] Query: '{query}'")
        
        found_relevant = False
        
        # Iterate through the retrieved results
        for j in range(len(results['documents'][0])):
            document = results['documents'][0][j]
            distance = results['distances'][0][j]
            
            if distance <= L2_DISTANCE_THRESHOLD:
                # Result is accepted because it is semantically close enough
                print(f"    ✅ ACCEPTED (Distance: {distance:.4f}): '{document}'")
                found_relevant = True
            else:
                # Result is rejected because the distance is too high (semantically too far)
                print(f"    ❌ REJECTED (Distance: {distance:.4f}): '{document}'")

        if not found_relevant:
            print(f"    Note: No documents were found below the {L2_DISTANCE_THRESHOLD:.2f} threshold.")

if __name__ == "__main__":
    try:
        chroma_client = chromadb.Client()
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"Initialization Error: {e}")
        exit()
        
    collection = setup_chroma_and_embed_data(chroma_client, embedding_model, DATA_FILE_PATH)
    
    if collection:
        analyze_threshold(collection, embedding_model)