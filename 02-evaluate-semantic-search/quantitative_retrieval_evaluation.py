import json
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
import os

# --- Configuration ---
DATA_FILE_PATH = "./data/test_data_subset.json"
CHROMA_COLLECTION_NAME = "news_test_collection"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' 
K_VALUE = 3 # We will check if the correct answer is in the top 3 results
# Maximum number of items to process in one go, set conservatively low
BATCH_SIZE = 1000 

# [1] The Golden Test Set
# These are manually created queries paired with the ID of the document 
# that is known to be the perfect match (based on the sample data).
GOLDEN_TEST_SET = [
    # Based on doc_0: "Over 4 Million Americans Roll Up Sleeves For Omicron-Targeted COVID Boosters" (U.S. NEWS)
    ("News about the latest COVID-19 vaccination rates.", "doc_0"), 
    # Based on doc_1: "American Airlines Flyer Charged..." (U.S. NEWS)
    ("What happened with the passenger who hit a flight attendant?", "doc_1"),
    # Based on doc_8412: "Super PAC Screw-Up: Ad Favorably Compares GOP Candidate..." (POLITICS)
    ("Political scandal involving a candidate and a Super PAC ad.", "doc_8412"), 
    # Based on doc_8413: "'Ant-Man And The Wasp' Trailer..." (ENTERTAINMENT)
    ("Tell me about the new Ant-Man movie trailer.", "doc_8413"),
]

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

def evaluate_recall_at_k(collection: chromadb.Collection, model: SentenceTransformer, test_set: list):
    """Runs the quantitative Recall @ K evaluation."""
    
    correct_count = 0
    total_queries = len(test_set)
    
    print(f"\n--- Running Recall @ {K_VALUE} Evaluation ({total_queries} Queries) ---")
    
    for i, (query, expected_id) in enumerate(test_set):
        # 1. Convert the test query into an embedding
        query_embedding = model.encode([query]).tolist()
        
        # 2. Perform the vector search
        # FIX: The 'include' parameter must contain one of the valid items 
        # ('distances', 'embeddings', 'documents', 'metadatas'). IDs are always returned.
        # We explicitly request 'distances' here to satisfy the API.
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=K_VALUE, # Retrieve top K documents
            include=['distances'] 
        )
        
        # IDs are still retrieved by default.
        retrieved_ids = results['ids'][0]
        
        # 3. Check if the expected ID is present in the retrieved IDs
        if expected_id in retrieved_ids:
            correct_count += 1
            status = "✅ MATCH"
        else:
            status = "❌ FAIL"

        print(f"[{i+1}/{total_queries}] {status}: Query: '{query}'")
        print(f"      Expected ID: {expected_id} | Top {K_VALUE} Retrieved IDs: {retrieved_ids}")

    # 4. Calculate and report the final metric
    recall_at_k = correct_count / total_queries
    print("\n" + "="*70)
    print(f"FINAL RESULT: Recall @ {K_VALUE} Score: {recall_at_k:.2f} ({correct_count} out of {total_queries} correct)")
    print("="*70)

if __name__ == "__main__":
    try:
        chroma_client = chromadb.Client()
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"Initialization Error: {e}")
        print("Ensure 'chromadb' and 'sentence-transformers' are installed.")
        exit()
        
    collection = setup_chroma_and_embed_data(chroma_client, embedding_model, DATA_FILE_PATH)
    
    if collection:
        evaluate_recall_at_k(collection, embedding_model, GOLDEN_TEST_SET)