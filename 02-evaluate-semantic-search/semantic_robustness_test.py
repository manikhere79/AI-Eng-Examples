import json
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd

# --- Configuration ---
DATA_FILE_PATH = "data/test_data_subset.json"
CHROMA_COLLECTION_NAME = "news_test_collection"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' 
N_RESULTS = 5 # Focus on the top 2 results for quality check
BATCH_SIZE = 1000

# [1] Qualitative Test Cases
QUALITATIVE_TEST_CASES = [
    {
        "type": "Synonymy Test",
        "description": "Two different queries that mean the same thing should yield similar results and scores.",
        "queries": [
            "What kind of funny tweets were posted about pets recently?",  # Paraphrased
            "Best tweets about cats and dogs this past week.",             # Keyword focused
        ],
        "target_doc_id": "doc_2" # Based on doc_2: "23 Of The Funniest Tweets About Cats And Dogs..."
    },
    {
        "type": "Polysemy/Context Test",
        "description": "The same word ('pay' or 'salary') used in political vs. employment context.",
        "queries": [
            "News about the government official's political salary reduction.", # Context: Government/Politics
            "An article discussing high compensation for corporate executives.", # Context: Corporate/Finance
        ],
        "target_doc_id": "doc_8411" # Based on doc_8411: "CDC Director Requests Salary Cut..." (POLITICS)
    },
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

def analyze_robustness(collection: chromadb.Collection, model: SentenceTransformer):
    """Runs the qualitative semantic tests."""
    
    print("\n--- Running Qualitative Semantic Robustness Test (Manual Review Required) ---")
    
    for case in QUALITATIVE_TEST_CASES:
        print("\n" + "="*70)
        print(f"TEST TYPE: {case['type']}")
        print(f"Description: {case['description']}")
        
        for i, query in enumerate(case['queries']):
            query_embedding = model.encode([query]).tolist()
            
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=N_RESULTS,
                include=['documents','distances']
            )
            
            print(f"\n  Query {i+1}: '{query}'")
            
            for j in range(len(results['documents'][0])):
                document = results['documents'][0][j]
                distance = results['distances'][0][j]
                doc_id = results['ids'][0][j]
                
                # Check if this document is the intended target
                is_target = "🎯 TARGET" if doc_id == case.get('target_doc_id') else ""
                
                print(f"    [Top {j+1}] ID: {doc_id} | Dist: {distance:.4f} {is_target}")
                print(f"            Headline: '{document}'")
        
        # Manual Review Guidance
        if case['type'] == "Synonymy Test":
            print("\n>> MANUAL CHECK: Do the two queries return the TARGET (doc_2) with similar LOW distances (< 0.5)?")
        elif case['type'] == "Polysemy/Context Test":
            print(f"\n>> MANUAL CHECK: Query 1 should strongly match the target (doc_8411). Query 2 should find other, non-political business news.")


if __name__ == "__main__":
    try:
        chroma_client = chromadb.Client()
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"Initialization Error: {e}")
        exit()
        
    collection = setup_chroma_and_embed_data(chroma_client, embedding_model, DATA_FILE_PATH)
    
    if collection:
        analyze_robustness(collection, embedding_model)