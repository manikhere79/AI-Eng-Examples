import os
import json
import asyncio
from rag_setup import setup_chroma_and_embed_data, get_context_for_rag, EMBEDDING_MODEL_NAME, DATA_FILE_PATH

# --- Configuration for Gemini API ---
# NOTE: Leave this API key as an empty string. Canvas will provide the necessary credentials at runtime.
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"
MODEL_NAME = "gemini-2.5-flash-preview-05-20"

# --- Prompt Engineering: The RAG Template ---
# This prompt guides the LLM on how to use the context provided by the vector search.
SYSTEM_INSTRUCTION_TEXT = (
    "You are an expert news summarizer. Your task is to provide a concise, single-paragraph "
    "answer to the user's question based ONLY on the provided context. "
    "Do not use any external knowledge. If the context does not contain the answer, "
    "you must state, 'I cannot find the answer in the provided news headlines.'"
)

def create_rag_prompt(user_query: str, context: str) -> dict:
    """
    Constructs the final payload structure for the Gemini API, including the system instruction
    and the user query combined with the retrieved context.
    """
    
    # 1. Define the system instruction to guide the model's behavior
    system_instruction_part = {
        "parts": [{"text": SYSTEM_INSTRUCTION_TEXT}]
    }
    
    # 2. Define the main user query payload, which includes the retrieved context
    rag_query = f"User Query: {user_query}\n\n{context}\n\nBased on the headlines above, please answer the user's query."
    user_content_part = {
        "parts": [{"text": rag_query}]
    }
    
    payload = {
        "contents": [user_content_part],
        "systemInstruction": system_instruction_part
    }
    return payload

async def generate_grounded_answer(payload: dict) -> str:
    """
    Makes the non-streaming API call to the Gemini model and extracts the response.
    Includes basic error handling and exponential backoff for reliability.
    """
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # Note: We use the local fetch function provided by the Canvas environment
            # This handles authentication via the empty GEMINI_API_KEY string

            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    GEMINI_API_URL,
                    headers={'Content-Type': 'application/json'},
                    data=json.dumps(payload)
                ) as response:
                    if response.status != 200:
                        print(f"API Error (Status {response.status}): {await response.text()}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt) # Exponential backoff
                            continue
                        return "Error: Failed to connect to the LLM service after multiple retries."

                    result = await response.json()

                    # Safely extract the generated text
                    candidate = result.get('candidates', [{}])[0]
                    generated_text = candidate.get('content', {}).get('parts', [{}])[0].get('text', 'No response generated.')

                    return generated_text

        except Exception as e:
            print(f"Fetch failed on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt) # Exponential backoff
                continue
            return "Error: An unexpected error occurred during the API call."
            
    return "Error: Failed to generate response."


async def main():
    """
    Main function to orchestrate the RAG process: Retrieval -> Prompt -> Generation.
    """
    # 1. Initialize ChromaDB and Sentence Transformer (assuming it's already done by running rag_setup.py)
    # We re-initialize here to get client/model instances, but setup_chroma_and_embed_data handles idempotency.
    import chromadb
    from sentence_transformers import SentenceTransformer
    
    try:
        chroma_client = chromadb.Client()
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"Initialization Error: {e}")
        print("Please ensure 'chromadb' and 'sentence-transformers' libraries are installed.")
        return

    # Ensure the collection is set up before querying (runs embedding if necessary)
    setup_chroma_and_embed_data(chroma_client, embedding_model, DATA_FILE_PATH)

    print("\n" + "="*70)
    print("🔥 Running Complete RAG Pipeline (Retrieval-Augmented Generation)")
    print("="*70)

    # --- Test Case 1: Clear and direct question (should work) ---
    query_1 = "What happened to the American Airlines flyer who hit a flight attendant?"
    print(f"\n[Test 1] User Query: {query_1}")
    
    # Retrieval Step: Get context from the vector database
    context_1 = get_context_for_rag(chroma_client, embedding_model, query_1)
    
    # Prompt Engineering Step: Assemble the RAG payload
    payload_1 = create_rag_prompt(query_1, context_1)
    
    # Generation Step: Get the grounded answer from Gemini
    answer_1 = await generate_grounded_answer(payload_1)
    
    print("\n--- Retrieved Context ---")
    print(context_1)
    print("\n--- Grounded Answer from Gemini ---")
    print(answer_1)
    
    print("-" * 50)
    
    # --- Test Case 2: Unanswerable question (should fail gracefully) ---
    # The data contains headlines, not detailed text, and no sports scores.
    query_2 = "What was the final score of the Lakers game last night?"
    print(f"\n[Test 2] User Query: {query_2}")
    
    context_2 = get_context_for_rag(chroma_client, embedding_model, query_2)
    payload_2 = create_rag_prompt(query_2, context_2)
    answer_2 = await generate_grounded_answer(payload_2)
    
    print("\n--- Retrieved Context ---")
    print(context_2)
    print("\n--- Grounded Answer from Gemini (Expected to be 'Unanswerable') ---")
    print(answer_2)
    
    print("\n" + "="*70)
    print("RAG Pipeline Complete.")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(main())