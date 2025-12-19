# test_rag_mpnet_whole.py
"""
Test RAG agent with mpnet embeddings using PydanticAI tool pattern.
This creates a dedicated agent with mpnet-based retrieval tool.
"""
import asyncio
import lancedb
import numpy as np
from sentence_transformers import SentenceTransformer # creates MPNet embeddings
from backend.constants import VECTOR_DATABASE_PATH
from backend.data_models import RagResponse
from pydantic_ai import Agent
import time
from google.genai import errors as genai_errors  # SDK error class to catch Gemini server errors

MODEL = "all-mpnet-base-v2" # specifies the MPNet embedding model
DB = lancedb.connect(uri=VECTOR_DATABASE_PATH / "transcripts_mpnet_whole")
TBL = DB["transcripts"]

# Initialize the same embedding model used during ingestion, ensuring:
# Embedding dimensionality matches
# Semantic space is consistent
# Similarity scores are meaningful
sentence_model = SentenceTransformer(MODEL)

# Create dedicated agent with mpnet retrieval tool
mpnet_rag_agent = Agent(
    model="google-gla:gemini-2.5-flash",
    retries=2,
    model_settings={"temperature": 0.0},  # Set to 0 for reproducible outputs
    system_prompt=(
        "You are an expert in the data engineering field",
        "Always answer based on the retrieved knowledge, but you can mix in your expertise to make the answer more coherent",
        "Don't hallucinate, rather say you can't answer it if the user prompts outside of the retrieved knowledge",
        "Make sure to keep the answer clear and concise, getting to the point directly, max 6 sentences",
        "Also describe which file you have used as source",
    ),
    output_type=RagResponse,
)

@mpnet_rag_agent.tool_plain
def retrieve_top_documents(query: str, k=5) -> str:
    """
    Uses mpnet embeddings to find the closest k matching documents to the query.
    """
    # Compute embedding using SentenceTransformer (same as ingestion)
    q_emb = sentence_model.encode(query, convert_to_numpy=True).astype(float)
    
    # Manual cosine similarity search (matches test_embeddings_mpnet.py)
    df = TBL.to_pandas()
    embs = np.vstack(df["embedding"].apply(lambda x: np.array(x, dtype=float)).to_numpy())
    q_norm = np.linalg.norm(q_emb)
    emb_norms = np.linalg.norm(embs, axis=1)
    sims = (embs @ q_emb) / (emb_norms * q_norm + 1e-12)

    # Select top-k documents: sorts similarity scores
    topk = sims.argsort()[-k:][::-1]
    # Takes the top k highest values and converts selected rows into Python dictionaries
    results = [df.iloc[i].to_dict() for i in topk]
    
    # Debug: print retrieved documents
    print(f"\n[DEBUG] Retrieved {len(results)} documents:")
    for i, doc in enumerate(results):
        print(f"  {i+1}. {doc.get('filename', 'NO FILENAME')} (similarity: {sims[topk[i]]:.4f})")
    
    top_result = results[0]
    print(f"\n[DEBUG] Top result: {top_result.get('filename', 'MISSING')}")
    
    # The tool returns only the top document, formatted as text.
    # This string becomes context injected into the LLM!
    return f"""
    Filename: {top_result["filename"]},
    
    Filepath: {top_result.get("filepath", "N/A")},

    Content: {top_result["content"]}
    """


def call_agent_with_retries(query: str, attempts=4, initial_delay=1.0):
    """Call mpnet RAG agent with retry logic."""
    delay = initial_delay
    for i in range(attempts):
        try:
            # Agent will automatically call retrieve_top_documents tool when needed
            return asyncio.run(mpnet_rag_agent.run(query))
        except genai_errors.ServerError as e:
            print(f"LLM ServerError attempt {i+1}/{attempts}: {e}")
        except Exception as e:
            print(f"LLM call failed attempt {i+1}/{attempts}: {e}")
        if i < attempts - 1:
            time.sleep(delay)
            delay *= 2
    return f"LLM unavailable after {attempts} attempts."

if __name__ == "__main__":
    query = "Explain what ETL is and give a short example."
    print(f"\nQuerying with mpnet embeddings: {query}\n")
    result = call_agent_with_retries(query)
    print(result)