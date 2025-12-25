import os
from pydantic_ai import Agent
from backend.data_models import RagResponse
from backend.constants import VECTOR_DATABASE_PATH
import lancedb

# Connect to unified database
vector_db = lancedb.connect(uri=VECTOR_DATABASE_PATH / "transcripts_unified")

rag_agent = Agent(
    model="google-gla:gemini-2.5-flash",
    retries=2,
    system_prompt=(
        "You are a knowledgable data-engineering YouTuber with nerdy humor.",
        "Always answer based on the retrieved knowledge, but you can mix in your expertise to make the answer more coherent",
        "Don't hallucinate, rather say you can't answer it if the user prompts outside of the retrieved knowledge",
        "Keep answers concise (max 6 sentences), practical, and to-the-point. ",
        "Always cite the source filename in your answer, and keep the tone light with subtle nerdy jokes."
    ),
    output_type=RagResponse,
)

@rag_agent.tool_plain
def retrieve_top_documents(query: str, k=3) -> str:
    """
    Uses vector search on chunks for granular retrieval.
    The unified database uses token-based chunking for better context retrieval.
    """
    results = vector_db["video_chunks"].search(query=query).limit(k).to_list()
    
    if not results:
        return "No relevant documents found."
    
    top_result = results[0]
    
    # Extract information from chunk record
    md_id = top_result.get("md_id", "Unknown")
    chunk_id = top_result.get("chunk_id", 0)
    filepath = f"Chunk {chunk_id}"
    content = top_result.get("cleaned_content", "")

    return f"""
    Video ID: {md_id},

    Location: {filepath},

    Content: {content}
    """