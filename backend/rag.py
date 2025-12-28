import os
from pydantic_ai import Agent
from backend.data_models import RagResponse
from backend.constants import VECTOR_DATABASE_PATH, LLM_MODEL_NAME
import lancedb

# Connect to unified database
vector_db = lancedb.connect(uri=VECTOR_DATABASE_PATH / "transcripts_unified")

# Store retrieval mode in a context variable (will be set by API)
_retrieval_mode = "chunked"

rag_agent = Agent(
    model=LLM_MODEL_NAME,
    retries=2,
    system_prompt=(
        "You are a knowledgable data-engineering YouTuber with nerdy humor.",
        "Always answer based on the retrieved knowledge, but you can mix in your expertise to make the answer more coherent",
        "Don't hallucinate, rather say you can't answer it if the user prompts outside of the retrieved knowledge",
        "Keep answers concise (max 6 sentences), practical, and to-the-point. ",
        "IMPORTANT: Extract the 'Location' field from the retrieved context (format: filename (Chunk X)) and use it as the filepath in your response.",
        "Always cite the source filename in your answer, and keep the tone light with subtle nerdy jokes."
    ),
    output_type=RagResponse,
)

@rag_agent.tool_plain
def retrieve_top_documents(query: str, k=3) -> str:
    """
    Uses vector search to retrieve relevant documents.
    Retrieval mode determines source: 'chunked' uses granular chunks, 'whole' uses full documents.
    """
    global _retrieval_mode
    
    if _retrieval_mode == "whole":
        # Query whole documents for broader context
        results = vector_db["parent_videos"].search(query=query).limit(k).to_list()
        
        if not results:
            return "No relevant documents found."
        
        top_result = results[0]
        md_id = top_result.get("md_id", "Unknown")
        filename = top_result.get("filename", "Unknown")
        filepath = top_result.get("filepath", "Unknown")
        content = top_result.get("content", "")
        
        return f"""
    Filename: {filename},

    Filepath: {filepath},

    Content: {content}
    """
    else:
        # Query chunks for granular retrieval (default)
        results = vector_db["video_chunks"].search(query=query).limit(k).to_list()
        
        if not results:
            return "No relevant documents found."
        
        top_result = results[0]
        md_id = top_result.get("md_id", "Unknown")
        chunk_id = top_result.get("chunk_id", 0)
        content = top_result.get("cleaned_content", "")
        
        # Look up filename from parent_videos table using md_id
        parent_results = vector_db["parent_videos"].search().where(f"md_id = '{md_id}'").to_list()
        filename = "Unknown"
        if parent_results:
            filename = parent_results[0].get("filename", "Unknown")
        
        filepath = f"{filename} (Chunk {chunk_id})"
        
        return f"""
    Video ID: {md_id},

    Location: {filepath},

    Content: {content}
    """

def set_retrieval_mode(mode: str):
    """Set the retrieval mode for the RAG agent."""
    global _retrieval_mode
    _retrieval_mode = mode