import os
from pydantic_ai import Agent
from backend.data_models import RagResponse
from backend.constants import VECTOR_DATABASE_PATH
import lancedb

db_type = os.getenv("RAG_DB_TYPE", "whole")
vector_db = lancedb.connect(uri=VECTOR_DATABASE_PATH / f"transcripts_gemini_{db_type}")

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
    Uses vector search to find the closest k matching documents to the query
    """
    db_type = os.environ.get("RAG_DB_TYPE", "whole")
    
    if db_type == "chunked":
        table_name = "chunks"
    else:
        table_name = "transcripts"
    
    results = vector_db[table_name].search(query=query).limit(k).to_list()
    top_result = results[0]

    # Handle different column names for chunked vs whole
    if db_type == "chunked":
        filename = top_result.get("md_id", "Unknown")
        filepath = f"Chunk {top_result.get('chunk_id', 'N/A')}"
        content = top_result.get("cleaned_content", "")
    else:
        filename = top_result.get("filename", "Unknown")
        filepath = top_result.get("filepath", "Unknown")
        content = top_result.get("content", "")

    return f"""
    Filename: {filename},

    Filepath: {filepath},

    Content: {content}
    """