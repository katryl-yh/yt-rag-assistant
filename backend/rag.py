import os
from pydantic_ai import Agent
from backend.data_models import RagResponse
from backend.constants import VECTOR_DATABASE_PATH, LLM_MODEL_NAME
import lancedb

# Connect to unified database
# Lazy loading to prevent import-time errors
vector_db = None

def get_vector_db():
    global vector_db
    if vector_db is None:
        vector_db = lancedb.connect(uri=VECTOR_DATABASE_PATH / "transcripts_unified")
    return vector_db

# Store retrieval mode in a context variable (will be set by API)
_retrieval_mode = "chunked"

rag_agent = Agent(
    model=LLM_MODEL_NAME,
    retries=2,
    system_prompt=(
        "You are a knowledgable data-engineering YouTuber with nerdy humor.",
        "Use ONLY the retrieved context to answer. If the context does not clearly contain the answer, reply: 'I don't know based on the provided documents.'",
        "Do not rely on outside knowledge. Reject questions outside the retrieved context with the same refusal line.",
        "If the question is about a person, entity, or fact not clearly mentioned in the retrieved context, respond with the refusal line.",
        "You will see context blocks labeled [Result N]. If none of these blocks contain the asked entity/keywords, respond with the refusal line.",
        "Keep answers concise (max 6 sentences), practical, and to-the-point.",
        "Use the 'Filename' line from the retrieved context as your source identifier in the response.",
        "Copy the 'Filename' line EXACTLY as it appears in the context into the 'filepath' field of the response. Do not invent chunk numbers if they are not present.",
        "Only cite a source if you found the answer in the retrieved context; otherwise just state you don't know. Keep the tone light with subtle nerdy jokes."
    ),
    output_type=RagResponse,
)

@rag_agent.tool_plain
def retrieve_top_documents(query: str, k=3) -> str:
    """
    Uses vector search to retrieve relevant documents.
    Retrieval mode determines source: 'chunked' uses granular chunks, 'whole' uses full documents.
    Returns top-k contexts to reduce hallucination risk.
    """
    global _retrieval_mode
    
    db = get_vector_db()
    if _retrieval_mode == "whole":
        results = db["parent_videos"].search(query=query).limit(k).to_list()
        if not results:
            return "No relevant documents found."
        blocks = []
        for idx, r in enumerate(results, 1):
            filename = r.get("filename", "Unknown")
            content = r.get("content", "")
            blocks.append(
                f"[Result {idx}]\nFilename: {filename}\nContent: {content}"
            )
        return "\n\n".join(blocks)
    else:  # chunked mode (default)
        results = db["video_chunks"].search(query=query).limit(k).to_list()
        if not results:
            return "No relevant documents found."

        # Resolve md_id -> filename from parent table so chunk citations are stable
        # even when the chunk table does not store filename.
        md_ids = []
        for r in results:
            md_id = r.get("md_id")
            if md_id and md_id not in md_ids:
                md_ids.append(md_id)

        filename_by_md_id = {}
        if md_ids:
            try:
                parent_table = db["parent_videos"]
                where_expr = " OR ".join([f"md_id = '{mid}'" for mid in md_ids])
                parent_rows = parent_table.search().where(where_expr).limit(len(md_ids)).to_list()
                filename_by_md_id = {
                    pr.get("md_id"): pr.get("filename", "Unknown")
                    for pr in parent_rows
                    if pr.get("md_id")
                }
            except Exception:
                # Best-effort: if anything goes wrong, fall back to 'Unknown'.
                filename_by_md_id = {}

        blocks = []
        for idx, r in enumerate(results, 1):
            md_id = r.get("md_id")
            filename = filename_by_md_id.get(md_id) or r.get("filename", "Unknown")
            chunk_id = r.get("chunk_id", "Unknown")
            content = r.get("cleaned_content", "")
            blocks.append(
                f"[Result {idx}]\nFilename: {filename} (Chunk {chunk_id})\nContent: {content}"
            )
        return "\n\n".join(blocks)

def set_retrieval_mode(mode: str):
    """Set the retrieval mode for the RAG agent."""
    global _retrieval_mode
    _retrieval_mode = mode