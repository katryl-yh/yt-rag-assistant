"""Data models for RAG system.

Defines LanceDB schemas for embedding providers and ingestion strategies:
- TranscriptGeminiWhole: Gemini embeddings (768-dim), whole-document
- TranscriptGeminiChunk: Two-stream chunk model (raw + cleaned versions)
Shared models:
- Prompt: user query input
- RagResponse: structured LLM response with sources
"""
from pydantic import BaseModel, Field
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from dotenv import load_dotenv
from typing import Optional
from backend.constants import EMBEDDING_MODEL_NAME

load_dotenv()

# Gemini embedding setup
try:
    embedding_model = get_registry().get("gemini-text").create(name=EMBEDDING_MODEL_NAME)
except Exception as e:
    print(f"WARNING: Failed to initialize embedding model: {e}")
    print("If running on Azure, ensure GOOGLE_API_KEY is set in Environment Variables.")
    # We re-raise because LanceModel definitions below depend on it.
    # However, printing the error helps debugging in Log Stream.
    raise

EMBEDDING_DIM_GEMINI = 768  # text-embedding-004 is 768-dim


class TranscriptGeminiWhole(LanceModel):
    """Whole-document transcript with Gemini embeddings.
    
    Requires Gemini API credentials and embedding registry configured.
    Use ingestion_gemini_whole.py to populate.
    """
    md_id: str
    filepath: str
    filename: str = Field(description="stem of the file without suffix")
    content: str = embedding_model.SourceField()
    summary: str = Field(description="summary of the video based on whole stranscipt")
    keywords: str = Field(description="stores 20-40 keywords about a particular video")
    embedding: Optional[Vector(EMBEDDING_DIM_GEMINI)] = embedding_model.VectorField(default=None)
    embedding_model: str = Field(default=EMBEDDING_MODEL_NAME)
    embedding_provider: str = Field(default="google-generativeai")
    embedding_dim: int = Field(default=EMBEDDING_DIM_GEMINI)


class TranscriptGeminiChunk(LanceModel):
    """Two-stream chunk: raw (for vector) + cleaned (for LLM)."""
    md_id: str
    chunk_id: int
    raw_content: str = embedding_model.SourceField()   # used for embeddings
    cleaned_content: str = Field(description="Heavily cleaned for LLM context")
    token_count: int = Field(description="Approximate token count from tiktoken")
    embedding: Optional[Vector(EMBEDDING_DIM_GEMINI)] = embedding_model.VectorField(default=None)
    embedding_model: str = Field(default=EMBEDDING_MODEL_NAME)
    embedding_provider: str = Field(default="google-genai")
    embedding_dim: int = Field(default=EMBEDDING_DIM_GEMINI)


class Prompt(BaseModel):
    """User query input for RAG system."""
    prompt: str = Field(description="prompt from user, if empty consider it as missing")
    retrieval_mode: str = Field(default="chunked", description="'chunked' for granular results or 'whole' for full document context")


class QueryRequest(BaseModel):
    """Request model for RAG query with history from frontend (stateless)."""
    query: str = Field(description="user question")
    retrieval_mode: str = Field(default="chunked", description="'chunked' or 'whole'")
    history: list[dict] = Field(default=[], description="conversation history from frontend") 


class RagResponse(BaseModel):
    """Structured response from RAG agent including source provenance."""
    filename: str = Field(description="Filename of the retrieved source (no extension), as shown in the retrieved context under 'Filename'.")

    filepath: str = Field(
        description=(
            "Exact string found in the 'Filename' line of the retrieved context. "
            "Do not add or remove anything. If it says '(Chunk X)', include it. If not, do not add it."
        )
    )
    
    answer: str = Field(description="answer based on the retrieved file")

class VideoMetadata(BaseModel):
    """LLM-generated metadata for a video transcript.
    
    Used to structure Gemini's response when generating summary and keywords.
    """
    summary: str = Field(description="1-3 sentence summary suitable for YouTube description")
    keywords: str = Field(description="20-40 comma-separated keywords for YouTube tags")


class VideoMetadataResponse(BaseModel):
    """API response model for video metadata endpoints."""
    md_id: str = Field(description="unique hash identifier for the video")
    filename: str = Field(description="original filename without extension")
    summary: str = Field(description="video summary")
    keywords: str = Field(description="comma-separated keywords")