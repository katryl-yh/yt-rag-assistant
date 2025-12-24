"""Data models for RAG system.

Defines LanceDB schemas for embedding providers and ingestion strategies:
- TranscriptGeminiWhole: Gemini embeddings (3072-dim), whole-document
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

load_dotenv()

# Gemini embedding setup
embedding_model = get_registry().get("gemini-text").create(name="gemini-embedding-001")
EMBEDDING_DIM_GEMINI = 3072


class TranscriptGeminiWhole(LanceModel):
    """Whole-document transcript with Gemini embeddings (3072-dim).
    
    Requires Gemini API credentials and embedding registry configured.
    Use ingestion_gemini_whole.py to populate.
    """
    md_id: str
    filepath: str
    filename: str = Field(description="stem of the file without suffix")
    content: str = embedding_model.SourceField()
    embedding: Vector(EMBEDDING_DIM_GEMINI) = embedding_model.VectorField()
    embedding_model: str = Field(default="gemini-embedding-001")
    embedding_provider: str = Field(default="google-genai")
    embedding_dim: int = Field(default=3072)


class TranscriptGeminiChunk(LanceModel):
    """Two-stream chunk: raw (for vector) + cleaned (for LLM)."""
    md_id: str
    chunk_id: int
    raw_content: str = embedding_model.SourceField()   # used for embeddings
    cleaned_content: str = Field(description="Heavily cleaned for LLM context")
    token_count: int = Field(description="Approximate token count from tiktoken")
    embedding: Optional[Vector(EMBEDDING_DIM_GEMINI)] = embedding_model.VectorField(default=None)
    embedding_model: str = Field(default="gemini-embedding-001")
    embedding_provider: str = Field(default="google-genai")
    embedding_dim: int = Field(default=3072)


class Prompt(BaseModel):
    """User query input for RAG system."""
    prompt: str = Field(description="prompt from user, if empty consider it as missing")


class RagResponse(BaseModel):
    """Structured response from RAG agent including source provenance."""
    filename: str = Field(description="filename of retrieved file without suffix")
    filepath: str = Field(description="absolute path to the retrieved file")
    answer: str = Field(description="answer based on the retrieved file")