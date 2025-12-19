"""Data models for RAG system.

Defines LanceDB schemas for different embedding providers and ingestion strategies:
- TranscriptGeminiWhole: Gemini embeddings (3072-dim), whole-document
- TranscriptMpnetWhole: SentenceTransformers mpnet embeddings (768-dim), whole-document
- Future: TranscriptGeminiChunk, TranscriptMpnetChunk for chunk-level ingestion

Shared models:
- Prompt: user query input
- RagResponse: structured LLM response with sources
"""
from pydantic import BaseModel, Field
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from dotenv import load_dotenv

load_dotenv()

# Gemini embedding setup
embedding_model = get_registry().get("gemini-text").create(name="gemini-embedding-001")
EMBEDDING_DIM_GEMINI = 3072

# Local embedding setup
local_model = get_registry().get("sentence-transformers").create(
    name="all-mpnet-base-v2"
)
EMBEDDING_DIM_MPNET = 768


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


class TranscriptLocalWhole(LanceModel):
    """Whole-document transcript with SentenceTransformers mpnet embeddings (768-dim).
    
    Uses local model (no API required). Embeddings are computed by LanceDB.
    Use ingestion_mpnet_whole.py to populate. Supports search(query=...) directly.
    """
    md_id: str
    filepath: str
    filename: str = Field(description="stem of the file without suffix")
    content: str = local_model.SourceField()
    embedding: Vector(EMBEDDING_DIM_MPNET) = local_model.VectorField()
    embedding_model: str = Field(default="all-mpnet-base-v2")
    embedding_provider: str = Field(default="sentence-transformers")
    embedding_dim: int = Field(default=768)


# Placeholder for future chunk-level models
# class TranscriptGeminiChunk(LanceModel):
#     """Chunk-level transcript with Gemini embeddings."""



# class TranscriptMpnetChunk(LanceModel):
#     """Chunk-level transcript with mpnet embeddings."""



class Prompt(BaseModel):
    """User query input for RAG system."""
    prompt: str = Field(description="prompt from user, if empty consider it as missing")


class RagResponse(BaseModel):
    """Structured response from RAG agent including source provenance."""
    filename: str = Field(description="filename of retrieved file without suffix")
    filepath: str = Field(description="absolute path to the retrieved file")
    answer: str = Field(description="answer based on the retrieved file")