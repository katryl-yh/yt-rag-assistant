"""Data models for RAG system.

Defines LanceDB schemas for embedding providers and ingestion strategies:
- TranscriptGeminiWhole: Gemini embeddings (3072-dim), whole-document
- TranscriptGeminiChunked: Gemini embeddings (3072-dim), sentence-based chunks

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


class TranscriptGeminiChunked(LanceModel):
    """Sentence-based chunk with Gemini embeddings (3072-dim).
    
    Each chunk represents a semantically coherent segment of a transcript,
    created using sentence-boundary splitting with token-aware sizing.
    
    Requires Gemini API credentials and embedding registry configured.
    Use ingestion_gemini_chunked.py to populate.
    """
    md_id: str = Field(description="stem of the file without suffix")
    chunk_id: int = Field(description="chunk index within the source document")
    filepath: str = Field(description="absolute path to the original source file")
    filename: str = Field(description="stem of the file without suffix")
    content: str = embedding_model.SourceField()
    embedding: Vector(EMBEDDING_DIM_GEMINI) = embedding_model.VectorField()
    
    # Chunk-specific metadata
    token_count: int = Field(description="number of tokens in this chunk")
    start_sentence_idx: int = Field(description="starting sentence index in original document")
    end_sentence_idx: int = Field(description="ending sentence index in original document")
    
    # Model metadata
    embedding_model: str = Field(default="gemini-embedding-001")
    embedding_provider: str = Field(default="google-genai")
    embedding_dim: int = Field(default=3072)
    
    # Chunking strategy metadata
    chunking_strategy: str = Field(default="sentence-based-sentencepiece")
    target_tokens: int = Field(default=350)
    hard_max_tokens: int = Field(default=600)
    overlap_ratio: float = Field(default=0.15)


class Prompt(BaseModel):
    """User query input for RAG system."""
    prompt: str = Field(description="prompt from user, if empty consider it as missing")


class RagResponse(BaseModel):
    """Structured response from RAG agent including source provenance."""
    filename: str = Field(description="filename of retrieved file without suffix")
    filepath: str = Field(description="absolute path to the retrieved file")
    answer: str = Field(description="answer based on the retrieved file")
    # Chunk-specific fields if using chunked retrieval
    chunk_id: int | None = Field(default=None, description="chunk index if using chunked retrieval")
    token_count: int | None = Field(default=None, description="tokens in retrieved chunk")