from pathlib import Path

VECTOR_DATABASE_PATH = Path(__file__).parents[1] / "knowledge_base"

# LLM Configuration
GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"
EMBEDDING_MODEL_NAME = "text-embedding-004"