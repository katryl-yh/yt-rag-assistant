import os
from pathlib import Path

VECTOR_DATABASE_PATH = Path(__file__).parents[1] / "knowledge_base"

# LLM Model Configuration
# GEMINI_MODELS dictionary maps friendly model keys to their full provider-prefixed names.
# This allows easy model switching without code changes.
GEMINI_MODELS = {
    "flash-lite": "google-gla:gemini-2.5-flash-lite",  # 10M tokens/day (recommended)
    "flash-2.0": "google-gla:gemini-2.0-flash",        # 10M tokens/day (alternative)
    "pro-2.5": "google-gla:gemini-2.5-pro",            # 5M tokens/day (higher quality)
    "flash-2.5": "google-gla:gemini-2.5-flash",        # 3M tokens/day (strictest)
}

# GEMINI_MODEL_KEY: A key that selects which model from the GEMINI_MODELS dictionary to use.
# Read from environment variable, defaults to "flash-lite" for best daily quota (10M tokens/day).
# Change this in .env to switch models without restarting code.
GEMINI_MODEL_KEY = os.getenv("GEMINI_MODEL_KEY", "flash-lite")

# LLM_MODEL_NAME: The actual full model name resolved from the dictionary using GEMINI_MODEL_KEY.
# This is what gets passed to the PydanticAI Agent.
LLM_MODEL_NAME = GEMINI_MODELS.get(GEMINI_MODEL_KEY, GEMINI_MODELS["flash-lite"])

# Embedding Configuration
EMBEDDING_MODEL_NAME = "text-embedding-004"