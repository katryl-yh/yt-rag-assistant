import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Detect if running in Azure Functions
IS_AZURE = os.getenv("WEBSITE_SITE_NAME") is not None

# Define paths
BASE_DIR = Path(__file__).parents[1]
LOCAL_KNOWLEDGE_BASE = BASE_DIR / "knowledge_base"

if IS_AZURE:
    # In Azure, we must use /tmp for writable storage
    # LanceDB needs to write lock files even for read operations
    VECTOR_DATABASE_PATH = Path("/tmp/knowledge_base")
    
    # Copy the database to /tmp if it doesn't exist
    # Note: /tmp is ephemeral in Azure Functions, so this runs on cold starts
    if not VECTOR_DATABASE_PATH.exists():
        try:
            print(f"Copying knowledge base from {LOCAL_KNOWLEDGE_BASE} to {VECTOR_DATABASE_PATH}...")
            shutil.copytree(LOCAL_KNOWLEDGE_BASE, VECTOR_DATABASE_PATH, dirs_exist_ok=True)
            print("Copy complete.")
        except Exception as e:
            print(f"Error copying knowledge base: {e}")
            # Fallback to local path (might fail due to read-only FS)
            VECTOR_DATABASE_PATH = LOCAL_KNOWLEDGE_BASE
    else:
        print(f"Knowledge base already exists at {VECTOR_DATABASE_PATH}")
else:
    VECTOR_DATABASE_PATH = LOCAL_KNOWLEDGE_BASE

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