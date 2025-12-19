from backend.constants import VECTOR_DATABASE_PATH, DATA_PATH
from backend.data_models import TranscriptLocalWhole
import lancedb
from pathlib import Path
from tqdm import tqdm
import time
import os
import shutil

# Disable GPU and ONNX to avoid segmentation faults
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU only
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_vector_db(path: Path):
    """Setup vector DB, removing existing data for idempotent runs."""
    # Make idempotent: remove existing DB and recreate
    if path.exists():
        shutil.rmtree(path)
    
    path.mkdir(parents=True, exist_ok=True)
    vector_db = lancedb.connect(uri=path)
    vector_db.create_table("transcripts", schema=TranscriptLocalWhole, exist_ok=True)
    return vector_db


def ingest_mds_to_vector_db(table):
    """Ingest markdown files. LanceDB handles embedding computation via all-mpnet-base-v2."""
    for file in tqdm(list(DATA_PATH.glob("*.md")), desc="Ingesting files"):
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()

        md_id = file.stem

        # LanceDB will compute embeddings automatically via the registered model
        table.add(
            [
                {
                    "md_id": md_id,
                    "filepath": str(file),
                    "filename": file.stem,
                    "content": content,
                    "embedding_model": "all-mpnet-base-v2",
                    "embedding_provider": "sentence-transformers",
                    "embedding_dim": 768,
                }
            ]
        )

        # small pause to free resources
        time.sleep(0.1)


if __name__ == "__main__":
    # Ensure DB is saved in knowledge_base/transcripts_mpnet_whole
    db_path = VECTOR_DATABASE_PATH / "transcripts_mpnet_whole"
    print(f"Setting up vector DB at: {db_path}")
    
    vector_db = setup_vector_db(db_path)
    ingest_mds_to_vector_db(vector_db["transcripts"])
    
    print(f"\nIngestion complete! Vector DB saved at: {db_path}")
    print("You can now use search(query=...) instead of search(vector=...)")
