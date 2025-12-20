from backend.constants import VECTOR_DATABASE_PATH
import lancedb
from backend.data_models import TranscriptGeminiWhole
import time
from pathlib import Path
import shutil


def setup_vector_db(path: Path):
    """Create a fresh LanceDB at the given path, remove existing data for idempotent runs."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    vector_db = lancedb.connect(uri=path)
    vector_db.create_table("transcripts", schema=TranscriptGeminiWhole, exist_ok=True)

    return vector_db


def ingest_mds_to_vector_db(table, data_path: Path):
    for file in data_path.glob("*.md"):
        with open(file, "r") as f:
            content = f.read()

        md_id = file.stem
        table.delete(f"md_id = '{md_id}'")

        table.add(
            [
                {
                    "md_id": md_id,
                    "filepath": str(file),
                    "filename": file.stem,
                    "content": content,
                    "embedding_model": "gemini-embedding-001",
                    "embedding_provider": "google-genai",
                    "embedding_dim": 3072,
                }
            ]
        )

        print(table.to_pandas().shape)
        print(table.to_pandas()["filename"])
        time.sleep(30)


if __name__ == "__main__":
    # Use normalized transcripts (whole documents, no chunks)
    DATA_PATH = Path(__file__).parent / "data_cleaned" / "02_normalized"
    
    db_path = VECTOR_DATABASE_PATH / "transcripts_gemini_whole"
    print(f"Setting up vector DB at: {db_path}")
    print(f"Reading transcripts from: {DATA_PATH}")

    vector_db = setup_vector_db(db_path)
    ingest_mds_to_vector_db(vector_db["transcripts"], DATA_PATH)
    print("\nIngestion complete!")