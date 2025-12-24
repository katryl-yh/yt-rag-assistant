from backend.constants import VECTOR_DATABASE_PATH
import lancedb
from backend.data_models import TranscriptGeminiChunk as ChunkModel
from lancedb.embeddings import get_registry
from pydantic import Field
import time
from pathlib import Path
import shutil
import json
from typing import List


# Define LanceDB schema with embeddings for chunks
embedding_model = get_registry().get("gemini-text").create(name="gemini-embedding-001")
EMBEDDING_DIM_GEMINI = 3072


def setup_vector_db(path: Path):
    """Create a fresh LanceDB at the given path, remove existing data for idempotent runs."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    vector_db = lancedb.connect(uri=path)
    vector_db.create_table("chunks", schema=ChunkModel, exist_ok=True)
    return vector_db


def load_chunks_from_markdown(data_path: Path) -> List[ChunkModel]:
    """
    Load chunks from markdown debug files.
    
    Expects format from chunk_transcripts.py:
    - Frontmatter with md_id, chunk_id, token_count
    - ## RAW section
    - ## CLEANED section
    """
    chunks: List[ChunkModel] = []
    
    for file in sorted(data_path.glob("*.md")):
        content = file.read_text(encoding="utf-8")
        
        # Parse frontmatter
        if not content.startswith("<!--"):
            print(f"⚠️  Skipping {file.name}: No frontmatter")
            continue
        
        try:
            # Extract metadata from frontmatter
            frontmatter_end = content.index("-->")
            frontmatter = content[4:frontmatter_end].strip()
            
            md_id = None
            chunk_id = None
            token_count = None
            
            for line in frontmatter.split("\n"):
                if "md_id:" in line:
                    md_id = line.split("md_id:")[1].strip()
                elif "chunk_id:" in line:
                    chunk_id = int(line.split("chunk_id:")[1].strip())
                elif "token_count:" in line:
                    token_count = int(line.split("token_count:")[1].strip())
            
            # Extract RAW and CLEANED sections
            raw_section = content.split("## RAW (for embeddings)")[1].split("## CLEANED")[0].strip()
            cleaned_section = content.split("## CLEANED (for LLM)")[1].strip()
            
            chunks.append(ChunkModel(
                md_id=md_id,
                chunk_id=chunk_id,
                raw_content=raw_section,
                cleaned_content=cleaned_section,
                token_count=token_count
            ))
            
        except Exception as e:
            print(f"⚠️  Error parsing {file.name}: {e}")
            continue
    
    return chunks


def ingest_chunks_to_vector_db(table, chunks: List[ChunkModel], batch_size: int = 10):
    """
    Ingest chunks into LanceDB with batching to handle API rate limits.
    
    Args:
        table: LanceDB table
        chunks: List of Chunk objects
        batch_size: Number of chunks per batch (default 10 to avoid rate limits)
    """
    total_chunks = len(chunks)
    print(f"\n📦 Ingesting {total_chunks} chunks in batches of {batch_size}...")
    
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        # Exclude embedding so LanceDB can auto-generate from raw_content
        batch_data = [chunk.model_dump(exclude={"embedding"}) for chunk in batch]
        
        table.add(batch_data)
        
        print(f"✓ Batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size} "
              f"({len(batch)} chunks) | Total rows: {table.count_rows()}")
        
        if i + batch_size < total_chunks:
            print("   ⏳ Waiting 30s for API rate limits...")
            time.sleep(30)


def main():
    # Paths
    CHUNK_DIR = Path(__file__).parent / "data_cleaned" / "03_chunked"
    DB_PATH = VECTOR_DATABASE_PATH / "transcripts_gemini_chunked"
    
    print(f"📂 Reading chunks from: {CHUNK_DIR}")
    print(f"🗄️  Setting up vector DB at: {DB_PATH}")
    
    # Setup database
    vector_db = setup_vector_db(DB_PATH)
    table = vector_db["chunks"]
    
    # Load chunks from markdown files
    print("\n📖 Loading chunks from markdown...")
    chunks = load_chunks_from_markdown(CHUNK_DIR)
    print(f"✓ Loaded {len(chunks)} chunks")
    
    # Show sample chunk
    if chunks:
        print("\n📋 Sample chunk:")
        print(json.dumps(chunks[0].model_dump(), indent=2))
    
    # Ingest to LanceDB
    ingest_chunks_to_vector_db(table, chunks, batch_size=10)
    
    print("\n✅ Ingestion complete!")
    print(f"   Total rows in DB: {table.count_rows()}")
    
    # Show sample embedded chunk
    print("\n🔍 Sample from database:")
    sample = table.to_pandas().head(1)
    print(sample[["md_id", "chunk_id", "token_count", "embedding_dim"]])


if __name__ == "__main__":
    main()