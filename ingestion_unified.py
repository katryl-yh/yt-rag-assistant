"""Unified ingestion pipeline for YouTube transcripts.

This script processes markdown transcripts and creates both:
1. parent_videos table: whole document embeddings + metadata (summary, keywords)
2. video_chunks table: chunked embeddings for granular retrieval

Features:
- Single-pass processing (efficient, avoids redundant LLM calls)
- Checkpointing (resume from failures)
- Rate limiting (respects Gemini free tier)
- Deterministic IDs (MD5 hash of filename)
"""

import warnings
import os

# Suppress all FutureWarnings from the generativeai package
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")
# This environment variable often helps with underlying gRPC/Google log noise
os.environ["GRPC_VERBOSITY"] = "ERROR"

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import shutil

import lancedb
from pydantic_ai import Agent
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

from backend.constants import VECTOR_DATABASE_PATH, LLM_MODEL_NAME
from backend.data_models import (
    TranscriptGeminiWhole,
    TranscriptGeminiChunk,
    VideoMetadata,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = Path(__file__).parent / "data_cleaned" / "02_normalized"
DB_PATH = VECTOR_DATABASE_PATH / "transcripts_unified"
CHECKPOINT_FILE = Path(__file__).parent / "ingestion_checkpoint.json"

# Chunking parameters (token-based)
CHUNK_SIZE = 400  # tokens (target 300-400)
CHUNK_OVERLAP = 100  # tokens (25% overlap)

# Rate limiting
SLEEP_AFTER_LLM_CALL = 10  # seconds (conservative to avoid 429s)


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def load_checkpoint() -> Dict:
    """Load processing checkpoint from disk."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "processed_files": {},
        "total_processed": 0,
        "last_error": None,
        "started_at": None,
        "last_updated": None
    }


def save_checkpoint(checkpoint: Dict):
    """Save processing checkpoint to disk."""
    from datetime import datetime
    checkpoint["last_updated"] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)


def mark_file_processed(checkpoint: Dict, filename: str):
    """Mark a file as successfully processed."""
    from datetime import datetime
    checkpoint["processed_files"][filename] = datetime.now().isoformat()
    checkpoint["total_processed"] = len(checkpoint["processed_files"])
    save_checkpoint(checkpoint)


# ============================================================================
# ID GENERATION
# ============================================================================

def generate_md_id(filename: str) -> str:
    """Generate deterministic MD5 hash from filename (without extension)."""
    # Remove .md extension
    name_without_ext = filename.replace(".md", "")
    return hashlib.md5(name_without_ext.encode()).hexdigest()


# ============================================================================
# LLM METADATA GENERATION
# ============================================================================

def create_metadata_agent() -> Agent:
    """Create PydanticAI agent for generating summary + keywords."""
    agent = Agent(
        model=LLM_MODEL_NAME,
        retries=3,  # Retry on failures
        system_prompt=(
            "You are a YouTube content analyzer. Generate concise metadata for video transcripts.",
            "For the summary: Write 1-3 sentences suitable for a YouTube video description. Make it engaging and informative.",
            "For keywords: Extract exactly 20-40 relevant keywords/phrases that would work as YouTube tags.",
            "Keywords should be comma-separated, lowercase, and cover: main topics, technologies, concepts, and use cases.",
            "Example keywords format: python, data engineering, api tutorial, rest api, fastapi, backend development"
        ),
        output_type=VideoMetadata,
    )
    return agent


async def generate_metadata(content: str, filename: str) -> VideoMetadata:
    """Generate summary and keywords using Gemini with structured output."""
    agent = create_metadata_agent()

    prompt = f"""Video title: {filename}

Transcript:
{content}"""

    while True:  # Keep trying until success
        try:
            result = await agent.run(prompt)
            return result.output
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower() or "resource_exhausted" in str(e).lower():
                # If we hit 20/day (or other limit), we might need to wait.
                # Let's try waiting 120 seconds and trying again.
                print(f"  ⚠️  Quota hit (429). Waiting 120s to retry...")
                await asyncio.sleep(120)
            else:
                # If it's a different error (like a 500), re-raise it
                raise e


def heavy_clean_text(text: str) -> str:
    """Aggressive cleanup for LLM consumption (remove fillers, collapse whitespace)."""
    import re

    fillers = ["uh", "um", "basically", "like", "you know", "i mean"]
    pattern = re.compile(r"\b(" + "|".join(fillers) + r")\b", re.IGNORECASE)
    text = pattern.sub(" ", text)
    # Collapse repeated spaces/newlines while preserving paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


async def embed_with_retry(text_or_texts, max_retries: int = 3, base_sleep: float = 1.0):
    """Deprecated: embeddings are generated by LanceDB on insert. Kept for future use."""
    return None


def normalize_keywords(raw: str) -> str:
    """Normalize LLM keywords output to comma-separated tags."""
    import re

    parts = re.split(r"[\n,]", raw)
    cleaned = []
    for part in parts:
        item = part.strip()
        # Drop bullet/number prefixes like '- ', '1. ', '• '
        item = re.sub(r"^(?:[-*•]\s+|\d+[.)]\s+)", "", item)
        if item:
            cleaned.append(item)
    return ", ".join(cleaned)


# ============================================================================
# CHUNKING
# ============================================================================

async def chunk_content(content: str, md_id: str) -> List[TranscriptGeminiChunk]:
    """Chunk content using token-based splitting and build LanceModel records."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        encoding_name="cl100k_base",  # GPT-4 tokenizer
    )
    
    chunks = splitter.split_text(content)

    enc = tiktoken.get_encoding("cl100k_base")
    chunk_tokens = [len(enc.encode(c)) for c in chunks]

    chunk_records: List[TranscriptGeminiChunk] = []
    for idx, chunk_text in enumerate(chunks):
        chunk_records.append(
            TranscriptGeminiChunk(
                md_id=md_id,
                chunk_id=idx,
                raw_content=chunk_text,
                cleaned_content=heavy_clean_text(chunk_text),
                token_count=chunk_tokens[idx],
            )
        )
    
    return chunk_records


# ============================================================================
# DATABASE SETUP
# ============================================================================

def setup_vector_db(path: Path):
    """Create or connect to LanceDB with parent and chunk tables."""
    if path.exists():
        print(f"⚠️  Database already exists at {path}")
        response = input("Delete and recreate? (yes/no): ")
        if response.lower() == "yes":
            shutil.rmtree(path)
            print("🗑️  Deleted existing database")
        else:
            print("📂 Using existing database")
    
    path.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(uri=path)
    
    # Create tables if they don't exist
    if "parent_videos" not in db.list_tables():
        db.create_table("parent_videos", schema=TranscriptGeminiWhole, exist_ok=True)
        print("✅ Created parent_videos table")
    
    if "video_chunks" not in db.list_tables():
        db.create_table("video_chunks", schema=TranscriptGeminiChunk, exist_ok=True)
        print("✅ Created video_chunks table")
    
    return db


# ============================================================================
# MAIN INGESTION PIPELINE
# ============================================================================

async def process_single_file(
    file: Path,
    db: lancedb.LanceDBConnection,
    checkpoint: Dict
) -> bool:
    """Process a single markdown file through the full pipeline.
    
    Steps:
        1. Load content
        2. Generate metadata (summary + keywords) via LLM
        3. Create whole document record
        4. Store in parent_videos table
        5. Chunk content
        6. Create chunk records
        7. Store in video_chunks table
    
    Returns:
        True if successful, False otherwise
    """
    filename = file.name
    
    # Skip if already processed
    if filename in checkpoint["processed_files"]:
        print(f"⏭️  Skipping {filename} (already processed)")
        return True
    
    try:
        print(f"📄 Processing: {filename}")
        
        # STEP 1: Load content
        # FIX: Robust file reading (prevents crash on encoding errors)
        with open(file, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        print(f"  ✓ Loaded {len(content)} characters")
        
        # STEP 2: Generate deterministic ID
        md_id = generate_md_id(filename)
        filename_without_ext = filename.replace(".md", "")
        print(f"  ✓ Generated md_id: {md_id[:12]}...")
        
        # STEP 3: Generate metadata (summary + keywords) via LLM
        print(f"  🤖 Generating metadata (summary + keywords)...")
        metadata = await generate_metadata(content, filename_without_ext)
        # FIX: Force lowercase for tag consistency
        keywords_clean = normalize_keywords(metadata.keywords).lower()
        print(f"  ✓ Summary: {metadata.summary[:60]}...")
        print(f"  ✓ Keywords: {len(keywords_clean.split(','))} tags")
        
        # Rate limiting (non-blocking)
        await asyncio.sleep(SLEEP_AFTER_LLM_CALL)
        
        # STEP 4: Create whole document record (embedding is auto-generated by LanceDB)
        parent_record = TranscriptGeminiWhole(
            md_id=md_id,
            filepath=str(file.absolute()),
            filename=filename_without_ext,
            content=content,
            summary=metadata.summary,
            keywords=keywords_clean,
        )
        
        # STEP 5: Upsert into parent_videos table
        # FIX: Atomic Upsert (Prevents data loss if script fails mid-operation)
        parent_table = db["parent_videos"]
        parent_row = parent_record.model_dump(exclude={"embedding"}, exclude_none=True)
        parent_table.merge_insert(on="md_id") \
                    .when_matched_update_all() \
                    .when_not_matched_insert_all() \
                    .execute([parent_row])
        print(f"  ✓ Upserted parent record")
        
        # STEP 6: Chunk content
        chunks = await chunk_content(content, md_id)
        print(f"  ✓ Created {len(chunks)} chunks")
        
        # STEP 7: Upsert chunks in video_chunks table using (md_id, chunk_id) as key
        # FIX: Atomic Chunk Upsert
        chunk_table = db["video_chunks"]
        chunk_rows = [c.model_dump(exclude={"embedding"}, exclude_none=True) for c in chunks]
        chunk_table.merge_insert(on=["md_id", "chunk_id"]) \
                   .when_matched_update_all() \
                   .when_not_matched_insert_all() \
                   .execute(chunk_rows)
        print(f"  ✓ Upserted {len(chunks)} chunks")
        
        # Mark as processed
        mark_file_processed(checkpoint, filename)
        print(f"  ✅ Completed: {filename}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error processing {filename}: {e}")
        checkpoint["last_error"] = {
            "file": filename,
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        save_checkpoint(checkpoint)
        return False


async def run_ingestion(limit: Optional[int] = None):
    """Run the complete ingestion pipeline.
    
    Args:
        limit: Process only first N files (for testing)
    """
    from datetime import datetime
    
    print("="*70)
    print("🚀 UNIFIED INGESTION PIPELINE")
    print("="*70)

    # Setup database first so we can detect a fresh/empty DB
    db = setup_vector_db(DB_PATH)
    parent_table = db["parent_videos"]
    chunk_table = db["video_chunks"]
    fresh_db = (parent_table.count_rows() == 0 and chunk_table.count_rows() == 0)

    # Load checkpoint (after DB setup). If DB is fresh, reset checkpoint.
    checkpoint = load_checkpoint()
    if fresh_db:
        checkpoint = {
            "processed_files": {},
            "total_processed": 0,
            "last_error": None,
            "started_at": None,
            "last_updated": None,
        }
        save_checkpoint(checkpoint)
        print("🧹 Fresh database detected — checkpoint reset.")
    if checkpoint["started_at"] is None:
        checkpoint["started_at"] = datetime.now().isoformat()
    
    # Get files to process
    all_files = sorted(DATA_PATH.glob("*.md"))
    total_files = len(all_files)
    
    if limit:
        all_files = all_files[:limit]
        print(f"⚠️  TESTING MODE: Processing only {limit} files")
    
    print(f"📊 Found {total_files} total files")
    print(f"📊 Already processed: {checkpoint['total_processed']}")
    print(f"📊 Remaining: {total_files - checkpoint['total_processed']}")
    
    # Process files
    successful = 0
    failed = 0
    
    for idx, file in enumerate(all_files, 1):
        print(f"[{idx}/{len(all_files)}] ", end="")
        success = await process_single_file(file, db, checkpoint)
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Final summary
    print("="*70)
    print("📈 INGESTION COMPLETE")
    print("="*70)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total processed: {checkpoint['total_processed']}")
    print(f"💾 Database location: {DB_PATH}")
    print(f"📝 Checkpoint saved to: {CHECKPOINT_FILE}")
    
    # Display table stats
    # RE-FETCH the tables to get the latest state from disk
    final_parent_table = db.open_table("parent_videos")
    final_chunk_table = db.open_table("video_chunks")

    print(f"📊 Database Stats:")
    print(f"  - parent_videos: {final_parent_table.count_rows()} records")
    print(f"  - video_chunks: {final_chunk_table.count_rows()} records")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import asyncio
    import sys
    
    # Check for test mode
    test_mode = "--test" in sys.argv
    
    if test_mode:
        print("🧪 RUNNING IN TEST MODE (2 files only)")
        asyncio.run(run_ingestion(limit=2))
    else:
        print("⚡ RUNNING FULL INGESTION")
        asyncio.run(run_ingestion())
