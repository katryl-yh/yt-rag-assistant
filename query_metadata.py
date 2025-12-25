"""Simple script to query and display video metadata (summary + keywords) from LanceDB."""

import lancedb
from pathlib import Path
from backend.constants import VECTOR_DATABASE_PATH

def query_video_by_filename(filename: str):
    """Query video metadata by filename (without .md extension)."""
    db = lancedb.connect(uri=str(VECTOR_DATABASE_PATH / "transcripts_unified"))
    parent_table = db["parent_videos"]
    
    # Search by filename
    results = parent_table.search().where(f"filename = '{filename}'").to_list()
    
    if not results:
        print(f"❌ No video found with filename: {filename}")
        return
    
    record = results[0]
    print("="*70)
    print(f"📹 VIDEO: {record['filename']}")
    print("="*70)
    print(f"\n📝 SUMMARY:\n{record['summary']}")
    print(f"\n🏷️  KEYWORDS:\n{record['keywords']}")
    print(f"\n📊 MD_ID: {record['md_id']}")
    print("="*70)


def list_all_videos():
    """List all ingested videos."""
    db = lancedb.connect(uri=str(VECTOR_DATABASE_PATH / "transcripts_unified"))
    parent_table = db["parent_videos"]
    
    results = parent_table.search().limit(100).to_list()
    
    if not results:
        print("❌ No videos found in database.")
        return
    
    print("="*70)
    print(f"📚 AVAILABLE VIDEOS ({len(results)} total)")
    print("="*70)
    for i, record in enumerate(results, 1):
        print(f"\n{i}. {record['filename']}")
        print(f"   Summary: {record['summary'][:80]}...")
        print(f"   Keywords: {record['keywords'][:100]}...")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Query specific video by filename
        filename = sys.argv[1]
        query_video_by_filename(filename)
    else:
        # List all videos
        list_all_videos()
        print("💡 Tip: Run with a filename to see full details:")
        print("   uv run python query_metadata.py 'An introduction to the vector database LanceDB'")
