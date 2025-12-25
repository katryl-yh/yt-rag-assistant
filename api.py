from fastapi import FastAPI
from backend.rag import rag_agent
from backend.data_models import Prompt, VideoMetadataResponse
from backend.constants import VECTOR_DATABASE_PATH
import lancedb

app = FastAPI()

# Initialize unified vector database
vector_db = lancedb.connect(uri=VECTOR_DATABASE_PATH / "transcripts_unified")

@app.get("/")
async def hello_message():
    return {
        "message": "Welcome to YT RAG Assistant API",
        "instructions": "To get started, call /videos to see all available videos and their identifiers",
        "endpoints": {
            "GET /videos": "List all available videos with their md_id (video identifier) and filename",
            "GET /video/description/{md_id}": "Get the pre-generated YouTube description for a video",
            "GET /video/keywords/{md_id}": "Get the pre-generated YouTube keywords/tags for a video",
            "POST /rag/query": "Query the RAG system with a custom question about the videos"
        }
    }

@app.get("/videos")
async def list_all_videos():
    """
    Get a list of all available videos in the knowledge base.
    
    Returns:
        List of videos with their md_id (video identifier) and filename
    """
    parent_table = vector_db["parent_videos"]
    results = parent_table.search().limit(1000).to_list()
    
    videos = [
        {
            "md_id": r.get("md_id", "Unknown"),
            "filename": r.get("filename", "Unknown")
        }
        for r in results
    ]
    
    return {
        "total": len(videos),
        "videos": videos
    }

@app.get("/keywords")
async def list_all_keywords():
    """
    Get a list of all unique keywords from all videos in the knowledge base.
    
    Returns:
        Sorted list of unique keywords with their frequency count
    """
    parent_table = vector_db["parent_videos"]
    results = parent_table.search().limit(1000).to_list()
    
    keyword_counts = {}
    
    for r in results:
        keywords_str = r.get("keywords", "")
        if keywords_str:
            # Split by comma and clean up whitespace
            keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]
            for keyword in keywords:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
    
    # Sort by count (descending), then alphabetically
    sorted_keywords = sorted(
        keyword_counts.items(),
        key=lambda x: (-x[1], x[0])
    )
    
    return {
        "total_unique_keywords": len(keyword_counts),
        "keywords": [
            {"keyword": k, "count": c}
            for k, c in sorted_keywords
        ]
    }

@app.post("/rag/query")
async def query_documentation(query: Prompt):
    from backend.rag import set_retrieval_mode
    
    # Set retrieval mode based on user input
    set_retrieval_mode(query.retrieval_mode)
    
    result = await rag_agent.run(query.prompt)

    return result.output

@app.get("/video/description/{video_id}")
async def get_video_description(video_id: str):
    """
    Get YouTube description (summary) for a video by its ID.
    
    Args:
        video_id: The MD5 hash identifier of the video in the knowledge base
        
    Returns:
        VideoMetadataResponse with summary field populated
    """
    # Retrieve the transcript from parent_videos table
    parent_table = vector_db["parent_videos"]
    results = parent_table.search().where(f"md_id = '{video_id}'").to_list()
    
    if not results:
        return {"error": f"Video with ID {video_id} not found"}
    
    result = results[0]
    filename = result.get("filename", "Unknown")
    content = result.get("content", "")
    
    # Return the pre-generated summary from the database
    summary = result.get("summary", "")
    
    return VideoMetadataResponse(
        md_id=video_id,
        filename=filename,
        summary=summary,
        keywords=""
    )

@app.get("/video/keywords/{video_id}")
async def get_video_keywords(video_id: str):
    """
    Get YouTube keywords/tags for a video by its ID.
    
    Args:
        video_id: The MD5 hash identifier of the video in the knowledge base
        
    Returns:
        VideoMetadataResponse with keywords field populated (comma-separated)
    """
    # Retrieve the transcript from parent_videos table
    parent_table = vector_db["parent_videos"]
    results = parent_table.search().where(f"md_id = '{video_id}'").to_list()
    
    if not results:
        return {"error": f"Video with ID {video_id} not found"}
    
    result = results[0]
    filename = result.get("filename", "Unknown")
    
    # Return the pre-generated keywords from the database
    keywords = result.get("keywords", "")
    
    return VideoMetadataResponse(
        md_id=video_id,
        filename=filename,
        summary="",
        keywords=keywords
    )