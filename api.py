from fastapi import FastAPI
from backend.rag import rag_agent, set_retrieval_mode
from backend.data_models import Prompt, QueryRequest, VideoMetadataResponse
from backend.constants import VECTOR_DATABASE_PATH, GEMINI_MODELS, GEMINI_MODEL_KEY
import lancedb
from typing import Optional, List
import uuid
from fastapi import HTTPException
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import UserPromptPart, ModelResponse, TextPart
import time
from datetime import datetime, timedelta

app = FastAPI()

# In-memory session storage for conversation history
sessions = {}  # {session_id: [{"role": "user", "content": "..."}, ...]}
SESSION_TTL_MINUTES = 60  # Clean up sessions older than 1 hour

def get_or_create_session() -> str:
    """Create a new session ID for tracking conversation history"""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "history": [], 
        "created_at": time.time()  # Add this line
    }
    return session_id

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
            "POST /session": "Create a new conversation session",
            "GET /sessions": "List all active session IDs",
            "POST /query": "Query the RAG system with a custom question about the videos",
            "GET /history/{session_id}": "Get conversation history for a session"
        }
    }

@app.post("/session")
async def create_session() -> dict:
    """Create a new conversation session and return session ID"""
    session_id = get_or_create_session()
    return {"session_id": session_id}

@app.get("/sessions")
async def list_sessions() -> dict:
    """List all active session IDs"""
    return {
        "count": len(sessions),
        "sessions": list(sessions.keys())
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
    # Set retrieval mode based on user input
    set_retrieval_mode(query.retrieval_mode)
    
    result = await rag_agent.run(query.prompt)

    return result.output

@app.post("/query")
async def query_rag(request: QueryRequest):
    """Query RAG using server-side session history"""
    cleanup_old_sessions()  # ← Add this single line here
    set_retrieval_mode(request.retrieval_mode)
    
    # 1. Validate Session
    if request.session_id not in sessions:
         raise HTTPException(status_code=404, detail="Session not found")

    # 2. Get History from Server Memory and convert to proper message format
    server_history = sessions[request.session_id]["history"]
    message_history = []
    for msg in server_history:
        if msg["role"] == "user":
            message_history.append(UserPromptPart(content=msg["content"]))
        elif msg["role"] == "model":
            message_history.append(ModelResponse(parts=[TextPart(content=msg["content"])]))
    
    # 3. Run Agent with 429 error handling
    try:
        result = await rag_agent.run(request.query, message_history=message_history)
    except ModelHTTPError as e:
        if e.status_code == 429:
            # Quota exceeded: provide helpful fallback guidance
            current_model = GEMINI_MODEL_KEY
            available_high_quota = [k for k, v in GEMINI_MODELS.items() if k != current_model]
            raise HTTPException(
                status_code=429,
                detail=f"Quota exceeded for model '{current_model}'. "
                       f"Switch to a high-quota model in .env (GEMINI_MODEL_KEY): {', '.join(available_high_quota)}. "
                       f"Recommended: 'flash-lite' or 'flash-2.0' (10M tokens/day each). "
                       f"Restart your server after changing .env."
            )
        raise  # Re-raise other HTTP errors
    
    # 4. Update Server Memory
    sessions[request.session_id]["history"].append({"role": "user", "content": request.query})
    
    # CHANGED: Use 'model' role and remove extra fields to keep history clean for the LLM
    sessions[request.session_id]["history"].append({
        "role": "model", 
        "content": result.output.answer
    })
    
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

@app.get("/history/{session_id}")
async def get_session_history(session_id: str) -> dict:
    """Get conversation history for a session"""
    # CHANGED: Raise proper 404 exception instead of returning 200 with error dict
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    return {
        "session_id": session_id,
        "history": sessions[session_id]["history"],
        "message_count": len(sessions[session_id]["history"])
    }

@app.post("/session/{session_id}/clear")
async def clear_session(session_id: str):
    """Manually clear a session"""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"Session {session_id} cleared"}
    return {"message": f"Session {session_id} not found"}


def cleanup_old_sessions():
    """Remove sessions older than SESSION_TTL_MINUTES"""
    current_time = time.time()
    expired_sessions = [
        sid for sid, data in sessions.items()
        if current_time - data["created_at"] > SESSION_TTL_MINUTES * 60
    ]
    for sid in expired_sessions:
        del sessions[sid]
    return len(expired_sessions)
