import streamlit as st
import requests
from pathlib import Path

ASSETS_PATH = Path(__file__).absolute().parents[1] / "assets"
API_BASE_URL = "http://127.0.0.1:8000"

@st.cache_data
def fetch_videos():
    """Fetch all available videos from the API"""
    try:
        response = requests.get(f"{API_BASE_URL}/videos")
        data = response.json()
        return {v["filename"]: v["md_id"] for v in data["videos"]}
    except Exception as e:
        st.error(f"Failed to fetch videos: {e}")
        return {}

def layout():
    st.set_page_config(page_title="YT RAG Assistant", layout="wide")
    
    st.markdown("# YT RAG Assistant")
    st.markdown("Ask questions about YouTube video transcripts or explore video metadata")
    
    # Sidebar for video selection and metadata
    with st.sidebar:
        st.markdown("## 📺 Video Explorer")
        
        # Fetch videos
        videos = fetch_videos()
        
        if videos:
            selected_filename = st.selectbox(
                "Select a video:",
                options=list(videos.keys()),
                index=0
            )
            
            if selected_filename:
                video_id = videos[selected_filename]
                
                # Fetch and display video description
                try:
                    desc_response = requests.get(f"{API_BASE_URL}/video/description/{video_id}")
                    desc_data = desc_response.json()
                    
                    st.markdown("### 📝 Description")
                    st.info(desc_data.get("summary", "No description available"))
                except Exception as e:
                    st.error(f"Failed to fetch description: {e}")
                
                # Fetch and display video keywords
                try:
                    kw_response = requests.get(f"{API_BASE_URL}/video/keywords/{video_id}")
                    kw_data = kw_response.json()
                    
                    st.markdown("### 🏷️ Keywords/Tags")
                    keywords = kw_data.get("keywords", "")
                    if keywords:
                        st.caption(keywords)
                    else:
                        st.caption("No keywords available")
                except Exception as e:
                    st.error(f"Failed to fetch keywords: {e}")
        else:
            st.warning("⚠️ No videos available. Please run the ingestion pipeline first.")
    
    # Main content area for RAG query
    st.markdown("## 🔍 Ask a Question")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        # Retrieval mode selector
        retrieval_mode = st.radio(
            "Retrieval mode:",
            options=["chunked", "whole"],
            help="Chunked: Focused results | Whole: Full document context"
        )
    
    with col1:
        # Question input
        text_input = st.text_input(label="Your question", placeholder="e.g., How does FastAPI work?")
    
    if st.button("Send", use_container_width=True) and text_input.strip() != "":
        try:
            response = requests.post(
                f"{API_BASE_URL}/rag/query",
                json={"prompt": text_input, "retrieval_mode": retrieval_mode}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                st.markdown("---")
                st.markdown("### ❓ Question:")
                st.markdown(f"*{text_input}*")
                
                st.markdown("### ✅ Answer:")
                st.markdown(data["answer"])
                
                st.markdown("### 📍 Source:")
                st.code(data["filepath"])
            else:
                st.error(f"API error: {response.status_code}")
        except Exception as e:
            st.error(f"Failed to get response: {e}")

if __name__ == "__main__":
    layout()