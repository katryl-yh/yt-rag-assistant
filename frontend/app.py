import streamlit as st
import requests
from rag_bot import RAGBot

API_BASE_URL = "http://127.0.0.1:8000"


def init_session_states():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "bot" not in st.session_state:
        st.session_state.bot = None
    if "retrieval_mode" not in st.session_state:
        st.session_state.retrieval_mode = "chunked"


@st.cache_data(ttl=600)
def fetch_videos():
    """Fetch all available videos from the API"""
    try:
        response = requests.get(f"{API_BASE_URL}/videos")
        if response.status_code == 200:
            data = response.json()
            return {v["filename"]: v["md_id"] for v in data["videos"]}
        return {}
    except Exception as e:
        st.error(f"Failed to fetch videos: {e}")
        return {}


@st.cache_data(ttl=3600)
def fetch_video_metadata(video_id):
    """Fetch description and keywords for a video"""
    desc = "No description available"
    keywords = ""
    
    try:
        # Description
        desc_response = requests.get(f"{API_BASE_URL}/video/description/{video_id}")
        if desc_response.status_code == 200:
            desc_data = desc_response.json()
            # Handle potential key variations (summary vs description)
            desc = desc_data.get("summary") or desc_data.get("description", "No description available")
            
        # Keywords
        kw_response = requests.get(f"{API_BASE_URL}/video/keywords/{video_id}")
        if kw_response.status_code == 200:
            kw_data = kw_response.json()
            keywords = kw_data.get("keywords", "")
            
    except Exception:
        pass
        
    return desc, keywords


def display_chat_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("source"):
                st.caption(f"📍 Source: {message['source']}")


def handle_user_input():
    if prompt := st.chat_input("Ask a question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.spinner("Thinking..."):
            bot_response = st.session_state.bot.chat(prompt)
        
        with st.chat_message("assistant"):
            st.markdown(bot_response["bot"])
            st.caption(f"📍 Source: {bot_response['source']}")
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": bot_response["bot"],
            "source": bot_response["source"]
        })


def layout():
    st.set_page_config(page_title="YT RAG Assistant", layout="wide")
    st.markdown("# YT RAG Assistant")
    
    with st.sidebar:
        st.header("Settings")
        retrieval_mode = st.radio("Retrieval mode:", ["chunked", "whole"])
        if retrieval_mode != st.session_state.retrieval_mode:
            st.session_state.retrieval_mode = retrieval_mode
            st.session_state.bot = RAGBot(retrieval_mode=retrieval_mode)
            st.session_state.messages = []
            
        st.markdown("---")
        st.header("📺 Video Explorer")
        
        videos = fetch_videos()
        
        if videos:
            selected_filename = st.selectbox(
                "Select a video:",
                options=list(videos.keys()),
                index=0
            )
            
            if selected_filename:
                video_id = videos[selected_filename]
                desc, keywords = fetch_video_metadata(video_id)
                
                st.markdown("### 📝 Description")
                st.info(desc)
                
                st.markdown("### 🏷️ Keywords")
                if keywords:
                    st.caption(keywords)
                else:
                    st.caption("No keywords available")
        else:
            st.warning("⚠️ No videos available.")
    
    if st.session_state.bot is None:
        st.session_state.bot = RAGBot(retrieval_mode=st.session_state.retrieval_mode)
    
    display_chat_messages()
    handle_user_input()


if __name__ == "__main__":
    init_session_states()
    layout()