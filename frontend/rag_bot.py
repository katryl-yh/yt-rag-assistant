from typing import Optional, List
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


class RAGBot:
    def __init__(self, retrieval_mode: str = "chunked"):
        self.api_url = API_BASE_URL  
        self.retrieval_mode = retrieval_mode
        self.message_history: List[dict] = []  # Store messages locally
        self.session_id = self._create_session()  # Create new session on init

    def _create_session(self) -> str:
        """Create a new session on the backend"""
        try:
            response = requests.post(f"{API_BASE_URL}/session")
            data = response.json()
            return data.get("session_id", None)
        except Exception as e:
            print(f"Warning: Failed to create session: {e}")
            return None

    def chat(self, user_query: str) -> dict:
        """Send query to RAG API and return formatted response"""
        payload = {
            "query": user_query,
            "retrieval_mode": self.retrieval_mode,
            "session_id": self.session_id
        }
        
        try:
            response = requests.post(f"{self.api_url}/query", json=payload)
            
            # CHANGED: Check status code first
            if response.status_code != 200:
                return {
                    "bot": f"⚠️ Error {response.status_code}: {response.text}",
                    "source": "API Error"
                }
                
            return {
                "bot": response.json().get("answer", "No answer provided."),
                "source": response.json().get("filepath", "Unknown source")
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "bot": f"⚠️ Connection error: {str(e)}",
                "source": "Network Error"
            }

    def clear_history(self):
        """Clear conversation history and create new session"""
        self.message_history = []
        self.session_id = self._create_session()