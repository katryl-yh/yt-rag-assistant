from typing import List
import requests
from config import get_api_base_url, build_api_url


class RAGBot:
    def __init__(self, retrieval_mode: str = "chunked"):
        self.api_url = get_api_base_url()
        self.retrieval_mode = retrieval_mode
        self.message_history: List[dict] = []  # Store messages locally in frontend

    def chat(self, user_query: str) -> dict:
        """Send query to RAG API with conversation history and return formatted response"""
        # Build payload with history from frontend
        payload = {
            "query": user_query,
            "retrieval_mode": self.retrieval_mode,
            "history": self.message_history  # Send full history to backend
        }
        
        try:
            response = requests.post(build_api_url("query"), json=payload, timeout=30)
            
            if response.status_code != 200:
                return {
                    "bot": f"⚠️ Error {response.status_code}: {response.text}",
                    "source": "API Error"
                }
                
            result = response.json()
            answer = result.get("answer", "No answer provided.")
            source = result.get("filepath", "Unknown source")
            
            # Update local history after successful response
            self.message_history.append({"role": "user", "content": user_query})
            self.message_history.append({"role": "assistant", "content": answer})
            
            return {
                "bot": answer,
                "source": source
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "bot": f"⚠️ Connection error: {str(e)}",
                "source": "Network Error"
            }

    def clear_history(self):
        """Clear conversation history"""
        self.message_history = []