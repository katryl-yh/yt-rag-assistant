import os
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
import posixpath
from dotenv import load_dotenv

load_dotenv()

DEFAULT_API_BASE_URL = "http://127.0.0.1:7071/"


def _append_host_key_if_needed(base_url: str) -> str:
    host_key = os.getenv("HOST_KEY")
    if not host_key:
        return base_url

    parts = urlsplit(base_url)
    query = dict(parse_qsl(parts.query))
    if "code" in query:
        return base_url

    query["code"] = host_key
    new_query = urlencode(query)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))


def get_api_base_url() -> str:
    """Resolve API base URL from Streamlit secrets, env vars, or default, then append HOST_KEY if present and not already in the URL."""
    secret_url = None
    try:
        import streamlit as st

        secret_url = st.secrets.get("api_base_url") if hasattr(st, "secrets") else None
    except Exception:
        secret_url = None

    env_url = os.getenv("API_BASE_URL")
    raw_url = (secret_url or env_url or DEFAULT_API_BASE_URL).rstrip("/")
    return _append_host_key_if_needed(raw_url)


def build_api_url(path: str) -> str:
    """Join base URL with path while preserving existing query params (e.g., ?code=host_key)."""
    base = get_api_base_url()
    parts = urlsplit(base)
    # normalize path join on POSIX style
    new_path = posixpath.join(parts.path, path.lstrip("/"))
    return urlunsplit((parts.scheme, parts.netloc, new_path, parts.query, parts.fragment))
