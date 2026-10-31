import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def get_secret(name: str, default: str = "") -> str:
    """Read a setting from .env first, then from Streamlit Cloud secrets."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default


OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
CHAT_MODEL = get_secret("OPENAI_MODEL", "gpt-4o-mini")
APP_PASSWORD = get_secret("APP_PASSWORD")
EMBED_MODEL = "text-embedding-3-small"

KB_DIR = BASE_DIR / "knowledge_base"
INDEX_DIR = BASE_DIR / "vector_store"
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "tickets.csv"

# One knowledge collection (and one FAISS index) per agent
COLLECTIONS = {
    "triage": KB_DIR / "triage",                  # Query Agent
    "policies": KB_DIR / "customer_policies",     # Knowledge Agent
    "style": KB_DIR / "response_style",           # Resolution Agent
    "escalation": KB_DIR / "escalation",          # Log Agent
}

TOP_K = 3           # passages returned per search
MIN_SCORE = 0.35    # similarity below this counts as "no matching document"
LOW_CONFIDENCE = 0.6

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env (local) or Streamlit secrets (cloud).")