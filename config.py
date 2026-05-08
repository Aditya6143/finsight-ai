"""
config.py — Centralised configuration for RockyBot / FinSight

Reads settings from environment variables (local dev via .env)
or Streamlit secrets (Streamlit Cloud deployment).
All other modules import from here — never read env vars directly.
"""

import os
from dotenv import load_dotenv
load_dotenv()  # reads .env into os.environ at import time

# ---------------------------------------------------------------------------
# Helper — read from st.secrets first, fall back to os.getenv
# ---------------------------------------------------------------------------

def _get(key: str, default: str = "") -> str:
    """
    Read a config value. Priority:
      1. Streamlit secrets (st.secrets) — available on Streamlit Cloud
      2. os.environ / .env loaded by python-dotenv
      3. Provided default
    """
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------

PROVIDER: str = _get("PROVIDER", "ollama").lower()   # "ollama" | "openai"

# ---------------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------------

# Ollama (local, default)
OLLAMA_LLM_MODEL: str    = "llama3:instruct"
OLLAMA_EMBED_MODEL: str  = "mxbai-embed-large:latest"

# OpenAI
OPENAI_LLM_MODEL: str    = "gpt-4o-mini"
OPENAI_EMBED_MODEL: str  = "text-embedding-3-small"

# ---------------------------------------------------------------------------
# OpenAI credentials
# ---------------------------------------------------------------------------

OPENAI_API_KEY: str = _get("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Pipeline settings (previously scattered across rag_pipeline.py)
# ---------------------------------------------------------------------------

FAISS_STORE_PATH: str            = "faiss_store_finsight"
CHUNK_SIZE: int                  = 1000
CHUNK_OVERLAP: int               = 100
MIN_WORD_COUNT: int              = 30
EMBEDDINGS_SIMILARITY_THRESHOLD: float = 0.2   # FinSight-tuned value

# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------

def validate_config() -> None:
    """
    Validate that the active provider is properly configured.
    Raises ValueError with an actionable message on misconfiguration.
    Should be called once at app startup (top of main.py).
    """
    if PROVIDER not in ("ollama", "openai"):
        raise ValueError(
            f"Invalid PROVIDER={PROVIDER!r}. Must be 'ollama' or 'openai'."
        )

    if PROVIDER == "openai":
        if not OPENAI_API_KEY or not OPENAI_API_KEY.strip():
            raise ValueError(
                "PROVIDER is set to 'openai' but OPENAI_API_KEY is missing or empty.\n"
                "  • Local dev:  add OPENAI_API_KEY=sk-... to your .env file\n"
                "  • Streamlit Cloud:  add it under App Settings → Secrets"
            )

# ---------------------------------------------------------------------------
# Convenience display helpers (used in Streamlit sidebar caption)
# ---------------------------------------------------------------------------

def llm_display_name() -> str:
    return OPENAI_LLM_MODEL if PROVIDER == "openai" else OLLAMA_LLM_MODEL

def embed_display_name() -> str:
    return OPENAI_EMBED_MODEL if PROVIDER == "openai" else OLLAMA_EMBED_MODEL

def provider_display_name() -> str:
    return "OpenAI" if PROVIDER == "openai" else "Ollama"
