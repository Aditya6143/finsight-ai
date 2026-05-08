"""
embeddings_provider.py — Embeddings Factory for RockyBot / FinSight

Returns a LangChain-compatible embeddings object based on config.PROVIDER.
Used in build_vectorstore(), load_vectorstore(), and get_retriever().
"""

import config


def get_embeddings():
    """
    Return the configured embeddings object.

    Returns:
        OllamaEmbeddings  (PROVIDER=ollama, default) — existing behaviour
        OpenAIEmbeddings  (PROVIDER=openai)           — langchain-openai
    Both are LangChain-compatible and work as drop-in replacements.
    """
    if config.PROVIDER == "openai":
        print(f"[embeddings_provider] Using OpenAI embeddings: {config.OPENAI_EMBED_MODEL}")
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as e:
            raise ImportError(
                "langchain-openai package is required for PROVIDER=openai.\n"
                "Install it with:  pip install langchain-openai"
            ) from e
        return OpenAIEmbeddings(
            model=config.OPENAI_EMBED_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
        )
    else:
        print(f"[embeddings_provider] Using Ollama embeddings: {config.OLLAMA_EMBED_MODEL}")
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=config.OLLAMA_EMBED_MODEL)
