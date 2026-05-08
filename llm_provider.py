"""
llm_provider.py — LLM Factory for RockyBot / FinSight

Returns the correct LLM object based on config.PROVIDER.
Both providers expose an identical interface: .invoke(prompt: str) -> str
so rag_pipeline.query() needs exactly one line changed.
"""

import config


# ---------------------------------------------------------------------------
# OpenAI wrapper — thin shim matching OllamaLLM's .invoke() interface
# ---------------------------------------------------------------------------

class _OpenAILLM:
    """
    Minimal wrapper around openai.OpenAI().chat.completions.create()
    that exposes .invoke(prompt) -> str, matching the OllamaLLM interface.
    """

    def __init__(self, model: str, temperature: float = 0.9, max_tokens: int = 1024):
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "openai package is required for PROVIDER=openai.\n"
                "Install it with:  pip install openai"
            ) from e

        self._client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def invoke(self, prompt: str) -> str:
        try:
            import openai
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            return response.choices[0].message.content or ""
        except openai.AuthenticationError as e:
            raise ValueError(
                "OpenAI API key is invalid or missing.\n"
                "Check that OPENAI_API_KEY is set correctly in your .env or Streamlit secrets."
            ) from e
        except openai.APIConnectionError as e:
            raise ConnectionError(
                "Could not connect to the OpenAI API.\n"
                "Check your network connection or OpenAI quota."
            ) from e


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def get_llm():
    """
    Return the configured LLM object.

    Returns:
        OllamaLLM  (PROVIDER=ollama, default) — existing behaviour, zero change
        _OpenAILLM (PROVIDER=openai)          — wraps openai SDK
    Both expose .invoke(prompt: str) -> str
    """
    if config.PROVIDER == "openai":
        print(f"[llm_provider] Using OpenAI LLM: {config.OPENAI_LLM_MODEL}")
        return _OpenAILLM(
            model=config.OPENAI_LLM_MODEL,
            temperature=0.9,
            max_tokens=1024,
        )
    else:
        print(f"[llm_provider] Using Ollama LLM: {config.OLLAMA_LLM_MODEL}")
        from langchain_ollama import OllamaLLM
        return OllamaLLM(model=config.OLLAMA_LLM_MODEL, temperature=0.9)
