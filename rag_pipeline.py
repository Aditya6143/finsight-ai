"""
rag_pipeline.py — RAG Engine for RockyBot / FinSight

Responsibilities:
- Load article text from URLs (UnstructuredURLLoader)
- Split into chunks (RecursiveCharacterTextSplitter)
- Layer 1: Filter noisy/irrelevant chunks (post-chunk heuristics)
- Embed clean chunks and store in FAISS (provider-selectable via config)
- Persist / load FAISS vectorstore to/from disk (with embed metadata guard)
- Layer 2: Cosine-equivalent filter using FAISS native L2 scores (no per-chunk re-embedding)
- Query via manual prompt chain + provider-selectable LLM

No Streamlit — pure pipeline logic, fully testable in isolation.
"""

import json
import os
import re

import numpy as np

import config
import embeddings_provider
import llm_provider
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Configuration — imported from config.py (single source of truth)
# ---------------------------------------------------------------------------

FAISS_STORE_PATH              = config.FAISS_STORE_PATH
CHUNK_SIZE                    = config.CHUNK_SIZE
CHUNK_OVERLAP                 = config.CHUNK_OVERLAP
MIN_WORD_COUNT                = config.MIN_WORD_COUNT
EMBEDDINGS_SIMILARITY_THRESHOLD = config.EMBEDDINGS_SIMILARITY_THRESHOLD

# Finance domain keywords — at least one must appear in a chunk to pass Layer 1
FINANCE_KEYWORDS = {
    "stock", "share", "market", "equity", "index", "sensex", "nifty",
    "bse", "nse", "rbi", "repo", "rate", "inflation", "gdp", "economy",
    "revenue", "profit", "loss", "earnings", "ipo", "fund", "investment",
    "bond", "yield", "dividend", "quarter", "fiscal", "budget", "trade",
    "export", "import", "rupee", "dollar", "crude", "oil", "gold",
    "bank", "finance", "financial", "analyst", "forecast", "growth",
    "recession", "rally", "correction", "bull", "bear", "sector",
    "acquisition", "merger", "valuation", "startup", "venture", "capital",
}

# Boilerplate patterns — chunks matching any of these are dropped
BOILERPLATE_PATTERNS = [
    r"cookie(s)?\s*(policy|consent|banner|notice)",
    r"(accept|agree)\s*(all\s*)?cookies",
    r"privacy\s*policy",
    r"terms\s*(of\s*service|and\s*conditions)",
    r"subscribe\s*(now|to\s*our\s*newsletter)",
    r"you\s*may\s*also\s*like",
    r"related\s*articles?",
    r"follow\s*us\s*on",
    r"share\s*(this|on)\s*(facebook|twitter|whatsapp|linkedin)",
    r"(all\s*rights?\s*reserved|copyright\s*©)",
    r"(read\s*more|continue\s*reading)\s*[:\-»]",
    r"advertisement",
    r"(sign\s*in|log\s*in|register)\s*to\s*(continue|read)",
    r"(download|get)\s*(the\s*)?app",
]
_BOILERPLATE_RE = re.compile(
    "|".join(BOILERPLATE_PATTERNS), re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Step 1 — Load articles
# ---------------------------------------------------------------------------

def load_articles(urls: list[str]) -> list:
    """
    Fetch and extract full article text from a list of URLs.

    Args:
        urls: Article URLs returned by feed_fetcher.fetch_article_urls()

    Returns:
        List of LangChain Document objects.
    """
    urls = [u for u in urls if u.strip()]
    print(f"[finsight] Loading {len(urls)} articles...")
    loader = UnstructuredURLLoader(urls=urls)
    docs = loader.load()
    print(f"[finsight] Loaded {len(docs)} documents")
    return docs


# ---------------------------------------------------------------------------
# Step 2 — Split
# ---------------------------------------------------------------------------

def split_docs(docs: list) -> list:
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.

    Args:
        docs: Documents from load_articles()

    Returns:
        List of chunk Documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"[finsight] Split into {len(chunks)} chunks")
    return chunks


# ---------------------------------------------------------------------------
# Step 3 — Layer 1: Chunk filter (ingestion-time noise reduction)
# ---------------------------------------------------------------------------

def _is_boilerplate(text: str) -> bool:
    return bool(_BOILERPLATE_RE.search(text))


def _has_finance_keyword(text: str) -> bool:
    words = set(re.findall(r"\b\w+\b", text.lower()))
    return bool(words & FINANCE_KEYWORDS)


def filter_chunks(chunks: list) -> list:
    """
    Layer 1 — Drop noisy or irrelevant chunks using heuristic rules.

    Rules applied (a chunk is dropped if ANY rule fails):
      1. Word count >= MIN_WORD_COUNT
      2. Does not match known boilerplate patterns
      3. Contains at least one finance-domain keyword

    Args:
        chunks: Raw chunks from split_docs()

    Returns:
        Filtered list of clean chunks.
    """
    clean = []
    dropped_short = dropped_boilerplate = dropped_no_keyword = 0

    for chunk in chunks:
        text = chunk.page_content

        if len(text.split()) < MIN_WORD_COUNT:
            dropped_short += 1
            continue

        if _is_boilerplate(text):
            dropped_boilerplate += 1
            continue

        if not _has_finance_keyword(text):
            dropped_no_keyword += 1
            continue

        clean.append(chunk)

    total_dropped = dropped_short + dropped_boilerplate + dropped_no_keyword
    print(
        f"[finsight] Layer 1 filter: {len(chunks)} → {len(clean)} chunks "
        f"(dropped {total_dropped}: {dropped_short} short, "
        f"{dropped_boilerplate} boilerplate, {dropped_no_keyword} off-topic)"
    )
    return clean


# ---------------------------------------------------------------------------
# Step 4 — Embed and store in FAISS
# ---------------------------------------------------------------------------

def _metadata_path(store_path: str) -> str:
    return os.path.join(store_path, "metadata.json")


def _write_metadata(store_path: str) -> None:
    """Write provider/model metadata alongside the FAISS index."""
    meta = {
        "embed_provider": config.PROVIDER,
        "embed_model": config.embed_display_name(),
    }
    with open(_metadata_path(store_path), "w") as f:
        json.dump(meta, f, indent=2)


def _check_metadata(store_path: str) -> str | None:
    """
    Read saved metadata and compare against current config.
    Returns a warning string if there is a mismatch, else None.
    """
    meta_file = _metadata_path(store_path)
    if not os.path.exists(meta_file):
        return None  # old index without metadata — can't verify, skip

    with open(meta_file) as f:
        meta = json.load(f)

    saved_provider = meta.get("embed_provider", "")
    saved_model    = meta.get("embed_model", "")
    current_model  = config.embed_display_name()

    if saved_provider != config.PROVIDER or saved_model != current_model:
        return (
            f"⚠️ Index was built with {saved_provider}/{saved_model} but current "
            f"provider is {config.PROVIDER}/{current_model}. "
            f"Answers may be wrong — click 'Fetch & Process News' to rebuild the index."
        )
    return None


def build_vectorstore(chunks: list, store_path: str = FAISS_STORE_PATH) -> object:
    """
    Embed clean chunks and store in a FAISS index.
    Uses provider-selected embeddings (Ollama or OpenAI).
    Saves provider/model metadata alongside the index for mismatch detection.

    Args:
        chunks:     Clean chunks from filter_chunks()
        store_path: Directory path to save the FAISS index

    Returns:
        FAISS vectorstore object.
    """
    print(f"[finsight] Building FAISS vectorstore with {len(chunks)} chunks...")
    embeddings = embeddings_provider.get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(store_path)
    _write_metadata(store_path)
    print(f"[finsight] Vectorstore saved to {store_path!r}")
    return vectorstore


def load_vectorstore(store_path: str = FAISS_STORE_PATH) -> tuple[object | None, str | None]:
    """
    Load a previously persisted FAISS vectorstore from disk.
    Also checks for provider/model mismatch and returns a warning if found.

    Args:
        store_path: Directory path written by build_vectorstore()

    Returns:
        (vectorstore, warning_message)
        vectorstore is None if no index exists.
        warning_message is None if no mismatch detected.
    """
    if not os.path.exists(store_path):
        print(f"[finsight] No vectorstore found at {store_path!r}")
        return None, None

    warning = _check_metadata(store_path)
    if warning:
        print(f"[finsight] {warning}")

    embeddings = embeddings_provider.get_embeddings()
    vectorstore = FAISS.load_local(
        store_path, embeddings, allow_dangerous_deserialization=True
    )
    print(f"[finsight] Vectorstore loaded from {store_path!r}")
    return vectorstore, warning


# ---------------------------------------------------------------------------
# Step 5 — Layer 2: Similarity filter using FAISS native L2 scores
#
# Phase 3 cost fix: replaces per-chunk embed_query() calls with FAISS's
# native similarity_search_with_score(), which returns L2 distances
# computed from vectors already stored in the index — no extra API calls.
#
# L2 → similarity conversion: score = 1 / (1 + l2_distance)
# This maps [0, ∞) to (0, 1] and preserves ordering.
# The same EMBEDDINGS_SIMILARITY_THRESHOLD is applied to this score.
# ---------------------------------------------------------------------------

class SimilarityFilterRetriever:
    """
    Layer 2 — Query-time retriever that drops chunks below a cosine-equivalent
    similarity threshold. Uses FAISS native L2 scores — no per-chunk re-embedding.

    Drop-in replacement for the broken ContextualCompressionRetriever.
    """

    def __init__(self, vectorstore, threshold: float, top_k: int = 10):
        self.vectorstore = vectorstore
        self.threshold = threshold
        self.top_k = top_k

    def get_relevant_documents(self, query: str) -> list[Document]:
        # Retrieve top_k candidates with L2 distances from FAISS
        candidates_with_scores = self.vectorstore.similarity_search_with_score(
            query, k=self.top_k
        )

        # Convert L2 distance → similarity score and apply threshold
        kept = []
        for doc, l2_distance in candidates_with_scores:
            score = 1.0 / (1.0 + float(l2_distance))
            if score >= self.threshold:
                kept.append(doc)

        print(
            f"[finsight] Layer 2: {len(candidates_with_scores)} candidates → "
            f"{len(kept)} kept (threshold={self.threshold})"
        )
        return kept

    # LangChain chain compatibility
    def invoke(self, query: str) -> list[Document]:
        return self.get_relevant_documents(query)


def get_retriever(vectorstore) -> SimilarityFilterRetriever:
    """
    Layer 2 — Return a SimilarityFilterRetriever wrapping the FAISS vectorstore.
    Chunks retrieved at query time are filtered by converted L2 similarity score.
    No embeddings object needed — uses FAISS native scoring.

    Args:
        vectorstore: FAISS vectorstore from build_vectorstore() or load_vectorstore()

    Returns:
        SimilarityFilterRetriever instance.
    """
    retriever = SimilarityFilterRetriever(
        vectorstore=vectorstore,
        threshold=EMBEDDINGS_SIMILARITY_THRESHOLD,
    )
    print(
        f"[finsight] Layer 2: SimilarityFilterRetriever active "
        f"(threshold={EMBEDDINGS_SIMILARITY_THRESHOLD}, L2-based scoring)"
    )
    return retriever


# ---------------------------------------------------------------------------
# Step 6 — Query
# ---------------------------------------------------------------------------

def query(question: str, retriever) -> dict:
    """
    Run a question through the RAG chain and return the answer with sources.

    Manual chain (RetrievalQAWithSourcesChain removed in langchain v1.x):
      1. Retrieve relevant chunks via SimilarityFilterRetriever
      2. Build a prompt with context + source URLs
      3. Call provider LLM via llm_provider.get_llm()
      4. Return answer + sources

    Args:
        question:  User's natural language question
        retriever: SimilarityFilterRetriever from get_retriever()

    Returns:
        Dict with keys:
          "answer"  — LLM-generated answer string
          "sources" — newline-separated source URLs (may be empty string)
    """
    # Step 1 — retrieve relevant chunks
    docs = retriever.get_relevant_documents(question)

    if not docs:
        return {
            "answer": "I couldn't find relevant information to answer your question.",
            "sources": "",
        }

    # Step 2 — build context block with sources
    context_parts = []
    sources = []
    for doc in docs:
        context_parts.append(doc.page_content)
        source = doc.metadata.get("source", "")
        if source and source not in sources:
            sources.append(source)

    context = "\n\n---\n\n".join(context_parts)

    # Step 3 — build prompt
    prompt = f"""You are a financial news analyst. Use the context below to answer the question.
Be concise and factual. Only use information from the provided context.

Context:
{context}

Question: {question}

Answer:"""

    # Step 4 — call provider LLM (one line changed from original)
    llm = llm_provider.get_llm()
    answer = llm.invoke(prompt)

    return {
        "answer": answer.strip(),
        "sources": "\n".join(sources),
    }


# ---------------------------------------------------------------------------
# Convenience — full ingestion pipeline in one call
# ---------------------------------------------------------------------------

def ingest(urls: list[str], store_path: str = FAISS_STORE_PATH) -> object:
    """
    Run the full ingestion pipeline:
      load → split → filter (Layer 1) → embed → store

    Args:
        urls:       Article URLs from feed_fetcher
        store_path: Where to persist the FAISS index

    Returns:
        Built FAISS vectorstore.
    """
    docs = load_articles(urls)
    chunks = split_docs(docs)
    clean_chunks = filter_chunks(chunks)
    vectorstore = build_vectorstore(clean_chunks, store_path=store_path)
    return vectorstore


# ---------------------------------------------------------------------------
# Quick smoke test — run directly to test pipeline with hardcoded URLs
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TEST_URLS = [
        "https://economictimes.indiatimes.com/markets/stocks/news/sensex-today-bse-nse-live-updates/articleshow/latest",
    ]

    print("=== Ingestion ===")
    vs = ingest(TEST_URLS)

    print("\n=== Query ===")
    retriever = get_retriever(vs)
    q = "What is happening with Sensex today?"
    result = query(q, retriever)
    print(f"\nQ: {q}")
    print(f"A: {result['answer']}")
    print(f"Sources: {result['sources']}")
