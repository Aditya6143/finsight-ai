"""
rag_pipeline.py — RAG Engine for RockyBot

Responsibilities:
- Load article text from URLs (UnstructuredURLLoader)
- Split into chunks (RecursiveCharacterTextSplitter)
- Layer 1: Filter noisy/irrelevant chunks (post-chunk heuristics)
- Embed clean chunks and store in FAISS (OllamaEmbeddings)
- Persist / load FAISS vectorstore to/from disk
- Layer 2: Manual cosine similarity filter at retrieval time
- Query via manual prompt chain + OllamaLLM

Fixes applied:
  Fix 2 — FINANCE_KEYWORDS expanded to include political / macro-event terms
           so that election, policy, and government-related chunks are not
           silently dropped by the Layer 1 filter.
  Fix 3 — EMBEDDINGS_SIMILARITY_THRESHOLD lowered to 0.60 for cross-domain
           queries; a guaranteed fallback returns the top-3 FAISS candidates
           by raw score when all candidates fall below the threshold, so the
           LLM always receives some context rather than an empty list.
  Fix 4 — query() prompt rewritten to allow the LLM to reason from partial or
           indirect context and to supplement with general financial knowledge
           when the retrieved context is incomplete, while clearly signalling
           when it does so.

No Streamlit — pure pipeline logic, fully testable in isolation.
"""

import os
import re

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FAISS_STORE_PATH = "faiss_store_rockybot"   # directory, not a .pkl file

LLM_MODEL = "llama3:instruct"               # ollama pull llama3
EMBED_MODEL = "mxbai-embed-large:latest"    # ollama pull mxbai-embed-large

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Layer 1 filter settings
MIN_WORD_COUNT = 30

# Layer 2 — Fix 3: threshold lowered from 0.76 → 0.60 to allow cross-domain
# queries (e.g. "election + market") to survive retrieval.
EMBEDDINGS_SIMILARITY_THRESHOLD = 0.60

# Fix 3: if ALL candidates fall below the threshold, return at least this many
# by raw FAISS score so the LLM always has something to reason from.
SIMILARITY_FALLBACK_K = 3

# ---------------------------------------------------------------------------
# Fix 2 — Expanded keyword set
#
# Original set covered pure finance terms only. Added:
#   - Political / electoral terms  (election, vote, mandate, party, coalition…)
#   - Government / regulatory      (government, minister, policy, parliament,
#                                   regulatory, reform, tariff, sanction…)
#   - Macro-economic events        (crisis, sanctions, geopolitical, war,
#                                   supply chain, commodity, energy…)
#   - Indian market-specific names (sebi, sensex is already present via
#                                   "sensex"; added rbi-adjacent terms)
# ---------------------------------------------------------------------------

FINANCE_KEYWORDS = {
    # Core market terms (original)
    "stock", "share", "market", "equity", "index", "sensex", "nifty",
    "bse", "nse", "rbi", "repo", "rate", "inflation", "gdp", "economy",
    "revenue", "profit", "loss", "earnings", "ipo", "fund", "investment",
    "bond", "yield", "dividend", "quarter", "fiscal", "budget", "trade",
    "export", "import", "rupee", "dollar", "crude", "oil", "gold",
    "bank", "finance", "financial", "analyst", "forecast", "growth",
    "recession", "rally", "correction", "bull", "bear", "sector",
    "acquisition", "merger", "valuation", "startup", "venture", "capital",
    # Fix 2: Political & electoral
    "election", "elections", "electoral", "vote", "votes", "voting",
    "voter", "voters", "ballot", "poll", "polls", "polling", "mandate",
    "party", "parties", "coalition", "government", "governor", "minister",
    "parliament", "legislature", "assembly", "candidate", "campaign",
    "result", "results", "winner", "victory", "defeat", "majority",
    "opposition", "incumbent", "ruling", "bjp", "congress", "tmc",
    "aap", "nda", "upa", "alliance",
    # Fix 2: Regulatory & policy
    "policy", "policies", "reform", "reforms", "regulatory", "regulation",
    "regulations", "sebi", "deregulation", "tariff", "tariffs", "sanction",
    "sanctions", "stimulus", "subsidy", "subsidies", "privatisation",
    "divestment", "disinvestment", "taxation", "tax",
    # Fix 2: Macro-economic events
    "geopolitical", "geopolitics", "crisis", "war", "conflict", "supply",
    "demand", "commodity", "commodities", "energy", "power", "infrastructure",
    "development", "project", "projects", "tender", "contract", "fdi",
    "foreign", "global", "international", "domestic", "regional", "state",
    "sentiment", "confidence", "uncertainty", "volatility", "impact",
    "effect", "affects", "influenced", "driven", "surge", "slump",
    "pressure", "concerns", "outlook", "guidance",
}

# Boilerplate patterns — chunks matching any of these are dropped (unchanged)
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
    print(f"[rag_pipeline] Loading {len(urls)} articles...")
    loader = UnstructuredURLLoader(urls=urls)
    docs = loader.load()
    print(f"[rag_pipeline] Loaded {len(docs)} documents")
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
    print(f"[rag_pipeline] Split into {len(chunks)} chunks")
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
      3. Contains at least one finance/market/political keyword (Fix 2: expanded set)

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
        f"[rag_pipeline] Layer 1 filter: {len(chunks)} → {len(clean)} chunks "
        f"(dropped {total_dropped}: {dropped_short} short, "
        f"{dropped_boilerplate} boilerplate, {dropped_no_keyword} off-topic)"
    )
    return clean


# ---------------------------------------------------------------------------
# Step 4 — Embed and store in FAISS
# ---------------------------------------------------------------------------

def build_vectorstore(chunks: list, store_path: str = FAISS_STORE_PATH) -> object:
    """
    Embed clean chunks with OllamaEmbeddings and store in a FAISS index.
    Persists using FAISS native save_local().

    Args:
        chunks:     Clean chunks from filter_chunks()
        store_path: Directory path to save the FAISS index

    Returns:
        FAISS vectorstore object.
    """
    print(f"[rag_pipeline] Building FAISS vectorstore with {len(chunks)} chunks...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(store_path)
    print(f"[rag_pipeline] Vectorstore saved to {store_path!r}")
    return vectorstore


def load_vectorstore(store_path: str = FAISS_STORE_PATH) -> object | None:
    """
    Load a previously persisted FAISS vectorstore from disk.

    Args:
        store_path: Directory path written by build_vectorstore()

    Returns:
        FAISS vectorstore object, or None if directory does not exist.
    """
    if not os.path.exists(store_path):
        print(f"[rag_pipeline] No vectorstore found at {store_path!r}")
        return None

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = FAISS.load_local(
        store_path, embeddings, allow_dangerous_deserialization=True
    )
    print(f"[rag_pipeline] Vectorstore loaded from {store_path!r}")
    return vectorstore


# ---------------------------------------------------------------------------
# Step 5 — Layer 2: Manual similarity filter (retrieval-time compression)
# ---------------------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class SimilarityFilterRetriever:
    """
    Layer 2 — Query-time retriever that drops chunks below a cosine similarity
    threshold.

    Fix 3 changes:
      - Threshold lowered to 0.60 (was 0.76) to handle cross-domain queries
        where the query vector sits between two topic clusters.
      - Guaranteed fallback: if every candidate falls below the threshold, the
        top SIMILARITY_FALLBACK_K chunks by raw FAISS score are returned instead
        of an empty list, so the LLM always receives some context.
    """

    def __init__(
        self,
        vectorstore,
        embeddings,
        threshold: float = EMBEDDINGS_SIMILARITY_THRESHOLD,
        top_k: int = 10,
        fallback_k: int = SIMILARITY_FALLBACK_K,
    ):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.threshold = threshold
        self.top_k = top_k
        self.fallback_k = fallback_k

    def get_relevant_documents(self, query: str) -> list[Document]:
        # Embed the query
        query_vec = np.array(self.embeddings.embed_query(query))

        # Retrieve top_k candidates from FAISS (ordered by L2 score, best first)
        candidates = self.vectorstore.similarity_search(query, k=self.top_k)

        # Score and filter by cosine similarity
        scored: list[tuple[float, Document]] = []
        for doc in candidates:
            doc_vec = np.array(self.embeddings.embed_query(doc.page_content))
            score = _cosine_similarity(query_vec, doc_vec)
            scored.append((score, doc))

        kept = [doc for score, doc in scored if score >= self.threshold]

        # Fix 3: guaranteed fallback — never return empty-handed
        if not kept:
            kept = [doc for _, doc in scored[: self.fallback_k]]
            print(
                f"[rag_pipeline] Layer 2: all {len(candidates)} candidates below "
                f"threshold {self.threshold}. Falling back to top-{self.fallback_k} "
                f"by FAISS score."
            )
        else:
            print(
                f"[rag_pipeline] Layer 2: {len(candidates)} candidates → "
                f"{len(kept)} kept (threshold={self.threshold})"
            )

        return kept

    def invoke(self, query: str) -> list[Document]:
        return self.get_relevant_documents(query)


def get_retriever(vectorstore) -> SimilarityFilterRetriever:
    """
    Return a SimilarityFilterRetriever wrapping the FAISS vectorstore.

    Args:
        vectorstore: FAISS vectorstore from build_vectorstore() or load_vectorstore()

    Returns:
        SimilarityFilterRetriever instance.
    """
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    retriever = SimilarityFilterRetriever(
        vectorstore=vectorstore,
        embeddings=embeddings,
        threshold=EMBEDDINGS_SIMILARITY_THRESHOLD,
        fallback_k=SIMILARITY_FALLBACK_K,
    )
    print(
        f"[rag_pipeline] Layer 2: SimilarityFilterRetriever active "
        f"(threshold={EMBEDDINGS_SIMILARITY_THRESHOLD}, "
        f"fallback_k={SIMILARITY_FALLBACK_K})"
    )
    return retriever


# ---------------------------------------------------------------------------
# Step 6 — Query
# ---------------------------------------------------------------------------

def query(question: str, retriever) -> dict:
    """
    Run a question through the RAG chain and return the answer with sources.

    Fix 4: Prompt rewritten to allow the LLM to reason from partial or indirect
    context and supplement with general financial knowledge when the retrieved
    context does not fully cover the question. The LLM is asked to explicitly
    label which parts of its answer come from retrieved articles versus general
    knowledge, so the user can calibrate trust accordingly.

    Args:
        question:  User's natural language question
        retriever: SimilarityFilterRetriever from get_retriever()

    Returns:
        Dict with keys:
          "answer"  — LLM-generated answer string
          "sources" — newline-separated source URLs (may be empty string)
    """
    # Step 1 — retrieve relevant chunks (Fix 3 guarantees at least fallback_k docs)
    docs = retriever.get_relevant_documents(question)

    if not docs:
        # This path is only reached if the vectorstore itself is empty
        return {
            "answer": (
                "No articles have been indexed yet. Please fetch and process "
                "news from the sidebar before asking questions."
            ),
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

    # Step 3 — Fix 4: prompt that permits partial-context reasoning
    prompt = f"""You are a senior financial news analyst with deep knowledge of Indian and global markets.

You have been provided with excerpts from recent news articles as context. Your task is to answer the user's question as accurately and helpfully as possible.

Guidelines:
- Use the provided context as your primary source of information.
- If the context directly answers the question, base your answer on it and cite the relevant details.
- If the context is only partially relevant or covers a related but not identical topic, use it as supporting evidence and clearly supplement with your general financial and economic knowledge to give a complete answer.
- If the context contains no information related to the question at all, answer using your general knowledge of financial markets and economics, and explicitly state that your answer is based on general knowledge rather than today's fetched articles.
- Always be clear when switching between context-based information and general knowledge. For example: "Based on the fetched articles, ..." or "Based on general market knowledge, ..."
- Be concise, factual, and avoid speculation beyond reasonable financial analysis.

Context from fetched articles:
{context}

Question: {question}

Answer:"""

    # Step 4 — call LLM
    llm = OllamaLLM(model=LLM_MODEL, temperature=0.7)
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
    q = "What is the effect of Bengal election on Indian stock market?"
    result = query(q, retriever)
    print(f"\nQ: {q}")
    print(f"A: {result['answer']}")
    print(f"Sources: {result['sources']}")
