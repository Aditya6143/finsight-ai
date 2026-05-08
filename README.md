# FinSight 📈

> Finance news research tool powered by RAG — ask questions, get answers with cited sources.

FinSight automatically pulls the latest articles from major finance RSS feeds, embeds them into a local vector store, and lets you ask natural language questions about today's market news. Runs fully locally via Ollama, or on Streamlit Cloud via OpenAI.

---

## Features

- **Automated news ingestion** — pulls from Reuters, Economic Times, Moneycontrol, Yahoo Finance, and Mint on every fetch
- **Keyword search** — type a term like "RBI policy" or "Sensex" to also fetch matching Google News articles
- **Two-layer noise filter** — removes cookie banners, nav menus, and off-topic chunks before they reach the vector store
- **Provider-selectable** — run locally with Ollama (no API key) or deploy to Streamlit Cloud with OpenAI
- **Cited answers** — every answer includes the source URLs it was derived from
- **Data stays local** — when using Ollama, no article text, query, or answer ever leaves the machine

---

## Architecture

```
RSS feeds (hardcoded + Google News keyword)
        ↓
  feedparser → article URLs
        ↓
  UnstructuredURLLoader → raw article text
        ↓
  RecursiveCharacterTextSplitter → chunks
        ↓
  Layer 1 filter (word count · boilerplate · finance keywords)
        ↓
  Embeddings → FAISS vector store (persisted to disk)
        ↓  (at query time)
  Layer 2 filter (L2 similarity score via FAISS native scoring)
        ↓
  LLM → answer + cited sources
```

---

## Quickstart

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running (for local mode)

### Install

```bash
git clone https://github.com/your-username/finsight.git
cd finsight
pip install -r requirements.txt
```

### Pull Ollama models (local mode only)

```bash
ollama pull llama3:instruct
ollama pull mxbai-embed-large
```

### Configure

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```env
# Local mode (default — no API key needed)
PROVIDER=ollama

# Cloud mode
# PROVIDER=openai
# OPENAI_API_KEY=sk-...
```

### Run

```bash
streamlit run main.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Deployment (Streamlit Cloud)

1. Build the FAISS index locally with OpenAI embeddings:

```bash
PROVIDER=openai OPENAI_API_KEY=sk-... python -c "
import feed_fetcher, rag_pipeline
urls = feed_fetcher.fetch_article_urls(keyword='Sensex', max_per_feed=3)
rag_pipeline.ingest(urls)
"
```

2. Commit the index directory:

```bash
git add faiss_store_finsight/
git commit -m "add pre-built index for cold-start deploy"
git push
```

3. Deploy at [share.streamlit.io](https://share.streamlit.io) → connect repo → set `main.py` as entry point.

4. Add secrets under **App Settings → Secrets**:

```toml
PROVIDER = "openai"
OPENAI_API_KEY = "sk-..."
```

The app will detect the pre-built index on first load — no ingestion wait for users.

---

## Module Overview

| File | Responsibility |
|---|---|
| `config.py` | All settings; reads from `.env` (local) or `st.secrets` (Streamlit Cloud) |
| `feed_fetcher.py` | RSS ingestion and article URL extraction |
| `rag_pipeline.py` | Chunking, filtering, embedding, FAISS, retrieval, querying |
| `llm_provider.py` | LLM factory — returns Ollama or OpenAI LLM with unified `.invoke()` |
| `embeddings_provider.py` | Embeddings factory — returns Ollama or OpenAI embeddings |
| `main.py` | Streamlit UI — wires sidebar inputs to the modules above |

---

## Noise Reduction

Web-scraped articles contain structural noise (cookie banners, nav menus, related article links, footer text) that degrades answer quality if it reaches the vector store. FinSight uses a two-layer approach:

**Layer 1 — ingestion time (chunk filter)**
Drops chunks that are too short (< 30 words), match known boilerplate patterns, or contain no finance-domain keywords.

**Layer 2 — query time (similarity filter)**
Uses FAISS native L2 distances to score each retrieved chunk against the user's query. Chunks below the similarity threshold are dropped before being passed to the LLM. No extra embedding API calls — scores are derived from vectors already in the index.

---

## Configuration Reference

All settings live in `config.py` and can be overridden via environment variables or Streamlit secrets.

| Variable | Default | Description |
|---|---|---|
| `PROVIDER` | `ollama` | `ollama` or `openai` |
| `OPENAI_API_KEY` | _(empty)_ | Required when `PROVIDER=openai` |
| `FAISS_STORE_PATH` | `faiss_store_finsight` | Directory for the persisted FAISS index |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `MIN_WORD_COUNT` | `30` | Layer 1: minimum words per chunk |
| `EMBEDDINGS_SIMILARITY_THRESHOLD` | `0.15` | Layer 2: minimum L2-derived similarity score |

---

## Smoke Tests

Each module is independently runnable without Streamlit:

```bash
# Test RSS ingestion
python feed_fetcher.py

# Test full pipeline (ingestion + query)
python rag_pipeline.py
```

---

## Tech Stack

- [Streamlit](https://streamlit.io) — UI
- [LangChain](https://python.langchain.com) — document loading and text splitting
- [FAISS](https://faiss.ai) — vector store
- [Ollama](https://ollama.com) — local LLM and embeddings (`llama3:instruct`, `mxbai-embed-large`)
- [OpenAI](https://platform.openai.com) — cloud LLM and embeddings (`gpt-4.1`, `text-embedding-3-small`)
- [feedparser](https://feedparser.readthedocs.io) — RSS parsing

---

## License

MIT — see [LICENSE](LICENSE).