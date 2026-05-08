"""
main.py — Streamlit UI for RockyBot / FinSight

Responsibilities (UI only):
- validate_config() at startup — surfaces misconfiguration immediately
- Sidebar: keyword input + "Fetch & Process News" button + dynamic model caption
- Status messages / spinners during ingestion
- Provider mismatch warning when loading a pre-built index
- Question input + answer rendering with error handling
- AuthenticationError caught and shown as st.error() (not a traceback)

All business logic lives in:
  feed_fetcher.py        — RSS ingestion, URL extraction
  rag_pipeline.py        — chunking, filtering, embedding, querying
  config.py              — all settings, provider selection
  llm_provider.py        — LLM factory
  embeddings_provider.py — embeddings factory
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Config validation — must run before any Streamlit rendering
# ---------------------------------------------------------------------------

import config

try:
    config.validate_config()
except ValueError as _cfg_err:
    st.error(f"**Configuration error:** {_cfg_err}")
    st.stop()

import feed_fetcher
import rag_pipeline

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FinSight: News Research Tool",
    page_icon="📈",
    layout="centered",
)

st.title("FinSight: News Research Tool 📈")

# ---------------------------------------------------------------------------
# Sidebar — news fetching controls
# ---------------------------------------------------------------------------

st.sidebar.title("📰 News Sources")
st.sidebar.markdown(
    "FinSight automatically pulls from **Reuters, Economic Times, "
    "Moneycontrol, Yahoo Finance, and Mint**.\n\n"
    "Optionally add a keyword to also fetch matching Google News articles."
)

keyword = st.sidebar.text_input(
    "Keyword (optional)",
    placeholder="e.g. Sensex, RBI policy, Nifty 50",
)

fetch_clicked = st.sidebar.button("🔄 Fetch & Process News", use_container_width=True)

st.sidebar.divider()

# Dynamic caption — reflects the active provider from config
st.sidebar.caption(
    f"Models: {config.llm_display_name()} (LLM) · "
    f"{config.embed_display_name()} (embeddings) via {config.provider_display_name()}"
)

# ---------------------------------------------------------------------------
# Session state — persist vectorstore and retriever across reruns
# ---------------------------------------------------------------------------

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "ingestion_done" not in st.session_state:
    st.session_state.ingestion_done = False

if "index_warning" not in st.session_state:
    st.session_state.index_warning = None

# ---------------------------------------------------------------------------
# Try loading a pre-built index on first run (before the button is clicked)
# This lets a deployed app serve queries immediately without re-ingesting.
# ---------------------------------------------------------------------------

if st.session_state.retriever is None and not fetch_clicked:
    vectorstore, warning = rag_pipeline.load_vectorstore()
    if vectorstore is not None:
        st.session_state.retriever = rag_pipeline.get_retriever(vectorstore)
        st.session_state.ingestion_done = True
        st.session_state.index_warning = warning

# ---------------------------------------------------------------------------
# Show index mismatch warning if detected
# ---------------------------------------------------------------------------

if st.session_state.index_warning:
    st.warning(st.session_state.index_warning)

# ---------------------------------------------------------------------------
# Ingestion flow — triggered by sidebar button
# ---------------------------------------------------------------------------

if fetch_clicked:
    st.session_state.retriever = None
    st.session_state.ingestion_done = False
    st.session_state.index_warning = None

    with st.status("Fetching and processing news...", expanded=True) as status:

        # Step 1 — fetch URLs
        st.write("📡 Fetching article URLs from RSS feeds...")
        urls = feed_fetcher.fetch_article_urls(
            keyword=keyword if keyword.strip() else None,
            max_per_feed=5,
        )

        if not urls:
            status.update(label="❌ No articles found. Try a different keyword.", state="error")
            st.stop()

        st.write(f"✅ Found **{len(urls)}** articles")

        # Step 2 — load articles
        st.write("📄 Loading article content...")
        docs = rag_pipeline.load_articles(urls)

        if not docs:
            status.update(label="❌ Could not load article content.", state="error")
            st.stop()

        st.write(f"✅ Loaded **{len(docs)}** documents")

        # Step 3 — split
        st.write("✂️ Splitting into chunks...")
        chunks = rag_pipeline.split_docs(docs)
        st.write(f"✅ **{len(chunks)}** raw chunks")

        # Step 4 — Layer 1 filter
        st.write("🧹 Filtering noisy chunks (Layer 1)...")
        clean_chunks = rag_pipeline.filter_chunks(chunks)
        st.write(f"✅ **{len(clean_chunks)}** clean chunks retained")

        if not clean_chunks:
            status.update(
                label="❌ All chunks were filtered out. Try a different keyword.",
                state="error",
            )
            st.stop()

        # Step 5 — embed + store
        st.write("🧠 Building FAISS vectorstore (this may take a minute)...")
        vectorstore = rag_pipeline.build_vectorstore(clean_chunks)
        st.write("✅ Vectorstore ready")

        # Step 6 — Layer 2 retriever
        st.write("🔍 Setting up similarity retriever (Layer 2)...")
        st.session_state.retriever = rag_pipeline.get_retriever(vectorstore)
        st.session_state.index_warning = None

        st.session_state.ingestion_done = True
        status.update(label="✅ News processed — ask your question below!", state="complete")

# ---------------------------------------------------------------------------
# Query flow
# ---------------------------------------------------------------------------

st.divider()

if not st.session_state.ingestion_done:
    st.info("👆 Click **Fetch & Process News** in the sidebar to get started.")
else:
    question = st.text_input(
        "Ask a question about today's finance news:",
        placeholder="e.g. What is happening with Sensex today?",
    )

    if question and question.strip():
        with st.spinner("Thinking..."):
            try:
                result = rag_pipeline.query(question, st.session_state.retriever)
            except ValueError as auth_err:
                # Catches OpenAI AuthenticationError re-raised by llm_provider
                st.error(f"**API key error:** {auth_err}")
                st.stop()
            except ConnectionError as conn_err:
                st.error(f"**Connection error:** {conn_err}")
                st.stop()

        st.header("Answer")
        st.write(result["answer"])

        sources = result.get("sources", "").strip()
        if sources:
            st.subheader("Sources")
            for source in sources.split("\n"):
                if source.strip():
                    st.write(source.strip())
