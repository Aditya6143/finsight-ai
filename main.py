"""
main.py — Streamlit UI for FinSight AI

Responsibilities (UI only):
- validate_config() at startup — surfaces misconfiguration immediately
- Sidebar: keyword input + "Fetch & Process News" button + dynamic model caption
- Status messages / spinners during ingestion
- Provider mismatch warning when loading a pre-built index
- Fix 1: auto-extract topic keywords from the user's question at query time,
         re-fetch Google News for those keywords, and augment the index so
         cross-domain questions always have relevant context.
- Question input with explicit Send button (no emoji)
- Answer rendering with error handling
- AuthenticationError caught and shown as st.error() (not a traceback)

All business logic lives in:
  feed_fetcher.py        — RSS ingestion, URL extraction, keyword extraction
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

import embeddings_provider
import feed_fetcher
import rag_pipeline

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FinSight AI: News Research Tool",
    layout="centered",
)

st.title("FinSight AI: News Research Tool")

# ---------------------------------------------------------------------------
# Sidebar — news fetching controls
# ---------------------------------------------------------------------------

st.sidebar.title("News Sources")
st.sidebar.markdown(
    "FinSight AI automatically pulls from **Reuters, Economic Times, "
    "Moneycontrol, Yahoo Finance, Bloomberg, and Mint**.\n\n"
    "Optionally add a keyword to also fetch matching Google News articles.\n\n"
    "**Tip:** If your question is about a specific event (e.g. Bengal elections, "
    "RBI policy), entering it as a keyword before fetching gives the most relevant "
    "results. Even without a keyword, FinSight AI will auto-fetch topic articles "
    "when you ask a question."
)

keyword = st.sidebar.text_input(
    "Keyword (optional)",
    placeholder="e.g. Bengal election, RBI policy, Nifty 50",
)

fetch_clicked = st.sidebar.button("Fetch & Process News", use_container_width=True)

st.sidebar.divider()

# Dynamic caption — reflects the active provider from config (unchanged)
st.sidebar.caption(
    f"Models: {config.llm_display_name()} (LLM) · "
    f"{config.embed_display_name()} (embeddings) via {config.provider_display_name()}"
)

# ---------------------------------------------------------------------------
# Session state — persist vectorstore and retriever across reruns
# ---------------------------------------------------------------------------

for key, default in [
    ("retriever", None),
    ("vectorstore", None),
    ("ingestion_done", False),
    ("index_warning", None),
    ("last_result", None),
    ("last_question", ""),
    ("ingested_keywords", set()),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Try loading a pre-built index on first run (before the button is clicked)
# This lets a deployed app serve queries immediately without re-ingesting.
# ---------------------------------------------------------------------------

if st.session_state.retriever is None and not fetch_clicked:
    vectorstore, warning = rag_pipeline.load_vectorstore()
    if vectorstore is not None:
        st.session_state.vectorstore = vectorstore
        st.session_state.retriever = rag_pipeline.get_retriever(vectorstore)
        st.session_state.ingestion_done = True
        st.session_state.index_warning = warning

# ---------------------------------------------------------------------------
# Show index mismatch warning if detected (emoji removed)
# ---------------------------------------------------------------------------

if st.session_state.index_warning:
    st.warning(st.session_state.index_warning)

# ---------------------------------------------------------------------------
# Helper — Fix 1: augment index with topic-specific articles for a query
# ---------------------------------------------------------------------------

def _augment_for_query(question: str) -> bool:
    """
    Extract topic keywords from the user's question. If those keywords have not
    yet been fetched, query Google News for them, load the articles, filter,
    embed, and merge the new chunks into the existing FAISS index.

    Returns True if the index was augmented, False if nothing new was added.
    """
    auto_kw = feed_fetcher.extract_query_keywords(question)
    if not auto_kw:
        return False

    kw_norm = auto_kw.strip().lower()
    if kw_norm in st.session_state.ingested_keywords:
        return False

    with st.status(
        f"Fetching topic-specific articles for: '{auto_kw}'...", expanded=True
    ) as status:

        st.write(f"Querying Google News for '{auto_kw}'...")
        new_urls = feed_fetcher.fetch_article_urls(
            keyword=None,
            query_keyword=auto_kw,
            max_per_feed=5,
            include_hardcoded=False,   # hardcoded feeds already in index
        )

        if not new_urls:
            status.update(
                label="No additional articles found for this topic.",
                state="complete",
            )
            st.session_state.ingested_keywords.add(kw_norm)
            return False

        st.write(f"Found {len(new_urls)} new articles. Loading content...")
        new_docs = rag_pipeline.load_articles(new_urls)

        if not new_docs:
            status.update(label="Could not load new article content.", state="complete")
            st.session_state.ingested_keywords.add(kw_norm)
            return False

        st.write("Splitting and filtering...")
        new_chunks = rag_pipeline.split_docs(new_docs)
        new_clean = rag_pipeline.filter_chunks(new_chunks)

        if not new_clean:
            status.update(
                label="New articles contained no relevant content after filtering.",
                state="complete",
            )
            st.session_state.ingested_keywords.add(kw_norm)
            return False

        st.write(f"Merging {len(new_clean)} new chunks into the index...")
        embeddings_model = embeddings_provider.get_embeddings()
        new_vs = rag_pipeline.FAISS.from_documents(new_clean, embeddings_model)

        st.session_state.vectorstore.merge_from(new_vs)
        st.session_state.vectorstore.save_local(rag_pipeline.FAISS_STORE_PATH)
        st.session_state.retriever = rag_pipeline.get_retriever(
            st.session_state.vectorstore
        )

        st.session_state.ingested_keywords.add(kw_norm)
        status.update(
            label=f"Index updated with {len(new_clean)} topic-specific chunks.",
            state="complete",
        )

    return True

# ---------------------------------------------------------------------------
# Ingestion flow — triggered by sidebar button
# ---------------------------------------------------------------------------

if fetch_clicked:
    st.session_state.retriever = None
    st.session_state.vectorstore = None
    st.session_state.ingestion_done = False
    st.session_state.index_warning = None
    st.session_state.last_result = None
    st.session_state.last_question = ""
    st.session_state.ingested_keywords = set()

    with st.status("Fetching and processing news...", expanded=True) as status:

        # Step 1 — fetch URLs
        st.write("Fetching article URLs from RSS feeds...")
        urls = feed_fetcher.fetch_article_urls(
            keyword=keyword if keyword.strip() else None,
            max_per_feed=5,
        )

        if not urls:
            status.update(label="No articles found. Try a different keyword.", state="error")
            st.stop()

        st.write(f"Found **{len(urls)}** articles")

        # Step 2 — load articles
        st.write("Loading article content...")
        docs = rag_pipeline.load_articles(urls)

        if not docs:
            status.update(label="Could not load article content.", state="error")
            st.stop()

        st.write(f"Loaded **{len(docs)}** documents")

        # Step 3 — split
        st.write("Splitting into chunks...")
        chunks = rag_pipeline.split_docs(docs)
        st.write(f"**{len(chunks)}** raw chunks")

        # Step 4 — Layer 1 filter
        st.write("Filtering noisy chunks (Layer 1)...")
        clean_chunks = rag_pipeline.filter_chunks(chunks)
        st.write(f"**{len(clean_chunks)}** clean chunks retained")

        if not clean_chunks:
            status.update(
                label="All chunks were filtered out. Try a different keyword.",
                state="error",
            )
            st.stop()

        # Step 5 — embed + store
        st.write("Building FAISS vectorstore (this may take a minute)...")
        vectorstore = rag_pipeline.build_vectorstore(clean_chunks)
        st.session_state.vectorstore = vectorstore
        st.write("Vectorstore ready")

        # Step 6 — Layer 2 retriever
        st.write("Setting up similarity retriever (Layer 2)...")
        st.session_state.retriever = rag_pipeline.get_retriever(vectorstore)
        st.session_state.index_warning = None

        # Record sidebar keyword as already ingested
        if keyword and keyword.strip():
            st.session_state.ingested_keywords.add(keyword.strip().lower())

        st.session_state.ingestion_done = True
        status.update(label="News processed — ask your question below.", state="complete")

# ---------------------------------------------------------------------------
# Query flow
# ---------------------------------------------------------------------------

st.divider()

if not st.session_state.ingestion_done:
    st.info("Click **Fetch & Process News** in the sidebar to get started.")
else:
    question = st.text_input(
        "Ask a question about today's finance news:",
        placeholder="e.g. What is the effect of Bengal election on the stock market?",
    )

    send_clicked = st.button("Send", type="primary")

    if send_clicked and question and question.strip():

        # Fix 1: augment index with topic-specific articles if needed
        _augment_for_query(question)

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

        st.session_state.last_result = result
        st.session_state.last_question = question

    elif send_clicked and not (question and question.strip()):
        st.warning("Please enter a question before submitting.")

    # Render last result (persists across reruns)
    if st.session_state.last_result:
        result = st.session_state.last_result

        # Show notice if topic articles were auto-fetched for this question
        if st.session_state.last_question:
            auto_kw = feed_fetcher.extract_query_keywords(st.session_state.last_question)
            if auto_kw and auto_kw.strip().lower() in st.session_state.ingested_keywords:
                st.info(
                    f"Additional articles about **{auto_kw}** were automatically "
                    f"fetched and added to the index to answer this question."
                )

        st.header("Answer")
        st.write(result["answer"])

        sources = result.get("sources", "").strip()
        if sources:
            st.subheader("Sources")
            for source in sources.split("\n"):
                if source.strip():
                    st.write(source.strip())