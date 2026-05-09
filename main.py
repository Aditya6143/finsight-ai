"""
main.py — Streamlit UI for RockyBot

Responsibilities (UI only):
- Sidebar: keyword input + "Fetch & Process News" button
- Status messages / spinners during ingestion
- Question input with explicit Send button
- Fix 1: auto-extract topic keywords from the user's question at query time,
         re-fetch Google News for those keywords, and augment / rebuild the
         index so cross-domain questions always have relevant context.
- Render answer + sources

All business logic lives in:
  feed_fetcher.py  — RSS ingestion, URL extraction, keyword extraction
  rag_pipeline.py  — chunking, filtering, embedding, querying
"""

import streamlit as st
import feed_fetcher
import rag_pipeline

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RockyBot — Finance News Research",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;500;600&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'EB Garamond', serif;
        letter-spacing: -0.02em;
    }

    .block-container {
        padding-top: 2.5rem;
        max-width: 780px;
    }

    section[data-testid="stSidebar"] {
        background-color: #f7f6f3;
        border-right: 1px solid #e4e0d8;
    }

    section[data-testid="stSidebar"] * {
        font-family: 'DM Sans', sans-serif;
    }

    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #1a1a1a;
        color: #ffffff;
        border: none;
        border-radius: 4px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        font-size: 0.875rem;
        letter-spacing: 0.04em;
        padding: 0.5rem 1.5rem;
        transition: background-color 0.15s ease;
    }

    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #333333;
    }

    div[data-testid="stButton"] > button[kind="secondary"] {
        border: 1px solid #c8c4bc;
        border-radius: 4px;
        color: #1a1a1a;
        font-family: 'DM Sans', sans-serif;
        font-weight: 400;
        font-size: 0.875rem;
    }

    .answer-block {
        background: #fafaf8;
        border-left: 3px solid #1a1a1a;
        padding: 1.25rem 1.5rem;
        border-radius: 0 4px 4px 0;
        margin-top: 1rem;
        font-size: 0.97rem;
        line-height: 1.7;
        color: #1a1a1a;
    }

    .sources-block {
        margin-top: 1.25rem;
        font-size: 0.82rem;
        color: #666;
    }

    .sources-block a {
        color: #444;
        word-break: break-all;
    }

    .notice-block {
        background: #fdf8ee;
        border-left: 3px solid #b07d2a;
        padding: 0.75rem 1rem;
        border-radius: 0 4px 4px 0;
        font-size: 0.85rem;
        color: #5a4010;
        margin-top: 0.75rem;
        line-height: 1.6;
    }

    hr {
        border: none;
        border-top: 1px solid #e4e0d8;
        margin: 1.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("## RockyBot")
st.markdown(
    "<p style='color:#666; font-size:0.95rem; margin-top:-0.5rem;'>"
    "Finance news research, powered by local inference.</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — news fetching controls
# ---------------------------------------------------------------------------

st.sidebar.markdown("### News Sources")
st.sidebar.markdown(
    "<p style='font-size:0.85rem; color:#555; line-height:1.55;'>"
    "Pulls automatically from Reuters, Economic Times, Moneycontrol, "
    "Yahoo Finance, and Mint. Add a keyword to also include matching "
    "Google News articles.<br><br>"
    "<strong>Tip:</strong> If your question is about a specific event "
    "(e.g. Bengal elections, RBI policy), entering it as a keyword "
    "before fetching gives the most relevant results. Even without a "
    "keyword, RockyBot will auto-fetch topic articles when you ask a "
    "question.</p>",
    unsafe_allow_html=True,
)

keyword = st.sidebar.text_input(
    "Keyword (optional)",
    placeholder="e.g. Bengal election, RBI policy, Nifty 50",
)

fetch_clicked = st.sidebar.button(
    "Fetch & Process News",
    use_container_width=True,
)

st.sidebar.divider()
st.sidebar.markdown(
    "<p style='font-size:0.78rem; color:#888;'>"
    "LLM: llama3:instruct &nbsp;·&nbsp; Embeddings: mxbai-embed-large<br>"
    "Inference via Ollama — no data leaves this machine.</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

for key, default in [
    ("retriever", None),
    ("vectorstore", None),
    ("ingestion_done", False),
    ("last_result", None),
    ("last_question", ""),
    ("ingested_keywords", set()),
]:
    if key not in st.session_state:
        st.session_state[key] = default

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
            include_hardcoded=False,
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
        embeddings_model = rag_pipeline.OllamaEmbeddings(model=rag_pipeline.EMBED_MODEL)
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
    st.session_state.last_result = None
    st.session_state.last_question = ""
    st.session_state.ingested_keywords = set()

    with st.status("Processing news feeds...", expanded=True) as status:

        st.write("Fetching article URLs from RSS feeds...")
        urls = feed_fetcher.fetch_article_urls(
            keyword=keyword if keyword.strip() else None,
            max_per_feed=5,
        )

        if not urls:
            status.update(label="No articles found. Try a different keyword.", state="error")
            st.stop()

        st.write(f"Found **{len(urls)}** articles.")

        st.write("Loading article content...")
        docs = rag_pipeline.load_articles(urls)

        if not docs:
            status.update(label="Could not load article content.", state="error")
            st.stop()

        st.write(f"Loaded **{len(docs)}** documents.")

        st.write("Splitting into chunks...")
        chunks = rag_pipeline.split_docs(docs)
        st.write(f"**{len(chunks)}** raw chunks produced.")

        st.write("Filtering noisy chunks (Layer 1)...")
        clean_chunks = rag_pipeline.filter_chunks(chunks)
        st.write(f"**{len(clean_chunks)}** clean chunks retained.")

        if not clean_chunks:
            status.update(
                label="All chunks were filtered out. Try a different keyword.",
                state="error",
            )
            st.stop()

        st.write("Building FAISS vectorstore — this may take a moment...")
        vectorstore = rag_pipeline.build_vectorstore(clean_chunks)
        st.session_state.vectorstore = vectorstore
        st.write("Vectorstore ready.")

        st.write("Initialising retriever (Layer 2)...")
        st.session_state.retriever = rag_pipeline.get_retriever(vectorstore)

        if keyword and keyword.strip():
            st.session_state.ingested_keywords.add(keyword.strip().lower())

        st.session_state.ingestion_done = True
        status.update(label="News processed. Ask your question below.", state="complete")

# ---------------------------------------------------------------------------
# Query flow
# ---------------------------------------------------------------------------

st.markdown("<hr>", unsafe_allow_html=True)

if not st.session_state.ingestion_done:
    st.markdown(
        "<p style='color:#888; font-size:0.92rem;'>"
        "Use the sidebar to fetch and process today's news before querying.</p>",
        unsafe_allow_html=True,
    )
else:
    question = st.text_input(
        "Your question",
        placeholder="e.g. What is the effect of Bengal election on the stock market?",
        label_visibility="collapsed",
    )

    send_clicked = st.button("Send", type="primary")

    if send_clicked and question and question.strip():

        # Fix 1: augment index with topic-specific articles if needed
        _augment_for_query(question)

        with st.spinner("Retrieving relevant context and generating answer..."):
            result = rag_pipeline.query(question, st.session_state.retriever)

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
                st.markdown(
                    f"<div class='notice-block'>Additional articles about "
                    f"<strong>{auto_kw}</strong> were automatically fetched "
                    f"and added to the index to answer this question.</div>",
                    unsafe_allow_html=True,
                )

        st.markdown(
            f"<div class='answer-block'>{result['answer']}</div>",
            unsafe_allow_html=True,
        )

        sources = result.get("sources", "").strip()
        if sources:
            source_links = "".join(
                f"<div><a href='{s.strip()}' target='_blank'>{s.strip()}</a></div>"
                for s in sources.split("\n")
                if s.strip()
            )
            st.markdown(
                f"<div class='sources-block'><strong>Sources</strong>{source_links}</div>",
                unsafe_allow_html=True,
            )
