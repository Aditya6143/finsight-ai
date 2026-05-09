"""
feed_fetcher.py — News Ingestion Layer for RockyBot

Responsibilities:
- Define hardcoded finance RSS feed sources
- Build dynamic Google News RSS URLs from a user keyword
- Parse all feeds via feedparser
- Return a deduplicated, clean list of article URLs

Fix 1: extract_query_keywords() pulls topic terms from a user question so
        that Google News is queried for the right subject even when the user
        did not pre-fill the sidebar keyword field.

No LangChain, no Streamlit — pure Python, fully testable in isolation.
"""

import re
import feedparser
from urllib.parse import quote_plus

# ---------------------------------------------------------------------------
# Hardcoded finance RSS feeds
# ---------------------------------------------------------------------------

FINANCE_RSS_FEEDS = [
    # Reuters Business
    "https://feeds.reuters.com/reuters/businessNews",
    # Economic Times Markets
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    # Moneycontrol News
    "https://www.moneycontrol.com/rss/MCtopnews.xml",
    # Yahoo Finance
    "https://finance.yahoo.com/news/rssindex",
    # Mint Markets
    "https://www.livemint.com/rss/markets",
]

# ---------------------------------------------------------------------------
# Google News RSS — dynamic keyword-based feed
# ---------------------------------------------------------------------------

GOOGLE_NEWS_RSS_TEMPLATE = (
    "https://news.google.com/rss/search?q={keyword}&hl=en-IN&gl=IN&ceid=IN:en"
)

# Common English stop-words to strip when auto-extracting query keywords
_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "and", "or", "but", "if", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "as", "about", "into", "through", "during", "before",
    "after", "above", "below", "between", "what", "how", "when", "where",
    "which", "who", "whom", "why", "this", "that", "these", "those",
    "it", "its", "me", "my", "we", "our", "you", "your", "he", "she",
    "they", "their", "there", "then", "than", "so", "up", "out", "no",
    "not", "any", "some", "tell", "give", "explain", "describe",
    "effect", "impact", "influence", "affect", "relation", "relationship",
    "like", "get", "got", "go", "going",
}


def extract_query_keywords(question: str, max_keywords: int = 3) -> str | None:
    """
    Extract meaningful topic keywords from a natural language question so that
    Google News can be queried for relevant articles automatically.

    Strategy:
      1. Tokenise the question into lowercase words.
      2. Drop single-character tokens and common stop-words.
      3. Keep words that are either capitalised in the original (proper nouns /
         named entities) or longer than 4 characters (likely substantive terms).
      4. Join up to max_keywords terms into a single query string.

    Examples:
      "What is the effect of Bengal election on Indian stock market?"
        → "Bengal election market"
      "How is RBI repo rate affecting Nifty?"
        → "RBI repo Nifty"

    Args:
        question:     Raw user question string.
        max_keywords: Maximum number of keywords to include.

    Returns:
        A space-joined keyword string, or None if nothing meaningful was found.
    """
    if not question or not question.strip():
        return None

    tokens = re.findall(r"\b[A-Za-z]+\b", question)

    # Keep proper nouns (capitalised mid-sentence) first, then long lower-case words
    proper = [t for t in tokens if t[0].isupper() and t.lower() not in _STOP_WORDS]
    common = [
        t for t in tokens
        if not t[0].isupper() and len(t) > 4 and t.lower() not in _STOP_WORDS
    ]

    # Deduplicate while preserving order (proper nouns take priority)
    seen: set[str] = set()
    chosen: list[str] = []
    for word in proper + common:
        key = word.lower()
        if key not in seen:
            seen.add(key)
            chosen.append(word)
        if len(chosen) == max_keywords:
            break

    if not chosen:
        return None

    kw = " ".join(chosen)
    print(f"[feed_fetcher] Auto-extracted query keywords: {kw!r}")
    return kw


def build_google_news_url(keyword: str) -> str:
    """
    Construct a Google News RSS URL for a given keyword.

    Args:
        keyword: Search term e.g. "Sensex", "RBI policy", "Nifty 50"

    Returns:
        Fully formed RSS URL string.
    """
    return GOOGLE_NEWS_RSS_TEMPLATE.format(keyword=quote_plus(keyword.strip()))


# ---------------------------------------------------------------------------
# Feed parsing
# ---------------------------------------------------------------------------

def _parse_feed(url: str, max_articles: int = 5) -> list[str]:
    """
    Parse a single RSS feed URL and extract article links.

    Args:
        url:          RSS feed URL
        max_articles: Max entries to pull from this feed

    Returns:
        List of article URLs (may be empty if feed is unreachable or malformed).
    """
    try:
        feed = feedparser.parse(url)
        urls = []
        for entry in feed.entries[:max_articles]:
            link = entry.get("link", "").strip()
            if link:
                urls.append(link)
        return urls
    except Exception as e:
        print(f"[feed_fetcher] Warning: could not parse feed {url!r} — {e}")
        return []


def fetch_article_urls(
    keyword: str | None = None,
    query_keyword: str | None = None,
    max_per_feed: int = 5,
    include_hardcoded: bool = True,
) -> list[str]:
    """
    Fetch article URLs from all configured RSS sources.

    Combines:
      1. Hardcoded finance feeds  (if include_hardcoded=True)
      2. Google News RSS for the sidebar keyword (if keyword is provided)
      3. Google News RSS auto-derived from the user's question (query_keyword)
         — Fix 1: ensures topic-specific content is fetched even when the user
           did not pre-fill the sidebar keyword.

    Deduplicates the final list before returning.

    Args:
        keyword:           User-supplied sidebar keyword (optional).
        query_keyword:     Keyword auto-extracted from the user's question (optional).
                           Passed by main.py at query time to re-fetch if needed.
        max_per_feed:      Max articles pulled per individual RSS feed.
        include_hardcoded: Whether to include the preset finance feeds.

    Returns:
        Deduplicated list of article URLs ready for UnstructuredURLLoader.
    """
    all_urls: list[str] = []

    # --- Hardcoded feeds ---
    if include_hardcoded:
        for feed_url in FINANCE_RSS_FEEDS:
            fetched = _parse_feed(feed_url, max_articles=max_per_feed)
            print(f"[feed_fetcher] {feed_url!r} → {len(fetched)} articles")
            all_urls.extend(fetched)

    # --- Sidebar keyword → Google News ---
    if keyword and keyword.strip():
        google_url = build_google_news_url(keyword)
        fetched = _parse_feed(google_url, max_articles=max_per_feed)
        print(f"[feed_fetcher] Google News (sidebar: {keyword!r}) → {len(fetched)} articles")
        all_urls.extend(fetched)

    # --- Fix 1: auto-extracted query keyword → Google News ---
    # Only fetch if it is different from the sidebar keyword (avoid duplicate call)
    if query_keyword and query_keyword.strip():
        if not keyword or query_keyword.strip().lower() != keyword.strip().lower():
            google_url = build_google_news_url(query_keyword)
            fetched = _parse_feed(google_url, max_articles=max_per_feed)
            print(
                f"[feed_fetcher] Google News (auto: {query_keyword!r}) → {len(fetched)} articles"
            )
            all_urls.extend(fetched)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_urls: list[str] = []
    for url in all_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    print(f"[feed_fetcher] Total unique article URLs: {len(unique_urls)}")
    return unique_urls


# ---------------------------------------------------------------------------
# Quick smoke test — run this file directly to verify feeds are reachable
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    question = input("Enter a question to test keyword extraction: ").strip()
    kw = extract_query_keywords(question)
    print(f"Extracted keyword: {kw!r}")
    urls = fetch_article_urls(query_keyword=kw, max_per_feed=3)
    print("\n--- Fetched URLs ---")
    for i, u in enumerate(urls, 1):
        print(f"  {i:>3}. {u}")
