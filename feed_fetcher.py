"""
feed_fetcher.py — News Ingestion Layer for FinSight AI - PhillipCapital Research Assistant

Responsibilities:
- Define hardcoded finance RSS feed sources
- Build dynamic Google News RSS URLs from a user keyword
- Parse all feeds via feedparser
- Return a deduplicated, clean list of article URLs

No LangChain, no Streamlit — pure Python, fully testable in isolation.
"""

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
    # Bloomberg Markets
    "https://feeds.bloomberg.com/markets/news.rss",
]

# ---------------------------------------------------------------------------
# Google News RSS — dynamic keyword-based feed
# ---------------------------------------------------------------------------

GOOGLE_NEWS_RSS_TEMPLATE = (
    "https://news.google.com/rss/search?q={keyword}&hl=en-IN&gl=IN&ceid=IN:en"
)


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
        max_articles: Max entries to pull from this feed (avoids overwhelming
                      the downstream loader with hundreds of URLs)

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
        print(f"[finsight] Warning: could not parse feed {url!r} — {e}")
        return []


def fetch_article_urls(
    keyword: str | None = None,
    max_per_feed: int = 5,
    include_hardcoded: bool = True,
) -> list[str]:
    """
    Fetch article URLs from all configured RSS sources.

    Combines:
      1. Hardcoded finance feeds  (if include_hardcoded=True)
      2. Google News RSS for the user's keyword (if keyword is provided)

    Deduplicates the final list before returning.

    Args:
        keyword:           User-supplied search term (optional).
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
            print(f"[finsight] {feed_url!r} → {len(fetched)} articles")
            all_urls.extend(fetched)

    # --- Dynamic Google News feed ---
    if keyword and keyword.strip():
        google_url = build_google_news_url(keyword)
        fetched = _parse_feed(google_url, max_articles=max_per_feed)
        print(f"[finsight] Google News ({keyword!r}) → {len(fetched)} articles")
        all_urls.extend(fetched)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_urls: list[str] = []
    for url in all_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    print(f"[finsight] Total unique article URLs: {len(unique_urls)}")
    return unique_urls


# ---------------------------------------------------------------------------
# Quick smoke test — run this file directly to verify feeds are reachable
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    keyword = input("Enter a finance keyword to test (or press Enter to skip): ").strip()
    urls = fetch_article_urls(keyword=keyword or None, max_per_feed=3)
    print("\n--- Fetched URLs ---")
    for i, u in enumerate(urls, 1):
        print(f"  {i:>3}. {u}")