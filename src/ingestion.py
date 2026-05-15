import re
from datetime import datetime, timezone

import feedparser
from newsapi import NewsApiClient
from tqdm import tqdm

from src.scraper import scrape_full_text


# ── RSS feed config ───────────────────────────────────────────────────────────

# Each entry is (outlet_name, feed_url)
# These are the six outlets the project tracks for political diversity
RSS_FEEDS = [
    ("bbc",         "http://feeds.bbci.co.uk/news/rss.xml"),
    ("reuters",     "https://feeds.reuters.com/reuters/topNews"),
    ("the_guardian","https://www.theguardian.com/world/rss"),
    ("al_jazeera",  "https://www.aljazeera.com/xml/rss/all.xml"),
    ("fox_news",    "https://feeds.foxnews.com/foxnews/latest"),
    ("npr",         "https://feeds.npr.org/1001/rss.xml"),
]


# ── Outlet normalisation ──────────────────────────────────────────────────────

# Maps substrings found in raw source names to a clean, consistent outlet slug.
# NewsAPI returns names like "BBC News", "Fox News", "Reuters" — this converts them.
_OUTLET_MAP = {
    "bbc":       "bbc",
    "reuters":   "reuters",
    "guardian":  "the_guardian",
    "al jazeera":"al_jazeera",
    "fox":       "fox_news",
    "npr":       "npr",
    "cnn":       "cnn",
    "nbc":       "nbc_news",
    "abc":       "abc_news",
    "associated press": "ap",
    "washington post":  "washington_post",
    "new york times":   "new_york_times",
    "politico":  "politico",
    "the hill":  "the_hill",
}

def normalise_outlet(raw_name: str) -> str:
    """
    Convert a raw outlet name from NewsAPI or an RSS feed title into a
    consistent lowercase slug e.g. "BBC News" -> "bbc".
    Falls back to a slugified version of the raw name if no match is found.
    """
    lower = raw_name.lower()

    for keyword, slug in _OUTLET_MAP.items():
        if keyword in lower:
            return slug

    # Fallback: strip non-alphanumeric characters and replace spaces with underscores
    return re.sub(r"[^a-z0-9]+", "_", lower).strip("_")


# ── NewsAPI ───────────────────────────────────────────────────────────────────

def fetch_newsapi_articles(topic: str, api_key: str, page_size: int = 20) -> list[dict]:
    """
    Search NewsAPI for articles on a given topic and return a list of article dicts.

    page_size controls how many results to request (max 100 on free tier).
    Each article is scraped for full body text before being returned.
    """
    client = NewsApiClient(api_key=api_key)

    # get_everything searches across all outlets for the query string
    response = client.get_everything(
        q=topic,
        language="en",
        sort_by="publishedAt",   # newest first
        page_size=page_size,
    )

    articles = []

    # tqdm wraps the list to show a progress bar while scraping
    for item in tqdm(response.get("articles", []), desc=f"Scraping NewsAPI [{topic}]"):
        url = item.get("url", "")
        if not url:
            continue

        body, body_source = scrape_full_text(url)

        # If scraping failed, fall back to the truncated summary NewsAPI provides
        if body_source == "summary_only":
            body = item.get("description", "") or ""

        raw_outlet = item.get("source", {}).get("name", "unknown")

        articles.append({
            "url":          url,
            "outlet":       normalise_outlet(raw_outlet),
            "headline":     item.get("title", ""),
            "body":         body,
            "body_source":  body_source,
            "published_at": item.get("publishedAt", ""),
            "topic":        topic,
            "source":       "newsapi",
            "ingested_at":  datetime.now(timezone.utc).isoformat(),
        })

    return articles


# ── RSS feeds ─────────────────────────────────────────────────────────────────

def _parse_rss_date(entry) -> str:
    """
    Extract a published date from a feedparser entry.
    feedparser normalises dates into a time.struct_time under `published_parsed`.
    Falls back to an empty string if no date is present.
    """
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        # Convert the struct_time to a datetime, then to an ISO 8601 string
        dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        return dt.isoformat()
    return ""


def _extract_rss_body(entry) -> tuple[str, str]:
    """
    Try to get article text directly from the RSS entry before scraping.

    Some outlets (e.g. The Guardian) provide the full article body in the feed.
    If the feed content is long enough to be useful, return it directly.
    Otherwise return empty string so the caller falls back to scraping.
    """
    # `content` is a list of content objects — check the first one
    if hasattr(entry, "content") and entry.content:
        text = entry.content[0].get("value", "")
        # Strip HTML tags to get plain text
        clean = re.sub(r"<[^>]+>", " ", text).strip()
        if len(clean) > 200:
            return clean, "rss_full"

    return "", ""


def fetch_rss_articles() -> list[dict]:
    """
    Parse all configured RSS feeds and return a list of article dicts.
    Scrapes full text for any article where the feed does not provide it.
    """
    all_articles = []

    for outlet_slug, feed_url in RSS_FEEDS:
        feed = feedparser.parse(feed_url)

        for entry in tqdm(feed.entries, desc=f"Scraping RSS [{outlet_slug}]"):
            url = entry.get("link", "")
            if not url:
                continue

            # Try to use the body already in the RSS entry
            body, body_source = _extract_rss_body(entry)

            # If the feed didn't provide useful content, scrape the full page
            if not body:
                body, body_source = scrape_full_text(url)

            # Last resort: use the short summary the feed always provides
            if body_source == "summary_only":
                body = re.sub(r"<[^>]+>", " ", entry.get("summary", "")).strip()

            all_articles.append({
                "url":          url,
                "outlet":       outlet_slug,
                "headline":     entry.get("title", ""),
                "body":         body,
                "body_source":  body_source,
                "published_at": _parse_rss_date(entry),
                "topic":        "",   # RSS feeds aren't topic-specific
                "source":       "rss",
                "ingested_at":  datetime.now(timezone.utc).isoformat(),
            })

    return all_articles
