import re
from datetime import datetime, timezone

import feedparser
import requests
from newsapi import NewsApiClient
from tqdm import tqdm

from src.scraper import scrape_full_text

# Outlets whose RSS feeds fail with SSL errors in WSL - pre-fetch content with
# requests (verify=False) and pass the raw bytes to feedparser instead
_SSL_PROBLEMATIC_FEEDS = {"the_wire", "republic_world"}


# -- RSS feed config -----------------------------------------------------------

# Each entry is (outlet_slug, feed_url). Test each URL before running - Indian
# outlets occasionally change or remove their RSS endpoints.
RSS_FEEDS = [
    ("the_hindu",       "https://www.thehindu.com/news/national/feeder/default.rss"),
    ("ndtv",            "https://feeds.feedburner.com/ndtvnews-top-stories"),
    ("times_of_india",  "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"),
    ("the_wire",        "https://thewire.in/feed"),
    ("hindustan_times", "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml"),
    ("india_today",     "https://www.indiatoday.in/rss/home"),
    ("scroll",          "https://scroll.in/feed"),
    ("indian_express",  "https://indianexpress.com/section/india/feed/"),
    ("republic_world",  "https://www.republicworld.com/feeds/top-stories.xml"),
    ("zee_news",        "https://zeenews.india.com/rss/india-national-news.xml"),
    ("news18",          "https://www.news18.com/rss/india.xml"),
]


# -- Outlet allowlist ----------------------------------------------------------

# NewsAPI searches on Indian topics also return foreign coverage of India (Fox News,
# BBC, Reuters, RT, etc.). These are not useful for an Indian media bias analysis,
# so I drop any article whose outlet slug is not in this set.
# RSS articles are always from known Indian outlets (the feeds are hardcoded above),
# so the filter is only applied to the NewsAPI path.
_INDIAN_OUTLETS = {
    "the_hindu", "ndtv", "times_of_india", "the_wire", "hindustan_times",
    "india_today", "scroll", "indian_express", "republic_world",
    "zee_news", "news18", "opindia", "the_print", "newslaundry",
    "businessline", "livemint", "pti", "ani", "the_quint",
    "deccan_herald", "deccan_chronicle", "tribune_india", "firstpost",
    "the_telegraph", "free_press_journal", "national_herald", "wire_science",
}


# -- Outlet normalisation ------------------------------------------------------

# Maps substrings found in raw NewsAPI source names to a consistent outlet slug.
_OUTLET_MAP = {
    "the hindu":           "the_hindu",
    "ndtv":                "ndtv",
    "times of india":      "times_of_india",
    "the wire":            "the_wire",
    "hindustan times":     "hindustan_times",
    "india today":         "india_today",
    "scroll":              "scroll",
    "scroll.in":           "scroll",
    "indian express":      "indian_express",
    "the indian express":  "indian_express",
    "republic world":      "republic_world",
    "republic":            "republic_world",
    "republicworld":       "republic_world",
    "zee news":            "zee_news",
    "zeenews":             "zee_news",
    "news18":              "news18",
    "opindia":             "opindia",
    "the print":           "the_print",
    "the wire science":    "the_wire",
    "newslaundry":         "newslaundry",
    "bbc":                 "bbc",
    "reuters":             "reuters",
    "associated press":    "ap",
    "pti":                 "pti",
    "ani":                 "ani",
}

def normalise_outlet(raw_name: str) -> str:
    """
    Convert a raw outlet name from NewsAPI or an RSS feed title into a
    consistent lowercase slug e.g. "Hindustan Times" -> "hindustan_times".
    Falls back to a slugified version of the raw name if no match is found.
    """
    lower = raw_name.lower()

    for keyword, slug in _OUTLET_MAP.items():
        if keyword in lower:
            return slug

    # Fallback: strip non-alphanumeric characters and replace spaces with underscores
    return re.sub(r"[^a-z0-9]+", "_", lower).strip("_")


# -- NewsAPI -------------------------------------------------------------------

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

        raw_outlet = item.get("source", {}).get("name", "unknown")
        outlet = normalise_outlet(raw_outlet)

        # Drop foreign outlets - NewsAPI returns international coverage of India
        # (Fox News, BBC, Reuters, etc.) which I don't want in an Indian bias analysis.
        if outlet not in _INDIAN_OUTLETS:
            continue

        body, body_source = scrape_full_text(url)

        # If scraping failed, fall back to the truncated summary NewsAPI provides
        if body_source == "summary_only":
            body = item.get("description", "") or ""

        articles.append({
            "url":          url,
            "outlet":       outlet,
            "headline":     item.get("title", ""),
            "body":         body,
            "body_source":  body_source,
            "published_at": item.get("publishedAt", ""),
            "topic":        topic,
            "source":       "newsapi",
            "ingested_at":  datetime.now(timezone.utc).isoformat(),
        })

    return articles


# -- RSS feeds -----------------------------------------------------------------

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
    # `content` is a list of content objects - check the first one
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
        if outlet_slug in _SSL_PROBLEMATIC_FEEDS:
            try:
                resp = requests.get(feed_url, timeout=10, verify=False)
                feed = feedparser.parse(resp.content)
            except Exception:
                feed = feedparser.parse(feed_url)
        else:
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


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from src.db import get_connection, create_tables, insert_article

    load_dotenv()
    api_key = os.getenv("NEWSAPI_KEY", "")

    TOPICS = [
        "India Pakistan", "BJP Modi", "Indian economy", "India China",
        "Kashmir", "communal violence India", "India democracy", "NEET India",
    ]

    conn = get_connection()
    create_tables(conn)
    total_inserted = 0

    if api_key:
        for topic in TOPICS:
            print(f"\n--- NewsAPI: {topic} ---")
            articles = fetch_newsapi_articles(topic, api_key, page_size=20)
            for a in articles:
                if insert_article(conn, a):
                    total_inserted += 1
        print(f"\nNewsAPI done. Inserted {total_inserted} articles.")
    else:
        print("No NEWSAPI_KEY in .env - skipping NewsAPI.")

    print("\n--- RSS feeds ---")
    rss_articles = fetch_rss_articles()
    rss_inserted = 0
    for a in rss_articles:
        if insert_article(conn, a):
            rss_inserted += 1
    total_inserted += rss_inserted
    print(f"RSS done. Inserted {rss_inserted} articles.")

    total_in_db = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    print(f"\nTotal articles in DB: {total_in_db}")
    conn.close()
