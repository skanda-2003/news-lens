"""
Collects weakly-supervised training data for the Indian bias classifier.

Phase 1 - GDELT: pulls neutral articles from major Indian papers (The Hindu,
Indian Express, Hindustan Times, ToI). These are well-indexed in GDELT.

Phase 2 - RSS: collects bjp_aligned and opposition_aligned articles directly
from outlet RSS feeds. GDELT barely indexes thewire.in, scroll.in, thequint.com,
republicworld.com, etc. - their 15-minute snapshots skew heavily toward mainstream
papers. RSS is far more reliable for minority-class collection.

Labels come from outlet identity (distant supervision). The outlet's documented
editorial alignment is the label source, not per-article annotation.

Target: ~200 articles per label (600 total), enough to fine-tune RoBERTa.

Output: data/india_training/india_bias_data.csv  (columns: text, label, outlet, url, published_date)

Run from the project root:
    PYTHONPATH=. venv/bin/python3 src/collect_training_data.py
"""

import os
import re
import time
from datetime import datetime, timedelta, timezone

import feedparser
import gdelt
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.scraper import scrape_full_text

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


# -- Config --------------------------------------------------------------------

# GDELT is only used for neutral - it covers major Indian papers well
GDELT_NEUTRAL_DOMAINS = {
    "thehindu.com":                 "neutral",
    "indianexpress.com":            "neutral",
    "hindustantimes.com":           "neutral",
    "timesofindia.indiatimes.com":  "neutral",
}

# RSS feeds for bjp_aligned - zeenews and news18 are reliably accessible
RSS_FEEDS_BY_LABEL = {
    "bjp_aligned": [
        ("zeenews",  "https://zeenews.india.com/rss/india-national-news.xml"),
        ("news18",   "https://www.news18.com/rss/india.xml"),
        ("opindia",  "https://www.opindia.com/feed/"),
    ],
}

# Opposition sources - scraped directly because RSS feeds are broken in WSL.
# Three outlets to improve class diversity beyond single-outlet theprint.in.
OPPOSITION_SOURCES = [
    {
        "outlet":   "theprint",
        "sections": [
            "https://theprint.in/politics/",
            "https://theprint.in/india/governance/",
            "https://theprint.in/india/",
            "https://theprint.in/opinion/",
            "https://theprint.in/judiciary/",
        ],
        "domain":   "theprint.in",
        "url_segs": ["/politics/", "/india/", "/opinion/", "/judiciary/"],
    },
    {
        "outlet":   "thewire",
        "sections": [
            "https://thewire.in/politics",
            "https://thewire.in/government",
            "https://thewire.in/law",
            "https://thewire.in/rights",
        ],
        "domain":   "thewire.in",
        "url_segs": ["/politics/", "/government/", "/law/", "/rights/"],
    },
    {
        "outlet":   "scroll",
        "sections": [
            "https://scroll.in/topic/politics",
            "https://scroll.in/topic/government",
        ],
        "domain":   "scroll.in",
        "url_segs": ["/article/"],
    },
]

# Target per opposition outlet - 3 outlets × 70 ≈ 210 total opposition articles
OPPOSITION_TARGET_PER_OUTLET = 70

TARGET_PER_LABEL    = 200   # per class - achievable via RSS even for small outlets
MIN_WORDS           = 100
GDELT_MONTHS_BACK   = 6
GDELT_SNAPSHOTS_PER_MONTH = 4

OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "india_training")
OUT_FILE = os.path.join(OUT_DIR, "india_bias_data.csv")


# -- GDELT helpers -------------------------------------------------------------

def _sample_dates() -> list[str]:
    """Generate dates spread evenly across the lookback window."""
    dates = []
    today = datetime.now(timezone.utc)
    offsets = [0, 8, 16, 24][:GDELT_SNAPSHOTS_PER_MONTH]

    for month_offset in range(GDELT_MONTHS_BACK, 0, -1):
        anchor = today - timedelta(days=month_offset * 30)
        for day_offset in offsets:
            dates.append((anchor + timedelta(days=day_offset)).strftime("%Y %b %d"))

    return dates


def _query_snapshot(date_str: str) -> pd.DataFrame | None:
    try:
        gd = gdelt.gdelt(version=2)
        df = gd.Search([date_str], table="gkg", coverage=False)
        if df is None or df.empty:
            return None
        pattern = "|".join(GDELT_NEUTRAL_DOMAINS.keys())
        mask = df["DocumentIdentifier"].str.contains(pattern, case=False, na=False)
        filtered = df[mask].copy()
        return filtered if not filtered.empty else None
    except Exception as e:
        print(f"  Warning: GDELT query failed for {date_str} - {e}")
        return None


def _detect_neutral_domain(url: str) -> str | None:
    for domain in GDELT_NEUTRAL_DOMAINS:
        if domain in url:
            return domain
    return None


# -- RSS helpers ---------------------------------------------------------------

def _collect_opposition_from_source(source: dict, target: int) -> list[dict]:
    """
    Scrape article links from a list of section pages for one outlet, then
    scrape full text. Works for any outlet whose section pages are accessible
    via requests (verify=False handles WSL SSL issues).
    """
    seen_urls: set[str] = set()
    rows: list[dict] = []

    for section_url in source["sections"]:
        if len(rows) >= target:
            break
        try:
            r = requests.get(section_url, headers=_HEADERS, timeout=10, verify=False)
            soup = BeautifulSoup(r.text, "html.parser")
            links = [
                a["href"] for a in soup.find_all("a", href=True)
                if source["domain"] in a["href"]
                and any(seg in a["href"] for seg in source["url_segs"])
                and a["href"] not in seen_urls
                and len(a["href"]) > 40
            ]
        except Exception as e:
            print(f"  Warning: could not load {section_url} - {e}")
            continue

        for url in links[:30]:
            if len(rows) >= target:
                break
            if url in seen_urls:
                continue
            seen_urls.add(url)

            body, _ = scrape_full_text(url)
            if not body or len(body.split()) < MIN_WORDS:
                continue

            rows.append({
                "text":           body,
                "label":          "opposition_aligned",
                "outlet":         source["outlet"],
                "url":            url,
                "published_date": "",
            })

    return rows


def _collect_from_rss(label: str, target: int) -> list[dict]:
    """Collect up to target articles for the given label via RSS feeds."""
    feeds = RSS_FEEDS_BY_LABEL.get(label, [])
    rows  = []

    for outlet_slug, feed_url in feeds:
        if len(rows) >= target:
            break

        print(f"  RSS [{outlet_slug}] ...")
        try:
            feed = feedparser.parse(feed_url)
        except Exception as e:
            print(f"    Failed to parse feed: {e}")
            continue

        entries = feed.entries[:60]  # cap per feed to avoid scraping forever

        for entry in tqdm(entries, desc=f"    Scraping {outlet_slug}", leave=False):
            if len(rows) >= target:
                break

            url = entry.get("link", "")
            if not url:
                continue

            # Try body from RSS content first, fall back to scraping
            body = ""
            if hasattr(entry, "content") and entry.content:
                raw = entry.content[0].get("value", "")
                body = re.sub(r"<[^>]+>", " ", raw).strip()

            if len(body.split()) < MIN_WORDS:
                body, _ = scrape_full_text(url)

            if not body or len(body.split()) < MIN_WORDS:
                continue

            published = ""
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat()

            rows.append({
                "text":           body,
                "label":          label,
                "outlet":         outlet_slug,
                "url":            url,
                "published_date": published,
            })

    return rows


# -- Main collection -----------------------------------------------------------

def collect() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    all_rows: list[dict] = []

    # ── Phase 1: GDELT for neutral ────────────────────────────────────────────
    print("=== Phase 1: GDELT neutral collection ===")
    dates = _sample_dates()
    print(f"Querying {len(dates)} snapshots over {GDELT_MONTHS_BACK} months...\n")

    neutral_count = 0

    for date_str in tqdm(dates, desc="GDELT snapshots"):
        if neutral_count >= TARGET_PER_LABEL:
            print("\nNeutral target reached - stopping GDELT early.")
            break

        df = _query_snapshot(date_str)
        if df is None:
            time.sleep(1)
            continue

        for _, row in df.iterrows():
            if neutral_count >= TARGET_PER_LABEL:
                break
            url    = row.get("DocumentIdentifier", "")
            domain = _detect_neutral_domain(url)
            if not domain:
                continue
            body, _ = scrape_full_text(url)
            if not body or len(body.split()) < MIN_WORDS:
                continue
            all_rows.append({
                "text":           body,
                "label":          "neutral",
                "outlet":         domain.split(".")[0],
                "url":            url,
                "published_date": str(row.get("DATE", "")),
            })
            neutral_count += 1

        time.sleep(1)

    print(f"\nNeutral collected: {neutral_count}")

    # ── Phase 2: RSS for bjp_aligned ─────────────────────────────────────────
    print("\n=== Phase 2: RSS bjp_aligned collection ===")
    bjp_rows = _collect_from_rss("bjp_aligned", TARGET_PER_LABEL)
    print(f"bjp_aligned collected: {len(bjp_rows)}")
    all_rows.extend(bjp_rows)

    # ── Phase 3: Scrape opposition outlets (theprint, thewire, scroll) ───────
    print("\n=== Phase 3: opposition_aligned collection (theprint + thewire + scroll) ===")
    opp_rows: list[dict] = []
    for source in OPPOSITION_SOURCES:
        print(f"  Scraping {source['outlet']}...")
        rows = _collect_opposition_from_source(source, OPPOSITION_TARGET_PER_OUTLET)
        print(f"    {source['outlet']}: {len(rows)} articles")
        opp_rows.extend(rows)
    print(f"opposition_aligned collected: {len(opp_rows)}")
    all_rows.extend(opp_rows)

    # ── Save and split ────────────────────────────────────────────────────────
    if not all_rows:
        print("\nNo articles collected.")
        return

    df_out = pd.DataFrame(all_rows).drop_duplicates(subset=["url"])
    df_out.to_csv(OUT_FILE, index=False)

    print(f"\nTotal saved: {len(df_out)}")
    print("\nLabel distribution:")
    print(df_out["label"].value_counts().to_string())

    min_class = df_out["label"].value_counts().min()
    stratify_col = df_out["label"] if min_class >= 10 else None
    if stratify_col is None:
        print(f"Warning: smallest class has {min_class} members - using non-stratified split.")

    train, tmp = train_test_split(df_out, test_size=0.2, stratify=stratify_col, random_state=42)
    tmp_min = tmp["label"].value_counts().min()
    stratify_tmp = tmp["label"] if tmp_min >= 2 else None
    val, test = train_test_split(tmp, test_size=0.5, stratify=stratify_tmp, random_state=42)

    train.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    val.to_csv(  os.path.join(OUT_DIR, "val.csv"),   index=False)
    test.to_csv( os.path.join(OUT_DIR, "test.csv"),  index=False)

    print(f"\nSplits: train={len(train)}, val={len(val)}, test={len(test)}")


if __name__ == "__main__":
    collect()
