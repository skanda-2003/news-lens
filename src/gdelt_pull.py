"""
GDELT historical data pull for drift monitoring.

GDELT archives news from thousands of outlets worldwide. I use it to get
historical articles spread over 6 months for Indian outlets - far more than
NewsAPI's one-month free-tier limit.

Each GDELT GKG snapshot covers one 15-minute window. I query 3 snapshots per
month spread across different weeks, giving 18 total queries for a 6-month window.
"""

import os
import time
from datetime import datetime, timedelta, timezone

import gdelt
import pandas as pd
from tqdm import tqdm

from src.db import get_connection, insert_article
from src.scraper import scrape_full_text

# Where to save the raw GDELT CSVs before inserting into SQLite
GDELT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "gdelt")

# Indian outlet domains to filter GDELT results to
TARGET_DOMAINS = {
    "thehindu.com":                  "the_hindu",
    "ndtv.com":                      "ndtv",
    "timesofindia.indiatimes.com":   "times_of_india",
    "thewire.in":                    "the_wire",
    "hindustantimes.com":            "hindustan_times",
    "indianexpress.com":             "indian_express",
    "scroll.in":                     "scroll",
    "republicworld.com":             "republic_world",
    "zeenews.india.com":             "zee_news",
    "news18.com":                    "news18",
    "indiatoday.in":                 "india_today",
}


# -- Date sampling -------------------------------------------------------------

def _sample_dates(months_back: int = 6, snapshots_per_month: int = 3) -> list[str]:
    """
    Generate a list of dates to query GDELT for, spread evenly across the lookback window.

    Returns dates formatted as 'YYYY Mon DD' which is what the gdelt package expects.
    snapshots_per_month controls how many data points per month - more = more articles
    but also more download time.
    """
    dates = []
    today = datetime.now(timezone.utc)

    for month_offset in range(months_back, 0, -1):
        # Anchor to roughly the same point each month
        anchor = today - timedelta(days=month_offset * 30)

        # Spread snapshots across three different weeks of the month
        offsets = [0, 8, 18][:snapshots_per_month]

        for day_offset in offsets:
            target = anchor + timedelta(days=day_offset)
            # gdelt package expects format like '2025 Nov 15'
            dates.append(target.strftime("%Y %b %d"))

    return dates


# -- GDELT query ---------------------------------------------------------------

def _query_gdelt_snapshot(date_str: str) -> pd.DataFrame | None:
    """
    Query GDELT GKG for one snapshot on a given date, filtered to our target outlets.

    Returns a filtered DataFrame or None if the query fails.
    """
    try:
        gd = gdelt.gdelt(version=2)
        # coverage=False fetches just one 15-minute snapshot instead of the full day
        df = gd.Search([date_str], table="gkg", coverage=False)

        if df is None or df.empty:
            return None

        # Filter to only rows where the article URL contains one of our target domains
        domain_pattern = "|".join(TARGET_DOMAINS.keys())
        mask = df["DocumentIdentifier"].str.contains(domain_pattern, case=False, na=False)
        filtered = df[mask].copy()

        return filtered if not filtered.empty else None

    except Exception as e:
        print(f"  Warning: GDELT query failed for {date_str} - {e}")
        return None


def _detect_outlet(url: str) -> str:
    """Match a URL against known domains and return the outlet slug."""
    for domain, slug in TARGET_DOMAINS.items():
        if domain in url:
            return slug
    return "unknown"


# -- Main pull -----------------------------------------------------------------

def pull_gdelt_historical(months_back: int = 6, snapshots_per_month: int = 3) -> str:
    """
    Pull historical GDELT articles for our target outlets and save to CSV.

    Returns the path to the saved CSV file.
    """
    os.makedirs(GDELT_DATA_DIR, exist_ok=True)

    dates = _sample_dates(months_back, snapshots_per_month)
    print(f"Querying {len(dates)} GDELT snapshots across {months_back} months...")

    all_rows = []

    for date_str in tqdm(dates, desc="GDELT queries"):
        df = _query_gdelt_snapshot(date_str)

        if df is not None:
            # Extract just the columns I need for ingestion
            for _, row in df.iterrows():
                url = row.get("DocumentIdentifier", "")
                outlet = _detect_outlet(url)
                # V2Tone is a comma-separated string - the first value is the overall tone score
                tone_raw = str(row.get("V2Tone", ""))
                tone = tone_raw.split(",")[0] if tone_raw else ""

                all_rows.append({
                    "url":          url,
                    "outlet":       outlet,
                    "headline":     str(row.get("SourceCommonName", "")),
                    "published_at": str(row.get("DATE", "")),
                    "gdelt_tone":   tone,
                    "query_date":   date_str,
                })

        # Brief pause between queries to avoid hammering the GDELT servers
        time.sleep(1)

    if not all_rows:
        print("No articles found matching target outlets.")
        return ""

    result_df = pd.DataFrame(all_rows).drop_duplicates(subset=["url"])
    print(f"\nTotal unique articles found: {len(result_df)}")
    print("Outlet breakdown:")
    print(result_df["outlet"].value_counts().to_string())

    # Save raw CSV before inserting into SQLite
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(GDELT_DATA_DIR, f"gdelt_historical_{timestamp}.csv")
    result_df.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")

    return csv_path


# -- SQLite ingestion ----------------------------------------------------------

def ingest_gdelt_csv(csv_path: str, scrape: bool = True) -> None:
    """
    Read a GDELT CSV from data/gdelt/ and insert the articles into SQLite.

    If scrape=True, fetches full body text for each article via newspaper3k.
    Set scrape=False to do a fast insert with empty body - useful for a quick
    check that the pipeline works before committing to a long scraping run.
    """
    df = pd.read_csv(csv_path)
    conn = get_connection()

    inserted = 0
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Inserting GDELT articles"):
        url = row["url"]
        body = ""
        body_source = "summary_only"

        if scrape:
            body, body_source = scrape_full_text(url)

        success = insert_article(conn, {
            "url":          url,
            "outlet":       row["outlet"],
            # GDELT doesn't provide headlines - use the source domain name as a placeholder
            "headline":     row.get("headline", ""),
            "body":         body,
            "body_source":  body_source,
            "published_at": str(row.get("published_at", "")),
            "topic":        "",
            "source":       "gdelt",
            "ingested_at":  datetime.now(timezone.utc).isoformat(),
        })

        inserted += int(success)
        skipped += int(not success)

    print(f"Inserted: {inserted} | Skipped (duplicates): {skipped}")
