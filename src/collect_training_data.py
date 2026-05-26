"""
Collects weakly-supervised training data for the Indian bias classifier.

Pulls articles from GDELT using known-leaning Indian outlet domains and assigns
outlet-level labels. This is distant supervision: the label comes from the outlet's
documented editorial alignment, not per-article expert annotation.

Outlet alignment sources: Reporters Without Borders India Press Freedom Index,
Committee to Protect Journalists, Media Ownership Monitor India.

Target: ~1200 articles per label.
Output: data/india_training/india_bias_data.csv  (columns: text, label, outlet, url, published_date)

Run from the project root:
    python src/collect_training_data.py
"""

import os
import time
from datetime import datetime, timedelta, timezone

import gdelt
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.scraper import scrape_full_text


# Maps GDELT domain → bias label (distant supervision)
OUTLET_LABELS = {
    "republicworld.com":              "bjp_aligned",
    "zeenews.india.com":              "bjp_aligned",
    "news18.com":                     "bjp_aligned",
    "opindia.com":                    "bjp_aligned",
    "thewire.in":                     "opposition_aligned",
    "scroll.in":                      "opposition_aligned",
    "newslaundry.com":                "opposition_aligned",
    "thehindu.com":                   "neutral",
    "indianexpress.com":              "neutral",
    "hindustantimes.com":             "neutral",
    "timesofindia.indiatimes.com":    "neutral",
}

TARGET_PER_LABEL   = 1200   # articles per label before balancing
MIN_WORDS          = 100    # articles shorter than this are too short to be useful
MONTHS_BACK        = 8
SNAPSHOTS_PER_MONTH = 3

OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "india_training")
OUT_FILE = os.path.join(OUT_DIR, "india_bias_data.csv")


def _sample_dates() -> list[str]:
    """Generate query dates spread evenly across the lookback window."""
    dates = []
    today = datetime.now(timezone.utc)

    for month_offset in range(MONTHS_BACK, 0, -1):
        anchor = today - timedelta(days=month_offset * 30)
        # Spread snapshots across different weeks of the month
        for day_offset in [0, 8, 18][:SNAPSHOTS_PER_MONTH]:
            target = anchor + timedelta(days=day_offset)
            dates.append(target.strftime("%Y %b %d"))

    return dates


def _query_snapshot(date_str: str) -> pd.DataFrame | None:
    """Query one GDELT GKG snapshot, filtered to our training outlet domains."""
    try:
        gd = gdelt.gdelt(version=2)
        df = gd.Search([date_str], table="gkg", coverage=False)

        if df is None or df.empty:
            return None

        domain_pattern = "|".join(OUTLET_LABELS.keys())
        mask = df["DocumentIdentifier"].str.contains(domain_pattern, case=False, na=False)
        filtered = df[mask].copy()

        return filtered if not filtered.empty else None

    except Exception as e:
        print(f"  Warning: GDELT query failed for {date_str} - {e}")
        return None


def _detect_domain(url: str) -> str | None:
    """Return the matching domain key if the URL contains a known training domain."""
    for domain in OUTLET_LABELS:
        if domain in url:
            return domain
    return None


def collect() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    dates = _sample_dates()
    print(f"Querying {len(dates)} GDELT snapshots over {MONTHS_BACK} months...")
    print(f"Target: {TARGET_PER_LABEL} articles per label\n")

    label_counts = {label: 0 for label in set(OUTLET_LABELS.values())}
    rows = []

    for date_str in tqdm(dates, desc="GDELT snapshots"):
        # Stop early if all labels have hit their target
        if all(count >= TARGET_PER_LABEL for count in label_counts.values()):
            print("\nTarget reached for all labels - stopping early.")
            break

        df = _query_snapshot(date_str)
        if df is None:
            time.sleep(1)
            continue

        for _, gdelt_row in df.iterrows():
            url    = gdelt_row.get("DocumentIdentifier", "")
            domain = _detect_domain(url)
            if not domain:
                continue

            label = OUTLET_LABELS[domain]

            # Skip if we already have enough articles for this label
            if label_counts[label] >= TARGET_PER_LABEL:
                continue

            body, _ = scrape_full_text(url)
            if not body or len(body.split()) < MIN_WORDS:
                continue

            rows.append({
                "text":           body,
                "label":          label,
                "outlet":         domain.split(".")[0],
                "url":            url,
                "published_date": str(gdelt_row.get("DATE", "")),
            })
            label_counts[label] += 1

        # Brief pause to avoid hammering GDELT servers
        time.sleep(1)

    print(f"\nCollection complete. Counts per label: {label_counts}")

    if not rows:
        print("No articles collected. Check GDELT connectivity and try again.")
        return

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_FILE, index=False)
    print(f"Saved {len(df_out)} articles to {OUT_FILE}")

    print("\nLabel distribution:")
    print(df_out["label"].value_counts().to_string())

    # Create train/val/test splits (80/10/10, stratified by label)
    train, tmp  = train_test_split(df_out, test_size=0.2,  stratify=df_out["label"], random_state=42)
    val,   test = train_test_split(tmp,    test_size=0.5,  stratify=tmp["label"],    random_state=42)

    train.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    val.to_csv(  os.path.join(OUT_DIR, "val.csv"),   index=False)
    test.to_csv( os.path.join(OUT_DIR, "test.csv"),  index=False)

    print(f"\nSplits saved to {OUT_DIR}/")
    print(f"  train: {len(train)}, val: {len(val)}, test: {len(test)}")


if __name__ == "__main__":
    collect()
