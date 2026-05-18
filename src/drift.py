"""
Bias drift calculation module.

Reads classified articles from SQLite, groups them by outlet and calendar month,
and detects shifts in the bias label distribution over time.

Design note: the original plan called for 7-day rolling averages. The GDELT historical
data averages 2-5 articles per outlet per month, so weekly rolling windows would be
mostly empty. Monthly aggregation is the correct granularity for this dataset.
"""

import os
import sqlite3
from datetime import datetime

import pandas as pd

# Anchor path to this file's location so it works from any working directory
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "newslens.db")

# The four outlets that have GDELT historical data - only these can be shown on a time axis
DRIFT_OUTLETS = ["bbc", "npr", "fox_news", "the_guardian"]


def parse_date(date_str: str) -> datetime | None:
    """
    Parse a date string into a naive datetime object.

    Handles two formats found in the DB:
    - GDELT format:  20251124234500  (14 digits, YYYYMMDDHHMMSS)
    - ISO 8601:      2026-05-15T07:38:17+00:00  (with or without timezone)
    """
    if not date_str:
        return None

    s = str(date_str).strip()

    # GDELT format - 14 digit string like 20251124234500
    if len(s) == 14 and s.isdigit():
        try:
            return datetime.strptime(s, "%Y%m%d%H%M%S")
        except ValueError:
            return None

    # ISO 8601 - slice off timezone suffix before parsing
    # "2026-05-15T07:38:17+00:00" -> "2026-05-15T07:38:17"
    # "2026-05-13T14:32:36Z"      -> "2026-05-13T14:32:36"
    try:
        return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        pass

    return None


def load_drift_data(outlets: list[str] = DRIFT_OUTLETS) -> pd.DataFrame:
    """
    Load all classified articles for the given outlets from SQLite.

    Returns a DataFrame with columns:
      outlet, published_at, bias_label, bias_confidence, source, date, month

    Rows with unparseable dates or missing bias_label are dropped.
    Articles from before 2025 are excluded - the one 2017 BBC article in the DB
    is an RSS feed artefact, not representative historical data.
    """
    conn = sqlite3.connect(DB_PATH)

    placeholders = ",".join("?" for _ in outlets)
    rows = conn.execute(f"""
        SELECT outlet, published_at, bias_label, bias_confidence, source
        FROM articles
        WHERE outlet IN ({placeholders})
          AND bias_label IS NOT NULL
    """, outlets).fetchall()
    conn.close()

    df = pd.DataFrame(rows, columns=["outlet", "published_at", "bias_label", "bias_confidence", "source"])

    # Parse dates into datetime objects - drop rows where parsing fails
    df["date"] = df["published_at"].apply(parse_date)
    df = df.dropna(subset=["date"])

    # Exclude articles before November 2025 - the start of the GDELT historical window.
    # A few isolated BBC RSS articles from Jan/Apr 2025 exist in the DB but are not
    # representative historical data; using them as a baseline produces misleading results.
    df = df[df["date"] >= datetime(2025, 11, 1)]

    # Add a year-month label for grouping: "2025-11", "2026-05", etc.
    df["month"] = df["date"].apply(lambda d: d.strftime("%Y-%m"))

    return df.reset_index(drop=True)


def monthly_bias_proportions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group articles by outlet and month, then compute the proportion of each bias label.

    Returns a DataFrame with columns:
      outlet, month, n_articles, left_pct, centre_pct, right_pct

    This is the core data structure for the drift time series charts.
    """
    rows = []

    for (outlet, month), group in df.groupby(["outlet", "month"]):
        n = len(group)
        left_pct   = (group["bias_label"] == "left").sum()   / n * 100
        centre_pct = (group["bias_label"] == "centre").sum() / n * 100
        right_pct  = (group["bias_label"] == "right").sum()  / n * 100

        rows.append({
            "outlet":     outlet,
            "month":      month,
            "n_articles": n,
            "left_pct":   round(left_pct,   1),
            "centre_pct": round(centre_pct, 1),
            "right_pct":  round(right_pct,  1),
        })

    result = pd.DataFrame(rows).sort_values(["outlet", "month"]).reset_index(drop=True)
    return result


def compute_baselines(monthly_df: pd.DataFrame, n_months: int = 2) -> dict:
    """
    Compute baseline bias proportions for each outlet using its first n_months of data.

    The baseline represents the outlet's "normal" bias distribution during the earliest
    months of the historical window. Later months are compared against this baseline to
    detect drift.

    Returns a dict keyed by outlet name. Each value is a dict with keys:
      left_mean, centre_mean, right_mean  - mean proportions during baseline period
      n_baseline_months                   - how many months were used
      baseline_months                     - list of month strings used
    """
    baselines = {}

    for outlet in monthly_df["outlet"].unique():
        outlet_df = monthly_df[monthly_df["outlet"] == outlet].sort_values("month")

        # Use the first n_months as the baseline window
        baseline_rows = outlet_df.head(n_months)

        if len(baseline_rows) == 0:
            continue

        baselines[outlet] = {
            "left_mean":          baseline_rows["left_pct"].mean(),
            "centre_mean":        baseline_rows["centre_pct"].mean(),
            "right_mean":         baseline_rows["right_pct"].mean(),
            "n_baseline_months":  len(baseline_rows),
            "baseline_months":    baseline_rows["month"].tolist(),
        }

    return baselines


def detect_drift_events(
    monthly_df: pd.DataFrame,
    baselines: dict,
    threshold_pct: float = 20.0,
    min_articles: int = 5,
) -> pd.DataFrame:
    """
    Flag months where a label's proportion deviates from baseline by more than threshold_pct
    percentage points.

    A threshold of 20pp is used rather than a standard-deviation-based threshold because
    monthly sample sizes are too small (1-7 articles/month) for SD to be meaningful.
    A 20pp absolute shift is a large and visible change in the label distribution.

    min_articles filters out sparse months - a single article flipping label looks like
    drift but is just noise. Only months with at least min_articles are evaluated.

    Returns a DataFrame of drift events with columns:
      outlet, month, label, baseline_pct, observed_pct, deviation
    """
    events = []

    for outlet, baseline in baselines.items():
        outlet_df = monthly_df[monthly_df["outlet"] == outlet].sort_values("month")

        # Only check months after the baseline window
        n_base = baseline["n_baseline_months"]
        post_baseline = outlet_df.iloc[n_base:]

        for _, row in post_baseline.iterrows():
            # Skip months with too few articles to draw conclusions from
            if row["n_articles"] < min_articles:
                continue
            for label in ["left", "centre", "right"]:
                observed  = row[f"{label}_pct"]
                base_mean = baseline[f"{label}_mean"]
                deviation = abs(observed - base_mean)

                if deviation >= threshold_pct:
                    events.append({
                        "outlet":       outlet,
                        "month":        row["month"],
                        "label":        label,
                        "baseline_pct": round(base_mean, 1),
                        "observed_pct": round(observed, 1),
                        "deviation":    round(deviation, 1),
                    })

    if not events:
        return pd.DataFrame(columns=["outlet", "month", "label", "baseline_pct", "observed_pct", "deviation"])

    return pd.DataFrame(events).sort_values(["outlet", "month"]).reset_index(drop=True)
