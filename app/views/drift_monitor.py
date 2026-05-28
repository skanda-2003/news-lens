import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.drift import (
    DRIFT_OUTLETS,
    load_drift_data,
    monthly_bias_proportions,
    compute_baselines,
    detect_drift_events,
)

BIAS_COLORS = {
    "bjp_aligned":        "#FF6B00",
    "opposition_aligned": "#2563EB",
    "neutral":            "#6B7280",
}
DRIFT_COLOR = "#FF4D00"

LABEL_DISPLAY = {
    "bjp_aligned":        "BJP-aligned",
    "opposition_aligned": "Opposition-aligned",
    "neutral":            "Neutral",
}


@st.cache_data(ttl=3600, show_spinner="Loading drift data...")
def _load():
    df      = load_drift_data(DRIFT_OUTLETS)
    monthly = monthly_bias_proportions(df)
    bases   = compute_baselines(monthly)
    events  = detect_drift_events(monthly, bases)
    return monthly, bases, events


def _chart(outlet: str, monthly: pd.DataFrame, baselines: dict, events: pd.DataFrame):
    outlet_df = monthly[monthly["outlet"] == outlet].sort_values("month")
    if outlet_df.empty:
        return None

    months = outlet_df["month"].tolist()
    fig    = go.Figure()

    for label in ["bjp_aligned", "opposition_aligned", "neutral"]:
        fig.add_trace(go.Scatter(
            x=months,
            y=outlet_df[f"{label}_pct"],
            mode="lines+markers",
            name=LABEL_DISPLAY[label],
            line=dict(color=BIAS_COLORS[label], width=2),
            marker=dict(size=6),
        ))

    fig.add_trace(go.Bar(
        x=months,
        y=outlet_df["n_articles"],
        name="Articles",
        marker_color="#E5E7EB",
        opacity=0.5,
        yaxis="y2",
    ))

    # Baseline bands: shaded region showing the +-20pp threshold window
    base = baselines.get(outlet, {})
    for label in ["bjp_aligned", "opposition_aligned", "neutral"]:
        mean_val = base.get(f"{label}_mean", 0)
        fig.add_hrect(
            y0=max(0, mean_val - 20),
            y1=min(100, mean_val + 20),
            fillcolor=BIAS_COLORS[label],
            opacity=0.05,
            line_width=0,
        )

    outlet_events = events[events["outlet"] == outlet] if not events.empty else pd.DataFrame()
    for month in outlet_events["month"].unique():
        fig.add_vline(
            x=month,
            line_color=DRIFT_COLOR,
            line_dash="dash",
            line_width=1.5,
        )

    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=16, b=0),
        paper_bgcolor="white",
        plot_bgcolor="#F9FAFB",
        yaxis=dict(title="Bias %", range=[0, 100], gridcolor="#E5E7EB"),
        yaxis2=dict(
            title="Articles",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        xaxis=dict(gridcolor="#E5E7EB"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def page_drift_monitor():
    st.header("Drift Monitor")
    st.caption(
        "Tracks whether an outlet's political lean has shifted over time. "
        "Shaded bands show the +-20pp baseline threshold. Orange dashed lines mark drift events."
    )

    monthly, baselines, events = _load()

    if monthly.empty:
        st.warning("No drift data available. Run the GDELT ingestion pipeline first.")
        return

    # ── All-outlets overview table ────────────────────────────────────────────
    st.subheader("All outlets overview")
    summary_rows = []
    for o in sorted(monthly["outlet"].unique()):
        base    = baselines.get(o, {})
        o_df    = monthly[monthly["outlet"] == o].sort_values("month")
        current = o_df["bjp_aligned_pct"].iloc[-1] if not o_df.empty else 0
        n_events = int((events["outlet"] == o).sum()) if not events.empty else 0
        summary_rows.append({
            "Outlet":                 o,
            "Baseline BJP-aligned %": f"{base.get('bjp_aligned_mean', 0):.0f}%",
            "Current BJP-aligned %":  f"{current:.0f}%",
            "Drift events":           n_events,
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    st.divider()

    # ── Per-outlet detail ─────────────────────────────────────────────────────
    available = sorted(monthly["outlet"].unique().tolist())
    outlet    = st.selectbox("Select outlet", available)

    outlet_df = monthly[monthly["outlet"] == outlet]
    if len(outlet_df) < 3:
        st.warning(
            "Not enough data to compute a meaningful baseline for this outlet. "
            "Need at least 3 months of articles."
        )

    # ── Chart ─────────────────────────────────────────────────────────────────
    fig = _chart(outlet, monthly, baselines, events)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # Baseline info
    base = baselines.get(outlet)
    if base:
        st.caption(
            f"Baseline months: {', '.join(base['baseline_months'])}  ·  "
            f"Threshold: +-20 percentage points  ·  Min 5 articles/month to qualify"
        )

    st.divider()

    # ── Drift events table ────────────────────────────────────────────────────
    st.subheader("Drift events")

    outlet_events = (
        events[events["outlet"] == outlet].copy()
        if not events.empty else pd.DataFrame()
    )

    if outlet_events.empty:
        st.info("No drift events detected for this outlet.")
    else:
        st.dataframe(
            outlet_events[["month", "label", "baseline_pct", "observed_pct", "deviation"]],
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.caption(
        "Methodology: baseline = first 2 months of data per outlet. "
        "Drift = absolute deviation >= 20pp from baseline. "
        "Historical data sourced from GDELT."
    )
