import sys
import os
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import plotly.graph_objects as go

from src.db import get_connection, get_articles, get_distinct_outlets, get_distinct_topics

BIAS_COLOR = {"bjp_aligned": "orange", "opposition_aligned": "blue", "neutral": "gray"}

PAGE_SIZE = 20


@st.cache_resource
def _conn():
    return get_connection()


@st.cache_data(ttl=300)
def _load_outlets():
    return get_distinct_outlets(_conn())


@st.cache_data(ttl=300)
def _load_topics():
    return get_distinct_topics(_conn())


@st.cache_data(ttl=300)
def _load_articles(topic, outlets_tuple, bias_tuple, source_tuple):
    # Tuples as args so Streamlit can cache correctly (lists aren't hashable)
    articles = get_articles(_conn(), topic=topic or None)
    if outlets_tuple:
        articles = [a for a in articles if a["outlet"] in outlets_tuple]
    if bias_tuple:
        articles = [a for a in articles if a.get("bias_label") in bias_tuple]
    if source_tuple:
        articles = [a for a in articles if a.get("source") in source_tuple]
    return articles


def _parse_date(raw: str) -> datetime.date | None:
    if not raw:
        return None
    s = str(raw)[:10]
    try:
        return datetime.date.fromisoformat(s)
    except ValueError:
        return None


def page_live_feed():
    st.header("Live Feed")
    st.caption("All ingested articles. Filter by topic, outlet, bias label, or source.")

    outlets = _load_outlets()
    topics  = _load_topics()

    # ── Filters ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        topic_sel = st.selectbox("Topic", ["All topics"] + topics)
    with c2:
        outlet_sel = st.multiselect("Outlet", outlets, placeholder="All outlets")
    with c3:
        bias_sel = st.multiselect(
            "Bias", ["bjp_aligned", "opposition_aligned", "neutral"], placeholder="All"
        )
    with c4:
        source_sel = st.multiselect("Source", ["newsapi", "rss", "gdelt"], placeholder="All")

    articles = _load_articles(
        None if topic_sel == "All topics" else topic_sel,
        tuple(outlet_sel),
        tuple(bias_sel),
        tuple(source_sel),
    )

    # ── Date range slider ─────────────────────────────────────────────────────
    # GDELT 14-digit timestamps (20260403234500) fail fromisoformat - silently skip those
    dates = [d for a in articles if (d := _parse_date(a.get("published_at"))) is not None]

    if dates and min(dates) < max(dates):
        date_range = st.slider(
            "Date range",
            min_value=min(dates),
            max_value=max(dates),
            value=(min(dates), max(dates)),
            format="YYYY-MM-DD",
        )
        articles = [
            a for a in articles
            if (d := _parse_date(a.get("published_at"))) is not None
            and date_range[0] <= d <= date_range[1]
        ]

    # ── Summary row ───────────────────────────────────────────────────────────
    total     = len(articles)
    trusted   = sum(1 for a in articles if a.get("bias_trusted"))
    uncertain = total - trusted

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Articles", total)
    m2.metric("Trusted predictions", trusted)
    m3.metric("Uncertain", uncertain)

    # ── Bias distribution bar ─────────────────────────────────────────────────
    if articles:
        counts = {
            "bjp_aligned":       sum(1 for a in articles if a.get("bias_label") == "bjp_aligned"),
            "opposition_aligned": sum(1 for a in articles if a.get("bias_label") == "opposition_aligned"),
            "neutral":            sum(1 for a in articles if a.get("bias_label") == "neutral"),
        }
        total_n = sum(counts.values())
        if total_n > 0:
            colors = {"bjp_aligned": "#FF6B00", "opposition_aligned": "#2563EB", "neutral": "#6B7280"}
            fig = go.Figure()
            for label, count in counts.items():
                fig.add_trace(go.Bar(
                    x=[count], y=["Bias distribution"], orientation="h",
                    name=label, marker_color=colors[label],
                    text=[f"{label}: {count} ({count/total_n*100:.0f}%)"],
                    textposition="inside", insidetextanchor="middle",
                ))
            fig.update_layout(
                barmode="stack", height=60,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="white", plot_bgcolor="white",
                showlegend=False,
                xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                yaxis=dict(showgrid=False, showticklabels=False),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    if not articles:
        st.info("No articles match these filters.")
        return

    # ── Pagination ────────────────────────────────────────────────────────────
    total_pages = max(1, (len(articles) + PAGE_SIZE - 1) // PAGE_SIZE)

    # Reset to page 0 when filters change so I don't land on a nonexistent page
    filter_key = (topic_sel, tuple(outlet_sel), tuple(bias_sel), tuple(source_sel))
    if st.session_state.get("feed_filter_key") != filter_key:
        st.session_state.feed_page = 0
        st.session_state.feed_filter_key = filter_key

    page = st.session_state.get("feed_page", 0)
    page_articles = articles[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]

    col_prev, col_info, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("← Prev", disabled=(page == 0)):
            st.session_state.feed_page -= 1
            st.rerun()
    with col_info:
        st.caption(f"Page {page + 1} of {total_pages}  ({len(articles)} articles total)")
    with col_next:
        if st.button("Next →", disabled=(page >= total_pages - 1)):
            st.session_state.feed_page += 1
            st.rerun()

    # ── Article cards ─────────────────────────────────────────────────────────
    for article in page_articles:
        headline   = article.get("headline") or "No headline"
        outlet     = (article.get("outlet") or "unknown").upper()
        published  = (article.get("published_at") or "")[:10]
        source     = (article.get("source") or "").upper()
        label      = article.get("bias_label") or ""
        trusted_fl = article.get("bias_trusted") or 0
        confidence = article.get("bias_confidence") or 0.0
        body       = article.get("body") or ""

        with st.container(border=True):
            left_col, right_col = st.columns([6, 1])
            with left_col:
                st.markdown(f"**{headline}**")
                st.caption(f"{outlet}  ·  {published}  ·  {source}")
            with right_col:
                if label and trusted_fl:
                    color = BIAS_COLOR.get(label, "gray")
                    st.markdown(f":{color}[**{label.upper()}**]")
                else:
                    st.markdown(":gray[**?**]")
                st.caption(f"{confidence:.2f}")
            if body:
                with st.expander("Preview"):
                    st.caption(body[:150] + ("..." if len(body) > 150 else ""))
