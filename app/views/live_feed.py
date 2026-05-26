import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
from src.db import get_connection, get_articles, get_distinct_outlets, get_distinct_topics

# Streamlit's built-in color names for bias labels
BIAS_COLOR = {"left": "blue", "centre": "gray", "right": "red"}


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
        bias_sel = st.multiselect("Bias", ["left", "centre", "right"], placeholder="All")
    with c4:
        source_sel = st.multiselect("Source", ["newsapi", "rss", "gdelt"], placeholder="All")

    articles = _load_articles(
        None if topic_sel == "All topics" else topic_sel,
        tuple(outlet_sel),
        tuple(bias_sel),
        tuple(source_sel),
    )

    # ── Summary row ───────────────────────────────────────────────────────────
    total     = len(articles)
    trusted   = sum(1 for a in articles if a.get("bias_trusted"))
    uncertain = total - trusted

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Articles", total)
    m2.metric("Trusted predictions", trusted)
    m3.metric("Uncertain", uncertain)
    st.divider()

    if not articles:
        st.info("No articles match these filters.")
        return

    # ── Article cards ─────────────────────────────────────────────────────────
    for article in articles:
        headline   = article.get("headline") or "No headline"
        outlet     = (article.get("outlet") or "unknown").upper()
        published  = (article.get("published_at") or "")[:10]
        source     = (article.get("source") or "").upper()
        label      = article.get("bias_label") or ""
        trusted_fl = article.get("bias_trusted") or 0
        confidence = article.get("bias_confidence") or 0.0

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
