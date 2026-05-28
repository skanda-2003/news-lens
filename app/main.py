"""
NewsLens dashboard entry point.
Run with: streamlit run app/main.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

from app.views.live_feed        import page_live_feed
from app.views.narrative_map    import page_narrative_map
from app.views.framing_explorer import page_framing_explorer
from app.views.drift_monitor    import page_drift_monitor
from app.views.synthesis        import page_synthesis
from app.views.about            import page_about
from src.db import get_connection, get_article_count, get_distinct_outlets

st.set_page_config(
    page_title="NewsLens",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar navigation ---
@st.cache_data(ttl=3600)
def _sidebar_stats():
    conn = get_connection()
    return get_article_count(conn), len(get_distinct_outlets(conn))


with st.sidebar:
    st.title("NewsLens")
    st.caption("Media bias & narrative detection")
    st.divider()
    page = st.radio(
        "Navigate",
        ["Live Feed", "Narrative Map", "Framing Explorer", "Drift Monitor", "Synthesis", "About"],
        label_visibility="collapsed",
    )
    st.divider()
    n_articles, n_outlets = _sidebar_stats()
    st.caption(f"{n_articles} articles  ·  {n_outlets} outlets")
    st.caption("Topics: India-Pakistan, BJP, economy, China, Kashmir, communal, democracy, NEET")
    st.divider()
    st.markdown("[GitHub](https://github.com/skanda-2003/news-lens)")

# --- Page router ---
PAGES = {
    "Live Feed":        page_live_feed,
    "Narrative Map":    page_narrative_map,
    "Framing Explorer": page_framing_explorer,
    "Drift Monitor":    page_drift_monitor,
    "Synthesis":        page_synthesis,
    "About":            page_about,
}

PAGES[page]()
