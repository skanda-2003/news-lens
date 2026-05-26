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

st.set_page_config(
    page_title="NewsLens",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.title("NewsLens")
    st.caption("Media bias & narrative detection")
    st.divider()
    page = st.radio(
        "Navigate",
        ["Live Feed", "Narrative Map", "Framing Explorer", "Drift Monitor"],
        label_visibility="collapsed",
    )

# ── Page router ───────────────────────────────────────────────────────────────
PAGES = {
    "Live Feed":        page_live_feed,
    "Narrative Map":    page_narrative_map,
    "Framing Explorer": page_framing_explorer,
    "Drift Monitor":    page_drift_monitor,
}

PAGES[page]()
