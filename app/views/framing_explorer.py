import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from collections import Counter, defaultdict

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.db import get_connection, get_distinct_topics, get_framing_articles

ACCENT = "#FF6B00"


@st.cache_resource
def _conn():
    return get_connection()


@st.cache_data(ttl=300)
def _load_topics():
    return get_distinct_topics(_conn())


@st.cache_data(ttl=3600)
def _load_framing(topic):
    return get_framing_articles(_conn(), topic=topic or None)


def _top_phrases(articles: list, field: str, n: int = 10) -> list[tuple[str, int]]:
    counter: Counter = Counter()
    for a in articles:
        phrase = a.get(field)
        # Skip nulls and single-word outputs (Ollama sometimes produces these)
        if phrase and len(phrase.strip().split()) >= 2:
            counter[phrase.strip().lower()] += 1
    return counter.most_common(n)


def _bar_chart(phrases: list[tuple[str, int]], title: str):
    if not phrases:
        return None
    labels = [p[0][:45] + ("..." if len(p[0]) > 45 else "") for p in phrases]
    counts = [p[1] for p in phrases]
    fig = go.Figure(go.Bar(
        x=counts,
        y=labels,
        orientation="h",
        marker_color=ACCENT,
        text=counts,
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        height=320,
        margin=dict(l=0, r=40, t=36, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
    )
    return fig


def _coverage_table(articles: list) -> pd.DataFrame:
    stats = defaultdict(lambda: {"total": 0, "villain": 0, "victim": 0, "solution": 0})
    for a in articles:
        outlet = a.get("outlet") or "unknown"
        stats[outlet]["total"] += 1
        if a.get("framing_villain"):
            stats[outlet]["villain"] += 1
        if a.get("framing_victim"):
            stats[outlet]["victim"] += 1
        if a.get("framing_solution"):
            stats[outlet]["solution"] += 1
    rows = []
    for outlet, s in sorted(stats.items()):
        t = s["total"] or 1
        rows.append({
            "Outlet":        outlet,
            "Articles":      s["total"],
            "Villain fill":  f"{s['villain']/t*100:.0f}%",
            "Victim fill":   f"{s['victim']/t*100:.0f}%",
            "Solution fill": f"{s['solution']/t*100:.0f}%",
        })
    return pd.DataFrame(rows)


def _outlet_top_phrase(articles: list, field: str) -> dict[str, tuple[str, int]]:
    """Returns each outlet's single most common non-null phrase (>= 2 words) for the field."""
    outlet_phrases: dict[str, Counter] = defaultdict(Counter)
    for a in articles:
        phrase = a.get(field)
        outlet = a.get("outlet") or "unknown"
        if phrase and len(phrase.strip().split()) >= 2:
            outlet_phrases[outlet][phrase.strip().lower()] += 1
    return {
        outlet: counter.most_common(1)[0]
        for outlet, counter in outlet_phrases.items()
        if counter
    }


def _outlet_bar_chart(outlet_data: dict, title: str):
    """Horizontal bar: y=outlet, x=count, text=top phrase. Uses indigo to distinguish from global charts."""
    if not outlet_data:
        return None
    sorted_items = sorted(outlet_data.items(), key=lambda x: x[1][1], reverse=True)
    outlets = [item[0] for item in sorted_items]
    counts  = [item[1][1] for item in sorted_items]
    phrases = [item[1][0][:45] + ("..." if len(item[1][0]) > 45 else "") for item in sorted_items]
    fig = go.Figure(go.Bar(
        x=counts, y=outlets, orientation="h",
        text=phrases, textposition="outside",
        marker_color="#6366F1",
        hovertemplate="<b>%{y}</b><br>Phrase: %{text}<br>Count: %{x}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        height=max(200, len(outlets) * 35 + 60),
        margin=dict(l=0, r=180, t=36, b=0),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(tickfont=dict(size=11)),
    )
    return fig


def page_framing_explorer():
    st.header("Framing Explorer")
    st.caption(
        "Which outlets assign which villain, victim, and solution to the same event? "
        "Based on Entman's (1993) framing theory."
    )

    st.info(
        "Framing values are extracted by Llama 3.2 3B and are not human-verified. "
        "Treat them as indicative, not definitive. "
        "Solution fill rates are typically low (~20%) - news rarely proposes explicit fixes.",
        icon="ℹ️",
    )

    topics = _load_topics()
    if not topics:
        st.warning("No topics found in the database.")
        return

    topic_sel = st.selectbox("Select topic", ["All topics"] + topics)
    topic_filter = None if topic_sel == "All topics" else topic_sel
    articles = _load_framing(topic_filter)

    if not articles:
        st.info("No framing data found for this topic.")
        return

    # ── Outlet filter ─────────────────────────────────────────────────────────
    all_outlets = sorted({a["outlet"] for a in articles if a.get("outlet")})
    outlet_filter = st.multiselect("Filter outlets", all_outlets, placeholder="All outlets")
    if outlet_filter:
        articles = [a for a in articles if a.get("outlet") in outlet_filter]

    st.caption(f"{len(articles)} articles with framing data")
    st.divider()

    # ── Three global phrase charts ────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Villain")
        phrases = _top_phrases(articles, "framing_villain")
        if phrases:
            st.plotly_chart(_bar_chart(phrases, "Top villains"), use_container_width=True)
        else:
            st.info("No villain data.")

    with c2:
        st.subheader("Victim")
        phrases = _top_phrases(articles, "framing_victim")
        if phrases:
            st.plotly_chart(_bar_chart(phrases, "Top victims"), use_container_width=True)
        else:
            st.info("No victim data.")

    with c3:
        st.subheader("Solution")
        phrases = _top_phrases(articles, "framing_solution")
        if phrases:
            st.plotly_chart(_bar_chart(phrases, "Top solutions"), use_container_width=True)
        else:
            st.info("No solution data.")

    # ── Coverage table ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Coverage by outlet")
    st.dataframe(_coverage_table(articles), use_container_width=True, hide_index=True)

    # ── Outlet-level framing breakdown ────────────────────────────────────────
    st.divider()
    st.subheader("Framing by outlet")
    st.caption(
        "Each bar shows one outlet's most frequently extracted framing phrase. "
        "This reveals systematic editorial patterns - which outlet assigns which villain "
        "to the same event."
    )
    o1, o2, o3 = st.columns(3)
    for col, field, label in [
        (o1, "framing_villain",  "Villain"),
        (o2, "framing_victim",   "Victim"),
        (o3, "framing_solution", "Solution"),
    ]:
        with col:
            st.markdown(f"**{label}**")
            fig = _outlet_bar_chart(
                _outlet_top_phrase(articles, field),
                f"Top {label.lower()} phrase by outlet",
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption(f"No {label.lower()} data.")
