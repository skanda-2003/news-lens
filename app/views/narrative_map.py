import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from umap import UMAP

from src.db import get_connection, get_distinct_topics
from src.clustering import cluster_topic

CLUSTER_COLORS = ["#FF4D00", "#2563EB", "#16A34A", "#9333EA", "#CA8A04", "#0891B2"]
BIAS_COLOR_MAP  = {"left": "🔵", "centre": "⚫", "right": "🔴"}


@st.cache_resource
def _conn():
    return get_connection()


@st.cache_data(ttl=300)
def _load_topics():
    return get_distinct_topics(_conn())


@st.cache_data(ttl=3600, show_spinner="Running clustering...")
def _cluster(topic: str) -> dict:
    return cluster_topic(topic)


@st.cache_data(ttl=3600, show_spinner="Reducing to 2D...")
def _umap(topic: str, embeddings: np.ndarray) -> np.ndarray:
    # topic arg makes the cache key topic-specific
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    return reducer.fit_transform(embeddings)


def page_narrative_map():
    st.header("Narrative Map")
    st.caption(
        "Articles on the same topic clustered by semantic similarity. "
        "Clusters reveal how different outlets frame the same event."
    )

    topics = _load_topics()
    if not topics:
        st.warning("No topics found in the database.")
        return

    topic = st.selectbox("Select topic", topics)

    if st.button("Run clustering"):
        st.session_state["nm_topic"] = topic

    active_topic = st.session_state.get("nm_topic")
    if not active_topic:
        st.info("Select a topic and click Run clustering.")
        return

    result   = _cluster(active_topic)
    coords   = _umap(active_topic, result["embeddings"])
    labels   = result["labels"]
    headlines = result["headlines"]
    metadatas = result["metadatas"]
    clusters  = result["clusters"]
    method    = result["method"]
    best_k    = result["best_k"]
    silhouette = result["best_silhouette"]

    # ── Metrics ───────────────────────────────────────────────────────────────
    method_str = f"KMeans k={best_k}" if method == "kmeans" else "DBSCAN"
    sil_str    = f"{silhouette:.3f}" if silhouette is not None else "N/A"

    m1, m2, m3 = st.columns(3)
    m1.metric("Articles", len(headlines))
    m2.metric("Clustering method", method_str)
    m3.metric("Silhouette score", sil_str)

    st.divider()

    # ── Scatter plot ──────────────────────────────────────────────────────────
    unique_labels = sorted(set(labels))
    traces = []

    for idx, cid in enumerate(unique_labels):
        mask  = labels == cid
        color = "#D1D5DB" if cid == -1 else CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]
        name  = "Noise" if cid == -1 else f"Cluster {cid}"
        hover = [
            f"{headlines[i][:80]}<br><i>{metadatas[i].get('outlet','')}</i>"
            for i, m in enumerate(mask) if m
        ]
        traces.append(go.Scatter(
            x=coords[mask, 0],
            y=coords[mask, 1],
            mode="markers",
            name=name,
            marker=dict(color=color, size=7),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        height=450,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white",
        plot_bgcolor="#F9FAFB",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Per-cluster breakdown ─────────────────────────────────────────────────
    st.subheader("Cluster breakdown")
    for cid, cluster in clusters.items():
        if cid == -1:
            label_str = f"Noise — {cluster['size']} articles"
        else:
            color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
            label_str = f"Cluster {cid} — {cluster['size']} articles"

        with st.expander(label_str, expanded=(cid != -1)):
            dist = cluster.get("bias_distribution", {})
            if dist:
                d1, d2, d3 = st.columns(3)
                d1.metric("Left", f"{dist.get('left', 0):.0f}%")
                d2.metric("Centre", f"{dist.get('centre', 0):.0f}%")
                d3.metric("Right", f"{dist.get('right', 0):.0f}%")

            rep = cluster.get("representative_headlines", [])
            if rep:
                st.markdown("**Representative headlines:**")
                for h in rep[:5]:
                    st.markdown(f"- {h}")
