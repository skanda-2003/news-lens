import sys
import os
import json
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import requests
import streamlit as st

from src.db import get_connection, get_distinct_topics

OLLAMA_URL = "http://localhost:11434"
MODEL      = "llama3.2:3b"

BIAS_LABELS = ["bjp_aligned", "opposition_aligned", "neutral"]
BIAS_COLOR  = {
    "bjp_aligned":        "#FF6B00",
    "opposition_aligned": "#2563EB",
    "neutral":            "#6B7280",
}
LABEL_DISPLAY = {
    "bjp_aligned":        "BJP-aligned",
    "opposition_aligned": "Opposition-aligned",
    "neutral":            "Neutral",
}


@st.cache_resource
def _conn():
    return get_connection()


@st.cache_data(ttl=300)
def _load_topics():
    return get_distinct_topics(_conn())


@st.cache_data(ttl=3600)
def _get_synthesis_articles(topic: str) -> dict[str, list[dict]]:
    """
    Fetch up to 2 articles per bias label for the given topic.
    Prefers trusted predictions with framing data; falls back to trusted-only,
    then any classified article if still short.
    """
    conn = _conn()
    result = {}
    for label in BIAS_LABELS:
        articles = []

        # Tier 1: trusted + framing parsed
        rows = conn.execute("""
            SELECT id, outlet, headline, body, bias_label, bias_trusted,
                   framing_villain, framing_victim, framing_solution, published_at
            FROM articles
            WHERE topic = ? AND bias_label = ? AND bias_trusted = 1 AND framing_parsed = 1
            ORDER BY published_at DESC
            LIMIT 2
        """, (topic, label)).fetchall()
        articles = [dict(r) for r in rows]

        # Tier 2: trusted only (no framing requirement)
        if len(articles) < 2:
            seen = {a["id"] for a in articles}
            rows2 = conn.execute("""
                SELECT id, outlet, headline, body, bias_label, bias_trusted,
                       framing_villain, framing_victim, framing_solution, published_at
                FROM articles
                WHERE topic = ? AND bias_label = ? AND bias_trusted = 1
                ORDER BY published_at DESC
                LIMIT 4
            """, (topic, label)).fetchall()
            for r in rows2:
                if r["id"] not in seen and len(articles) < 2:
                    articles.append(dict(r))

        # Tier 3: any classified article
        if len(articles) < 2:
            seen = {a["id"] for a in articles}
            rows3 = conn.execute("""
                SELECT id, outlet, headline, body, bias_label, bias_trusted,
                       framing_villain, framing_victim, framing_solution, published_at
                FROM articles
                WHERE topic = ? AND bias_label = ?
                ORDER BY published_at DESC
                LIMIT 4
            """, (topic, label)).fetchall()
            for r in rows3:
                if r["id"] not in seen and len(articles) < 2:
                    articles.append(dict(r))

        result[label] = articles
    return result


@st.cache_data(ttl=3600)
def _run_synthesis(topic: str, articles_by_label: str) -> tuple[dict | None, str]:
    """
    Call Ollama with articles from each bias group and return parsed synthesis JSON.
    articles_by_label is a JSON string so it's hashable as a cache key.
    """
    by_label = json.loads(articles_by_label)

    def _article_block(articles: list[dict]) -> str:
        if not articles:
            return "(no articles available)"
        lines = []
        for i, a in enumerate(articles, 1):
            headline = a.get("headline") or ""
            body     = (a.get("body") or "")[:300]
            lines.append(f"Article {i}: {headline}\n{body}")
        return "\n\n".join(lines)

    prompt = f"""You are a neutral journalism analyst. Read these Indian news articles about "{topic}" from outlets with different political leanings.

BJP-ALIGNED COVERAGE:
{_article_block(by_label.get('bjp_aligned', []))}

OPPOSITION-ALIGNED COVERAGE:
{_article_block(by_label.get('opposition_aligned', []))}

NEUTRAL COVERAGE:
{_article_block(by_label.get('neutral', []))}

Respond with a JSON object only. Do not include any text before or after the JSON.
Do not use quotation marks inside string values - rephrase to avoid them.
Every string value MUST begin AND end with a double quote character.
Use this exact structure:
{{
  "factual_summary": "2-3 sentences describing what actually happened based on facts present across multiple articles",
  "agreed_facts": ["fact that appears in multiple outlets", "another agreed fact", "a third agreed fact"],
  "framing_note": "1 sentence describing the key framing difference between BJP-aligned and opposition-aligned coverage"
}}"""

    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=60,
    )
    resp.raise_for_status()
    raw = resp.json().get("response", "")

    # Strip markdown code fences if the model wraps its output
    clean = raw.strip()
    if clean.startswith("```"):
        parts = clean.split("```")
        clean = parts[1] if len(parts) > 1 else clean
        if clean.startswith("json"):
            clean = clean[4:]
    clean = clean.strip()

    # Try direct parse first
    try:
        return json.loads(clean), raw
    except json.JSONDecodeError:
        pass

    # Repair pass: small LLMs sometimes emit  "key": value"  (missing opening quote).
    # Add the opening quote for any string value that starts with a letter but has no opening quote.
    repaired = re.sub(
        r'(":\s*)([A-Za-z][^"\n]*?")',
        lambda m: m.group(1) + '"' + m.group(2),
        clean,
    )
    try:
        return json.loads(repaired), raw
    except json.JSONDecodeError:
        pass

    # Regex fallback: find the outermost {...} block in case the model added preamble/postamble
    match = re.search(r'\{.*\}', clean, re.DOTALL)
    if match:
        extracted = match.group()
        try:
            return json.loads(extracted), raw
        except json.JSONDecodeError:
            pass
        # Repair pass on extracted block too
        try:
            return json.loads(re.sub(
                r'(":\s*)([A-Za-z][^"\n]*?")',
                lambda m: m.group(1) + '"' + m.group(2),
                extracted,
            )), raw
        except json.JSONDecodeError:
            pass

    # All parsing failed - return None so caller can show the raw output
    return None, raw


def _ollama_running() -> bool:
    try:
        requests.get(OLLAMA_URL, timeout=3)
        return True
    except Exception:
        return False


def _bias_badge(label: str) -> str:
    color = BIAS_COLOR.get(label, "#6B7280")
    display = LABEL_DISPLAY.get(label, label)
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:0.78em">{display}</span>'


def page_synthesis():
    st.header("What Actually Happened")
    st.caption(
        "Select a topic to see a neutral factual synthesis alongside how BJP-aligned and "
        "opposition-aligned outlets framed the same story."
    )

    topics = _load_topics()
    if not topics:
        st.warning("No topics found in the database. Run the ingestion pipeline first.")
        return

    topic = st.selectbox("Select topic", topics)

    if not st.button("Synthesise"):
        st.info("Select a topic and click Synthesise. This calls a local Ollama model and takes 10-20 seconds.")
        return

    # ── Ollama health check ───────────────────────────────────────────────────
    if not _ollama_running():
        st.error(
            "Ollama is not running. Start it with:\n\n"
            "```\nsudo systemctl start ollama\n```\n\n"
            "Then reload this page."
        )
        return

    # ── Fetch articles ────────────────────────────────────────────────────────
    articles_by_label = _get_synthesis_articles(topic)
    total_articles    = sum(len(v) for v in articles_by_label.values())

    if total_articles == 0:
        st.info(f"No classified articles found for topic: {topic}")
        return

    present_labels = [lbl for lbl, arts in articles_by_label.items() if arts]
    if len(present_labels) == 1:
        st.warning(
            f"Only {LABEL_DISPLAY[present_labels[0]]} articles found for this topic. "
            "The synthesis will be limited - framing comparison needs multiple perspectives."
        )

    # ── Run synthesis ─────────────────────────────────────────────────────────
    with st.spinner("Calling Ollama... this takes 10-20 seconds"):
        try:
            synthesis, raw_output = _run_synthesis(topic, json.dumps(articles_by_label))
        except requests.RequestException as e:
            st.error(f"Ollama request failed: {e}")
            return

    if synthesis is None:
        st.warning("Ollama response could not be parsed as JSON. Raw output below.")
        st.code(raw_output)
        return

    # ── Section 1: What happened ──────────────────────────────────────────────
    st.divider()
    st.subheader("What happened")

    factual = synthesis.get("factual_summary", "")
    if factual:
        st.markdown(factual)

    agreed = synthesis.get("agreed_facts", [])
    if agreed:
        st.markdown("**Points most outlets agree on:**")
        for fact in agreed:
            st.markdown(f"- {fact}")

    framing_note = synthesis.get("framing_note", "")
    if framing_note:
        st.caption(f"Framing difference: {framing_note}")

    # ── Section 2: How it was framed ──────────────────────────────────────────
    st.divider()
    st.subheader("How it was framed")

    cols = st.columns(3)
    for col, label in zip(cols, BIAS_LABELS):
        with col:
            st.markdown(
                f"<div style='border-top:3px solid {BIAS_COLOR[label]};padding-top:8px'>"
                f"<b>{LABEL_DISPLAY[label]}</b></div>",
                unsafe_allow_html=True,
            )
            label_articles = articles_by_label.get(label, [])
            if not label_articles:
                st.caption("No articles for this label.")
            for a in label_articles:
                outlet    = (a.get("outlet") or "unknown").replace("_", " ").title()
                headline  = a.get("headline") or "No headline"
                body      = (a.get("body") or "")[:200]
                villain   = a.get("framing_villain") or ""
                victim    = a.get("framing_victim") or ""

                with st.container(border=True):
                    st.markdown(f"**{outlet}**")
                    st.markdown(f"*{headline}*")
                    if body:
                        st.caption(body + ("..." if len(a.get("body") or "") > 200 else ""))
                    st.markdown(_bias_badge(label), unsafe_allow_html=True)
                    if villain:
                        st.caption(f"Villain: {villain}")
                    if victim:
                        st.caption(f"Victim: {victim}")

    # ── Section 3: Framing at a glance ────────────────────────────────────────
    st.divider()
    st.subheader("Framing at a glance")

    glance_rows = []
    for label in BIAS_LABELS:
        for a in articles_by_label.get(label, []):
            outlet = (a.get("outlet") or "unknown").replace("_", " ").title()
            glance_rows.append({
                "Outlet":   outlet,
                "Label":    LABEL_DISPLAY[label],
                "Villain":  a.get("framing_villain") or "-",
                "Victim":   a.get("framing_victim") or "-",
                "Solution": a.get("framing_solution") or "-",
            })

    if glance_rows:
        import pandas as pd
        st.dataframe(pd.DataFrame(glance_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No framing data available for these articles.")
