"""
Synthesis module - takes articles from different-leaning outlets on the same topic
and uses Ollama to produce a neutral factual summary alongside framing differences.

This is the backend for the Synthesis dashboard page (app/views/synthesis.py).
get_articles_for_synthesis() fetches the articles; synthesize() calls Ollama.
"""

import json

import requests

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"
N_PER_LABEL  = 2   # articles per bias label included in each synthesis call

# Prompt designed for factual extraction, not open-ended summarisation.
# The rules block is important - without it Llama 3.2 will include framing language.
_PROMPT_TEMPLATE = """You are a factual news editor working with Indian news sources.

You have {n} articles from different outlets covering the same story.
Extract only the verifiable facts that multiple sources agree on.

Rules:
- Write in neutral, direct language
- No adjectives that carry political weight
- Do not assign blame or credit
- Do not include framing that only one source uses
- Use names and specific facts, not vague summaries

Articles:
{articles_text}

Return ONLY a raw JSON object with these exact keys:
- "factual_summary": 3-5 sentences stating what happened
- "agreed_facts": list of 3-6 bullet points all sources confirm
- "framing_note": one sentence describing how the sources frame it differently

Example output format:
{{"factual_summary": "...", "agreed_facts": ["...", "..."], "framing_note": "..."}}"""


def check_ollama_running() -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        r = requests.get("http://localhost:11434", timeout=3)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


def get_articles_for_synthesis(conn, topic: str, n_per_label: int = N_PER_LABEL) -> list[dict]:
    """
    Fetch n_per_label articles per bias label for the given topic.

    Prefers recently published, trusted predictions with successful framing extraction.
    Falls back to any article for a label if trusted ones are scarce.
    """
    articles = []

    for label in ["bjp_aligned", "opposition_aligned", "neutral"]:
        rows = conn.execute("""
            SELECT headline, body, outlet, bias_label, bias_confidence,
                   framing_villain, framing_victim
            FROM articles
            WHERE topic = ?
              AND bias_label = ?
              AND bias_trusted = 1
              AND framing_parsed = 1
            ORDER BY published_at DESC
            LIMIT ?
        """, (topic, label, n_per_label)).fetchall()

        # If not enough trusted+framed articles, relax constraints
        if len(rows) < n_per_label:
            rows = conn.execute("""
                SELECT headline, body, outlet, bias_label, bias_confidence,
                       framing_villain, framing_victim
                FROM articles
                WHERE topic = ?
                  AND bias_label = ?
                ORDER BY published_at DESC
                LIMIT ?
            """, (topic, label, n_per_label)).fetchall()

        for row in rows:
            articles.append(dict(row))

    return articles


def _format_articles(articles: list[dict]) -> str:
    """Format article list into numbered text blocks for the prompt."""
    parts = []
    for i, a in enumerate(articles, 1):
        outlet   = (a.get("outlet") or "unknown").upper()
        label    = a.get("bias_label") or ""
        headline = a.get("headline") or ""
        body     = (a.get("body") or "")[:600]
        parts.append(
            f"[Article {i} - {outlet} ({label})]\nHeadline: {headline}\n{body}"
        )
    return "\n\n".join(parts)


def _parse_response(raw: str) -> dict | None:
    """
    Parse Ollama JSON output. Returns None on any failure so the caller can
    show a graceful error rather than crashing.
    """
    try:
        cleaned = raw.strip().strip("```json").strip("```").strip()
        parsed  = json.loads(cleaned)
        if not all(k in parsed for k in ("factual_summary", "agreed_facts", "framing_note")):
            return None
        return parsed
    except (json.JSONDecodeError, AttributeError):
        return None


def synthesize(articles: list[dict]) -> dict | None:
    """
    Call Ollama with the given articles and return the parsed synthesis.

    Returns None if Ollama is unreachable or the response cannot be parsed.
    The caller is responsible for checking check_ollama_running() before calling this.
    """
    if not articles:
        return None

    prompt = _PROMPT_TEMPLATE.format(
        n=len(articles),
        articles_text=_format_articles(articles),
    )

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        raw = response.json().get("response", "")
        return _parse_response(raw)
    except requests.RequestException:
        return None
