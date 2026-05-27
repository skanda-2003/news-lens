import json
import logging
import os
import time

import requests

from src.db import get_connection

# Ollama runs on Windows and is reachable from WSL at this address
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"

# Articles shorter than this word count don't have enough content for meaningful framing
MIN_WORDS = 50

# Log parse failures here so I can inspect bad Ollama outputs later
FAILURE_LOG = os.path.join(os.path.dirname(__file__), "..", "data", "framing_failures.log")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Prompt template - {article_text} is replaced with the actual article before sending
PROMPT_TEMPLATE = """You are a media framing analyst using Entman's (1993) framing theory.
Analyse the following news article excerpt and extract three framing dimensions.

Return ONLY a raw JSON object. No explanation. No markdown fences. No preamble. No trailing text.
Use exactly these keys: "villain", "victim", "solution".
Values must be short phrases (3-8 words). Use JSON null (not the string "null") if a dimension is absent from the article.

Example of correct output:
{{"villain": "federal government inaction", "victim": "coastal communities", "solution": null}}

Article:
{article_text}"""


# ── Ollama health check ───────────────────────────────────────────────────────

def check_ollama_running() -> None:
    """Raise RuntimeError if Ollama is not reachable at localhost:11434."""
    try:
        r = requests.get(OLLAMA_URL, timeout=3)
        if r.status_code != 200:
            raise RuntimeError(f"Ollama returned status {r.status_code}. Is the right model loaded?")
    except requests.ConnectionError:
        raise RuntimeError(
            "Ollama is not running. Start it from Windows before running this script."
        )


# ── Ollama call ───────────────────────────────────────────────────────────────

def call_ollama(article_text: str) -> str:
    """Send article text to Ollama and return the raw string response."""
    # Trim to 600 words - Ollama context window is limited and framing is in the first few paragraphs
    words = article_text.split()
    trimmed = " ".join(words[:600])

    prompt = PROMPT_TEMPLATE.format(article_text=trimmed)

    # Ollama's /api/generate endpoint takes model + prompt and returns a streaming JSON response
    # stream=False tells it to wait and return the full response in one go
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=60)
    response.raise_for_status()

    # The response JSON has a "response" key containing the model's output text
    return response.json()["response"]


# ── JSON validation ───────────────────────────────────────────────────────────

def parse_framing_response(raw: str) -> dict | None:
    """Parse and validate Ollama's JSON output. Returns None on any failure."""
    try:
        # Strip accidental markdown fences - e.g. ```json ... ``` or ``` ... ```
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Remove the opening fence line and closing fence
            lines = cleaned.splitlines()
            # Drop first line (```json or ```) and last line (```)
            cleaned = "\n".join(lines[1:-1]).strip()

        parsed = json.loads(cleaned)

        # Make sure all three expected keys are present
        if not all(k in parsed for k in ("villain", "victim", "solution")):
            return None

        return parsed

    except (json.JSONDecodeError, AttributeError, ValueError):
        return None


def log_failure(article_id: int, raw: str) -> None:
    """Append a failed parse to the failure log with the article ID and raw output."""
    with open(FAILURE_LOG, "a", encoding="utf-8") as f:
        f.write(f"--- article_id={article_id} ---\n")
        f.write(raw.strip())
        f.write("\n\n")


# ── Main extraction loop ──────────────────────────────────────────────────────

def extract_framing(limit: int = None) -> None:
    """
    Run framing extraction on all articles that haven't been processed yet.
    Writes villain/victim/solution back to SQLite for each article.
    Set limit to a small number (e.g. 10) to do a quick test run first.
    """
    check_ollama_running()

    conn = get_connection()

    # Only fetch articles where framing hasn't been attempted yet
    # framing_parsed IS NULL means we've never run framing on this article
    query = """
        SELECT id, headline, body
        FROM articles
        WHERE framing_parsed IS NULL
        AND body IS NOT NULL
        AND body != ''
    """
    if limit:
        query += f" LIMIT {limit}"

    rows = conn.execute(query).fetchall()
    logger.info(f"Found {len(rows)} articles to process")

    success = 0
    failed = 0

    for i, row in enumerate(rows):
        article_id = row["id"]
        headline = row["headline"] or ""
        body = row["body"] or ""

        # Skip articles that are too short to have meaningful framing
        word_count = len(body.split())
        if word_count < MIN_WORDS:
            logger.info(f"[{i+1}/{len(rows)}] Skipping article {article_id} - only {word_count} words")
            # Mark as attempted but skipped so I don't revisit it
            conn.execute(
                "UPDATE articles SET framing_parsed = 0 WHERE id = ?",
                (article_id,)
            )
            conn.commit()
            continue

        # Combine headline and body - same pattern as the classifier input
        article_text = f"{headline}\n\n{body}"

        logger.info(f"[{i+1}/{len(rows)}] Processing article {article_id} ({word_count} words)")

        try:
            raw = call_ollama(article_text)
            parsed = parse_framing_response(raw)

            if parsed:
                # Write the three framing dimensions and mark as successfully parsed
                conn.execute(
                    """UPDATE articles
                       SET framing_villain = ?,
                           framing_victim  = ?,
                           framing_solution = ?,
                           framing_parsed  = 1
                       WHERE id = ?""",
                    (parsed["villain"], parsed["victim"], parsed["solution"], article_id)
                )
                success += 1
            else:
                # Parsing failed - log raw output and mark as failed in the database
                log_failure(article_id, raw)
                conn.execute(
                    """UPDATE articles
                       SET framing_villain = NULL,
                           framing_victim  = NULL,
                           framing_solution = NULL,
                           framing_parsed  = 0
                       WHERE id = ?""",
                    (article_id,)
                )
                failed += 1
                logger.warning(f"  Parse failed for article {article_id}")

        except requests.RequestException as e:
            # Network error mid-run - log and continue rather than crash
            logger.error(f"  Ollama request failed for article {article_id}: {e}")
            failed += 1

        conn.commit()

        # Small delay between requests so I don't overwhelm Ollama
        time.sleep(0.5)

    logger.info(f"Done. Success: {success} | Failed/skipped: {failed}")
    conn.close()


if __name__ == "__main__":
    extract_framing()