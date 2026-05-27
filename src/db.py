import sqlite3
import os
from datetime import datetime, timezone

# Path to the SQLite database file - it lives in data/ alongside the raw and processed folders
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "newslens.db")


# ── Connection ────────────────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    """Return a connection to the SQLite database."""
    # check_same_thread=False is needed when multiple parts of the code share one connection
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)

    # This makes rows come back as dict-like objects so I can access columns by name
    # e.g. row["headline"] instead of row[2]
    conn.row_factory = sqlite3.Row

    return conn


# ── Schema ────────────────────────────────────────────────────────────────────

def create_tables(conn: sqlite3.Connection) -> None:
    """Create the articles table if it doesn't already exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,

            -- Core article content
            url              TEXT UNIQUE,   -- UNIQUE enforces deduplication at the DB level
            outlet           TEXT,          -- normalised outlet name e.g. "bbc", "fox_news"
            headline         TEXT,
            body             TEXT,          -- full article text scraped by newspaper3k
            body_source      TEXT,          -- "scraped" | "rss_full" | "summary_only"

            -- Provenance
            published_at     TEXT,          -- ISO 8601 datetime string from the source
            topic            TEXT,          -- search query used to retrieve this article
            source           TEXT,          -- ingestion source: "newsapi" | "rss" | "gdelt"
            ingested_at      TEXT,          -- when this row was inserted, set automatically

            -- Bias classification (filled in later by the classifier)
            bias_label       TEXT,          -- "bjp_aligned" | "opposition_aligned" | "neutral"
            bias_confidence  REAL,          -- model confidence score between 0 and 1
            bias_trusted     INTEGER,       -- 1 if confidence is above threshold, 0 if uncertain

            -- ChromaDB link (filled in after embedding is stored)
            chroma_id        TEXT,          -- the ID of this article's vector in ChromaDB

            -- Framing extraction (filled in later by Ollama)
            framing_villain  TEXT,          -- implied villain extracted from the article
            framing_victim   TEXT,          -- implied victim extracted from the article
            framing_solution TEXT,          -- implied solution extracted from the article
            framing_parsed   INTEGER        -- 1 if Ollama output parsed cleanly, 0 if it failed
        )
    """)
    conn.commit()


# ── Insert ────────────────────────────────────────────────────────────────────

def insert_article(conn: sqlite3.Connection, article: dict) -> bool:
    """
    Insert one article into the database.

    Returns True if the article was inserted, False if it was skipped because
    the URL already exists (duplicate).
    """
    try:
        conn.execute(
            """
            INSERT INTO articles (
                url, outlet, headline, body, body_source,
                published_at, topic, source, ingested_at
            ) VALUES (
                :url, :outlet, :headline, :body, :body_source,
                :published_at, :topic, :source, :ingested_at
            )
            """,
            {
                "url":          article["url"],
                "outlet":       article["outlet"],
                "headline":     article["headline"],
                "body":         article.get("body", ""),
                "body_source":  article.get("body_source", "summary_only"),
                "published_at": article.get("published_at", ""),
                "topic":        article.get("topic", ""),
                "source":       article["source"],
                # If the caller didn't provide ingested_at, use the current UTC time
                "ingested_at":  article.get("ingested_at", datetime.now(timezone.utc).isoformat()),
            },
        )
        conn.commit()
        return True

    except sqlite3.IntegrityError:
        # This fires when the URL already exists in the table - just skip it silently
        return False


# ── Query ─────────────────────────────────────────────────────────────────────

def get_articles(conn: sqlite3.Connection, topic: str = None, outlet: str = None) -> list:
    """
    Fetch articles from the database, with optional filters.

    If topic is provided, only articles matching that topic are returned.
    If outlet is provided, only articles from that outlet are returned.
    Both filters can be combined.
    """
    # Start with a query that selects everything
    query = "SELECT * FROM articles WHERE 1=1"
    params = []

    # Append filters only if the caller passed them in
    if topic:
        query += " AND topic = ?"
        params.append(topic)
    if outlet:
        query += " AND outlet = ?"
        params.append(outlet)

    query += " ORDER BY published_at DESC"

    # fetchall() returns every matching row as a list
    rows = conn.execute(query, params).fetchall()

    # Convert each sqlite3.Row into a plain dict so callers can work with it normally
    return [dict(row) for row in rows]


def get_article_count(conn: sqlite3.Connection) -> int:
    """Return the total number of articles in the database."""
    row = conn.execute("SELECT COUNT(*) FROM articles").fetchone()
    return row[0]


def get_distinct_outlets(conn: sqlite3.Connection) -> list[str]:
    """Return a sorted list of every outlet slug that appears in the database."""
    rows = conn.execute(
        "SELECT DISTINCT outlet FROM articles WHERE outlet IS NOT NULL ORDER BY outlet"
    ).fetchall()
    return [row[0] for row in rows]


def get_distinct_topics(conn: sqlite3.Connection) -> list[str]:
    """Return a sorted list of every non-empty topic string in the database."""
    rows = conn.execute(
        "SELECT DISTINCT topic FROM articles WHERE topic IS NOT NULL AND topic != '' ORDER BY topic"
    ).fetchall()
    return [row[0] for row in rows]


def get_framing_articles(
    conn: sqlite3.Connection,
    topic: str = None,
    outlet: str = None,
) -> list[dict]:
    """
    Return articles where framing was successfully extracted (framing_parsed = 1).

    Columns returned: outlet, headline, framing_villain, framing_victim,
    framing_solution, bias_label, published_at.

    Optional filters: topic, outlet.
    """
    query = """
        SELECT outlet, headline, framing_villain, framing_victim,
               framing_solution, bias_label, published_at
        FROM articles
        WHERE framing_parsed = 1
    """
    params = []

    if topic:
        query += " AND topic = ?"
        params.append(topic)
    if outlet:
        query += " AND outlet = ?"
        params.append(outlet)

    query += " ORDER BY published_at DESC"

    rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]
