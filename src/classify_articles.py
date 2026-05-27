"""
Runs the fine-tuned RoBERTa classifier over all ingested articles that don't
have a bias label yet, writes results back to SQLite, and updates ChromaDB
metadata so clusters can show per-cluster bias distributions.

Run this once after ingestion and again whenever new articles are added.
"""

import sqlite3

from tqdm import tqdm

from src.classifier import predict_batch, CONFIDENCE_THRESHOLD
from src.db import get_connection
from src.chroma_store import get_collection

# Labels match directly - no remapping needed
LABEL_MAP = {
    "bjp_aligned":        "bjp_aligned",
    "opposition_aligned":  "opposition_aligned",
    "neutral":             "neutral",
}

# How many articles to classify at once - 32 fits comfortably in VRAM
BATCH_SIZE = 32


def run(overwrite: bool = False) -> None:
    """
    Classify all unclassified articles and write results to SQLite + ChromaDB.

    Args:
        overwrite: if True, re-classify articles that already have a label.
                   Defaults to False so re-running is safe and fast.
    """
    conn = get_connection()
    collection = get_collection()

    if overwrite:
        rows = conn.execute(
            "SELECT id, headline, body, chroma_id FROM articles"
        ).fetchall()
    else:
        # Only fetch articles that haven't been classified yet
        rows = conn.execute(
            "SELECT id, headline, body, chroma_id FROM articles "
            "WHERE bias_label IS NULL"
        ).fetchall()

    if not rows:
        print("No unclassified articles found.")
        return

    print(f"Classifying {len(rows)} articles...")

    # Split into batches and process each one
    for batch_start in tqdm(range(0, len(rows), BATCH_SIZE)):
        batch = rows[batch_start : batch_start + BATCH_SIZE]

        # predict_batch expects a list of dicts with "headline" and "body" keys
        articles_input = [
            {"headline": row["headline"] or "", "body": row["body"] or ""}
            for row in batch
        ]
        predictions = predict_batch(articles_input, batch_size=BATCH_SIZE)

        # Collect the ChromaDB updates so we can send them in one batch call
        chroma_ids_to_update = []
        chroma_metadatas_to_update = []

        for row, pred in zip(batch, predictions):
            # Map "center" → "centre" before writing to the database
            label      = LABEL_MAP[pred["label"]]
            confidence = pred["confidence"]
            trusted    = 1 if pred["trusted"] else 0

            # Write bias results to SQLite
            conn.execute(
                """
                UPDATE articles
                SET bias_label = ?, bias_confidence = ?, bias_trusted = ?
                WHERE id = ?
                """,
                (label, confidence, trusted, row["id"]),
            )

            # Queue the ChromaDB metadata update if this article has a chroma_id
            if row["chroma_id"]:
                chroma_ids_to_update.append(row["chroma_id"])
                chroma_metadatas_to_update.append({"bias_label": label})

        conn.commit()

        # Update ChromaDB metadata for this batch in one call - much faster than
        # calling update() once per article
        if chroma_ids_to_update:
            collection.update(
                ids=chroma_ids_to_update,
                metadatas=chroma_metadatas_to_update,
            )

    # Summary
    total = conn.execute("SELECT COUNT(*) FROM articles WHERE bias_label IS NOT NULL").fetchone()[0]
    trusted = conn.execute("SELECT COUNT(*) FROM articles WHERE bias_trusted = 1").fetchone()[0]
    bjp  = conn.execute("SELECT COUNT(*) FROM articles WHERE bias_label = 'bjp_aligned'").fetchone()[0]
    opp  = conn.execute("SELECT COUNT(*) FROM articles WHERE bias_label = 'opposition_aligned'").fetchone()[0]
    neu  = conn.execute("SELECT COUNT(*) FROM articles WHERE bias_label = 'neutral'").fetchone()[0]

    print(f"\nDone. {total} articles classified, {trusted} trusted (confidence >= {CONFIDENCE_THRESHOLD}).")
    print(f"Labels: bjp_aligned={bjp}, opposition_aligned={opp}, neutral={neu}")

    conn.close()


if __name__ == "__main__":
    run()
