import sqlite3
import uuid

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.db import get_connection
from src.chroma_store import get_collection, add_article

# The embedding model - all-MiniLM-L6-v2 produces 384-dimensional vectors.
# SentenceTransformer downloads it automatically on first use (~80MB).
MODEL_NAME = "all-MiniLM-L6-v2"


def build_input_text(headline: str, body: str) -> str:
    """
    Combine headline and body into a single string for embedding.

    I use the first 400 chars of body so the embedding captures the article's
    opening framing rather than being dominated by filler text at the end.
    The pipe separator gives the model a clear boundary between the two fields.
    """
    body_excerpt = (body or "").strip()[:400]
    return f"{headline} | {body_excerpt}"


def generate_embeddings(batch_size: int = 64) -> None:
    """
    Embed all articles in SQLite that don't have a chroma_id yet, then store
    the embeddings in ChromaDB and write the chroma_id back to SQLite.

    Skips any article that already has a chroma_id so it's safe to re-run
    without duplicating work.

    batch_size controls how many articles are sent to the model at once.
    Larger batches are faster but use more RAM. 64 is safe on most machines.
    """
    print(f"Loading embedding model: {MODEL_NAME}")
    # SentenceTransformer loads the model into memory here - first run downloads it
    model = SentenceTransformer(MODEL_NAME)

    conn = get_connection()
    collection = get_collection()

    # Only fetch articles that haven't been embedded yet
    rows = conn.execute(
        "SELECT id, headline, body, outlet, topic, bias_label, published_at "
        "FROM articles WHERE chroma_id IS NULL"
    ).fetchall()

    if not rows:
        print("No new articles to embed - all articles already have a chroma_id.")
        return

    print(f"Embedding {len(rows)} articles in batches of {batch_size}...")

    # Process in batches so I don't load all texts into memory at once
    for batch_start in tqdm(range(0, len(rows), batch_size)):
        batch = rows[batch_start : batch_start + batch_size]

        # Build the input strings for every article in this batch
        texts = [build_input_text(row["headline"], row["body"]) for row in batch]

        # encode() runs the model and returns a numpy array of shape (batch_size, 384)
        # Each row is one article's embedding vector
        embeddings = model.encode(texts, show_progress_bar=False)

        for row, embedding in zip(batch, embeddings):
            # Generate a unique ID to link SQLite and ChromaDB
            chroma_id = str(uuid.uuid4())

            # Only pass metadata fields that have a value - ChromaDB rejects None values
            metadata = {
                k: v for k, v in {
                    "outlet":       row["outlet"],
                    "topic":        row["topic"],
                    "bias_label":   row["bias_label"],
                    "published_at": row["published_at"],
                }.items() if v is not None
            }

            # Store the embedding in ChromaDB
            add_article(
                collection=collection,
                chroma_id=chroma_id,
                embedding=embedding.tolist(),  # ChromaDB expects a plain Python list, not numpy
                headline=row["headline"] or "",
                metadata=metadata,
            )

            # Write the chroma_id back to SQLite so the two stores are linked
            conn.execute(
                "UPDATE articles SET chroma_id = ? WHERE id = ?",
                (chroma_id, row["id"]),
            )

        # Commit after each batch so progress is saved even if the run is interrupted
        conn.commit()

    final_count = collection.count()
    print(f"Done. ChromaDB now contains {final_count} embeddings.")
    conn.close()


if __name__ == "__main__":
    generate_embeddings()
