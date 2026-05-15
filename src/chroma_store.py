import os

import chromadb

# Path to the folder where ChromaDB writes its files to disk
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_store")


# -- Connection ----------------------------------------------------------------

def get_collection() -> chromadb.Collection:
    """
    Open the ChromaDB store on disk and return the articles collection.

    PersistentClient writes to disk at CHROMA_PATH so embeddings survive
    between runs. get_or_create_collection is safe to call every time -
    it creates the collection on first run and opens the existing one after.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name="articles",
        # cosine similarity is better than Euclidean distance for comparing
        # text embeddings - it measures the angle between vectors, not their length
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# -- Write ---------------------------------------------------------------------

def add_article(
    collection: chromadb.Collection,
    chroma_id: str,
    embedding: list[float],
    headline: str,
    metadata: dict,
) -> None:
    """
    Store one article's embedding in ChromaDB.

    chroma_id must match the chroma_id written to SQLite so the two stores
    can be linked. Uses upsert so re-running the pipeline doesn't crash on
    articles that are already stored.

    metadata can include: outlet, topic, bias_label, published_at.
    headline is stored as the document string - it appears in search results.
    """
    # upsert = insert if not exists, update if already exists
    collection.upsert(
        ids=[chroma_id],
        embeddings=[embedding],
        documents=[headline],
        metadatas=[metadata],
    )


# -- Read ----------------------------------------------------------------------

def query_similar(
    collection: chromadb.Collection,
    embedding: list[float],
    n_results: int = 10,
    filters: dict = None,
) -> list[dict]:
    """
    Find the N articles whose embeddings are most similar to the given embedding.

    filters is an optional ChromaDB `where` clause to narrow results by metadata,
    e.g. {"topic": "climate change"} or {"bias_label": "right"}.

    Returns a list of dicts with keys: id, headline, distance, metadata.
    """
    kwargs = {
        "query_embeddings": [embedding],
        "n_results": n_results,
        # include specifies what ChromaDB returns alongside the ids
        "include": ["documents", "distances", "metadatas"],
    }

    if filters:
        kwargs["where"] = filters

    results = collection.query(**kwargs)

    # ChromaDB returns parallel lists, so zip them into a more usable list of dicts
    output = []
    for i, chroma_id in enumerate(results["ids"][0]):
        output.append({
            "id":       chroma_id,
            "headline": results["documents"][0][i],
            "distance": results["distances"][0][i],
            "metadata": results["metadatas"][0][i],
        })

    return output


def get_by_topic(collection: chromadb.Collection, topic: str) -> dict:
    """
    Fetch all stored embeddings for a given topic.

    Used by the clustering module - it needs the full matrix of embeddings
    for a topic to run KMeans or DBSCAN on.

    Returns a dict with keys: ids, embeddings, headlines, metadatas.
    """
    results = collection.get(
        where={"topic": topic},
        include=["embeddings", "documents", "metadatas"],
    )

    return {
        "ids":        results["ids"],
        "embeddings": results["embeddings"],
        "headlines":  results["documents"],
        "metadatas":  results["metadatas"],
    }
