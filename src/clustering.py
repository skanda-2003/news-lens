import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

from src.chroma_store import get_collection, get_by_topic

# If the best silhouette score across all k values is below this, KMeans is
# not finding meaningful structure - fall back to DBSCAN instead
SILHOUETTE_FALLBACK_THRESHOLD = 0.3


# --- Helpers ---

def _representative_headlines(
    embeddings: np.ndarray,
    labels: np.ndarray,
    headlines: list[str],
    cluster_id: int,
    n: int = 5,
) -> list[str]:
    """
    Return the n headlines whose embeddings are closest to the centroid of
    cluster_id. These are the most "central" articles in that cluster - the
    ones that best represent what the cluster is about.
    """
    mask = labels == cluster_id
    cluster_embeddings = embeddings[mask]
    cluster_headlines = [h for h, m in zip(headlines, mask) if m]

    centroid = cluster_embeddings.mean(axis=0)

    # Euclidean distance from each article to the centroid
    distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)

    closest_indices = np.argsort(distances)[:n]
    return [cluster_headlines[i] for i in closest_indices]


def _bias_distribution(
    labels: np.ndarray,
    metadatas: list[dict],
    cluster_id: int,
) -> dict[str, float]:
    """
    Count bjp_aligned/opposition_aligned/neutral bias labels for articles in
    cluster_id and return them as percentages. Articles with no bias label are
    counted as "unknown".
    """
    counts = {"bjp_aligned": 0, "opposition_aligned": 0, "neutral": 0, "unknown": 0}

    for label, meta in zip(labels, metadatas):
        if label != cluster_id:
            continue
        bias = meta.get("bias_label") or "unknown"
        if bias not in counts:
            bias = "unknown"
        counts[bias] += 1

    total = sum(counts.values())
    if total == 0:
        return counts

    # Convert raw counts to percentages rounded to 1 decimal place
    return {k: round(v / total * 100, 1) for k, v in counts.items()}


# --- KMeans ---

def _run_kmeans(
    embeddings: np.ndarray,
    k_values: tuple[int, ...],
) -> dict:
    """
    Run KMeans for each value of k and collect silhouette score and inertia.

    Returns a dict mapping k → {"labels", "silhouette", "inertia"}.
    """
    results = {}

    for k in k_values:
        if k >= len(embeddings):
            # Can't have more clusters than articles
            continue

        km = KMeans(
            n_clusters=k,
            # n_init=10 means KMeans tries 10 different random starting
            # positions and keeps the best result - reduces bad luck from
            # a poor initialisation
            n_init=10,
            random_state=42,
        )
        labels = km.fit_predict(embeddings)

        # silhouette_score needs at least 2 distinct cluster labels
        score = silhouette_score(embeddings, labels)

        results[k] = {
            "labels":     labels,
            "silhouette": round(float(score), 4),
            # inertia = sum of squared distances from each point to its centroid
            # lower means tighter, more compact clusters
            "inertia":    round(float(km.inertia_), 2),
        }

        print(f"  k={k}  silhouette={score:.4f}  inertia={km.inertia_:.1f}")

    return results


# --- DBSCAN ---

def _run_dbscan(embeddings: np.ndarray) -> np.ndarray:
    """
    Run DBSCAN with eps derived from the data's own distance distribution.

    eps is the neighbourhood radius - points within eps of each other are
    considered neighbours. Setting it to the 10th percentile of all pairwise
    distances is a heuristic that avoids having to choose eps manually.

    Returns cluster labels. Label -1 means noise (no cluster).
    """
    # Compute pairwise Euclidean distances between all articles
    # np.linalg.norm along axis=2 gives the distance between each pair
    diff = embeddings[:, np.newaxis, :] - embeddings[np.newaxis, :, :]
    pairwise = np.linalg.norm(diff, axis=2)

    # Use the 10th percentile of non-zero distances as eps
    nonzero_distances = pairwise[pairwise > 0]
    eps = float(np.percentile(nonzero_distances, 10))

    print(f"  DBSCAN eps={eps:.4f} (10th percentile of pairwise distances)")

    db = DBSCAN(eps=eps, min_samples=3, metric="euclidean")
    labels = db.fit_predict(embeddings)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"  DBSCAN found {n_clusters} clusters, {n_noise} noise articles")

    return labels


# --- Main entry point ---

def cluster_topic(
    topic: str,
    k_values: tuple[int, ...] = (3, 4, 5),
) -> dict:
    """
    Run narrative clustering for a given topic.

    Pulls all embeddings for the topic from ChromaDB, tries KMeans at each k
    in k_values, picks the k with the highest silhouette score, and falls back
    to DBSCAN if no k produces a silhouette score above the threshold.

    Returns a dict with everything the notebook needs to plot and analyse:
      - embeddings: numpy array of shape (n_articles, 384)
      - headlines: list of article headlines
      - metadatas: list of metadata dicts (outlet, bias_label, topic, ...)
      - labels: numpy array of cluster assignments, one per article
      - method: "kmeans" or "dbscan"
      - best_k: the chosen k (None if DBSCAN was used)
      - kmeans_metrics: dict of k → {silhouette, inertia} for the elbow plot
      - best_silhouette: silhouette score for the chosen clustering
      - clusters: dict of cluster_id → {representative_headlines, bias_distribution}
    """
    collection = get_collection()
    raw = get_by_topic(collection, topic)

    if not raw["ids"]:
        raise ValueError(f"No articles found in ChromaDB for topic '{topic}'.")

    # Convert the list of lists from ChromaDB into a 2D numpy array
    # shape: (n_articles, 384)
    embeddings = np.array(raw["embeddings"], dtype=np.float32)
    headlines  = raw["headlines"]
    metadatas  = raw["metadatas"]

    n = len(headlines)
    print(f"Clustering {n} articles for topic '{topic}'")

    # --- Run KMeans for each k ---
    print("Running KMeans...")
    kmeans_results = _run_kmeans(embeddings, k_values)

    # Collect just the silhouette scores to find the best k
    kmeans_metrics = {
        k: {"silhouette": v["silhouette"], "inertia": v["inertia"]}
        for k, v in kmeans_results.items()
    }

    best_k = max(kmeans_results, key=lambda k: kmeans_results[k]["silhouette"])
    best_silhouette = kmeans_results[best_k]["silhouette"]

    # --- Choose method ---
    if best_silhouette >= SILHOUETTE_FALLBACK_THRESHOLD:
        method = "kmeans"
        labels = kmeans_results[best_k]["labels"]
        print(f"Using KMeans k={best_k} (silhouette={best_silhouette})")
    else:
        method = "dbscan"
        best_k = None
        print(
            f"Best KMeans silhouette {best_silhouette} < {SILHOUETTE_FALLBACK_THRESHOLD}. "
            f"Falling back to DBSCAN..."
        )
        labels = _run_dbscan(embeddings)
        # Compute silhouette for DBSCAN on non-noise points only
        non_noise = labels != -1
        if non_noise.sum() > 1 and len(set(labels[non_noise])) > 1:
            best_silhouette = round(float(silhouette_score(embeddings[non_noise], labels[non_noise])), 4)
        else:
            best_silhouette = None

    # --- Per-cluster summaries ---
    cluster_ids = sorted(set(labels))
    clusters = {}

    for cid in cluster_ids:
        size = int((labels == cid).sum())

        if cid == -1:
            # DBSCAN noise points - don't compute representative headlines
            clusters[cid] = {
                "label": "noise",
                "size": size,
                "representative_headlines": [],
                "bias_distribution": {},
            }
            continue

        rep_headlines = _representative_headlines(embeddings, labels, headlines, cid)
        bias_dist = _bias_distribution(labels, metadatas, cid)

        clusters[cid] = {
            "label": f"Cluster {cid}",
            "size": size,
            "representative_headlines": rep_headlines,
            "bias_distribution": bias_dist,
        }

    return {
        "embeddings":      embeddings,
        "headlines":       headlines,
        "metadatas":       metadatas,
        "labels":          labels,
        "method":          method,
        "best_k":          best_k,
        "kmeans_metrics":  kmeans_metrics,
        "best_silhouette": best_silhouette,
        "clusters":        clusters,
    }
