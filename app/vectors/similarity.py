from __future__ import annotations

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    Returns a value in [-1.0, 1.0]. Returns 0.0 if either vector is zero.
    """
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")

    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return round(dot / (mag_a * mag_b), 6)


def top_k_matches(
    query: list[float],
    candidates: list[tuple[str, list[float]]],  # (id, vector)
    k: int = 3,
    min_similarity: float = 0.0,
) -> list[tuple[str, float]]:
    """
    Return the top-k candidates most similar to the query vector.

    Args:
        query:          Query embedding vector
        candidates:     List of (id, embedding) pairs
        k:              How many results to return
        min_similarity: Minimum cosine similarity threshold

    Returns:
        Sorted list of (id, similarity) pairs, highest similarity first.
    """
    scored = [
        (cid, cosine_similarity(query, vec))
        for cid, vec in candidates
    ]
    scored = [(cid, sim) for cid, sim in scored if sim >= min_similarity]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def centroid(vectors: list[list[float]]) -> list[float]:
    """
    Compute the element-wise mean of a list of vectors.
    Used to update a child's tone profile by averaging successful phrase embeddings.
    """
    if not vectors:
        raise ValueError("Cannot compute centroid of empty list.")
    dims = len(vectors[0])
    result = [0.0] * dims
    for vec in vectors:
        for i, v in enumerate(vec):
            result[i] += v
    n = len(vectors)
    return [round(x / n, 6) for x in result]


def weighted_centroid(
    vectors: list[list[float]],
    weights: list[float],
) -> list[float]:
    """
    Weighted centroid — successful phrases are weighted higher.
    Used to bias the preferred_tone_embedding toward phrases that led to success.
    """
    if not vectors or not weights:
        raise ValueError("vectors and weights must be non-empty.")
    if len(vectors) != len(weights):
        raise ValueError("vectors and weights must have the same length.")

    dims = len(vectors[0])
    result = [0.0] * dims
    total_weight = sum(weights)
    if total_weight == 0.0:
        return centroid(vectors)

    for vec, w in zip(vectors, weights):
        for i, v in enumerate(vec):
            result[i] += v * w

    return [round(x / total_weight, 6) for x in result]
