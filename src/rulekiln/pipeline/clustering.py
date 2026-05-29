"""Rule embedding clustering service using DBSCAN and HDBSCAN."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN

from rulekiln.schemas.pipeline import RuleClusterSchema

_NOISE_LABEL = -1


def _cosine_distance_matrix(vectors: list[list[float]]) -> np.ndarray:  # type: ignore[type-arg]
    mat = np.array(vectors, dtype=np.float64)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = mat / norms
    dist = 1.0 - normed @ normed.T
    # Numerical jitter can produce tiny negative values (for example, -1e-16)
    # that violate sklearn's non-negative constraint for precomputed distances.
    dist = np.nan_to_num(dist, nan=1.0, posinf=2.0, neginf=2.0)
    np.clip(dist, 0.0, 2.0, out=dist)
    np.fill_diagonal(dist, 0.0)
    return (dist + dist.T) / 2.0


def _build_clusters(
    labels: np.ndarray,  # type: ignore[type-arg]
    rule_ids: list[str],
    strategy: str,
    algorithm: str,
) -> list[RuleClusterSchema]:
    cluster_map: dict[int, list[str]] = {}
    for rule_id, label in zip(rule_ids, labels, strict=False):
        if label == _NOISE_LABEL:
            continue
        cluster_map.setdefault(int(label), []).append(rule_id)

    clusters: list[RuleClusterSchema] = []
    for cluster_label, ids in sorted(cluster_map.items()):
        clusters.append(
            RuleClusterSchema(
                strategy=strategy,
                algorithm=algorithm,
                rule_ids=ids,
                cluster_metadata={"cluster_id": cluster_label, "size": len(ids)},
            )
        )
    return clusters


def cluster_dbscan(
    rule_ids: list[str],
    embeddings: list[list[float]],
    eps: float = 0.3,
    min_samples: int = 2,
) -> list[RuleClusterSchema]:
    """Cluster rules using DBSCAN on cosine distance."""
    if len(rule_ids) < 2:
        return [RuleClusterSchema(strategy="dbscan", algorithm="dbscan", rule_ids=rule_ids)]

    dist = _cosine_distance_matrix(embeddings)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels: np.ndarray = db.fit_predict(dist)  # type: ignore[type-arg]
    return _build_clusters(labels, rule_ids, strategy="dbscan", algorithm="dbscan")


def cluster_hdbscan(
    rule_ids: list[str],
    embeddings: list[list[float]],
    min_cluster_size: int = 2,
) -> list[RuleClusterSchema]:
    """Cluster rules using HDBSCAN on cosine distance."""
    if len(rule_ids) < 2:
        return [RuleClusterSchema(strategy="hdbscan", algorithm="hdbscan", rule_ids=rule_ids)]

    try:
        import hdbscan as hdbscan_lib  # pyright: ignore[reportMissingModuleSource]
    except ImportError:
        # Fall back to DBSCAN if hdbscan not available
        clusters = cluster_dbscan(rule_ids, embeddings)
        for c in clusters:
            c.strategy = "hdbscan"
            c.algorithm = "hdbscan_fallback_dbscan"
        return clusters

    dist = _cosine_distance_matrix(embeddings)
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="precomputed",
        cluster_selection_method="eom",
    )
    labels: np.ndarray = clusterer.fit_predict(dist)  # type: ignore[type-arg]
    return _build_clusters(labels, rule_ids, strategy="hdbscan", algorithm="hdbscan")
