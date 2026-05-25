"""Tests for clustering distance matrix stability."""

import numpy as np

from rulekiln.pipeline.clustering import _cosine_distance_matrix, cluster_dbscan


def test_cosine_distance_matrix_is_non_negative_and_finite() -> None:
    vectors = [
        [0.5, -0.4, 0.0, 1.2],
        [-0.6, 0.9, -0.1, 0.2],
        [0.5, -0.4, 0.0, 1.2],
    ]

    dist = _cosine_distance_matrix(vectors)

    assert np.isfinite(dist).all()
    assert (dist >= 0.0).all()
    assert np.allclose(np.diag(dist), 0.0)
    assert np.allclose(dist, dist.T)


def test_cluster_dbscan_accepts_negative_embedding_components() -> None:
    rule_ids = ["r1", "r2", "r3"]
    embeddings = [
        [0.2, -0.1, 0.3, -0.4],
        [0.21, -0.11, 0.29, -0.39],
        [-0.9, 0.8, -0.7, 0.6],
    ]

    clusters = cluster_dbscan(rule_ids=rule_ids, embeddings=embeddings, eps=0.15, min_samples=2)

    assert isinstance(clusters, list)
    assert all(cluster.strategy == "dbscan" for cluster in clusters)
