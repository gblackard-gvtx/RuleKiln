"""Unit tests for structure-level clustering diagnostics."""

from __future__ import annotations

import numpy as np

from rulekiln.pipeline.clustering import cosine_distance_matrix, dbscan_labels
from rulekiln.pipeline.clustering_diagnostics import (
    compute_cluster_health,
    derive_eps_candidates,
    duplicate_rule_rate,
    evaluate_eligibility,
    k_distance_report,
    kth_neighbor_distances,
    shannon_entropy,
    vector_norm_stats,
)
from rulekiln.schemas.pipeline import ClusterHealthMetrics, EligibilityThresholds

_HANDCRAFTED_DISTANCES = np.array(
    [
        [0.0, 0.1, 0.2, 0.3],
        [0.1, 0.0, 0.4, 0.5],
        [0.2, 0.4, 0.0, 0.6],
        [0.3, 0.5, 0.6, 0.0],
    ],
    dtype=np.float64,
)


def _two_blob_vectors(points_per_blob: int = 30, dim: int = 8) -> list[list[float]]:
    rng = np.random.default_rng(1729)
    center_a = np.zeros(dim)
    center_a[0] = 1.0
    center_b = np.zeros(dim)
    center_b[-1] = 1.0
    blob_a = center_a + rng.normal(scale=0.02, size=(points_per_blob, dim))
    blob_b = center_b + rng.normal(scale=0.02, size=(points_per_blob, dim))
    return np.vstack([blob_a, blob_b]).tolist()


def _health_with(
    cluster_count: int,
    noise_pct: float = 0.0,
    largest_cluster_pct: float = 10.0,
) -> ClusterHealthMetrics:
    return ClusterHealthMetrics(
        cluster_count_excluding_noise=cluster_count,
        noise_count=0,
        noise_pct=noise_pct,
        largest_cluster_size=1,
        largest_cluster_pct=largest_cluster_pct,
        median_cluster_size=1.0,
        p90_cluster_size=1.0,
        small_cluster_count=0,
    )


def test_kth_neighbor_distances_exact_values() -> None:
    assert np.allclose(
        kth_neighbor_distances(_HANDCRAFTED_DISTANCES, 1), [0.1, 0.1, 0.2, 0.3]
    )
    assert np.allclose(
        kth_neighbor_distances(_HANDCRAFTED_DISTANCES, 2), [0.2, 0.4, 0.4, 0.5]
    )
    assert np.allclose(
        kth_neighbor_distances(_HANDCRAFTED_DISTANCES, 3), [0.3, 0.5, 0.6, 0.6]
    )


def test_kth_neighbor_distances_clamps_large_k() -> None:
    clamped = kth_neighbor_distances(_HANDCRAFTED_DISTANCES, 99)
    assert np.allclose(clamped, kth_neighbor_distances(_HANDCRAFTED_DISTANCES, 3))


def test_k_distance_report_shape() -> None:
    rows = k_distance_report(_HANDCRAFTED_DISTANCES, ks=[1, 2])
    assert [row.k for row in rows] == [1, 2]
    for row in rows:
        assert "p50" in row.percentiles
        assert "p95" in row.percentiles
        assert "max" in row.percentiles
        assert row.percentiles["max"] >= row.percentiles["p50"]


def test_derive_eps_candidates_sorted_deduped_no_zeros() -> None:
    candidates = derive_eps_candidates(_HANDCRAFTED_DISTANCES, min_samples=1)
    eps_values = [eps for _, eps in candidates]
    assert eps_values == sorted(eps_values)
    assert len(eps_values) == len(set(eps_values))
    assert all(eps > 0.0 for eps in eps_values)


def test_derive_eps_candidates_identical_points_empty() -> None:
    dist = np.zeros((5, 5), dtype=np.float64)
    assert derive_eps_candidates(dist, min_samples=2) == []


def test_shannon_entropy() -> None:
    assert shannon_entropy({"a": 1, "b": 1, "c": 1, "d": 1}) == 2.0
    assert shannon_entropy({"a": 7}) == 0.0
    assert shannon_entropy({}) == 0.0


def test_duplicate_rule_rate() -> None:
    assert duplicate_rule_rate(["A rule", "a  RULE", "other"]) == 1.0 - 2 / 3
    assert duplicate_rule_rate(["x", "y", "z"]) == 0.0
    assert duplicate_rule_rate([]) == 0.0


def test_vector_norm_stats() -> None:
    norm_min, norm_median, norm_max = vector_norm_stats([[3.0, 4.0], [0.0, 1.0], [0.0, 2.0]])
    assert norm_min == 1.0
    assert norm_median == 2.0
    assert norm_max == 5.0


def test_two_blobs_cluster_with_percentile_eps() -> None:
    vectors = _two_blob_vectors()
    dist = cosine_distance_matrix(vectors)
    candidates = derive_eps_candidates(dist, min_samples=5)
    assert candidates, "expected eps candidates from the k-distance curve"
    _, eps = candidates[-1]
    labels = dbscan_labels(dist, eps=eps, min_samples=5)

    rule_labels = ["entailment"] * 30 + ["contradiction"] * 30
    health = compute_cluster_health(labels, dist, rule_labels)

    assert health.cluster_count_excluding_noise == 2
    assert health.noise_pct < 10.0
    assert 40.0 <= health.largest_cluster_pct <= 60.0
    # Blobs align with labels, so each cluster is label-pure.
    assert health.label_entropy_mean == 0.0
    assert health.intra_cluster_distance_mean is not None
    assert health.inter_cluster_distance_mean is not None
    assert health.inter_cluster_distance_mean > health.intra_cluster_distance_mean


def test_cluster_health_mixed_labels_and_noise() -> None:
    labels = np.array([0, 0, 0, 1, 1, -1], dtype=np.int64)
    dist = np.zeros((6, 6), dtype=np.float64)
    rule_labels = ["a", "b", "a", "c", "c", "minority"]
    hypotheses = ["h1", "h1", "h1", "h2", "h2", "h3"]
    health = compute_cluster_health(labels, dist, rule_labels, hypotheses)

    assert health.cluster_count_excluding_noise == 2
    assert health.noise_count == 1
    assert health.noise_label_counts == {"minority": 1}
    assert health.per_cluster_label_distribution == {
        "0": {"a": 2, "b": 1},
        "1": {"c": 2},
    }
    assert health.label_entropy_per_cluster["1"] == 0.0
    assert health.label_entropy_per_cluster["0"] > 0.0
    assert health.per_cluster_hypothesis_distribution == {
        "0": {"h1": 3},
        "1": {"h2": 2},
    }
    assert health.hypothesis_entropy_mean == 0.0
    assert health.small_cluster_count == 1  # cluster 1 has size 2 < threshold 3


def test_cluster_health_degenerate_single_cluster() -> None:
    labels = np.zeros(4, dtype=np.int64)
    dist = np.zeros((4, 4), dtype=np.float64)
    health = compute_cluster_health(labels, dist, ["a", "a", "a", "a"])
    assert health.cluster_count_excluding_noise == 1
    assert health.largest_cluster_pct == 100.0
    assert health.intra_cluster_distance_mean == 0.0
    assert health.inter_cluster_distance_mean is None


def test_evaluate_eligibility_boundaries() -> None:
    thresholds = EligibilityThresholds()

    assert not evaluate_eligibility(_health_with(7), thresholds).cluster_count_in_window
    assert evaluate_eligibility(_health_with(8), thresholds).eligible
    assert evaluate_eligibility(_health_with(40), thresholds).eligible
    assert not evaluate_eligibility(_health_with(41), thresholds).cluster_count_in_window

    assert evaluate_eligibility(_health_with(10, noise_pct=60.0), thresholds).noise_pct_ok
    flags = evaluate_eligibility(_health_with(10, noise_pct=60.1), thresholds)
    assert not flags.noise_pct_ok
    assert not flags.eligible

    assert evaluate_eligibility(
        _health_with(10, largest_cluster_pct=40.0), thresholds
    ).largest_cluster_pct_ok
    assert not evaluate_eligibility(
        _health_with(10, largest_cluster_pct=40.1), thresholds
    ).largest_cluster_pct_ok
