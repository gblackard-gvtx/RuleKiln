"""Structure-level clustering diagnostics: k-distance, cluster health, eligibility.

Pure functions over precomputed cosine-distance matrices and cluster label
arrays. Used by the offline clustering-sweep benchmark; no I/O, no LLM calls.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np

from rulekiln.schemas.pipeline import (
    ClusterHealthMetrics,
    EligibilityFlags,
    EligibilityThresholds,
    KDistanceRow,
)

_NOISE_LABEL = -1

KDISTANCE_KS: tuple[int, ...] = (3, 5, 6, 8, 10, 15)
KDISTANCE_PERCENTILES: tuple[float, ...] = (5, 10, 25, 50, 60, 70, 75, 80, 85, 90, 95, 99)
EPS_CANDIDATE_PERCENTILES: tuple[float, ...] = (60, 70, 75, 80, 85, 90, 95)
SMALL_CLUSTER_THRESHOLD: int = 3


def kth_neighbor_distances(
    distance_matrix: np.ndarray,  # type: ignore[type-arg]
    k: int,
) -> np.ndarray:  # type: ignore[type-arg]
    """Distance from each point to its k-th nearest neighbor (self excluded).

    k is clamped to n - 1 for small inputs.
    """
    n = distance_matrix.shape[0]
    if n < 2:
        return np.zeros(n, dtype=np.float64)
    k_eff = min(k, n - 1)
    # Column 0 after partition at index k_eff is not needed; the self-distance
    # (0 on the diagonal) occupies one of the first k_eff positions, so the
    # k-th neighbor lives at partition index k_eff.
    partitioned = np.partition(distance_matrix, k_eff, axis=1)
    return partitioned[:, k_eff]


def k_distance_report(
    distance_matrix: np.ndarray,  # type: ignore[type-arg]
    ks: Sequence[int] = KDISTANCE_KS,
) -> list[KDistanceRow]:
    """Percentile summary of k-th-neighbor distances for each candidate k."""
    rows: list[KDistanceRow] = []
    for k in ks:
        dists = kth_neighbor_distances(distance_matrix, k)
        percentiles: dict[str, float] = {
            f"p{int(p):02d}": float(np.percentile(dists, p)) for p in KDISTANCE_PERCENTILES
        }
        percentiles["max"] = float(dists.max()) if dists.size else 0.0
        rows.append(KDistanceRow(k=k, percentiles=percentiles))
    return rows


def derive_eps_candidates(
    distance_matrix: np.ndarray,  # type: ignore[type-arg]
    min_samples: int,
    percentiles: Sequence[float] = EPS_CANDIDATE_PERCENTILES,
) -> list[tuple[float, float]]:
    """(percentile, eps) candidates from the k-distance curve with k = min_samples.

    Rounded to 4 decimals, deduplicated on eps, zeros dropped, sorted by eps.
    """
    dists = kth_neighbor_distances(distance_matrix, min_samples)
    if dists.size == 0:
        return []
    seen: set[float] = set()
    candidates: list[tuple[float, float]] = []
    for p in percentiles:
        eps = round(float(np.percentile(dists, p)), 4)
        if eps <= 0.0 or eps in seen:
            continue
        seen.add(eps)
        candidates.append((p, eps))
    candidates.sort(key=lambda pair: pair[1])
    return candidates


def shannon_entropy(counts: Mapping[str, int]) -> float:
    """Shannon entropy in bits of a count distribution; 0.0 for empty/singleton."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def duplicate_rule_rate(texts: Sequence[str]) -> float:
    """Fraction of texts that are exact duplicates after casefold + whitespace collapse."""
    if not texts:
        return 0.0
    normalized = {" ".join(t.casefold().split()) for t in texts}
    return 1.0 - len(normalized) / len(texts)


def vector_norm_stats(embeddings: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    """(min, median, max) L2 norms of raw embedding vectors."""
    if not embeddings:
        return (0.0, 0.0, 0.0)
    norms = np.linalg.norm(np.array(embeddings, dtype=np.float64), axis=1)
    return (float(norms.min()), float(np.median(norms)), float(norms.max()))


def _distribution_by_cluster(
    labels: np.ndarray,  # type: ignore[type-arg]
    values: Sequence[str],
) -> dict[str, dict[str, int]]:
    per_cluster: dict[str, Counter[str]] = {}
    for label, value in zip(labels, values, strict=True):
        if label == _NOISE_LABEL:
            continue
        per_cluster.setdefault(str(int(label)), Counter())[value] += 1
    return {cluster: dict(counter) for cluster, counter in sorted(per_cluster.items())}


def _entropy_by_cluster(
    distribution: dict[str, dict[str, int]],
) -> tuple[dict[str, float], float | None]:
    entropies = {cluster: shannon_entropy(counts) for cluster, counts in distribution.items()}
    mean = sum(entropies.values()) / len(entropies) if entropies else None
    return entropies, mean


def compute_cluster_health(
    labels: np.ndarray,  # type: ignore[type-arg]
    distance_matrix: np.ndarray,  # type: ignore[type-arg]
    rule_labels: Sequence[str],
    rule_hypotheses: Sequence[str] | None = None,
) -> ClusterHealthMetrics:
    """Structure-level health metrics for one clustering outcome.

    `labels` are raw cluster assignments including -1 noise. `rule_labels` are
    the classification labels of each rule's source case, aligned by index.
    """
    n = int(labels.shape[0])
    noise_mask = labels == _NOISE_LABEL
    noise_count = int(noise_mask.sum())
    cluster_ids = sorted({int(label) for label in labels if label != _NOISE_LABEL})
    sizes = [int((labels == cid).sum()) for cid in cluster_ids]

    largest = max(sizes) if sizes else 0
    label_distribution = _distribution_by_cluster(labels, rule_labels)
    label_entropies, label_entropy_mean = _entropy_by_cluster(label_distribution)

    hypothesis_distribution: dict[str, dict[str, int]] = {}
    hypothesis_entropies: dict[str, float] = {}
    hypothesis_entropy_mean: float | None = None
    if rule_hypotheses is not None:
        hypothesis_distribution = _distribution_by_cluster(labels, rule_hypotheses)
        hypothesis_entropies, hypothesis_entropy_mean = _entropy_by_cluster(
            hypothesis_distribution
        )

    noise_label_counts = dict(
        Counter(
            value
            for label, value in zip(labels, rule_labels, strict=True)
            if label == _NOISE_LABEL
        )
    )

    intra_means: list[float] = []
    for cid, size in zip(cluster_ids, sizes, strict=True):
        if size < 2:
            continue
        member_idx = np.flatnonzero(labels == cid)
        sub = distance_matrix[np.ix_(member_idx, member_idx)]
        # Mean over off-diagonal pairs of the symmetric submatrix.
        intra_means.append(float((sub.sum() - np.trace(sub)) / (size * (size - 1))))
    intra_cluster_distance_mean = (
        sum(intra_means) / len(intra_means) if intra_means else None
    )

    inter_cluster_distance_mean: float | None = None
    if len(cluster_ids) >= 2:
        total = 0.0
        pair_count = 0
        for i, cid_a in enumerate(cluster_ids):
            idx_a = np.flatnonzero(labels == cid_a)
            for cid_b in cluster_ids[i + 1 :]:
                idx_b = np.flatnonzero(labels == cid_b)
                block = distance_matrix[np.ix_(idx_a, idx_b)]
                total += float(block.sum())
                pair_count += block.size
        inter_cluster_distance_mean = total / pair_count if pair_count else None

    return ClusterHealthMetrics(
        cluster_count_excluding_noise=len(cluster_ids),
        noise_count=noise_count,
        noise_pct=round(100.0 * noise_count / n, 2) if n else 0.0,
        largest_cluster_size=largest,
        largest_cluster_pct=round(100.0 * largest / n, 2) if n else 0.0,
        median_cluster_size=float(np.median(sizes)) if sizes else 0.0,
        p90_cluster_size=float(np.percentile(sizes, 90)) if sizes else 0.0,
        small_cluster_count=sum(1 for s in sizes if s < SMALL_CLUSTER_THRESHOLD),
        label_entropy_mean=label_entropy_mean,
        label_entropy_per_cluster=label_entropies,
        per_cluster_label_distribution=label_distribution,
        hypothesis_entropy_mean=hypothesis_entropy_mean,
        hypothesis_entropy_per_cluster=hypothesis_entropies,
        per_cluster_hypothesis_distribution=hypothesis_distribution,
        noise_label_counts=noise_label_counts,
        intra_cluster_distance_mean=intra_cluster_distance_mean,
        inter_cluster_distance_mean=inter_cluster_distance_mean,
    )


def evaluate_eligibility(
    health: ClusterHealthMetrics,
    thresholds: EligibilityThresholds,
) -> EligibilityFlags:
    """Structure-level smoke tests; an eligible arm is worth downstream eval."""
    cluster_count_in_window = (
        thresholds.min_cluster_count
        <= health.cluster_count_excluding_noise
        <= thresholds.max_cluster_count
    )
    noise_pct_ok = health.noise_pct <= thresholds.max_noise_pct
    largest_cluster_pct_ok = health.largest_cluster_pct <= thresholds.max_largest_cluster_pct
    return EligibilityFlags(
        cluster_count_in_window=cluster_count_in_window,
        noise_pct_ok=noise_pct_ok,
        largest_cluster_pct_ok=largest_cluster_pct_ok,
        eligible=cluster_count_in_window and noise_pct_ok and largest_cluster_pct_ok,
    )
