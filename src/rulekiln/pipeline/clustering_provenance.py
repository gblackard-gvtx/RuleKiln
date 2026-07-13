"""Runtime capture of clustering provenance (Workstream B).

Pure helpers that turn the *true runtime* clustering inputs/outputs into the typed
provenance artifacts. Kept free of DB/session/context types so they are unit-testable and
so capture happens deterministically from the exact values used at runtime, rather than
being reconstructed later from ``task.yaml``.

The distinction the integrity review demanded:

* ``requested_parameters`` mirror what the task config asked for (HDBSCAN ``min_samples``
  stays visibly ``None`` when unset).
* ``effective_parameters`` record the resolved library behavior (HDBSCAN ``min_samples``
  defaults to ``min_cluster_size`` when ``None``).
"""

from __future__ import annotations

import re
import statistics
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from importlib import metadata as importlib_metadata

import numpy as np

from rulekiln.artifacts.canonical_hash import canonical_hash, hash_matrix, hash_text
from rulekiln.schemas.pipeline import RuleClusterSchema, SynthesizedRuleSchema
from rulekiln.schemas.provenance import (
    ClusterAssignmentRecord,
    ClusteringAlgorithm,
    ClusteringHashes,
    ClusteringInput,
    ClusteringOutput,
    ClusteringParameters,
    ClusteringProvenance,
    ClusteringRuntime,
    ClusteringSource,
    ClusterSummary,
    ClusterSummaryEntry,
    EmbeddingProvenance,
)
from rulekiln.schemas.task_case import DbscanParams, HdbscanParams

_LIBRARY_BY_ALGORITHM: dict[str, str] = {"dbscan": "sklearn", "hdbscan": "hdbscan"}
_DISTRIBUTION_BY_LIBRARY: dict[str, str] = {"sklearn": "scikit-learn", "hdbscan": "hdbscan"}
_SMALL_CLUSTER_MAX_SIZE = 3


def parse_quantization(model: str) -> str | None:
    """Best-effort quantization tag parsed from a model id (e.g. ``Q6_K``); never invented."""
    match = re.search(r"Q\d+(?:_[A-Za-z0-9]+)?", model)
    return match.group(0) if match else None


def build_embedding_provenance(
    *,
    provider_profile: str,
    provider: str,
    model: str,
    embeddings: Sequence[Sequence[float]],
) -> EmbeddingProvenance:
    """Capture embedding runtime info, measuring normalization from the actual vectors."""
    dimension = len(embeddings[0]) if embeddings else 0
    vectors_prenormalized = False
    if embeddings:
        norms = np.linalg.norm(np.asarray(embeddings, dtype=np.float64), axis=1)
        vectors_prenormalized = bool(
            np.all(np.isfinite(norms)) and np.allclose(norms, 1.0, atol=1e-3)
        )
    return EmbeddingProvenance(
        provider_profile=provider_profile,
        provider=provider,
        model=model,
        dimension=dimension,
        quantization=parse_quantization(model),
        vectors_prenormalized=vectors_prenormalized,
        normalized_before_distance=True,
        distance_metric="cosine",
    )


def requested_effective_params(
    algorithm: ClusteringAlgorithm,
    dbscan: DbscanParams,
    hdbscan: HdbscanParams,
) -> tuple[ClusteringParameters, ClusteringParameters]:
    """Return (requested, effective) parameters for one algorithm.

    DBSCAN's HDBSCAN-only fields stay ``None``; HDBSCAN's ``eps`` stays ``None``. Requested
    HDBSCAN ``min_samples=None`` is preserved, while effective resolves it to
    ``min_cluster_size`` (the library default).
    """
    if algorithm == "dbscan":
        requested = ClusteringParameters(
            eps=dbscan.eps, min_samples=dbscan.min_samples, metric="cosine"
        )
        return requested, requested.model_copy(deep=True)

    requested = ClusteringParameters(
        min_cluster_size=hdbscan.min_cluster_size,
        min_samples=hdbscan.min_samples,
        cluster_selection_method=hdbscan.cluster_selection_method,
        cluster_selection_epsilon=hdbscan.cluster_selection_epsilon,
        metric="cosine",
    )
    effective_min_samples = (
        hdbscan.min_samples if hdbscan.min_samples is not None else hdbscan.min_cluster_size
    )
    effective = ClusteringParameters(
        min_cluster_size=hdbscan.min_cluster_size,
        min_samples=effective_min_samples,
        cluster_selection_method=hdbscan.cluster_selection_method,
        cluster_selection_epsilon=hdbscan.cluster_selection_epsilon,
        metric="cosine",
    )
    return requested, effective


def build_cluster_assignments(
    *,
    rule_ids: Sequence[str],
    rule_texts: Sequence[str],
    clusters: Sequence[RuleClusterSchema],
    label_by_rule: Mapping[str, str | None],
    source_case_by_rule: Mapping[str, str | None],
    hypothesis_by_rule: Mapping[str, str | None],
) -> list[ClusterAssignmentRecord]:
    """Deterministic per-rule assignment records (ordered by embedding row index)."""
    rule_to_cluster: dict[str, int] = {}
    for index, cluster in enumerate(clusters):
        raw = cluster.cluster_metadata.get("cluster_id")
        is_numeric = isinstance(raw, (int, float)) and not isinstance(raw, bool)
        cluster_id = int(raw) if is_numeric else index
        for rule_id in cluster.rule_ids:
            rule_to_cluster[rule_id] = cluster_id

    records: list[ClusterAssignmentRecord] = []
    for row_index, rule_id in enumerate(rule_ids):
        cluster_id = rule_to_cluster.get(rule_id)
        text = rule_texts[row_index] if row_index < len(rule_texts) else ""
        records.append(
            ClusterAssignmentRecord(
                rule_id=rule_id,
                source_case_id=source_case_by_rule.get(rule_id),
                cluster_id=cluster_id if cluster_id is not None else -1,
                is_noise=cluster_id is None,
                label=label_by_rule.get(rule_id),
                hypothesis_id=hypothesis_by_rule.get(rule_id),
                text_hash=hash_text(text),
                embedding_row_index=row_index,
            )
        )
    return records


def hash_assignments(records: Sequence[ClusterAssignmentRecord]) -> str:
    """Canonical hash of ordered assignment records (matches the persisted jsonl)."""
    return canonical_hash([record.model_dump(mode="json") for record in records])


def hash_synthesized_rules(rules: Sequence[SynthesizedRuleSchema]) -> str:
    """Canonical hash of a strategy's ordered synthesized rules."""
    return canonical_hash([rule.model_dump(mode="json") for rule in rules])


def build_cluster_summary(
    *,
    job_id: str | None,
    strategy: str,
    algorithm: ClusteringAlgorithm,
    assignments: Sequence[ClusterAssignmentRecord],
    rule_count: int,
) -> ClusterSummary:
    """Aggregate per-cluster stats plus noise statistics from assignment records."""
    by_cluster: dict[int, list[ClusterAssignmentRecord]] = {}
    noise_count = 0
    for record in assignments:
        if record.is_noise:
            noise_count += 1
            continue
        by_cluster.setdefault(record.cluster_id, []).append(record)

    entries: list[ClusterSummaryEntry] = []
    for cluster_id in sorted(by_cluster):
        members = by_cluster[cluster_id]
        label_dist = Counter(m.label for m in members if m.label is not None)
        hypothesis_dist = Counter(m.hypothesis_id for m in members if m.hypothesis_id is not None)
        entries.append(
            ClusterSummaryEntry(
                cluster_id=cluster_id,
                size=len(members),
                label_distribution=dict(sorted(label_dist.items())),
                hypothesis_distribution=dict(sorted(hypothesis_dist.items())),
                representative_rule_ids=[m.rule_id for m in members[:5]],
                medoid_index=members[0].embedding_row_index if members else None,
                synthesized_rule_id=None,
            )
        )

    noise_pct = round(noise_count / rule_count * 100, 4) if rule_count else 0.0
    return ClusterSummary(
        job_id=job_id,
        strategy=strategy,
        algorithm=algorithm,
        cluster_count=len(entries),
        noise_count=noise_count,
        noise_pct=noise_pct,
        clusters=entries,
    )


def _build_clustering_output(summary: ClusterSummary, rule_count: int) -> ClusteringOutput:
    sizes = [entry.size for entry in summary.clusters]
    largest = max(sizes, default=0)
    return ClusteringOutput(
        cluster_count_excluding_noise=len(summary.clusters),
        noise_count=summary.noise_count,
        noise_pct=summary.noise_pct,
        largest_cluster_size=largest,
        largest_cluster_pct=round(largest / rule_count * 100, 4) if rule_count else 0.0,
        median_cluster_size=float(statistics.median(sizes)) if sizes else 0.0,
        small_cluster_count=sum(1 for size in sizes if size < _SMALL_CLUSTER_MAX_SIZE),
        synthesized_rule_count=0,
    )


def _build_clustering_input(
    *,
    rule_ids: Sequence[str],
    rule_texts: Sequence[str],
    label_by_rule: Mapping[str, str | None],
    text_variant: str,
) -> ClusteringInput:
    text_hashes = [hash_text(text) for text in rule_texts]
    duplicate_count = len(text_hashes) - len(set(text_hashes))
    rule_count = len(rule_ids)
    label_counts = Counter(
        label_by_rule.get(rule_id)
        for rule_id in rule_ids
        if label_by_rule.get(rule_id) is not None
    )
    return ClusteringInput(
        rule_count=rule_count,
        duplicate_rule_count=duplicate_count,
        duplicate_rule_rate=round(duplicate_count / rule_count, 6) if rule_count else 0.0,
        label_counts={str(k): v for k, v in sorted(label_counts.items())},
        text_variant=text_variant,  # pyright: ignore[reportArgumentType]
    )


def _library_runtime(algorithm: ClusteringAlgorithm) -> tuple[str | None, str | None, str]:
    library = _LIBRARY_BY_ALGORITHM.get(algorithm)
    version: str | None = None
    distribution = _DISTRIBUTION_BY_LIBRARY.get(library or "")
    if distribution is not None:
        try:
            version = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            version = None
    return library, version, sys.version.split()[0]


def capture_clustering_provenance(
    *,
    job_id: str,
    strategy: str,
    algorithm: ClusteringAlgorithm,
    dbscan: DbscanParams,
    hdbscan: HdbscanParams,
    embedding: EmbeddingProvenance,
    rule_ids: Sequence[str],
    rule_texts: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    clusters: Sequence[RuleClusterSchema],
    label_by_rule: Mapping[str, str | None],
    source_case_by_rule: Mapping[str, str | None],
    hypothesis_by_rule: Mapping[str, str | None],
    text_variant: str = "rule_only",
    task_config_path: str | None = None,
    job_task_snapshot_path: str | None = "task.yaml",
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
) -> tuple[ClusteringProvenance, list[ClusterAssignmentRecord], ClusterSummary]:
    """Assemble the complete provenance triple for one algorithm from runtime values."""
    requested, effective = requested_effective_params(algorithm, dbscan, hdbscan)
    assignments = build_cluster_assignments(
        rule_ids=rule_ids,
        rule_texts=rule_texts,
        clusters=clusters,
        label_by_rule=label_by_rule,
        source_case_by_rule=source_case_by_rule,
        hypothesis_by_rule=hypothesis_by_rule,
    )
    rule_count = len(rule_ids)
    summary = build_cluster_summary(
        job_id=job_id,
        strategy=strategy,
        algorithm=algorithm,
        assignments=assignments,
        rule_count=rule_count,
    )
    output = _build_clustering_output(summary, rule_count)
    clustering_input = _build_clustering_input(
        rule_ids=rule_ids,
        rule_texts=rule_texts,
        label_by_rule=label_by_rule,
        text_variant=text_variant,
    )
    library, library_version, python_version = _library_runtime(algorithm)
    duration_ms: float | None = None
    if started_at is not None and completed_at is not None:
        duration_ms = (completed_at - started_at).total_seconds() * 1000.0
    runtime = ClusteringRuntime(
        library=library,  # pyright: ignore[reportArgumentType]
        library_version=library_version,
        python_version=python_version,
        started_at=started_at.isoformat() if started_at is not None else None,
        completed_at=completed_at.isoformat() if completed_at is not None else None,
        duration_ms=duration_ms,
    )
    hashes = ClusteringHashes(
        input_text_hash=canonical_hash(list(rule_texts)),
        embedding_matrix_hash=hash_matrix(embeddings),
        cluster_assignment_hash=hash_assignments(assignments),
        cluster_summary_hash=canonical_hash(summary),
        synthesized_rules_hash=None,
    )
    provenance = ClusteringProvenance(
        job_id=job_id,
        algorithm=algorithm,
        text_variant=text_variant,  # pyright: ignore[reportArgumentType]
        requested_parameters=requested,
        effective_parameters=effective,
        embedding=embedding,
        input=clustering_input,
        output=output,
        runtime=runtime,
        source=ClusteringSource(
            task_config_path=task_config_path,
            job_task_snapshot_path=job_task_snapshot_path,
            strategy=strategy,
        ),
        hashes=hashes,
    )
    return provenance, assignments, summary
