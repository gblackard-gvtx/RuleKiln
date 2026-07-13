"""Job-, clustering-, and strategy-level hash aggregation (Workstream D/E).

Every reproducibility hash flows through :mod:`rulekiln.artifacts.canonical_hash`; this
module only *assembles* the pieces into the persisted ``metadata/hashes.json`` shape
(:class:`ArtifactHashes`) and the per-strategy rows embedded in
``outputs/strategy_comparison.json`` (:class:`StrategyProvenanceRow`).

Nothing here recomputes model behavior: prompt/prediction/evaluation hashes are derived
from the artifacts that already exist for the run. Clustering hashes are read back from the
provenance object captured at clustering time (never reconstructed from ``task.yaml``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rulekiln.artifacts.canonical_hash import canonical_hash, hash_text
from rulekiln.pipeline.approved_configs import approved_config_hash
from rulekiln.pipeline.strategy_families import strategy_family
from rulekiln.schemas.pipeline import EvalResult
from rulekiln.schemas.promotion import ApprovedClusteringConfig, ArtifactHashes
from rulekiln.schemas.provenance import (
    ClusteringProvenance,
    StrategyProvenanceRow,
)
from rulekiln.schemas.task_case import RuleKilnTask


def runtime_clustering_config(provenance: ClusteringProvenance) -> ApprovedClusteringConfig:
    """Map a run's *effective* clustering parameters into the approved-config shape.

    Using the same shape (and therefore the same canonical hash) as the approved-config
    registry is what lets promotion compare a run's config against the approved config by
    hash (Workstream F/I).
    """
    effective = provenance.effective_parameters
    return ApprovedClusteringConfig(
        algorithm=provenance.algorithm,
        text_variant=provenance.text_variant,
        min_cluster_size=effective.min_cluster_size,
        min_samples=effective.min_samples,
        eps=effective.eps,
        cluster_selection_method=effective.cluster_selection_method,
        cluster_selection_epsilon=effective.cluster_selection_epsilon,
        distance_metric=effective.metric,
    )


def clustering_config_hash(provenance: ClusteringProvenance) -> str:
    """Canonical hash of a run's effective clustering config (comparable to approved)."""
    return approved_config_hash(runtime_clustering_config(provenance))


def prediction_hash(eval_result: EvalResult) -> str:
    """Order-sensitive hash of a strategy's per-case prediction outcomes.

    Uses the persisted case-level outcome fields (not model internals) so the same
    predictions re-read from disk reproduce the hash exactly.
    """
    records = [
        {
            "case_id": case.case_id,
            "passed": case.passed,
            "malformed": case.malformed,
            "score": case.score,
        }
        for case in eval_result.case_results
    ]
    return canonical_hash(records)


def evaluation_hash(eval_result: EvalResult) -> str:
    """Hash of a strategy's headline evaluation metrics (stable across reruns)."""
    payload = {
        "strategy": eval_result.strategy,
        "split": eval_result.split,
        "accuracy": eval_result.accuracy,
        "macro_f1": eval_result.macro_f1,
        "malformed_output_rate": eval_result.malformed_output_rate,
        "per_label_f1": {row.label: row.f1 for row in eval_result.per_label_metrics},
    }
    return canonical_hash(payload)


def ordered_case_ids_hash(ordered_case_ids: Sequence[str]) -> str:
    """Order-sensitive hash of the evaluation case-id list."""
    return canonical_hash(list(ordered_case_ids))


def build_job_hashes(
    *,
    task: RuleKilnTask,
    ordered_case_ids: Sequence[str],
    split: str | None,
    provider_config: Mapping[str, object],
    parser_config: Mapping[str, object],
    quality_gate_config: Mapping[str, object],
) -> dict[str, str | None]:
    """Compute the job-level hashes (dataset/task/provider/parser/gate)."""
    case_ids = list(ordered_case_ids)
    return {
        "dataset_split": canonical_hash(
            {
                "task_id": task.task_id,
                "split": split,
                "case_count": len(case_ids),
                "ordered_case_ids": case_ids,
            }
        ),
        "ordered_case_ids": ordered_case_ids_hash(case_ids),
        "task_config": canonical_hash(task),
        "provider_config": canonical_hash(dict(provider_config)),
        "parser_config": canonical_hash(dict(parser_config)),
        "quality_gate_config": canonical_hash(dict(quality_gate_config)),
    }


def build_strategy_provenance_row(
    *,
    strategy: str,
    compiled_prompt: str | None,
    eval_result: EvalResult | None,
    clustering_provenance: ClusteringProvenance | None,
    clustering_provenance_ref: str | None,
) -> StrategyProvenanceRow:
    """Assemble one comparison row.

    Clustering fields are ``None`` for non-clustering strategies; prompt/prediction/
    evaluation hashes are populated whenever the underlying artifact exists.
    """
    row = StrategyProvenanceRow(
        strategy=strategy,
        family=strategy_family(strategy),
        clustering_provenance_ref=clustering_provenance_ref,
        compiled_prompt_hash=hash_text(compiled_prompt) if compiled_prompt else None,
        prediction_hash=prediction_hash(eval_result) if eval_result is not None else None,
        evaluation_hash=evaluation_hash(eval_result) if eval_result is not None else None,
    )
    if clustering_provenance is not None:
        row.clustering_config_hash = clustering_config_hash(clustering_provenance)
        row.cluster_assignment_hash = clustering_provenance.hashes.cluster_assignment_hash
        row.synthesized_rules_hash = clustering_provenance.hashes.synthesized_rules_hash
    return row


def build_artifact_hashes(
    *,
    job_hashes: Mapping[str, str | None],
    clustering_provenance_by_strategy: Mapping[str, ClusteringProvenance],
    strategy_rows: Sequence[StrategyProvenanceRow],
    selected_strategy: str | None,
) -> ArtifactHashes:
    """Combine job + clustering + selected-strategy hashes into ``metadata/hashes.json``.

    The clustering-level slots reflect the *selected* strategy's clustering when it is a
    clustered strategy, so a reader of ``hashes.json`` sees the config that actually
    produced the selected prompt. Per-strategy hashes remain available in
    ``strategy_comparison.json``.
    """
    hashes: dict[str, str | None] = dict(job_hashes)

    selected_prov = (
        clustering_provenance_by_strategy.get(selected_strategy)
        if selected_strategy is not None
        else None
    )
    hashes["micro_rules"] = selected_prov.hashes.input_text_hash if selected_prov else None
    hashes["embedding_matrix"] = (
        selected_prov.hashes.embedding_matrix_hash if selected_prov else None
    )
    hashes["clustering_config"] = (
        clustering_config_hash(selected_prov) if selected_prov else None
    )
    hashes["cluster_assignments"] = (
        selected_prov.hashes.cluster_assignment_hash if selected_prov else None
    )
    hashes["synthesized_rules"] = (
        selected_prov.hashes.synthesized_rules_hash if selected_prov else None
    )

    selected_row = next(
        (row for row in strategy_rows if row.strategy == selected_strategy), None
    )
    hashes["compiled_prompt"] = selected_row.compiled_prompt_hash if selected_row else None
    hashes["decoding_config"] = None  # decoding is provider-runtime; not persisted per run
    hashes["predictions"] = selected_row.prediction_hash if selected_row else None
    hashes["evaluation_result"] = selected_row.evaluation_hash if selected_row else None

    return ArtifactHashes(hashes=hashes)
