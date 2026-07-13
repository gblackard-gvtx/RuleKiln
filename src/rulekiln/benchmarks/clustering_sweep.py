"""Offline clustering-tuning sweep over a completed job's micro-rules.

Phase A of the clustering-tuning harness: no LLM calls, no student eval.
Re-embeds persisted micro-rules under one or more text variants, then sweeps
DBSCAN/HDBSCAN parameter grids and reports structure-level health metrics per
arm so only sane settings graduate to expensive downstream evaluation.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rulekiln.db.models import Case
from rulekiln.db.repositories.jobs import get_micro_rules_for_job
from rulekiln.observability.logging import get_logger
from rulekiln.pipeline.clustering import cosine_distance_matrix, dbscan_labels, hdbscan_labels
from rulekiln.pipeline.clustering_diagnostics import (
    compute_cluster_health,
    derive_eps_candidates,
    duplicate_rule_rate,
    evaluate_eligibility,
    k_distance_report,
    vector_norm_stats,
)
from rulekiln.providers.contracts import EmbeddingClient, ProviderConfig
from rulekiln.schemas.pipeline import (
    ClusteringSweepArm,
    ClusteringSweepArtifact,
    EligibilityThresholds,
    EmbeddingSetupReport,
    EmbeddingTextVariant,
    KDistanceReport,
)
from rulekiln.schemas.task_case import DbscanParams, HdbscanParams

logger = get_logger(__name__)

DBSCAN_MIN_SAMPLES_GRID: tuple[int, ...] = (3, 5, 6, 8, 10)
HDBSCAN_MIN_CLUSTER_SIZE_GRID: tuple[int, ...] = (5, 10, 15, 25)
HDBSCAN_MIN_SAMPLES_GRID: tuple[int, ...] = (1, 3, 5, 10)
HDBSCAN_SELECTION_METHODS: tuple[Literal["eom", "leaf"], ...] = ("eom", "leaf")
HDBSCAN_SELECTION_EPSILON_GRID: tuple[float, ...] = (0.0, 0.05, 0.10)

ALL_TEXT_VARIANTS: tuple[EmbeddingTextVariant, ...] = (
    "rule_only",
    "label_plus_rule",
    "label_plus_hypothesis_plus_rule",
)

_HYPOTHESIS_KEY_MAX_CHARS = 80
_PRENORMALIZED_TOLERANCE = 1e-3


class RuleRecord(BaseModel):
    """A micro-rule joined with its source case's label and hypothesis."""

    rule_id: str
    topic: str
    condition: str
    expected_outcome: str
    label: str
    hypothesis: str | None = None


async def load_rule_records(session: AsyncSession, job_id: str) -> list[RuleRecord]:
    """Load micro-rules for a job joined with source-case label and hypothesis."""
    micro_rules = await get_micro_rules_for_job(session, job_id)
    result = await session.execute(select(Case).where(Case.job_id == job_id))
    cases = {case.id: case for case in result.scalars().all()}

    records: list[RuleRecord] = []
    for rule in micro_rules:
        case = cases.get(rule.case_id)
        label = "unknown"
        hypothesis: str | None = None
        if case is not None:
            expected = case.expected_json or {}
            raw_label = expected.get("label")
            if isinstance(raw_label, str) and raw_label:
                label = raw_label
            elif isinstance(case.expected_text, str) and case.expected_text:
                label = case.expected_text
            raw_hypothesis = (case.input_json or {}).get("hypothesis")
            if isinstance(raw_hypothesis, str) and raw_hypothesis:
                hypothesis = raw_hypothesis
        records.append(
            RuleRecord(
                rule_id=rule.id,
                topic=rule.topic,
                condition=rule.condition,
                expected_outcome=rule.expected_outcome,
                label=label,
                hypothesis=hypothesis,
            )
        )
    return records


def _rule_only_text(record: RuleRecord) -> str:
    # Must stay byte-identical to the pipeline's rule embedding text
    # (RuleEmbeddingStageService in workers/pipeline/early_stages.py).
    return f"{record.topic}: {record.condition} → {record.expected_outcome}"


def build_variant_texts(records: list[RuleRecord], variant: EmbeddingTextVariant) -> list[str]:
    """Render each rule as the text to embed for the given variant.

    In the hypothesis variant, rules without a hypothesis degrade per-rule to
    the label_plus_rule form.
    """
    texts: list[str] = []
    for record in records:
        base = _rule_only_text(record)
        if variant == "rule_only":
            texts.append(base)
        elif variant == "label_plus_rule" or record.hypothesis is None:
            texts.append(f"[label={record.label}] {base}")
        else:
            texts.append(f"[label={record.label}] [hypothesis={record.hypothesis}] {base}")
    return texts


def _dbscan_arms(
    variant: EmbeddingTextVariant,
    distance_matrix_eps_candidates: dict[int, list[tuple[float, float]]],
) -> list[tuple[str, DbscanParams, float]]:
    arms: list[tuple[str, DbscanParams, float]] = []
    for min_samples, candidates in distance_matrix_eps_candidates.items():
        for percentile, eps in candidates:
            arm_id = f"dbscan__{variant}__eps{eps:.4f}_ms{min_samples}"
            arms.append((arm_id, DbscanParams(eps=eps, min_samples=min_samples), percentile))
    return arms


def _hdbscan_arms(variant: EmbeddingTextVariant) -> list[tuple[str, HdbscanParams]]:
    arms: list[tuple[str, HdbscanParams]] = []
    for mcs, ms, method, eps in product(
        HDBSCAN_MIN_CLUSTER_SIZE_GRID,
        HDBSCAN_MIN_SAMPLES_GRID,
        HDBSCAN_SELECTION_METHODS,
        HDBSCAN_SELECTION_EPSILON_GRID,
    ):
        arm_id = f"hdbscan__{variant}__mcs{mcs}_ms{ms}_{method}_eps{eps:.2f}"
        arms.append(
            (
                arm_id,
                HdbscanParams(
                    min_cluster_size=mcs,
                    min_samples=ms,
                    cluster_selection_method=method,
                    cluster_selection_epsilon=eps,
                ),
            )
        )
    return arms


async def run_clustering_sweep(
    *,
    session: AsyncSession,
    embedding_client: EmbeddingClient,
    embedding_config: ProviderConfig,
    job_id: str,
    text_variants: list[EmbeddingTextVariant],
    thresholds: EligibilityThresholds,
) -> ClusteringSweepArtifact:
    """Run the structure-only clustering sweep for one completed job."""
    records = await load_rule_records(session, job_id)
    if len(records) < 2:
        msg = (
            f"Job {job_id} has {len(records)} micro-rules; need at least 2 to cluster. "
            "Run the pipeline through rule extraction first."
        )
        raise ValueError(msg)

    rule_labels = [record.label for record in records]
    has_hypotheses = any(record.hypothesis is not None for record in records)
    rule_hypotheses: list[str] | None = None
    if has_hypotheses:
        rule_hypotheses = [
            (record.hypothesis or "none")[:_HYPOTHESIS_KEY_MAX_CHARS] for record in records
        ]

    artifact = ClusteringSweepArtifact(
        job_id=job_id,
        generated_at=datetime.now(UTC).isoformat(),
        rule_count=len(records),
        label_counts=dict(Counter(rule_labels)),
        thresholds=thresholds,
        notes=[
            "Phase A caveat: the pipeline embeds rule_only text, so only rule_only-variant "
            "arms may be promoted into task.yaml clustering params. Other variants are "
            "evidence for a Phase B pipeline change.",
        ],
    )

    for variant in text_variants:
        if variant == "label_plus_hypothesis_plus_rule" and not has_hypotheses:
            artifact.notes.append(
                "Skipped variant label_plus_hypothesis_plus_rule: no rule has a hypothesis."
            )
            continue

        texts = build_variant_texts(records, variant)
        embedding_result = await embedding_client.embed_texts(texts=texts, config=embedding_config)
        embeddings = embedding_result.embeddings
        norm_min, norm_median, norm_max = vector_norm_stats(embeddings)
        artifact.embedding_setups.append(
            EmbeddingSetupReport(
                text_variant=variant,
                embedding_model=embedding_config.model,
                provider=embedding_config.provider,
                provider_profile=embedding_config.profile_name,
                dimension=len(embeddings[0]) if embeddings else 0,
                vectors_prenormalized=(
                    abs(norm_min - 1.0) <= _PRENORMALIZED_TOLERANCE
                    and abs(norm_max - 1.0) <= _PRENORMALIZED_TOLERANCE
                ),
                vector_norm_min=norm_min,
                vector_norm_median=norm_median,
                vector_norm_max=norm_max,
                rule_count=len(records),
                duplicate_rule_rate=duplicate_rule_rate(texts),
            )
        )

        distance_matrix = cosine_distance_matrix(embeddings)
        eps_candidates = {
            min_samples: derive_eps_candidates(distance_matrix, min_samples)
            for min_samples in DBSCAN_MIN_SAMPLES_GRID
        }
        artifact.k_distance_reports.append(
            KDistanceReport(
                text_variant=variant,
                rows=k_distance_report(distance_matrix),
                eps_candidates_by_min_samples={
                    str(min_samples): [eps for _, eps in candidates]
                    for min_samples, candidates in eps_candidates.items()
                },
            )
        )

        for arm_id, dbscan_params, percentile in _dbscan_arms(variant, eps_candidates):
            labels = dbscan_labels(
                distance_matrix,
                eps=dbscan_params.eps,
                min_samples=dbscan_params.min_samples,
            )
            health = compute_cluster_health(labels, distance_matrix, rule_labels, rule_hypotheses)
            artifact.arms.append(
                ClusteringSweepArm(
                    arm_id=arm_id,
                    algorithm="dbscan",
                    text_variant=variant,
                    dbscan_params=dbscan_params,
                    eps_source_percentile=percentile,
                    health=health,
                    eligibility=evaluate_eligibility(health, thresholds),
                )
            )

        for arm_id, hdbscan_params in _hdbscan_arms(variant):
            try:
                labels = hdbscan_labels(
                    distance_matrix,
                    min_cluster_size=hdbscan_params.min_cluster_size,
                    min_samples=hdbscan_params.min_samples,
                    cluster_selection_method=hdbscan_params.cluster_selection_method,
                    cluster_selection_epsilon=hdbscan_params.cluster_selection_epsilon,
                )
            except (ValueError, RuntimeError) as exc:
                artifact.notes.append(f"Arm {arm_id} failed: {exc}")
                logger.warning("clustering_sweep_arm_failed", arm_id=arm_id, error=str(exc))
                continue
            health = compute_cluster_health(labels, distance_matrix, rule_labels, rule_hypotheses)
            artifact.arms.append(
                ClusteringSweepArm(
                    arm_id=arm_id,
                    algorithm="hdbscan",
                    text_variant=variant,
                    hdbscan_params=hdbscan_params,
                    health=health,
                    eligibility=evaluate_eligibility(health, thresholds),
                )
            )

        logger.info(
            "clustering_sweep_variant_done",
            job_id=job_id,
            text_variant=variant,
            arms=len(artifact.arms),
        )

    return artifact


def rank_eligible_arms(artifact: ClusteringSweepArtifact) -> list[ClusteringSweepArm]:
    """Eligible arms sorted by mean label entropy asc, then noise pct asc."""
    eligible = [arm for arm in artifact.arms if arm.eligibility.eligible]
    return sorted(
        eligible,
        key=lambda arm: (
            arm.health.label_entropy_mean if arm.health.label_entropy_mean is not None else 999.0,
            arm.health.noise_pct,
        ),
    )


def write_clustering_sweep_json(output_path: Path, artifact: ClusteringSweepArtifact) -> Path:
    """Write the sweep artifact as JSON, creating parent directories."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    return output_path


def log_sweep_to_mlflow(artifact: ClusteringSweepArtifact, output_dir: Path) -> None:
    """Log the sweep as a single MLflow run: params, summary metrics, artifact dir."""
    from rulekiln.config.settings import get_settings
    from rulekiln.integrations.mlflow_tracker import (
        create_run,
        log_artifacts_dir,
        log_metrics,
        log_params,
    )

    settings = get_settings()
    tracking_uri = settings.mlflow_tracking_uri
    run_id = create_run(
        tracking_uri,
        settings.mlflow_experiment_name,
        job_id=artifact.job_id,
        task_id="clustering_sweep",
        task_name="clustering-sweep",
    )

    setups = artifact.embedding_setups
    log_params(
        tracking_uri,
        run_id,
        {
            "sweep.job_id": artifact.job_id,
            "sweep.schema_version": artifact.schema_version,
            "sweep.embedding_model": setups[0].embedding_model if setups else "unknown",
            "sweep.provider_profile": setups[0].provider_profile if setups else "unknown",
            "sweep.text_variants": ",".join(s.text_variant for s in setups),
            "sweep.rule_count": str(artifact.rule_count),
        },
    )

    metrics: dict[str, float] = {
        "sweep.arms_total": float(len(artifact.arms)),
        "sweep.arms_eligible": float(
            sum(1 for arm in artifact.arms if arm.eligibility.eligible)
        ),
    }
    ranked = rank_eligible_arms(artifact)
    for algorithm in ("dbscan", "hdbscan"):
        best = next((arm for arm in ranked if arm.algorithm == algorithm), None)
        if best is None:
            continue
        metrics[f"sweep.best_{algorithm}.cluster_count"] = float(
            best.health.cluster_count_excluding_noise
        )
        metrics[f"sweep.best_{algorithm}.noise_pct"] = best.health.noise_pct
        metrics[f"sweep.best_{algorithm}.largest_cluster_pct"] = best.health.largest_cluster_pct
    log_metrics(tracking_uri, run_id, metrics)
    log_artifacts_dir(tracking_uri, run_id, output_dir)
