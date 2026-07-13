"""Provenance mismatch detection (Workstream F).

Runs *before* promotion is decided and produces a structured
:class:`ProvenanceValidationReport`. The rule the integrity review demanded:

* **Selection** may still identify the best result for exploratory analysis even when
  provenance validation fails — selection is governed by quality gates only.
* **Promotion** must be blocked when provenance validation contains errors.

So this module never changes selection; it only records findings. Promotion consumes
``report.has_errors`` as a blocker (see :mod:`rulekiln.pipeline.promotion`).

Findings are read from persisted artifacts on disk so the validation is reproducible from
the run directory alone (not from live, in-memory state that a rerun could not see).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from rulekiln.artifacts.canonical_hash import canonical_hash, hash_text
from rulekiln.pipeline.approved_configs import load_approved_config
from rulekiln.pipeline.artifact_hashing import clustering_config_hash, ordered_case_ids_hash
from rulekiln.schemas.promotion import ProvenanceFinding, ProvenanceValidationReport
from rulekiln.schemas.provenance import ClusterAssignmentRecord, ClusteringProvenance

_DISTILLED_STRATEGIES = ("dbscan", "hdbscan")


def _clustering_dir(artifact_root: Path, algorithm: str) -> Path:
    return artifact_root / "outputs" / "clustering" / algorithm


def load_clustering_provenance(
    artifact_root: Path, algorithm: str
) -> ClusteringProvenance | None:
    """Load a persisted clustering provenance object, or ``None`` if absent."""
    path = _clustering_dir(artifact_root, algorithm) / "clustering_provenance.json"
    if not path.exists():
        return None
    return ClusteringProvenance.model_validate_json(path.read_text(encoding="utf-8"))


def _load_assignments(artifact_root: Path, algorithm: str) -> list[ClusterAssignmentRecord]:
    path = _clustering_dir(artifact_root, algorithm) / "cluster_assignments.jsonl"
    if not path.exists():
        return []
    records: list[ClusterAssignmentRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            records.append(ClusterAssignmentRecord.model_validate_json(stripped))
    return records


def _hash_assignment_records(records: Sequence[ClusterAssignmentRecord]) -> str:
    return canonical_hash([record.model_dump(mode="json") for record in records])


def validate_provenance(
    *,
    job_id: str,
    artifact_root: Path,
    ordered_case_ids: Sequence[str],
    compiled_prompts: Mapping[str, str],
    prompt_hash_by_strategy: Mapping[str, str | None],
    prediction_case_ids_by_strategy: Mapping[str, Sequence[str]],
    strategies: Sequence[str] = _DISTILLED_STRATEGIES,
) -> ProvenanceValidationReport:
    """Cross-check persisted provenance artifacts and return structured findings.

    Error findings (data-integrity failures) block promotion. Config-mismatch findings —
    a run that used a config different from the approved one — are *warnings* here because
    they do not indicate corruption; promotion applies its own approved-config blocker.
    """
    findings: list[ProvenanceFinding] = []
    expected_case_set_hash = ordered_case_ids_hash(list(ordered_case_ids))

    for strategy in strategies:
        provenance = load_clustering_provenance(artifact_root, strategy)
        if provenance is None:
            continue
        rel = f"outputs/clustering/{strategy}/clustering_provenance.json"

        # Cross-job artifact: provenance must belong to this job.
        if provenance.job_id is not None and provenance.job_id != job_id:
            findings.append(
                ProvenanceFinding(
                    code="CROSS_JOB_ARTIFACT",
                    severity="error",
                    message=(
                        f"Clustering provenance for {strategy!r} belongs to job "
                        f"{provenance.job_id!r}, not the current job {job_id!r}."
                    ),
                    expected={"job_id": job_id},
                    actual={"job_id": provenance.job_id},
                    artifact=rel,
                )
            )

        # Cluster-assignment hash must match the persisted assignment file.
        assignments = _load_assignments(artifact_root, strategy)
        actual_assignment_hash = _hash_assignment_records(assignments)
        recorded_assignment_hash = provenance.hashes.cluster_assignment_hash
        if (
            recorded_assignment_hash is not None
            and recorded_assignment_hash != actual_assignment_hash
        ):
            findings.append(
                ProvenanceFinding(
                    code="CLUSTER_ASSIGNMENT_HASH_MISMATCH",
                    severity="error",
                    message=(
                        f"Recorded cluster-assignment hash for {strategy!r} does not match "
                        "the persisted cluster_assignments.jsonl."
                    ),
                    expected={"cluster_assignment_hash": recorded_assignment_hash},
                    actual={"cluster_assignment_hash": actual_assignment_hash},
                    artifact=f"outputs/clustering/{strategy}/cluster_assignments.jsonl",
                )
            )

        # Approved-config comparison (warning: informs promotion, not a corruption).
        approved = load_approved_config(strategy)
        if approved is not None and approved.config_hash is not None:
            runtime_hash = clustering_config_hash(provenance)
            if runtime_hash != approved.config_hash:
                eff = provenance.effective_parameters
                findings.append(
                    ProvenanceFinding(
                        code="CLUSTER_CONFIG_MISMATCH",
                        severity="warning",
                        message=(
                            f"Runtime {strategy!r} clustering config does not match the "
                            "approved/promoted config."
                        ),
                        expected=approved.config.model_dump(mode="json"),
                        actual={
                            "min_cluster_size": eff.min_cluster_size,
                            "min_samples": eff.min_samples,
                            "eps": eff.eps,
                            "cluster_selection_method": eff.cluster_selection_method,
                            "cluster_selection_epsilon": eff.cluster_selection_epsilon,
                            "text_variant": provenance.text_variant,
                        },
                        artifact=rel,
                    )
                )

    # Prompt-hash integrity: the recorded per-strategy prompt hash must match the prompt
    # text actually compiled for that strategy.
    for strategy, recorded_prompt_hash in prompt_hash_by_strategy.items():
        prompt_text = compiled_prompts.get(strategy)
        if recorded_prompt_hash is None or prompt_text is None:
            continue
        actual_prompt_hash = hash_text(prompt_text)
        if actual_prompt_hash != recorded_prompt_hash:
            findings.append(
                ProvenanceFinding(
                    code="PROMPT_HASH_MISMATCH",
                    severity="error",
                    message=(
                        f"Recorded compiled-prompt hash for {strategy!r} does not match the "
                        "persisted prompt text."
                    ),
                    expected={"compiled_prompt_hash": recorded_prompt_hash},
                    actual={"compiled_prompt_hash": actual_prompt_hash},
                    artifact=f"outputs/prompts/{strategy}.txt",
                )
            )

    # Case-set integrity: every strategy's predictions must cover the same ordered case
    # set the job evaluated against.
    for strategy, case_ids in prediction_case_ids_by_strategy.items():
        actual_case_set_hash = ordered_case_ids_hash(list(case_ids))
        if actual_case_set_hash != expected_case_set_hash:
            findings.append(
                ProvenanceFinding(
                    code="CASE_SET_MISMATCH",
                    severity="error",
                    message=(
                        f"Prediction case set for {strategy!r} does not match the job's "
                        "ordered evaluation case set."
                    ),
                    expected={"ordered_case_ids_hash": expected_case_set_hash},
                    actual={"ordered_case_ids_hash": actual_case_set_hash},
                    artifact=f"outputs/{strategy}_eval.json",
                )
            )

    return ProvenanceValidationReport(job_id=job_id, findings=findings)


def read_provenance_validation(artifact_root: Path) -> ProvenanceValidationReport | None:
    """Load a persisted provenance validation report, or ``None`` for legacy runs."""
    path = artifact_root / "metadata" / "provenance_validation.json"
    if not path.exists():
        return None
    return ProvenanceValidationReport.model_validate(
        json.loads(path.read_text(encoding="utf-8"))
    )
