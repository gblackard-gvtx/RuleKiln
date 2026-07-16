"""Stale-artifact detection (Work Package 4).

Checks whether the hashes/predictions freshly computed for this invocation still match
what a prior invocation of the same job persisted to ``outputs/strategy_comparison.json``
and ``metadata/manifest.json``. Reuses the existing immutable-snapshot artifacts rather
than re-deriving config hashes from a live re-read of ``task.yaml`` or inferring strategy
identity from filenames.

Like :mod:`rulekiln.pipeline.provenance_validation`, this never changes selection — it
only records findings. Promotion consumes ``report.has_errors`` as a blocker (see
:mod:`rulekiln.pipeline.promotion`).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from rulekiln.artifacts.canonical_hash import canonical_hash
from rulekiln.artifacts.manifest import RunManifest, verify_manifest
from rulekiln.pipeline.artifact_hashing import evaluation_hash, prediction_hash
from rulekiln.schemas.pipeline import EvalResult, PairedComparisonSummary, StrategyComparison
from rulekiln.schemas.stale_artifact import StaleArtifactFinding, StaleArtifactValidationReport


def _load_strategy_comparison(artifact_root: Path) -> StrategyComparison | None:
    path = artifact_root / "outputs" / "strategy_comparison.json"
    if not path.exists():
        return None
    return StrategyComparison.model_validate_json(path.read_text(encoding="utf-8"))


def _load_manifest(artifact_root: Path) -> RunManifest | None:
    path = artifact_root / "metadata" / "manifest.json"
    if not path.exists():
        return None
    return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _recompute_paired_summary(
    *,
    baseline: EvalResult,
    candidate: EvalResult,
    baseline_strategy_id: str,
    candidate_strategy_id: str,
) -> PairedComparisonSummary:
    """Recompute paired-comparison counts from authoritative per-case ``passed`` flags.

    Counts only depend on case membership, not ordering, so this reproduces the same
    summary as :func:`rulekiln.pipeline.statistics.compute_paired_comparison` without
    needing the original label/prediction text.
    """
    baseline_by_id = {case.case_id: case.passed for case in baseline.case_results}
    candidate_by_id = {case.case_id: case.passed for case in candidate.case_results}
    common_ids = baseline_by_id.keys() & candidate_by_id.keys()

    fixed_count = 0
    broken_count = 0
    unchanged_correct_count = 0
    unchanged_wrong_count = 0
    for case_id in common_ids:
        base_passed = baseline_by_id[case_id]
        cand_passed = candidate_by_id[case_id]
        if not base_passed and cand_passed:
            fixed_count += 1
        elif base_passed and not cand_passed:
            broken_count += 1
        elif base_passed and cand_passed:
            unchanged_correct_count += 1
        else:
            unchanged_wrong_count += 1

    total_cases = len(common_ids)
    changed_outcomes = fixed_count + broken_count
    if changed_outcomes == 0:
        net_fix_rate: float | None = None
        net_fix_rate_status = "no_changed_outcomes"
    else:
        net_fix_rate = (fixed_count - broken_count) / changed_outcomes
        net_fix_rate_status = "ok"
    overall_net_fix_rate = (fixed_count - broken_count) / total_cases if total_cases > 0 else 0.0

    return PairedComparisonSummary(
        baseline_strategy_id=baseline_strategy_id,
        candidate_strategy_id=candidate_strategy_id,
        fixed_count=fixed_count,
        broken_count=broken_count,
        unchanged_correct_count=unchanged_correct_count,
        unchanged_wrong_count=unchanged_wrong_count,
        total_cases=total_cases,
        net_fix_rate=net_fix_rate,
        net_fix_rate_status=net_fix_rate_status,
        overall_net_fix_rate=overall_net_fix_rate,
    )


def validate_stale_artifacts(
    *,
    job_id: str,
    artifact_root: Path,
    eval_results_by_strategy: Mapping[str, EvalResult],
) -> StaleArtifactValidationReport:
    """Cross-check freshly recomputed hashes against persisted artifacts.

    A missing ``strategy_comparison.json``/``manifest.json`` (first invocation for this
    job) is not an error — the corresponding checks are simply skipped.
    """
    findings: list[StaleArtifactFinding] = []

    comparison = _load_strategy_comparison(artifact_root)
    if comparison is not None:
        rows_by_strategy = {row.strategy: row for row in comparison.strategy_provenance}
        for strategy, eval_result in eval_results_by_strategy.items():
            row = rows_by_strategy.get(strategy)
            if row is None:
                continue

            if row.evaluation_hash is not None:
                recomputed = evaluation_hash(eval_result)
                if recomputed != row.evaluation_hash:
                    findings.append(
                        StaleArtifactFinding(
                            code="EVALUATION_HASH_MISMATCH",
                            severity="error",
                            message=(
                                f"Recomputed evaluation hash for strategy {strategy!r} does not "
                                "match the persisted hash."
                            ),
                            expected={"evaluation_hash": row.evaluation_hash},
                            actual={"evaluation_hash": recomputed},
                            artifact="outputs/strategy_comparison.json",
                            strategy=strategy,
                        )
                    )

            if row.prediction_hash is not None:
                recomputed_pred = prediction_hash(eval_result)
                if recomputed_pred != row.prediction_hash:
                    findings.append(
                        StaleArtifactFinding(
                            code="PREDICTION_HASH_MISMATCH",
                            severity="error",
                            message=(
                                f"Recomputed prediction hash for strategy {strategy!r} does not "
                                "match the persisted hash."
                            ),
                            expected={"prediction_hash": row.prediction_hash},
                            actual={"prediction_hash": recomputed_pred},
                            artifact="outputs/strategy_comparison.json",
                            strategy=strategy,
                        )
                    )

        persisted_paired = comparison.paired_comparison
        if persisted_paired is not None:
            baseline = eval_results_by_strategy.get(persisted_paired.baseline_strategy_id)
            candidate = eval_results_by_strategy.get(persisted_paired.candidate_strategy_id)
            if baseline is not None and candidate is not None:
                recomputed_summary = _recompute_paired_summary(
                    baseline=baseline,
                    candidate=candidate,
                    baseline_strategy_id=persisted_paired.baseline_strategy_id,
                    candidate_strategy_id=persisted_paired.candidate_strategy_id,
                )
                recomputed_hash = canonical_hash(recomputed_summary.model_dump(mode="json"))
                persisted_hash = canonical_hash(persisted_paired.model_dump(mode="json"))
                if recomputed_hash != persisted_hash:
                    findings.append(
                        StaleArtifactFinding(
                            code="COMPARISON_HASH_MISMATCH",
                            severity="error",
                            message=(
                                "Recomputed paired-comparison summary does not match the "
                                "persisted summary."
                            ),
                            expected={"paired_comparison_hash": persisted_hash},
                            actual={"paired_comparison_hash": recomputed_hash},
                            artifact="outputs/strategy_comparison.json",
                        )
                    )

    manifest = _load_manifest(artifact_root)
    if manifest is not None:
        for mismatched_path in verify_manifest(artifact_root, manifest):
            findings.append(
                StaleArtifactFinding(
                    code="MANIFEST_HASH_MISMATCH",
                    severity="error",
                    message=(
                        f"On-disk artifact {mismatched_path!r} no longer matches the manifest "
                        "SHA-256, or is missing."
                    ),
                    artifact=mismatched_path,
                )
            )

    return StaleArtifactValidationReport(job_id=job_id, findings=findings)


def write_stale_artifact_validation(root: Path, report: StaleArtifactValidationReport) -> Path:
    """Write metadata/stale_artifact_validation.json."""
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    path = metadata / "stale_artifact_validation.json"
    path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def read_stale_artifact_validation(artifact_root: Path) -> StaleArtifactValidationReport | None:
    """Load a persisted stale-artifact validation report, or ``None`` for legacy runs."""
    path = artifact_root / "metadata" / "stale_artifact_validation.json"
    if not path.exists():
        return None
    return StaleArtifactValidationReport.model_validate(
        json.loads(path.read_text(encoding="utf-8"))
    )
