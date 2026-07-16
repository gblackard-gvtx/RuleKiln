"""Unit tests for stale-artifact validation (Work Package 4)."""

from __future__ import annotations

import json
from pathlib import Path

from rulekiln.artifacts.manifest import ManifestEntry, RunManifest
from rulekiln.pipeline.artifact_hashing import evaluation_hash, prediction_hash
from rulekiln.pipeline.stale_artifact_validation import (
    read_stale_artifact_validation,
    validate_stale_artifacts,
    write_stale_artifact_validation,
)
from rulekiln.schemas.pipeline import (
    CaseEvalResult,
    EvalResult,
    PairedComparisonSummary,
    StrategyComparison,
)
from rulekiln.schemas.provenance import StrategyProvenanceRow
from rulekiln.schemas.stale_artifact import StaleArtifactValidationReport


def _eval(strategy: str, passed: tuple[bool, ...] = (True, True)) -> EvalResult:
    return EvalResult(
        strategy=strategy,
        model="m",
        split="validation",
        case_results=[
            CaseEvalResult(case_id=f"c{i}", score=1.0 if p else 0.0, passed=p)
            for i, p in enumerate(passed)
        ],
    )


def _write_comparison(artifact_root: Path, comparison: StrategyComparison) -> None:
    outputs = artifact_root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "strategy_comparison.json").write_text(
        json.dumps(comparison.model_dump(mode="json")), encoding="utf-8"
    )


def test_first_invocation_produces_clean_report(tmp_path: Path) -> None:
    report = validate_stale_artifacts(
        job_id="job-1",
        artifact_root=tmp_path,
        eval_results_by_strategy={"baseline_scaffold": _eval("baseline_scaffold")},
    )
    assert report.findings == []
    assert not report.has_errors


def test_evaluation_hash_mismatch_detected(tmp_path: Path) -> None:
    eval_result = _eval("baseline_scaffold")
    row = StrategyProvenanceRow(
        strategy="baseline_scaffold",
        family="baseline",
        prediction_hash=prediction_hash(eval_result),
        evaluation_hash="stale-hash",
    )
    _write_comparison(tmp_path, StrategyComparison(strategy_provenance=[row]))

    report = validate_stale_artifacts(
        job_id="job-1",
        artifact_root=tmp_path,
        eval_results_by_strategy={"baseline_scaffold": eval_result},
    )
    codes = {f.code for f in report.findings}
    assert "EVALUATION_HASH_MISMATCH" in codes
    assert "PREDICTION_HASH_MISMATCH" not in codes
    assert report.has_errors


def test_prediction_hash_mismatch_detected(tmp_path: Path) -> None:
    eval_result = _eval("baseline_scaffold")
    row = StrategyProvenanceRow(
        strategy="baseline_scaffold",
        family="baseline",
        prediction_hash="stale-hash",
        evaluation_hash=evaluation_hash(eval_result),
    )
    _write_comparison(tmp_path, StrategyComparison(strategy_provenance=[row]))

    report = validate_stale_artifacts(
        job_id="job-1",
        artifact_root=tmp_path,
        eval_results_by_strategy={"baseline_scaffold": eval_result},
    )
    codes = {f.code for f in report.findings}
    assert codes == {"PREDICTION_HASH_MISMATCH"}


def test_matching_hashes_produce_no_findings(tmp_path: Path) -> None:
    eval_result = _eval("baseline_scaffold")
    row = StrategyProvenanceRow(
        strategy="baseline_scaffold",
        family="baseline",
        prediction_hash=prediction_hash(eval_result),
        evaluation_hash=evaluation_hash(eval_result),
    )
    _write_comparison(tmp_path, StrategyComparison(strategy_provenance=[row]))

    report = validate_stale_artifacts(
        job_id="job-1",
        artifact_root=tmp_path,
        eval_results_by_strategy={"baseline_scaffold": eval_result},
    )
    assert report.findings == []


def test_comparison_hash_mismatch_detected(tmp_path: Path) -> None:
    baseline = _eval("baseline_scaffold", passed=(True, False))
    candidate = _eval("dbscan", passed=(True, True))
    stale_paired = PairedComparisonSummary(
        baseline_strategy_id="baseline_scaffold",
        candidate_strategy_id="dbscan",
        fixed_count=99,
        broken_count=0,
        unchanged_correct_count=1,
        unchanged_wrong_count=0,
        total_cases=2,
        net_fix_rate=1.0,
        net_fix_rate_status="ok",
        overall_net_fix_rate=1.0,
    )
    _write_comparison(
        tmp_path, StrategyComparison(strategy_provenance=[], paired_comparison=stale_paired)
    )

    report = validate_stale_artifacts(
        job_id="job-1",
        artifact_root=tmp_path,
        eval_results_by_strategy={"baseline_scaffold": baseline, "dbscan": candidate},
    )
    codes = {f.code for f in report.findings}
    assert codes == {"COMPARISON_HASH_MISMATCH"}


def test_manifest_hash_mismatch_detected(tmp_path: Path) -> None:
    tracked = tmp_path / "outputs" / "eval_report.json"
    tracked.parent.mkdir(parents=True, exist_ok=True)
    tracked.write_text("original", encoding="utf-8")

    manifest = RunManifest(
        entries=[
            ManifestEntry(
                path="outputs/eval_report.json",
                artifact_type="eval_report",
                sha256="0" * 64,
                size_bytes=8,
                created_at="2026-01-01T00:00:00+00:00",
            )
        ]
    )
    metadata = tmp_path / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")

    report = validate_stale_artifacts(
        job_id="job-1", artifact_root=tmp_path, eval_results_by_strategy={}
    )
    assert len(report.findings) == 1
    assert report.findings[0].code == "MANIFEST_HASH_MISMATCH"
    assert report.findings[0].artifact == "outputs/eval_report.json"


def test_write_and_read_round_trip(tmp_path: Path) -> None:
    report = StaleArtifactValidationReport(job_id="job-1", findings=[])
    write_stale_artifact_validation(tmp_path, report)
    loaded = read_stale_artifact_validation(tmp_path)
    assert loaded == report


def test_read_returns_none_when_absent(tmp_path: Path) -> None:
    assert read_stale_artifact_validation(tmp_path) is None
