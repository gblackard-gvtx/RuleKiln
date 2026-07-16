"""Failure-visibility tests for ArtifactExportService._write_provenance_and_promotion.

Each of the four independent integrity checks (hash/provenance-row
construction, provenance validation, promotion evaluation, split-leakage
validation) must fail without cascading into the others, and every failure
must be persisted to metadata/integrity_check_failures.json rather than only
logged. See Work Package 3 of docs plan "Dataset integrity and authoritative
evaluation".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast
from unittest.mock import patch

from sqlalchemy.ext.asyncio import AsyncSession

from rulekiln.config.settings import AppSettings
from rulekiln.pipeline.split_policy import resolve_split_policy
from rulekiln.providers.tracking import ModelCallCollector
from rulekiln.schemas.job import DistillationRequest
from rulekiln.schemas.pipeline import CaseEvalResult, EvalResult, StrategyComparison
from rulekiln.schemas.task_case import ModelRoute, RuleKilnCase, RuleKilnTask
from rulekiln.workers.pipeline.context import PipelineContext
from rulekiln.workers.pipeline.output_stage_services import ArtifactExportService


def _task() -> RuleKilnTask:
    return RuleKilnTask(
        task_id="t1",
        task_name="Test",
        task_mode="classification",
        description="d",
        input_template="{x}",
    )


def _case(case_id: str) -> RuleKilnCase:
    return RuleKilnCase(
        id=case_id,
        split="validation",
        task_mode="classification",
        input={"x": "v"},
        expected={"label": "a"},
    )


def _eval(strategy: str) -> EvalResult:
    return EvalResult(
        strategy=strategy,
        model="m",
        split="validation",
        macro_f1=0.5,
        accuracy=0.5,
        case_results=[CaseEvalResult(case_id="c1", score=1.0, passed=True)],
    )


def _route() -> ModelRoute:
    return ModelRoute(provider_profile="local", model="m")


def _ctx(tmp_path: Path) -> PipelineContext:
    cases = [_case("c1")]
    payload = DistillationRequest(
        task=_task(), cases=cases, teacher=_route(), student=_route(), embedding=_route()
    )
    settings = AppSettings(
        DATABASE_URL="sqlite+aiosqlite://",
        MLFLOW_TRACKING_URI="file:///tmp/mlflow-ci",
        ARTIFACT_ROOT=str(tmp_path),
    )
    split_policy = resolve_split_policy(cases)
    return PipelineContext(
        job_id="job-1",
        payload=payload,
        settings=settings,
        task=_task(),
        cases=cases,
        case_by_id={c.id: c for c in cases},
        split_policy=split_policy,
        extraction_cases=split_policy.extraction_cases,
        eval_cases=split_policy.evaluation_cases,
        eval_split=split_policy.evaluation_split,
        golden_cases=[],
        primary_metric="macro_f1",
        dataset_id="cases_sha256:test",
        collector=ModelCallCollector(),
        evaluation_bootstrap_enabled=False,
        evaluation_bootstrap_iterations=100,
        evaluation_bootstrap_seed_offset=0,
    )


def _comparison() -> StrategyComparison:
    return StrategyComparison(strategy_evals={"baseline_scaffold": _eval("baseline_scaffold")})


def _service() -> ArtifactExportService:
    return ArtifactExportService(session=cast(AsyncSession, None))


def _read_failures(artifact_root: Path) -> dict[str, object]:
    path = artifact_root / "metadata" / "integrity_check_failures.json"
    assert path.exists()
    return json.loads(path.read_text(encoding="utf-8"))


def test_happy_path_writes_no_failures_artifact(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    comparison = _comparison()
    written = _service()._write_provenance_and_promotion(
        ctx, tmp_path, comparison, "baseline_scaffold", {}
    )
    assert not (tmp_path / "metadata" / "integrity_check_failures.json").exists()
    assert comparison.promotion_status is not None
    assert comparison.provenance_validation is not None
    assert written
    # baseline_scaffold is not distilled-family: promotion is not_applicable, not
    # "blocked" against distilled-vs-baseline blockers that carry no signal for it.
    assert comparison.promotion_status.promotion_status == "not_applicable"
    assert comparison.promotion_status.promotion_candidate is None
    assert comparison.promotion_status.selected_strategy_family == "baseline"
    assert [b.code for b in comparison.promotion_status.promotion_blockers] == [
        "NO_ELIGIBLE_DISTILLED_STRATEGY"
    ]


def test_hash_construction_failure_returns_early_with_failures_artifact(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    comparison = _comparison()
    with patch(
        "rulekiln.workers.pipeline.output_stage_services.build_job_hashes",
        side_effect=RuntimeError("boom"),
    ):
        written = _service()._write_provenance_and_promotion(
            ctx, tmp_path, comparison, "baseline_scaffold", {}
        )
    report = _read_failures(tmp_path)
    failures = cast(list[dict[str, str]], report["failures"])
    assert len(failures) == 1
    assert failures[0]["stage"] == "hash_and_provenance_row_construction"
    assert failures[0]["error_type"] == "RuntimeError"
    assert len(written) == 1
    # Downstream checks never ran: comparison was not enriched.
    assert comparison.promotion_status is None
    assert comparison.provenance_validation is None


def test_provenance_validation_failure_defaults_to_none_and_continues(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    comparison = _comparison()
    with patch(
        "rulekiln.workers.pipeline.output_stage_services.validate_provenance",
        side_effect=RuntimeError("boom"),
    ):
        _service()._write_provenance_and_promotion(
            ctx, tmp_path, comparison, "baseline_scaffold", {}
        )
    report = _read_failures(tmp_path)
    failures = cast(list[dict[str, str]], report["failures"])
    assert len(failures) == 1
    assert failures[0]["stage"] == "provenance_validation"
    assert comparison.provenance_validation is None
    # Promotion evaluation still ran (fail-closed: provenance_status becomes "missing").
    assert comparison.promotion_status is not None
    assert not (tmp_path / "metadata" / "provenance_validation.json").exists()


def test_stale_artifact_validation_failure_defaults_to_none_and_continues(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    comparison = _comparison()
    with patch(
        "rulekiln.workers.pipeline.output_stage_services.validate_stale_artifacts",
        side_effect=RuntimeError("boom"),
    ):
        _service()._write_provenance_and_promotion(
            ctx, tmp_path, comparison, "baseline_scaffold", {}
        )
    report = _read_failures(tmp_path)
    failures = cast(list[dict[str, str]], report["failures"])
    assert len(failures) == 1
    assert failures[0]["stage"] == "stale_artifact_validation"
    # Promotion evaluation still ran (fail-closed: stale_artifact_status becomes "missing").
    assert comparison.promotion_status is not None
    assert not (tmp_path / "metadata" / "stale_artifact_validation.json").exists()


def test_promotion_evaluation_failure_falls_back_to_not_evaluated(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    comparison = _comparison()
    with patch(
        "rulekiln.workers.pipeline.output_stage_services.evaluate_promotion",
        side_effect=RuntimeError("boom"),
    ):
        _service()._write_provenance_and_promotion(
            ctx, tmp_path, comparison, "baseline_scaffold", {}
        )
    report = _read_failures(tmp_path)
    failures = cast(list[dict[str, str]], report["failures"])
    assert len(failures) == 1
    assert failures[0]["stage"] == "promotion_evaluation"
    assert comparison.promotion_status is not None
    assert comparison.promotion_status.promotion_status == "not_evaluated"


def test_split_leakage_validation_failure_defaults_to_none_and_continues(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    comparison = _comparison()
    with patch(
        "rulekiln.workers.pipeline.output_stage_services.validate_split_leakage",
        side_effect=RuntimeError("boom"),
    ):
        _service()._write_provenance_and_promotion(
            ctx, tmp_path, comparison, "baseline_scaffold", {}
        )
    report = _read_failures(tmp_path)
    failures = cast(list[dict[str, str]], report["failures"])
    assert len(failures) == 1
    assert failures[0]["stage"] == "split_leakage_validation"
    assert not (tmp_path / "metadata" / "split_leakage_validation.json").exists()
    # Other artifacts (strategy comparison, hashes, promotion) still written.
    assert comparison.promotion_status is not None


def test_multiple_independent_failures_all_recorded(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    comparison = _comparison()
    with (
        patch(
            "rulekiln.workers.pipeline.output_stage_services.validate_provenance",
            side_effect=RuntimeError("provenance boom"),
        ),
        patch(
            "rulekiln.workers.pipeline.output_stage_services.validate_split_leakage",
            side_effect=RuntimeError("leakage boom"),
        ),
    ):
        _service()._write_provenance_and_promotion(
            ctx, tmp_path, comparison, "baseline_scaffold", {}
        )
    report = _read_failures(tmp_path)
    failures = cast(list[dict[str, str]], report["failures"])
    stages = {f["stage"] for f in failures}
    assert stages == {"provenance_validation", "split_leakage_validation"}
