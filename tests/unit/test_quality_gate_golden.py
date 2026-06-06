"""Tests confirming the golden quality gate fires only when golden cases are evaluated."""

from __future__ import annotations

from rulekiln.config.settings import QualityGateDefaults
from rulekiln.pipeline.quality_gates import check_quality_gates
from rulekiln.schemas.pipeline import CaseEvalResult, EvalResult
from rulekiln.schemas.task_case import EvaluationSpec, RuleKilnCase


def _eval_result(case_results: list[CaseEvalResult], strategy: str = "dbscan") -> EvalResult:
    return EvalResult(
        strategy=strategy,
        model="fake",
        split="validation",
        weighted_case_score=1.0,
        malformed_output_rate=0.0,
        case_results=case_results,
    )


def _case_result(case_id: str, passed: bool) -> CaseEvalResult:
    return CaseEvalResult(
        case_id=case_id,
        score=1.0 if passed else 0.0,
        passed=passed,
        malformed=False,
        assertion_scores={},
        actual_output=None,
    )


def _case(case_id: str, split: str) -> RuleKilnCase:
    return RuleKilnCase(
        id=case_id,
        split=split,  # type: ignore[arg-type]
        task_mode="classification",
        input={"text": f"input {case_id}"},
        expected="yes",
        evaluation=EvaluationSpec(assertions=[]),
    )


def _defaults() -> QualityGateDefaults:
    return QualityGateDefaults(max_golden_failures=0)


# ── original behavior (documenting the bug) ──────────────────────────────────

def test_golden_gate_silent_when_golden_cases_absent_from_eval_cases() -> None:
    """Gate passes with golden_failures=0 when golden cases were never evaluated.

    This documents the pre-fix behavior: if golden cases are not present in
    distilled_eval.case_results, the gate counts zero golden failures and passes.
    With the fix the caller must pass golden_eval separately.
    """
    golden_case = _case("golden-1", "golden")
    val_case = _case("val-1", "validation")

    # distilled_eval only covers the validation case — golden case is absent
    distilled_eval = _eval_result([_case_result("val-1", passed=True)])

    result = check_quality_gates(
        strategy="dbscan",
        distilled_eval=distilled_eval,
        baseline_eval=None,
        cases=[val_case, golden_case],
        task_mode="classification",
        task_gates={},
        settings_defaults=_defaults(),
    )

    # Gate cannot detect the golden failure because it was never evaluated
    assert result.golden_failures == 0
    assert result.passed is True


# ── fixed behavior ────────────────────────────────────────────────────────────

def test_golden_gate_fires_when_golden_eval_provided_with_failure() -> None:
    """Gate correctly detects a golden failure via the separate golden_eval."""
    golden_case = _case("golden-1", "golden")
    val_case = _case("val-1", "validation")

    distilled_eval = _eval_result([_case_result("val-1", passed=True)])
    golden_eval = _eval_result(
        [_case_result("golden-1", passed=False)],
        strategy="dbscan_golden",
    )

    result = check_quality_gates(
        strategy="dbscan",
        distilled_eval=distilled_eval,
        baseline_eval=None,
        cases=[val_case, golden_case],
        task_mode="classification",
        task_gates={},
        settings_defaults=_defaults(),
        golden_eval=golden_eval,
    )

    assert result.golden_failures == 1
    assert result.passed is False
    assert any("golden_failures" in v for v in result.violations)


def test_golden_gate_passes_when_golden_cases_all_pass() -> None:
    golden_case = _case("golden-1", "golden")

    distilled_eval = _eval_result([_case_result("val-1", passed=True)])
    golden_eval = _eval_result(
        [_case_result("golden-1", passed=True)],
        strategy="dbscan_golden",
    )

    result = check_quality_gates(
        strategy="dbscan",
        distilled_eval=distilled_eval,
        baseline_eval=None,
        cases=[_case("val-1", "validation"), golden_case],
        task_mode="classification",
        task_gates={},
        settings_defaults=_defaults(),
        golden_eval=golden_eval,
    )

    assert result.golden_failures == 0
    assert result.passed is True


def test_golden_gate_uses_golden_eval_not_distilled_eval() -> None:
    """When both contain the same case ID, golden_eval is authoritative for golden cases."""
    golden_case = _case("shared-1", "golden")

    # distilled_eval claims the case passed (e.g. it was accidentally in eval_cases)
    distilled_eval = _eval_result([_case_result("shared-1", passed=True)])
    # golden_eval says it failed — this should be the gate's source of truth
    golden_eval = _eval_result(
        [_case_result("shared-1", passed=False)],
        strategy="dbscan_golden",
    )

    result = check_quality_gates(
        strategy="dbscan",
        distilled_eval=distilled_eval,
        baseline_eval=None,
        cases=[golden_case],
        task_mode="classification",
        task_gates={},
        settings_defaults=_defaults(),
        golden_eval=golden_eval,
    )

    assert result.golden_failures == 1
    assert result.passed is False


def test_golden_gate_threshold_respected() -> None:
    """max_golden_failures > 0 should allow some golden failures."""
    golden_cases = [_case(f"g-{i}", "golden") for i in range(3)]

    distilled_eval = _eval_result([])
    golden_eval = _eval_result(
        [_case_result(f"g-{i}", passed=(i > 0)) for i in range(3)],
        strategy="dbscan_golden",
    )

    result = check_quality_gates(
        strategy="dbscan",
        distilled_eval=distilled_eval,
        baseline_eval=None,
        cases=golden_cases,
        task_mode="classification",
        task_gates={"max_golden_failures": 1},
        settings_defaults=_defaults(),
        golden_eval=golden_eval,
    )

    # Exactly 1 failure, threshold is 1 → should pass
    assert result.golden_failures == 1
    assert result.passed is True


def test_golden_gate_no_golden_cases_always_passes() -> None:
    """No golden cases in the task → gate passes regardless of golden_eval argument."""
    distilled_eval = _eval_result([_case_result("val-1", passed=True)])

    result = check_quality_gates(
        strategy="dbscan",
        distilled_eval=distilled_eval,
        baseline_eval=None,
        cases=[_case("val-1", "validation")],
        task_mode="classification",
        task_gates={},
        settings_defaults=_defaults(),
    )

    assert result.golden_failures == 0
    assert result.passed is True
