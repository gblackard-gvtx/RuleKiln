"""Boundary tests for the malformed-output quality gate.

Motivated by the ContractNLI HDBSCAN validation win
(docs/investigations/contractnli_hdbscan_validation_win.md): the winning run's
malformed rate was exactly 3/300 == 0.01, sitting precisely on the configured
gate threshold. The gate uses a strict ``>`` comparison, so a rate exactly equal
to the threshold PASSES (the gate is inclusive / ``<=``). These tests lock that
semantics and the immediate neighborhood, so a future refactor to ``>=`` (which
would flip the reported result) cannot land silently.
"""

from __future__ import annotations

from rulekiln.config.settings import QualityGateDefaults
from rulekiln.pipeline.quality_gates import check_quality_gates
from rulekiln.schemas.pipeline import CaseEvalResult, EvalResult
from rulekiln.schemas.task_case import EvaluationSpec, RuleKilnCase


def _eval_result(malformed_rate: float) -> EvalResult:
    # One passing case keeps the paired/regression gates satisfied; only the
    # malformed gate is under test here.
    return EvalResult(
        strategy="hdbscan",
        model="fake",
        split="validation",
        macro_f1=0.75,
        weighted_case_score=1.0,
        malformed_output_rate=malformed_rate,
        case_results=[
            CaseEvalResult(case_id="c1", score=1.0, passed=True, actual_output="entailment")
        ],
    )


def _baseline() -> EvalResult:
    return EvalResult(
        strategy="baseline",
        model="fake",
        split="validation",
        macro_f1=0.70,
        weighted_case_score=1.0,
        malformed_output_rate=0.0,
        case_results=[
            CaseEvalResult(case_id="c1", score=1.0, passed=True, actual_output="entailment")
        ],
    )


def _case() -> RuleKilnCase:
    return RuleKilnCase(
        id="c1",
        split="validation",  # type: ignore[arg-type]
        task_mode="classification",
        input={"text": "input c1"},
        expected="entailment",
        evaluation=EvaluationSpec(assertions=[]),
    )


def _has_malformed_violation(malformed_rate: float, threshold: float) -> bool:
    result = check_quality_gates(
        strategy="hdbscan",
        distilled_eval=_eval_result(malformed_rate),
        baseline_eval=_baseline(),
        cases=[_case()],
        task_mode="classification",
        task_gates={"max_malformed_output_rate": threshold},
        settings_defaults=QualityGateDefaults(),
    )
    return any(v.startswith("malformed_output_rate") for v in result.violations)


def test_three_of_300_equals_threshold_exactly() -> None:
    # The exact configuration from the investigated run.
    assert 3 / 300 == 0.01
    assert _has_malformed_violation(3 / 300, 0.01) is False


def test_just_below_threshold_passes() -> None:
    assert _has_malformed_violation(0.009999, 0.01) is False


def test_exactly_at_threshold_passes() -> None:
    # Strict ``>`` means the boundary is inclusive: rate == threshold is allowed.
    assert _has_malformed_violation(0.010000, 0.01) is False


def test_just_above_threshold_fails() -> None:
    assert _has_malformed_violation(0.010001, 0.01) is True


def test_four_of_300_would_fail_the_gate() -> None:
    # One more malformed case than the winning run flips the gate.
    assert _has_malformed_violation(4 / 300, 0.01) is True
