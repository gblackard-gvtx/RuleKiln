"""Unit tests for the updated failure analysis with rule mapping."""

from __future__ import annotations

from rulekiln.pipeline.failure_analysis import FailureAnalysisResult, analyze_failures
from rulekiln.schemas.pipeline import (
    CaseEvalResult,
    EvalResult,
    OutcomeCondition,
    SynthesizedRuleSchema,
)


def _eval_result(
    strategy: str = "dbscan",
    case_results: list[CaseEvalResult] | None = None,
) -> EvalResult:
    return EvalResult(
        strategy=strategy,
        model="fake",
        split="test",
        accuracy=0.8,
        macro_f1=0.75,
        weighted_case_score=0.8,
        per_outcome_precision={},
        per_outcome_recall={},
        malformed_output_rate=0.0,
        confusion_matrix={},
        case_results=case_results or [],
    )


def _case_result(case_id: str, passed: bool, score: float = 0.8) -> CaseEvalResult:
    return CaseEvalResult(
        case_id=case_id,
        passed=passed,
        score=score,
        malformed=False,
        assertion_scores={"path.a": 1.0 if passed else 0.0},
    )


def _rule(rule_id: str, topic: str) -> SynthesizedRuleSchema:
    return SynthesizedRuleSchema(
        id=rule_id,
        topic=topic,
        applies_when=["cond"],
        outcome_conditions={"out": OutcomeCondition(outcome="out", when=["c"], confidence="high")},
        tie_breakers=[],
        priority=1,
        source_case_ids=["c1"],
        source_micro_rule_ids=["m1"],
    )


def test_no_baseline_all_distilled_classified() -> None:
    cases = [_case_result("c1", passed=True), _case_result("c2", passed=False)]
    result = analyze_failures(None, _eval_result(case_results=cases))
    assert len(result.unchanged_passing) == 1
    assert len(result.unchanged_failing) == 1
    assert result.unchanged_passing[0]["case_id"] == "c1"


def test_fixed_case_detected() -> None:
    baseline = _eval_result(case_results=[_case_result("c1", passed=False)])
    distilled = _eval_result(case_results=[_case_result("c1", passed=True)])
    result = analyze_failures(baseline, distilled)
    assert len(result.fixed) == 1
    assert result.fixed[0]["case_id"] == "c1"


def test_broken_case_detected() -> None:
    baseline = _eval_result(case_results=[_case_result("c1", passed=True)])
    distilled = _eval_result(case_results=[_case_result("c1", passed=False)])
    result = analyze_failures(baseline, distilled)
    assert len(result.broken) == 1


def test_unchanged_passing() -> None:
    baseline = _eval_result(case_results=[_case_result("c1", passed=True)])
    distilled = _eval_result(case_results=[_case_result("c1", passed=True)])
    result = analyze_failures(baseline, distilled)
    assert len(result.unchanged_passing) == 1
    assert len(result.broken) == 0
    assert len(result.fixed) == 0


def test_structured_failures_populated_for_broken() -> None:
    baseline = _eval_result(case_results=[_case_result("c1", passed=True)])
    distilled = _eval_result(case_results=[_case_result("c1", passed=False)])
    rules = [_rule("rule_1", "path.a")]
    result = analyze_failures(baseline, distilled, selected_rules=rules)
    broken_failures = [f for f in result.structured_failures if f.failure_class == "broken"]
    assert len(broken_failures) == 1
    assert broken_failures[0].case_id == "c1"
    assert "path.a" in broken_failures[0].failed_assertion_paths


def test_to_jsonl_empty_category() -> None:
    result = FailureAnalysisResult()
    assert result.to_jsonl("fixed") == ""
    assert result.to_jsonl("broken") == ""


def test_missing_case_in_distilled_goes_unchanged_failing() -> None:
    baseline = _eval_result(case_results=[_case_result("c1", passed=True)])
    distilled = _eval_result(case_results=[])
    result = analyze_failures(baseline, distilled)
    assert len(result.unchanged_failing) == 1
