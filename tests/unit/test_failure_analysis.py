"""Unit tests for the updated failure analysis with rule mapping."""

from __future__ import annotations

from rulekiln.pipeline.failure_analysis import (
    UNATTRIBUTED_RULE_ID,
    FailureAnalysisResult,
    analyze_failures,
)
from rulekiln.schemas.pipeline import (
    CaseEvalResult,
    EvalResult,
    OutcomeCondition,
    SynthesizedRuleSchema,
)
from rulekiln.schemas.task_case import EvaluationAssertion, EvaluationSpec, RuleKilnCase

# ── Fixtures ────────────────────────────────────────────────────────────────


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


def _case_result_evaluator_style(case_id: str, passed: bool) -> CaseEvalResult:
    """CaseEvalResult using the real evaluator key format assertion_{i}."""
    return CaseEvalResult(
        case_id=case_id,
        passed=passed,
        score=1.0 if passed else 0.0,
        malformed=False,
        assertion_scores={"assertion_0": 1.0 if passed else 0.0},
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


def _rule_with_outcome(rule_id: str, outcome_label: str) -> SynthesizedRuleSchema:
    return SynthesizedRuleSchema(
        id=rule_id,
        topic=f"rule_topic_{outcome_label}",
        applies_when=["some condition"],
        outcome_conditions={
            outcome_label: OutcomeCondition(
                outcome=outcome_label,
                when=["some condition"],
                confidence="high",
            )
        },
        priority=1,
        source_case_ids=["c1"],
        source_micro_rule_ids=["m1"],
    )


def _case_with_assertion(
    case_id: str,
    assertion_type: str = "must_equal",
    assertion_value: str = "entailment",
) -> RuleKilnCase:
    return RuleKilnCase(
        id=case_id,
        task_mode="classification",
        input={"text": "sample"},
        expected={"label": assertion_value},
        evaluation=EvaluationSpec(
            assertions=[
                EvaluationAssertion(
                    type=assertion_type,  # type: ignore[arg-type]
                    path="label",
                    value=assertion_value,
                    weight=1.0,
                )
            ]
        ),
    )


# ── Existing tests (backward compatibility) ──────────────────────────────────


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


# ── New tests: real rule attribution (Task 1) ─────────────────────────────────


def test_violated_rule_ids_populated_for_broken_with_cases() -> None:
    """violated_rule_ids contains the expected rule ID when assertion key is assertion_{i}."""
    case = _case_with_assertion("c1", assertion_value="entailment")
    rule = _rule_with_outcome("rule_A", "entailment")
    baseline = _eval_result(case_results=[_case_result_evaluator_style("c1", passed=True)])
    distilled = _eval_result(case_results=[_case_result_evaluator_style("c1", passed=False)])
    result = analyze_failures(baseline, distilled, selected_rules=[rule], cases=[case])
    broken = [f for f in result.structured_failures if f.failure_class == "broken"]
    assert len(broken) == 1
    assert "rule_A" in broken[0].violated_rule_ids
    assert UNATTRIBUTED_RULE_ID not in broken[0].violated_rule_ids


def test_failed_assertion_types_populated() -> None:
    """failed_assertion_types contains the actual assertion type when cases are provided."""
    case = _case_with_assertion("c1", assertion_type="must_equal", assertion_value="entailment")
    rule = _rule_with_outcome("rule_A", "entailment")
    baseline = _eval_result(case_results=[_case_result_evaluator_style("c1", passed=True)])
    distilled = _eval_result(case_results=[_case_result_evaluator_style("c1", passed=False)])
    result = analyze_failures(baseline, distilled, selected_rules=[rule], cases=[case])
    broken = [f for f in result.structured_failures if f.failure_class == "broken"]
    assert "must_equal" in broken[0].failed_assertion_types


def test_unattributed_sentinel_when_no_rule_match() -> None:
    """UNATTRIBUTED_RULE_ID is used when no rule covers the expected outcome."""
    case = _case_with_assertion("c1", assertion_value="entailment")
    rule = _rule_with_outcome("rule_B", "contradiction")  # covers different outcome
    baseline = _eval_result(case_results=[_case_result_evaluator_style("c1", passed=True)])
    distilled = _eval_result(case_results=[_case_result_evaluator_style("c1", passed=False)])
    result = analyze_failures(baseline, distilled, selected_rules=[rule], cases=[case])
    broken = [f for f in result.structured_failures if f.failure_class == "broken"]
    assert broken[0].violated_rule_ids == [UNATTRIBUTED_RULE_ID]


def test_violated_rule_summary_non_empty_on_fixture() -> None:
    """violated_rule_summary() returns non-zero counts when attribution succeeds."""
    case = _case_with_assertion("c1", assertion_value="entailment")
    rule = _rule_with_outcome("rule_A", "entailment")
    baseline = _eval_result(case_results=[_case_result_evaluator_style("c1", passed=True)])
    distilled = _eval_result(case_results=[_case_result_evaluator_style("c1", passed=False)])
    result = analyze_failures(baseline, distilled, selected_rules=[rule], cases=[case])
    summary = result.violated_rule_summary()
    assert "rule_A" in summary
    assert summary["rule_A"]["broken_count"] >= 1
    assert summary["rule_A"]["violated_count"] >= 1


def test_unattributed_fraction_zero_when_all_attributed() -> None:
    """unattributed_fraction() is 0.0 when every failure maps to a real rule."""
    case = _case_with_assertion("c1", assertion_value="entailment")
    rule = _rule_with_outcome("rule_A", "entailment")
    baseline = _eval_result(case_results=[_case_result_evaluator_style("c1", passed=True)])
    distilled = _eval_result(case_results=[_case_result_evaluator_style("c1", passed=False)])
    result = analyze_failures(baseline, distilled, selected_rules=[rule], cases=[case])
    assert result.unattributed_fraction() == 0.0


def test_unattributed_fraction_below_threshold_on_fixture() -> None:
    """The unattributed fraction is reportable; with 1 match + 0 unattributed, it is 0.0 <= 0.5."""
    case_a = _case_with_assertion("c1", assertion_value="entailment")
    rule = _rule_with_outcome("rule_A", "entailment")
    baseline = _eval_result(case_results=[_case_result_evaluator_style("c1", passed=True)])
    distilled = _eval_result(case_results=[_case_result_evaluator_style("c1", passed=False)])
    result = analyze_failures(baseline, distilled, selected_rules=[rule], cases=[case_a])
    threshold = 0.5
    assert result.unattributed_fraction() <= threshold


def test_matched_rule_ids_populated_for_fixed() -> None:
    """matched_rule_ids is populated for fixed cases when cases are provided."""
    case = _case_with_assertion("c1", assertion_value="entailment")
    rule = _rule_with_outcome("rule_A", "entailment")
    baseline = _eval_result(case_results=[_case_result_evaluator_style("c1", passed=False)])
    distilled = _eval_result(case_results=[_case_result_evaluator_style("c1", passed=True)])
    result = analyze_failures(baseline, distilled, selected_rules=[rule], cases=[case])
    fixed = [f for f in result.structured_failures if f.failure_class == "fixed"]
    assert len(fixed) == 1
    assert "rule_A" in fixed[0].matched_rule_ids


def test_build_utility_signals_non_empty_with_broken() -> None:
    """build_utility_signals() returns non-empty dict when failures are attributed."""
    case = _case_with_assertion("c1", assertion_value="entailment")
    rule = _rule_with_outcome("rule_A", "entailment")
    baseline = _eval_result(case_results=[_case_result_evaluator_style("c1", passed=True)])
    distilled = _eval_result(case_results=[_case_result_evaluator_style("c1", passed=False)])
    result = analyze_failures(baseline, distilled, selected_rules=[rule], cases=[case])
    signals = result.build_utility_signals()
    assert "rule_A" in signals
    _fixed, broken = signals["rule_A"]
    assert broken >= 1
    assert UNATTRIBUTED_RULE_ID not in signals


def test_build_utility_signals_excludes_unattributed() -> None:
    """UNATTRIBUTED_RULE_ID is never included in build_utility_signals output."""
    case = _case_with_assertion("c1", assertion_value="entailment")
    rule = _rule_with_outcome("rule_B", "contradiction")  # no match → sentinel
    baseline = _eval_result(case_results=[_case_result_evaluator_style("c1", passed=True)])
    distilled = _eval_result(case_results=[_case_result_evaluator_style("c1", passed=False)])
    result = analyze_failures(baseline, distilled, selected_rules=[rule], cases=[case])
    signals = result.build_utility_signals()
    assert UNATTRIBUTED_RULE_ID not in signals


def test_violated_rule_summary_includes_fixed_count() -> None:
    """violated_rule_summary() tracks fixed_count from matched_rule_ids."""
    case = _case_with_assertion("c1", assertion_value="entailment")
    rule = _rule_with_outcome("rule_A", "entailment")
    baseline = _eval_result(case_results=[_case_result_evaluator_style("c1", passed=False)])
    distilled = _eval_result(case_results=[_case_result_evaluator_style("c1", passed=True)])
    result = analyze_failures(baseline, distilled, selected_rules=[rule], cases=[case])
    summary = result.violated_rule_summary()
    assert "rule_A" in summary
    assert summary["rule_A"]["fixed_count"] >= 1
