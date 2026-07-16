"""Legacy compatibility tests for classification-statistics refresh (Work Package 5).

A DB-loaded ``EvalResult`` with ``metric_semantics_version < 2`` predates strict/
valid-output metric semantics (Work Package 1). ``_refresh_eval_classification_statistics``
must recompute and stamp ``recomputed_legacy`` when ``case_results`` is available, and
must leave the row alone (aside from an ``incomplete_legacy`` status stamp) when it isn't
— it must never silently fabricate values it can't honestly reconstruct.
"""

from __future__ import annotations

from rulekiln.schemas.pipeline import CaseEvalResult, EvalResult
from rulekiln.schemas.task_case import RuleKilnCase, RuleKilnTask
from rulekiln.workers.pipeline.classification_metrics import (
    _refresh_eval_classification_statistics,
)


def _task() -> RuleKilnTask:
    return RuleKilnTask(
        task_id="contractnli",
        task_name="ContractNLI",
        task_mode="classification",
        description="d",
        input_template="{{ premise }}",
        output_schema={
            "type": "object",
            "required": ["label"],
            "properties": {
                "label": {"type": "string", "enum": ["entailment", "contradiction", "neutral"]}
            },
        },
    )


def _case(case_id: str, label: str) -> RuleKilnCase:
    return RuleKilnCase(
        id=case_id,
        split="train",
        task_mode="classification",
        input={"premise": f"premise {case_id}"},
        expected={"label": label},
    )


def _refresh(eval_result: EvalResult, cases: list[RuleKilnCase]) -> EvalResult:
    return _refresh_eval_classification_statistics(
        eval_result=eval_result,
        eval_cases=cases,
        task=_task(),
        bootstrap_enabled=False,
        bootstrap_iterations=100,
        bootstrap_seed=0,
    )


def test_legacy_row_with_case_results_is_recomputed_and_stamped() -> None:
    cases = [_case("c1", "entailment"), _case("c2", "neutral")]
    legacy = EvalResult(
        strategy="baseline_scaffold",
        model="m",
        split="train",
        metric_semantics_version=1,
        accuracy=0.0,
        macro_f1=0.0,
        case_results=[
            CaseEvalResult(
                case_id="c1", score=1.0, passed=True, actual_output={"label": "entailment"}
            ),
            CaseEvalResult(
                case_id="c2", score=1.0, passed=True, actual_output={"label": "neutral"}
            ),
        ],
    )
    refreshed = _refresh(legacy, cases)
    assert refreshed.metric_semantics_version == 2
    assert refreshed.metric_semantics_status == "recomputed_legacy"
    assert refreshed.accuracy == 1.0


def test_legacy_row_without_case_results_is_labeled_incomplete_and_left_alone() -> None:
    legacy = EvalResult(
        strategy="baseline_scaffold",
        model="m",
        split="train",
        metric_semantics_version=1,
        accuracy=0.42,
        macro_f1=0.37,
        case_results=[],
    )
    refreshed = _refresh(legacy, [_case("c1", "entailment")])
    assert refreshed.metric_semantics_status == "incomplete_legacy"
    # Left unmodified: original (possibly stale) scalar values are preserved, not fabricated.
    assert refreshed.metric_semantics_version == 1
    assert refreshed.accuracy == 0.42
    assert refreshed.macro_f1 == 0.37


def test_current_row_stays_current_after_refresh() -> None:
    cases = [_case("c1", "entailment")]
    current = EvalResult(
        strategy="baseline_scaffold",
        model="m",
        split="train",
        case_results=[
            CaseEvalResult(
                case_id="c1", score=1.0, passed=True, actual_output={"label": "entailment"}
            ),
        ],
    )
    assert current.metric_semantics_version == 2
    refreshed = _refresh(current, cases)
    assert refreshed.metric_semantics_version == 2
    assert refreshed.metric_semantics_status == "current"
