"""Unit tests for evaluation scoring contract (T028)."""

import pytest

from rulekiln.schemas.pipeline import CaseEvalResult, EvalResult
from rulekiln.schemas.task_case import EvaluationSpec, RuleKilnCase, RuleKilnTask


def _task(mode: str = "classification") -> RuleKilnTask:
    return RuleKilnTask(
        task_id="t1",
        task_name="T",
        task_mode=mode,
        description="d",
        input_template="{{input}}",
    )


def _case(
    case_id: str, expected: str = "yes", golden: bool = False, weight: float = 1.0
) -> RuleKilnCase:
    return RuleKilnCase(
        id=case_id,
        task_mode="classification",
        split="train",
        input={"q": "?"},
        expected=expected,
        evaluation=EvaluationSpec(assertions=[]),
        is_golden=golden,
        weight=weight,
    )


def _eval_result(**kwargs) -> EvalResult:  # type: ignore[no-untyped-def]
    defaults: dict[str, object] = {
        "strategy": "hdbscan",
        "model": "fake",
        "split": "train",
        "accuracy": 0.8,
        "macro_f1": 0.75,
        "weighted_case_score": 0.8,
        "malformed_output_rate": 0.0,
        "per_outcome_precision": {},
        "per_outcome_recall": {},
        "confusion_matrix": {},
        "case_results": [],
    }
    defaults.update(kwargs)
    return EvalResult(**defaults)  # type: ignore[arg-type]


def test_malformed_output_rate_in_eval_result() -> None:
    ev = _eval_result(malformed_output_rate=0.05)
    assert ev.malformed_output_rate == pytest.approx(0.05)


def test_accuracy_range() -> None:
    ev = _eval_result(accuracy=1.0)
    assert 0.0 <= ev.accuracy <= 1.0


def test_case_eval_result_passed_flag() -> None:
    cr = CaseEvalResult(
        case_id="c1",
        score=1.0,
        passed=True,
        malformed=False,
        assertion_scores={},
        actual_output={"label": "yes"},
    )
    assert cr.passed is True
    assert cr.malformed is False


def test_case_eval_result_malformed() -> None:
    cr = CaseEvalResult(
        case_id="c1",
        score=0.0,
        passed=False,
        malformed=True,
        assertion_scores={},
        actual_output=None,
        error="parse error",
    )
    assert cr.malformed is True
    assert cr.score == 0.0
