"""Unit tests for evaluation scoring contract (T028)."""

import pytest

from rulekiln.schemas.pipeline import CaseEvalResult, EvalResult
from rulekiln.schemas.task_case import EvaluationSpec, RuleKilnCase, RuleKilnTask, TaskMode


def _task(mode: TaskMode = "classification") -> RuleKilnTask:
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
        split="golden" if golden else "train",
        input={"q": "?"},
        expected=expected,
        evaluation=EvaluationSpec(assertions=[]),
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
    assert ev.accuracy is not None
    assert 0.0 <= ev.accuracy <= 1.0


def test_confidence_interval_object_shape() -> None:
    ev = _eval_result(
        accuracy_ci_95={
            "low": 0.6,
            "high": 0.8,
            "iterations": 1000,
            "seed": 12345,
        },
        macro_f1_ci_95={
            "low": 0.55,
            "high": 0.75,
            "iterations": 1000,
            "seed": 12345,
        },
    )
    assert ev.accuracy_ci_95 is not None
    assert ev.macro_f1_ci_95 is not None
    assert ev.accuracy_ci_95.method == "bootstrap"
    assert ev.accuracy_ci_95.iterations == 1000
    assert ev.accuracy_ci_95.seed == 12345
    assert ev.macro_f1_ci_95.method == "bootstrap"


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
