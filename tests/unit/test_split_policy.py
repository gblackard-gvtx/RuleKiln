"""Unit tests for centralized split policy resolution."""

from typing import Literal

from rulekiln.pipeline.split_policy import resolve_split_policy
from rulekiln.schemas.task_case import EvaluationSpec, RuleKilnCase


def _case(case_id: str, split: Literal["train", "validation", "test", "golden"]) -> RuleKilnCase:
    return RuleKilnCase(
        id=case_id,
        split=split,
        task_mode="classification",
        input={"text": f"input {case_id}"},
        expected="positive",
        evaluation=EvaluationSpec(assertions=[]),
    )


def test_split_policy_prefers_validation_for_evaluation() -> None:
    cases = [
        _case("train-1", "train"),
        _case("train-2", "train"),
        _case("val-1", "validation"),
    ]

    decision = resolve_split_policy(cases)

    assert decision.extraction_split == "train"
    assert [case.id for case in decision.extraction_cases] == ["train-1", "train-2"]
    assert decision.evaluation_split == "validation"
    assert [case.id for case in decision.evaluation_cases] == ["val-1"]
    assert decision.fallback_warning is None


def test_split_policy_falls_back_to_train_when_validation_missing() -> None:
    cases = [
        _case("train-1", "train"),
        _case("train-2", "train"),
    ]

    decision = resolve_split_policy(cases)

    assert decision.extraction_split == "train"
    assert [case.id for case in decision.extraction_cases] == ["train-1", "train-2"]
    assert decision.evaluation_split == "train"
    assert [case.id for case in decision.evaluation_cases] == ["train-1", "train-2"]
    assert decision.fallback_warning is not None
    assert "split=train" in decision.fallback_warning
