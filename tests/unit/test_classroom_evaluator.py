"""Unit tests for the classroom evaluator."""

from __future__ import annotations

import pytest

from rulekiln.pipeline.classroom_evaluator import anchor_eval, evaluate_classroom
from rulekiln.providers.chat.fake import FakeChatClient
from rulekiln.providers.contracts import ChatModelClient, ProviderConfig
from rulekiln.schemas.classroom import ClassroomConfig, StudentConfig
from rulekiln.schemas.pipeline import EvalResult
from rulekiln.schemas.task_case import (
    EvaluationAssertion,
    EvaluationSpec,
    RuleKilnCase,
    RuleKilnTask,
)


def _task() -> RuleKilnTask:
    return RuleKilnTask(
        schema_version="rulekiln.task.v1",
        task_id="test-task",
        task_name="Test Task",
        task_mode="classification",
        description="Classify intent",
        input_template="{{input.text}}",
    )


def _cases() -> list[RuleKilnCase]:
    return [
        RuleKilnCase(
            id=f"c{i}",
            task_mode="classification",
            input={"text": f"sample {i}"},
            expected={"label": "travel"},
            evaluation=EvaluationSpec(
                assertions=[
                    EvaluationAssertion(
                        type="must_equal",  # type: ignore[arg-type]
                        path="label",
                        value="travel",
                        weight=1.0,
                    )
                ]
            ),
        )
        for i in range(3)
    ]


def _fake_config(model: str = "fake-model") -> ProviderConfig:
    return ProviderConfig(profile_name="fake", model=model, provider="fake")


def _classroom_3students() -> ClassroomConfig:
    return ClassroomConfig(
        students=[
            StudentConfig(id="s1", provider="fake", model="m1", is_anchor=True),
            StudentConfig(id="s2", provider="fake", model="m2"),
            StudentConfig(id="s3", provider="fake", model="m3"),
        ],
        anchor_student_id="s1",
    )


@pytest.mark.asyncio
async def test_evaluate_classroom_returns_three_keys() -> None:
    """3-student classroom → 3 result keys."""
    cc = _classroom_3students()

    def get_client(student: StudentConfig) -> ChatModelClient:
        return FakeChatClient()

    def get_config(student: StudentConfig) -> ProviderConfig:
        return _fake_config(student.model)

    results = await evaluate_classroom(
        system_prompt="You are a classifier.",
        cases=_cases(),
        task=_task(),
        classroom_config=cc,
        get_chat_client=get_client,
        get_provider_config=get_config,
        strategy="dbscan",
        split="validation",
        bootstrap_enabled=False,
    )

    assert set(results.keys()) == {"s1", "s2", "s3"}
    for val in results.values():
        assert isinstance(val, EvalResult)


@pytest.mark.asyncio
async def test_evaluate_classroom_anchor_only_returns_one_key() -> None:
    """anchor_only=True evaluates only the anchor student."""
    cc = _classroom_3students()

    def get_client(student: StudentConfig) -> ChatModelClient:
        return FakeChatClient()

    def get_config(student: StudentConfig) -> ProviderConfig:
        return _fake_config(student.model)

    results = await evaluate_classroom(
        system_prompt="You are a classifier.",
        cases=_cases(),
        task=_task(),
        classroom_config=cc,
        get_chat_client=get_client,
        get_provider_config=get_config,
        strategy="dbscan",
        split="validation",
        bootstrap_enabled=False,
        anchor_only=True,
    )

    assert set(results.keys()) == {"s1"}


@pytest.mark.asyncio
async def test_evaluate_classroom_single_student_backward_compat() -> None:
    """Single-student classroom (flat migration) produces result keyed by 'default'."""
    cc = ClassroomConfig.from_provider_model("fake", "fake-model")

    def get_client(student: StudentConfig) -> ChatModelClient:
        return FakeChatClient()

    def get_config(student: StudentConfig) -> ProviderConfig:
        return _fake_config()

    results = await evaluate_classroom(
        system_prompt="prompt",
        cases=_cases(),
        task=_task(),
        classroom_config=cc,
        get_chat_client=get_client,
        get_provider_config=get_config,
        strategy="dbscan",
        split="validation",
        bootstrap_enabled=False,
    )

    assert "default" in results


@pytest.mark.asyncio
async def test_anchor_eval_helper_returns_anchor_result() -> None:
    cc = _classroom_3students()

    def get_client(student: StudentConfig) -> ChatModelClient:
        return FakeChatClient()

    def get_config(student: StudentConfig) -> ProviderConfig:
        return _fake_config(student.model)

    results = await evaluate_classroom(
        system_prompt="prompt",
        cases=_cases(),
        task=_task(),
        classroom_config=cc,
        get_chat_client=get_client,
        get_provider_config=get_config,
        strategy="dbscan",
        split="validation",
        bootstrap_enabled=False,
    )

    anchor = anchor_eval(results, cc)
    assert anchor is not None
    assert isinstance(anchor, EvalResult)


@pytest.mark.asyncio
async def test_results_keyed_by_student_id_not_index() -> None:
    """Result keys are student IDs, not integer positions."""
    cc = ClassroomConfig(
        students=[
            StudentConfig(id="alpha", provider="fake", model="m", is_anchor=True),
            StudentConfig(id="beta", provider="fake", model="m"),
        ],
        anchor_student_id="alpha",
    )

    def get_client(student: StudentConfig) -> ChatModelClient:
        return FakeChatClient()

    def get_config(student: StudentConfig) -> ProviderConfig:
        return _fake_config()

    results = await evaluate_classroom(
        system_prompt="prompt",
        cases=_cases(),
        task=_task(),
        classroom_config=cc,
        get_chat_client=get_client,
        get_provider_config=get_config,
        strategy="dbscan",
        split="validation",
        bootstrap_enabled=False,
    )

    assert "alpha" in results
    assert "beta" in results
    assert 0 not in results  # type: ignore[operator]
    assert 1 not in results  # type: ignore[operator]
