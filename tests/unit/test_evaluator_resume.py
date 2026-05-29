"""Unit tests for resumable evaluator behavior."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from rulekiln.pipeline.evaluator import evaluate_prompt
from rulekiln.providers.contracts import ChatModelClient, ProviderConfig
from rulekiln.schemas.pipeline import CaseEvalResult
from rulekiln.schemas.task_case import EvaluationSpec, RuleKilnCase, RuleKilnTask
from rulekiln.schemas.usage import ChatCompletionResult


class _FakeChatClient(ChatModelClient):
    """Minimal chat client that echoes labels from case input."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel],
        config: ProviderConfig,
    ) -> ChatCompletionResult:
        del system_prompt
        del config

        self.calls += 1
        payload = json.loads(user_prompt)
        predicted_label = str(payload.get("predicted_label", ""))
        parsed = output_schema.model_validate({"raw": {"label": predicted_label}})
        return ChatCompletionResult(content="", parsed=parsed, usage=None, raw_model="fake")


def _task() -> RuleKilnTask:
    return RuleKilnTask(
        task_id="task-1",
        task_name="Task",
        task_mode="classification",
        description="Test task",
        input_template="{{input}}",
    )


def _cases() -> list[RuleKilnCase]:
    return [
        RuleKilnCase(
            id="c1",
            task_mode="classification",
            split="train",
            input={"predicted_label": "alpha"},
            expected={"label": "alpha"},
            evaluation=EvaluationSpec(assertions=[]),
        ),
        RuleKilnCase(
            id="c2",
            task_mode="classification",
            split="train",
            input={"predicted_label": "beta"},
            expected={"label": "beta"},
            evaluation=EvaluationSpec(assertions=[]),
        ),
    ]


@pytest.mark.asyncio
async def test_evaluate_prompt_skips_completed_cases_and_persists_missing_only() -> None:
    client = _FakeChatClient()
    config = ProviderConfig(
        profile_name="fake-profile",
        provider="fake",
        model="fake-model",
    )

    completed = {
        "c1": CaseEvalResult(
            case_id="c1",
            score=1.0,
            passed=True,
            malformed=False,
            actual_output={"label": "alpha"},
        )
    }
    persisted: list[CaseEvalResult] = []

    async def _persist(result: CaseEvalResult) -> None:
        persisted.append(result)

    result = await evaluate_prompt(
        system_prompt="system",
        cases=_cases(),
        task=_task(),
        chat_client=client,
        config=config,
        strategy="baseline",
        split="train",
        completed_case_results=completed,
        on_case_result=_persist,
    )

    assert client.calls == 1
    assert [case_result.case_id for case_result in result.case_results] == ["c1", "c2"]
    assert len(persisted) == 1
    assert persisted[0].case_id == "c2"
