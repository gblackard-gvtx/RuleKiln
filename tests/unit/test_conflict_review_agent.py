"""Unit tests for conflict-review agent fallback behavior."""

from __future__ import annotations

import pytest

from rulekiln.agents.rule_conflict_review import review_rule_for_conflicts
from rulekiln.providers.contracts import ChatModelClient, ProviderConfig
from rulekiln.schemas.pipeline import (
    MicroRuleSchema,
    OutcomeCondition,
    RuleConflictReview,
    SynthesizedRuleSchema,
)
from rulekiln.schemas.task_case import RuleKilnTask
from rulekiln.schemas.usage import ChatCompletionResult


class _FailingChatClient(ChatModelClient):
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema,
        config: ProviderConfig,
    ) -> ChatCompletionResult:
        del system_prompt
        del user_prompt
        del output_schema
        del config
        raise self._exc


class _SuccessChatClient(ChatModelClient):
    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema,
        config: ProviderConfig,
    ) -> ChatCompletionResult:
        del system_prompt
        del user_prompt
        del output_schema
        del config
        review = RuleConflictReview(
            synthesized_rule_id="wrong-id",
            has_conflicts=True,
            conflict_summary="conflict found",
            conflicting_micro_rule_ids=["m1"],
            resolution="discard",
            resolved_rules=[],
        )
        return ChatCompletionResult(content="", parsed=review, raw_model="fake")


def _task() -> RuleKilnTask:
    return RuleKilnTask(
        task_id="t1",
        task_name="Conflict Task",
        task_mode="classification",
        description="test",
        input_template="{{input}}",
    )


def _rule() -> SynthesizedRuleSchema:
    return SynthesizedRuleSchema(
        id="rule-123",
        topic="topic",
        applies_when=["input present"],
        outcome_conditions={"approve": OutcomeCondition(outcome="approve", when=["input present"])},
        tie_breakers=[],
        priority=10,
        source_case_ids=["case-1"],
        source_micro_rule_ids=["m1"],
    )


def _micro_rules() -> list[MicroRuleSchema]:
    return [
        MicroRuleSchema(
            topic="topic",
            condition="if input present",
            expected_outcome="approve",
        )
    ]


def _config() -> ProviderConfig:
    return ProviderConfig(
        profile_name="fake-profile",
        provider="fake",
        model="fake-model",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "Exceeded maximum retries (3) for output validation",
        "Exceeded maximum retries (3) for result validation",
    ],
)
async def test_conflict_review_falls_back_when_validation_retries_exhausted(
    message: str,
) -> None:
    chat_client = _FailingChatClient(RuntimeError(message))

    review = await review_rule_for_conflicts(
        _task(),
        _rule(),
        _micro_rules(),
        chat_client,
        _config(),
    )

    assert review.synthesized_rule_id == "rule-123"
    assert review.resolution == "keep"
    assert review.has_conflicts is False
    assert review.conflicting_micro_rule_ids == []
    assert review.conflict_summary is not None
    assert "fallback" in review.conflict_summary.lower()


@pytest.mark.asyncio
async def test_conflict_review_falls_back_when_validation_retry_error_is_nested() -> None:
    def _nested_retry_error() -> Exception:
        try:
            raise RuntimeError("Exceeded maximum retries (3) for output validation")
        except RuntimeError as inner:
            try:
                raise RuntimeError("agent execution failed") from inner
            except RuntimeError as outer:
                return outer

    chat_client = _FailingChatClient(_nested_retry_error())

    review = await review_rule_for_conflicts(
        _task(),
        _rule(),
        _micro_rules(),
        chat_client,
        _config(),
    )

    assert review.synthesized_rule_id == "rule-123"
    assert review.resolution == "keep"
    assert review.has_conflicts is False


@pytest.mark.asyncio
async def test_conflict_review_does_not_swallow_unrelated_exceptions() -> None:
    chat_client = _FailingChatClient(RuntimeError("rate limiter unavailable"))

    with pytest.raises(RuntimeError, match="rate limiter unavailable"):
        await review_rule_for_conflicts(
            _task(),
            _rule(),
            _micro_rules(),
            chat_client,
            _config(),
        )


@pytest.mark.asyncio
async def test_conflict_review_overrides_model_returned_rule_id() -> None:
    review = await review_rule_for_conflicts(
        _task(),
        _rule(),
        _micro_rules(),
        _SuccessChatClient(),
        _config(),
    )

    assert review.synthesized_rule_id == "rule-123"
    assert review.resolution == "discard"
