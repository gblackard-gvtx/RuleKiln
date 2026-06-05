"""Unit tests for conflict-review agent fallback behavior and apply_conflict_reviews."""

from __future__ import annotations

import pytest

from rulekiln.agents.rule_conflict_review import apply_conflict_reviews, review_rule_for_conflicts
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


# ── apply_conflict_reviews ────────────────────────────────────────────────────


def _synth_rule(rule_id: str, topic: str = "topic") -> SynthesizedRuleSchema:
    return SynthesizedRuleSchema(
        id=rule_id,
        topic=topic,
        applies_when=["cond"],
        outcome_conditions={"approve": OutcomeCondition(outcome="approve", when=["cond"])},
    )


def _conflict_review(
    rule_id: str,
    resolution: str,
    resolved_rules: list[SynthesizedRuleSchema] | None = None,
) -> RuleConflictReview:
    return RuleConflictReview(
        synthesized_rule_id=rule_id,
        has_conflicts=resolution != "keep",
        conflict_summary=None,
        resolution=resolution,  # type: ignore[arg-type]
        resolved_rules=resolved_rules or [],
    )


def test_apply_conflict_reviews_keep_preserves_rule() -> None:
    rule = _synth_rule("r1")
    review = _conflict_review("r1", "keep")
    result = apply_conflict_reviews([rule], [review])
    assert len(result) == 1
    assert result[0].id == "r1"
    assert result[0].topic == "topic"


def test_apply_conflict_reviews_modify_replaces_rule_preserves_id() -> None:
    rule = _synth_rule("r1", "original")
    replacement = _synth_rule("ignored_id", "modified")
    review = _conflict_review("r1", "modify", resolved_rules=[replacement])
    result = apply_conflict_reviews([rule], [review])
    assert len(result) == 1
    assert result[0].id == "r1"  # ID preserved
    assert result[0].topic == "modified"


def test_apply_conflict_reviews_split_creates_deterministic_child_ids() -> None:
    rule = _synth_rule("r1")
    child_a = _synth_rule("a", "part1")
    child_b = _synth_rule("b", "part2")
    review = _conflict_review("r1", "split", resolved_rules=[child_a, child_b])
    result = apply_conflict_reviews([rule], [review])
    assert len(result) == 2
    assert result[0].id == "r1__split_1"
    assert result[1].id == "r1__split_2"


def test_apply_conflict_reviews_discard_removes_rule() -> None:
    rule_a = _synth_rule("r1")
    rule_b = _synth_rule("r2")
    review = _conflict_review("r1", "discard")
    result = apply_conflict_reviews([rule_a, rule_b], [review])
    assert len(result) == 1
    assert result[0].id == "r2"


def test_apply_conflict_reviews_no_review_keeps_rule() -> None:
    rule = _synth_rule("r1")
    result = apply_conflict_reviews([rule], [])
    assert len(result) == 1
    assert result[0].id == "r1"


# ── fallback review is distinguishable ───────────────────────────────────────


@pytest.mark.asyncio
async def test_conflict_review_fallback_has_fallback_status() -> None:
    """Fallback review has review_status='fallback_validation_failed', not 'completed'."""
    chat_client = _FailingChatClient(
        RuntimeError("Exceeded maximum retries (3) for output validation")
    )
    review = await review_rule_for_conflicts(
        _task(), _rule(), _micro_rules(), chat_client, _config()
    )
    assert review.review_status == "fallback_validation_failed"


@pytest.mark.asyncio
async def test_conflict_review_successful_has_completed_status() -> None:
    """A successful review has review_status='completed'."""
    review = await review_rule_for_conflicts(
        _task(), _rule(), _micro_rules(), _SuccessChatClient(), _config()
    )
    assert review.review_status == "completed"
