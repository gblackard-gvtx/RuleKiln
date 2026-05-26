"""Unit tests for model call idempotency key generation in provider tracking."""

from __future__ import annotations

import pytest

from rulekiln.providers.tracking import (
    ModelCallCollector,
    clear_tracking_context,
    set_tracking_context,
    tracked_chat_call,
    update_tracking_context,
)
from rulekiln.schemas.usage import ChatCompletionResult, ModelCallContext


@pytest.mark.asyncio
async def test_tracked_chat_call_uses_stable_idempotency_key_for_same_request() -> None:
    collector = ModelCallCollector()
    set_tracking_context(
        ModelCallContext(
            job_id="job-1",
            stage="evaluating_baseline",
            role="student",
            provider_profile="local",
            provider="openai_compatible",
            model="qwen-4b",
            strategy="baseline",
            student_id="qwen-4b",
            case_id="case-1",
        ),
        collector,
    )

    async def _call() -> ChatCompletionResult:
        return ChatCompletionResult(content="ok", parsed=None, usage=None, raw_model="qwen-4b")

    await tracked_chat_call(call=_call, fallback_input_text="same-input")
    await tracked_chat_call(call=_call, fallback_input_text="same-input")

    keys = [record.idempotency_key for record in collector.records]
    assert len(keys) == 2
    assert keys[0] is not None
    assert keys[0] == keys[1]

    clear_tracking_context()


@pytest.mark.asyncio
async def test_tracked_chat_call_changes_idempotency_key_when_case_changes() -> None:
    collector = ModelCallCollector()
    set_tracking_context(
        ModelCallContext(
            job_id="job-2",
            stage="evaluating_distilled",
            role="student",
            provider_profile="local",
            provider="openai_compatible",
            model="qwen-4b",
            strategy="dbscan",
            student_id="qwen-4b",
            case_id="case-a",
        ),
        collector,
    )

    async def _call() -> ChatCompletionResult:
        return ChatCompletionResult(content="ok", parsed=None, usage=None, raw_model="qwen-4b")

    await tracked_chat_call(call=_call, fallback_input_text="same-input")
    update_tracking_context(case_id="case-b")
    await tracked_chat_call(call=_call, fallback_input_text="same-input")

    keys = [record.idempotency_key for record in collector.records]
    assert len(keys) == 2
    assert keys[0] is not None
    assert keys[1] is not None
    assert keys[0] != keys[1]

    clear_tracking_context()
