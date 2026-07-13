"""Regression tests for evaluator JSON repair logic (T999 — Gemma malformed output).

Gemma 4B via llama.cpp returns plain JSON text (e.g. {"label": "entailment"}) instead
of a tool-call wrapper.  The evaluator's _extract_json_from_content helper and the
_call_student fallback path must survive all observed failure patterns.
"""

from __future__ import annotations

import pytest

from rulekiln.pipeline.evaluator import (
    _extract_json_from_content,  # pyright: ignore[reportPrivateUsage]
)
from rulekiln.providers.contracts import ChatModelClient, ProviderConfig
from rulekiln.schemas.task_case import EvaluationAssertion, EvaluationSpec, RuleKilnCase
from rulekiln.schemas.usage import ChatCompletionResult, ModelUsage

# ── _extract_json_from_content unit tests ─────────────────────────────────────


def test_plain_json() -> None:
    result = _extract_json_from_content('{"label": "entailment"}')
    assert result == {"label": "entailment"}


def test_plain_json_whitespace() -> None:
    result = _extract_json_from_content('  {"label": "contradiction"}  ')
    assert result == {"label": "contradiction"}


def test_markdown_fenced_with_language_tag() -> None:
    result = _extract_json_from_content('```json\n{"label": "neutral"}\n```')
    assert result == {"label": "neutral"}


def test_markdown_fenced_without_language_tag() -> None:
    result = _extract_json_from_content('```\n{"label": "entailment"}\n```')
    assert result == {"label": "entailment"}


def test_prose_before_json() -> None:
    """Model outputs explanation before the JSON object."""
    content = 'Based on the contract clause, the answer is:\n\n{"label": "contradiction"}'
    result = _extract_json_from_content(content)
    assert result == {"label": "contradiction"}


def test_prose_after_json() -> None:
    """Model outputs JSON then adds a trailing explanation."""
    content = '{"label": "entailment"}\n\nThe clause explicitly grants this right.'
    result = _extract_json_from_content(content)
    assert result == {"label": "entailment"}


def test_extra_primitive_fields_kept() -> None:
    """Extra primitive fields pass through — assertion only checks $.label."""
    result = _extract_json_from_content('{"label": "neutral", "confidence": 0.87}')
    assert result == {"label": "neutral", "confidence": 0.87}


def test_nested_object_fields_stripped() -> None:
    """Nested dicts are stripped because actual_output is dict[str, primitive]."""
    content = '{"label": "entailment", "details": {"step": "clause matched"}}'
    result = _extract_json_from_content(content)
    assert result == {"label": "entailment"}


def test_invalid_label_value_passes_through() -> None:
    """_extract_json_from_content does not validate label values — scorer does."""
    result = _extract_json_from_content('{"label": "UNKNOWN_LABEL"}')
    assert result == {"label": "UNKNOWN_LABEL"}


def test_multiple_json_objects_takes_first() -> None:
    """When prose contains multiple {...} blocks, only the first is used."""
    content = 'First: {"label": "neutral"} and also {"label": "entailment"}'
    result = _extract_json_from_content(content)
    assert result == {"label": "neutral"}


def test_empty_string_returns_none() -> None:
    assert _extract_json_from_content("") is None


def test_whitespace_only_returns_none() -> None:
    assert _extract_json_from_content("   \n  ") is None


def test_plain_prose_no_json_returns_none() -> None:
    result = _extract_json_from_content("The contract clause is entailment.")
    assert result is None


def test_invalid_json_returns_none() -> None:
    result = _extract_json_from_content("{label: entailment}")
    assert result is None


def test_json_array_returns_none() -> None:
    """A JSON array is not a valid structured output — return None."""
    result = _extract_json_from_content('["entailment"]')
    assert result is None


def test_string_scalar_returns_string() -> None:
    """A bare JSON string is returned as-is (e.g. some models return just the label)."""
    result = _extract_json_from_content('"entailment"')
    assert result == "entailment"


def test_markdown_fence_takes_priority_over_outer_text() -> None:
    """Fenced content is tried before treating the whole string as JSON."""
    content = 'Some preamble {"label": "wrong"}\n```json\n{"label": "correct"}\n```\nTrailing.'
    result = _extract_json_from_content(content)
    assert result == {"label": "correct"}


# ── Integration: _call_student with a provider returning plain JSON content ───


class _PlainJsonClient(ChatModelClient):
    """Fake provider that returns raw JSON text in content (no tool-call wrapper)."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text

    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type,
        config: ProviderConfig,
    ) -> ChatCompletionResult:
        return ChatCompletionResult(
            content=self._response_text,
            parsed=None,
            usage=ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            raw_model="gemma-4-E4B-it-Q4_K_M.gguf",
        )


def _config() -> ProviderConfig:
    return ProviderConfig(
        profile_name="local",
        provider="openai_compatible",
        model="gemma-4-E4B-it-Q4_K_M.gguf",
        max_retries=0,
    )


def _classification_case(case_id: str = "c1", label: str = "neutral") -> RuleKilnCase:
    return RuleKilnCase(
        id=case_id,
        task_mode="classification",
        input={"premise": "No additional rights granted.", "hypothesis": "No rights granted."},
        expected={"label": label},
        evaluation=EvaluationSpec(
            assertions=[
                EvaluationAssertion(type="must_equal", path="$.label", value=label, weight=1.0)
            ]
        ),
    )


@pytest.mark.asyncio
async def test_call_student_plain_json_is_not_malformed() -> None:
    """When the provider returns plain JSON, _call_student extracts it (not malformed)."""
    from rulekiln.pipeline.evaluator import _call_student  # pyright: ignore[reportPrivateUsage]

    client = _PlainJsonClient('{"label": "neutral"}')
    output, malformed, raw_content = await _call_student(
        "sys", _classification_case(), client, _config(), input_template="{{ premise }}"
    )
    assert malformed is False
    assert output == {"label": "neutral"}
    assert raw_content == '{"label": "neutral"}'


@pytest.mark.asyncio
async def test_call_student_markdown_fenced_is_not_malformed() -> None:
    from rulekiln.pipeline.evaluator import _call_student  # pyright: ignore[reportPrivateUsage]

    client = _PlainJsonClient('```json\n{"label": "entailment"}\n```')
    case = _classification_case(label="entailment")
    output, malformed, _raw_content = await _call_student(
        "sys", case, client, _config(), input_template="{{ premise }}"
    )
    assert malformed is False
    assert output == {"label": "entailment"}


@pytest.mark.asyncio
async def test_call_student_empty_content_not_malformed_from_call_but_none() -> None:
    """Empty content returns (None, False); _score_case will mark malformed via actual is None."""
    from rulekiln.pipeline.evaluator import _call_student  # pyright: ignore[reportPrivateUsage]

    client = _PlainJsonClient("")
    output, malformed, _raw_content = await _call_student(
        "sys", _classification_case(), client, _config(), input_template="{{ premise }}"
    )
    assert malformed is False
    assert output is None


@pytest.mark.asyncio
async def test_call_student_exception_is_malformed() -> None:
    """A provider that throws marks the case as explicitly malformed."""
    from rulekiln.pipeline.evaluator import _call_student  # pyright: ignore[reportPrivateUsage]

    class _ThrowingClient(ChatModelClient):
        async def complete_structured(  # type: ignore[override]
            self,
            *,
            system_prompt: str,
            user_prompt: str,
            output_schema: type,
            config: ProviderConfig,
        ) -> ChatCompletionResult:
            raise RuntimeError("LLM server unavailable")

    output, malformed, raw_content = await _call_student(
        "sys", _classification_case(), _ThrowingClient(), _config(), input_template="{{ premise }}"
    )
    assert malformed is True
    assert output is None
    assert raw_content is None
