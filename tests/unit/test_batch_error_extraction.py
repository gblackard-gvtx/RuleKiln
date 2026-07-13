"""Unit tests for OpenAI batch per-item error extraction.

Regression coverage for the incident where all 500 extraction items failed but
every error surfaced as the generic placeholder ``"Provider error"``, discarding
the real cause. The real OpenAI error lives in the top-level ``error`` field or
in ``response.body.error``; either may be populated.
"""

from __future__ import annotations

from rulekiln.providers.chat.openai_chat import (
    _extract_batch_item_error,
    _format_error_dict,
)
from rulekiln.workers.error_classification import classify_worker_error
from rulekiln.workers.pipeline.early_stages import (
    ExtractionYieldedNoRulesError,
    _dominant_failure_reason,
)


def test_format_error_dict_combines_message_and_fields() -> None:
    rendered = _format_error_dict(
        {
            "message": "Request too large",
            "code": "context_length_exceeded",
            "type": "invalid_request_error",
            "param": None,
        }
    )
    assert rendered is not None
    assert "Request too large" in rendered
    assert "code=context_length_exceeded" in rendered
    assert "type=invalid_request_error" in rendered
    # param is None → omitted rather than rendered as param=None
    assert "param=" not in rendered


def test_format_error_dict_empty_returns_none() -> None:
    assert _format_error_dict({}) is None
    assert _format_error_dict({"message": "", "code": None}) is None


def test_extract_top_level_error_dict() -> None:
    row: dict[str, object] = {
        "custom_id": "c1",
        "error": {"message": "You exceeded your rate limit", "code": "rate_limit_exceeded"},
        "response": None,
    }
    msg = _extract_batch_item_error(row)
    assert "You exceeded your rate limit" in msg
    assert "code=rate_limit_exceeded" in msg


def test_extract_top_level_error_string() -> None:
    row: dict[str, object] = {"custom_id": "c1", "error": "raw failure text"}
    assert _extract_batch_item_error(row) == "raw failure text"


def test_extract_nested_response_body_error_with_status() -> None:
    row: dict[str, object] = {
        "custom_id": "c1",
        "error": None,
        "response": {
            "status_code": 400,
            "body": {"error": {"message": "Invalid model", "type": "invalid_request_error"}},
        },
    }
    msg = _extract_batch_item_error(row)
    assert "status_code=400" in msg
    assert "Invalid model" in msg


def test_extract_non_200_without_body_error_falls_back_to_status() -> None:
    row: dict[str, object] = {
        "custom_id": "c1",
        "error": None,
        "response": {"status_code": 503, "body": {}},
    }
    msg = _extract_batch_item_error(row)
    assert "status_code=503" in msg


def test_extract_incident_shape_recovers_raw_row() -> None:
    """The incident's rows had error=null with no usable detail.

    Previously these defaulted to the literal ``"Provider error"``; now the raw
    row is preserved so the true cause is recoverable instead of discarded.
    """
    row: dict[str, object] = {"custom_id": "contractnli_train_000028", "error": None}
    msg = _extract_batch_item_error(row)
    assert msg != "Provider error"
    assert "unstructured provider error" in msg
    assert "contractnli_train_000028" in msg


def test_extract_truncates_very_long_raw_row() -> None:
    row: dict[str, object] = {"custom_id": "c1", "error": None, "blob": "x" * 2000}
    msg = _extract_batch_item_error(row)
    assert msg.endswith("…")
    assert len(msg) < 700


def test_dominant_failure_reason_picks_most_common() -> None:
    messages = [
        "status_code=429 rate limit",
        "status_code=429 rate limit",
        "status_code=400 bad request",
    ]
    assert _dominant_failure_reason(messages) == "status_code=429 rate limit"


def test_dominant_failure_reason_preserves_retryable_substrings() -> None:
    # The worker error classifier keys off substrings like "timeout"/"rate limit",
    # so the full message must be preserved (not summarized away).
    reason = _dominant_failure_reason(["gateway timeout after 120s"])
    assert "timeout" in reason


def test_dominant_failure_reason_empty() -> None:
    assert _dominant_failure_reason([]) == "no error detail reported by provider"


def test_no_rules_error_is_retryable_when_provider_error_is_transient() -> None:
    exc = ExtractionYieldedNoRulesError(
        "Extraction produced 0 rules: 500/500 batch items failed. "
        "Most common provider error: status_code=429 rate limit"
    )
    assert classify_worker_error(exc).retryable is True


def test_no_rules_error_is_terminal_when_provider_error_is_permanent() -> None:
    exc = ExtractionYieldedNoRulesError(
        "Extraction produced 0 rules: 500/500 batch items failed. "
        "Most common provider error: Invalid model code=model_not_found"
    )
    assert classify_worker_error(exc).retryable is False
