"""Unit tests for worker error classification."""

from __future__ import annotations

from pydantic import BaseModel
from pydantic import ValidationError

from rulekiln.api.validators.request_shape import RequestValidationError
from rulekiln.providers.contracts import ProviderNotConfiguredError
from rulekiln.workers.error_classification import classify_worker_error


def test_connection_error_is_retryable() -> None:
    classification = classify_worker_error(ConnectionError("connection refused"))
    assert classification.retryable is True


def test_timeout_message_is_retryable() -> None:
    classification = classify_worker_error(RuntimeError("HTTP 503 timeout from provider"))
    assert classification.retryable is True


def test_request_validation_error_is_terminal() -> None:
    classification = classify_worker_error(RequestValidationError("bad request"))
    assert classification.retryable is False


def test_provider_not_configured_error_is_terminal() -> None:
    classification = classify_worker_error(
        ProviderNotConfiguredError("openai", "missing api key")
    )
    assert classification.retryable is False


def test_pydantic_validation_error_is_terminal() -> None:
    class _Model(BaseModel):
        value: int

    try:
        _Model.model_validate({"value": "not-an-int"})
    except ValidationError as exc:
        classification = classify_worker_error(exc)

    assert classification.retryable is False
