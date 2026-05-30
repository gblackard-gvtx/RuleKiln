"""Classify worker exceptions into retryable or terminal categories."""

from __future__ import annotations

from pydantic import BaseModel, ValidationError

from rulekiln.api.validators.request_shape import RequestValidationError
from rulekiln.providers.contracts import ProviderNotConfiguredError, ProviderNotImplementedError

try:
    import httpx
except Exception:  # pragma: no cover - httpx is expected but keep guard explicit
    httpx = None  # type: ignore[assignment]


_RETRYABLE_SUBSTRINGS: tuple[str, ...] = (
    "connection refused",
    "connection reset",
    "temporarily unavailable",
    "timed out",
    "timeout",
    "502",
    "503",
    "504",
    "rate limit",
    "too many requests",
)


class ErrorClassification(BaseModel):
    """Result of classifying one worker exception."""

    retryable: bool
    error_type: str


def format_worker_error_message(exc: Exception) -> str:
    """Return a stable non-empty error message for logs and persistence."""
    message = str(exc).strip()
    if message:
        return message
    return type(exc).__name__


def classify_worker_error(exc: Exception) -> ErrorClassification:
    """Return retryability classification for a worker exception."""
    error_type = type(exc).__name__

    terminal_types: tuple[type[Exception], ...] = (
        ValueError,
        TypeError,
        KeyError,
        PermissionError,
        FileNotFoundError,
        RequestValidationError,
        ProviderNotConfiguredError,
        ProviderNotImplementedError,
        ValidationError,
    )
    if isinstance(exc, terminal_types):
        return ErrorClassification(retryable=False, error_type=error_type)

    retryable_types: tuple[type[Exception], ...] = (
        TimeoutError,
        ConnectionError,
    )
    if isinstance(exc, retryable_types):
        return ErrorClassification(retryable=True, error_type=error_type)

    if isinstance(exc, OSError):
        retryable_errnos = {104, 110, 111, 113}  # reset, timeout, refused, no route
        if exc.errno in retryable_errnos:
            return ErrorClassification(retryable=True, error_type=error_type)

    if httpx is not None:
        httpx_retryable_types: tuple[type[Exception], ...] = (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.TransportError,
        )
        if isinstance(exc, httpx_retryable_types):
            return ErrorClassification(retryable=True, error_type=error_type)

    message = format_worker_error_message(exc).lower()
    if any(token in message for token in _RETRYABLE_SUBSTRINGS):
        return ErrorClassification(retryable=True, error_type=error_type)

    return ErrorClassification(retryable=False, error_type=error_type)
