"""Security helpers: mask secrets and sensitive data before logging."""

from __future__ import annotations

import re

# Patterns for URL credentials (username:password@host)
_URL_CRED_RE = re.compile(r"(://)[^:@/]+:[^@/]+(@)", re.IGNORECASE)

# Query-string parameter patterns
_QS_PARAMS: list[re.Pattern[str]] = [
    re.compile(r"([?&]api[_-]?key=)[^&\s]*", re.IGNORECASE),
    re.compile(r"([?&]token=)[^&\s]*", re.IGNORECASE),
    re.compile(r"([?&]secret=)[^&\s]*", re.IGNORECASE),
    re.compile(r"([?&]password=)[^&\s]*", re.IGNORECASE),
    re.compile(r"([?&]key=)[^&\s]*", re.IGNORECASE),
]


def mask_url(url: str) -> str:
    """Replace credential portions of a URL with masked placeholders.

    Example:
        "https://user:pass@host/path?api_key=secret"
        → "https://***:***@host/path?api_key=***MASKED***"
    """
    masked = _URL_CRED_RE.sub(r"\1***:***\2", url)
    for pattern in _QS_PARAMS:
        masked = pattern.sub(r"\1***MASKED***", masked)
    return masked


def mask_dict_values(data: dict[str, str]) -> dict[str, str]:
    """Return a copy of *data* with secret-looking values replaced by '***REDACTED***'."""
    _SECRET_KEYS = re.compile(
        r"(api[_-]?key|secret|password|token|credential|auth)",
        re.IGNORECASE,
    )
    return {
        k: "***REDACTED***" if _SECRET_KEYS.search(k) else v
        for k, v in data.items()
    }
