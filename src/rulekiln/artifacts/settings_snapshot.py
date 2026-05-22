"""Settings snapshot export with secret redaction."""

from __future__ import annotations

import json
import re
from pathlib import Path

from rulekiln.config.settings import AppSettings

_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(api[_-]?key)", re.IGNORECASE),
    re.compile(r"(secret)", re.IGNORECASE),
    re.compile(r"(password)", re.IGNORECASE),
    re.compile(r"(token)", re.IGNORECASE),
    re.compile(r"(credentials)", re.IGNORECASE),
]

_URL_SECRET_PATTERN = re.compile(r"://([^:]+):([^@]+)@")


def _redact_value(key: str, value: object) -> object:
    key_lower = key.lower()
    for pattern in _SECRET_PATTERNS:
        if pattern.search(key_lower):
            return "***REDACTED***"
    if isinstance(value, str):
        return _URL_SECRET_PATTERN.sub("://***:***@", value)
    return value


def _redact_dict(d: dict[str, object]) -> dict[str, object]:
    return {k: _redact_value(k, v) for k, v in d.items()}


def build_settings_snapshot(settings: AppSettings) -> dict[str, object]:
    """Build a redacted snapshot of application settings for artifact export."""
    raw = settings.model_dump(mode="json", exclude={"openai_api_key"})

    # Redact top-level sensitive keys
    sanitized: dict[str, object] = {}
    for key, value in raw.items():
        redacted = _redact_value(key, value)
        if isinstance(redacted, dict):
            redacted = _redact_dict(redacted)  # type: ignore[arg-type]
        sanitized[key] = redacted

    return sanitized


def write_settings_snapshot(root: Path, settings: AppSettings) -> Path:
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    path = metadata / "settings_snapshot.json"
    snapshot = build_settings_snapshot(settings)
    path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
