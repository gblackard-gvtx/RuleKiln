"""Deterministic type inference for CSV cell values."""

from __future__ import annotations

import json
from typing import cast

type JsonValue = bool | int | float | str | None | dict[str, JsonValue] | list[JsonValue]

type InferredValue = JsonValue


def infer_value(raw: str) -> InferredValue:
    """Infer the Python type of a raw CSV string value."""
    stripped = raw.strip()
    if stripped == "":
        return None
    lower = stripped.lower()
    if lower in ("true", "false"):
        return lower == "true"
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        pass
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return cast(dict[str, JsonValue], parsed)
        if isinstance(parsed, list):
            return cast(list[JsonValue], parsed)
    except (json.JSONDecodeError, ValueError):
        pass
    return raw


def infer_column_type(samples: list[str]) -> str:
    """Return majority inferred type name across sample values."""
    if not samples:
        return "string"
    type_counts: dict[str, int] = {}
    for sample in samples:
        value = infer_value(sample)
        if value is None:
            type_name = "null"
        elif isinstance(value, bool):
            type_name = "boolean"
        elif isinstance(value, int):
            type_name = "integer"
        elif isinstance(value, float):
            type_name = "number"
        elif isinstance(value, dict):
            type_name = "object"
        elif isinstance(value, list):
            type_name = "array"
        else:
            type_name = "string"
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
    return max(type_counts, key=type_counts.__getitem__)
