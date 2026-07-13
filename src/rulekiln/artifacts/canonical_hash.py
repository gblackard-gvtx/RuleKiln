"""Canonical JSON serialization and SHA-256 hashing for reproducibility artifacts.

Every reproducibility hash in RuleKiln flows through this module so that hashes are
stable across dictionary ordering, process restarts, and insignificant JSON formatting
differences. The canonicalization rules are:

* Mappings are canonicalized by key (sorted); ordered lists remain order-sensitive.
* Strings are UTF-8 encoded with JSON escaping.
* Floats use Python's shortest round-trip ``repr`` (``-0.0`` normalized to ``0.0``);
  non-finite floats (NaN/Inf) are rejected because they are neither valid JSON nor
  reproducible.
* Integers and booleans are distinct from floats (``2`` and ``2.0`` hash differently).
* ``None`` is encoded as JSON ``null`` and is preserved. A ``null`` value is therefore
  distinct from an omitted key: the caller controls which keys are present, and two
  mappings that differ only by an explicit ``null`` vs. an absent key hash differently.

The algorithm and canonicalization version are recorded as module constants so callers
can persist them alongside the hashes they produce.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence

from pydantic import BaseModel

ALGORITHM = "sha256"
CANONICALIZATION_VERSION = "v1"


def _encode_float(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        raise ValueError("Cannot canonicalize a non-finite float (NaN/Inf).")
    if value == 0.0:
        # Normalize -0.0 to 0.0 so the sign of zero does not affect the hash.
        value = 0.0
    return repr(value)


def _encode(obj: object) -> str:
    """Return the canonical string encoding of a JSON-compatible object."""
    if isinstance(obj, BaseModel):
        return _encode(obj.model_dump(mode="json"))
    if obj is None:
        return "null"
    # bool must be checked before int, since bool is a subclass of int.
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, float):
        return _encode_float(obj)
    if isinstance(obj, str):
        return json.dumps(obj, ensure_ascii=False)
    if isinstance(obj, Mapping):
        items = sorted((str(key), value) for key, value in obj.items())
        body = ",".join(
            f"{json.dumps(key, ensure_ascii=False)}:{_encode(value)}" for key, value in items
        )
        return "{" + body + "}"
    if isinstance(obj, Sequence):
        return "[" + ",".join(_encode(item) for item in obj) + "]"
    raise TypeError(f"Cannot canonicalize object of type {type(obj)!r}.")


def canonical_json(obj: object) -> str:
    """Serialize an object to its canonical, formatting-insensitive string form."""
    return _encode(obj)


def canonical_hash(obj: object) -> str:
    """Return the hex SHA-256 of an object's canonical JSON encoding."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def hash_text(text: str) -> str:
    """Return the hex SHA-256 of a UTF-8 string (used for prompts and per-rule text)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_matrix(matrix: Sequence[Sequence[float]]) -> str:
    """Return a stable hash of an embedding matrix without persisting the vectors.

    The matrix is canonicalized as its shape plus row values, so the hash changes when
    any value or the dimensionality changes, while the raw vectors stay out of JSON.
    """
    rows = [[float(value) for value in row] for row in matrix]
    payload = {
        "rows": len(rows),
        "cols": len(rows[0]) if rows else 0,
        "values": rows,
    }
    return canonical_hash(payload)
