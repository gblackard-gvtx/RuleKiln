"""Extraction cache: on-disk, concurrent-safe cache for micro-rule extraction results.

Cache entries are keyed by a hash of (input, expected label, model_id, prompt_version).
Changing the extraction model or prompt version produces a new key; changing only the
student model, synthesis model, or conflict-resolution model does NOT invalidate entries
because extraction outputs are teacher-only artifacts.

Atomic writes (write to temp file then rename) prevent corrupt entries from crashed workers.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from rulekiln.schemas.pipeline import MicroRuleSchema


class ExtractionCacheKey(BaseModel):
    """Identifies one extraction cache entry deterministically."""

    input_hash: str
    gold_label: str
    model_id: str  # "{provider}/{model_name}" normalized to lowercase
    prompt_version: str

    def cache_key(self) -> str:
        """Return the hex digest used for filesystem path and lookup."""
        raw = json.dumps(self.model_dump(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @classmethod
    def build(
        cls,
        *,
        input_data: dict[str, object],
        gold_label: str,
        provider: str,
        model: str,
        prompt_version: str,
    ) -> ExtractionCacheKey:
        """Build a cache key from raw inputs."""
        serialized = json.dumps(input_data, sort_keys=True, ensure_ascii=False)
        input_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        model_id = f"{provider.lower()}/{model.lower()}"
        return cls(
            input_hash=input_hash,
            gold_label=gold_label,
            model_id=model_id,
            prompt_version=prompt_version,
        )


class ExtractionCacheEntry(BaseModel):
    """A cached extraction result for a single training case."""

    schema_version: Literal["rulekiln.extraction_cache.v1"] = "rulekiln.extraction_cache.v1"
    cache_key: str
    cached_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model_id: str
    prompt_version: str
    micro_rule: MicroRuleSchema
    reasoning_trace: str | None = None


class ExtractionCache:
    """Filesystem-backed extraction cache with atomic writes.

    Layout: ``{cache_root}/{dataset_name}/{key[:2]}/{key}.json``

    All writes use a write-to-temp-then-rename pattern so a crashed worker
    cannot leave a partial entry that subsequent readers mistake for a valid hit.
    """

    def __init__(self, cache_root: Path) -> None:
        self._root = cache_root
        self._hits = 0
        self._misses = 0

    def _entry_path(self, dataset_name: str, key: str) -> Path:
        return self._root / dataset_name / key[:2] / f"{key}.json"

    def get(self, dataset_name: str, cache_key: ExtractionCacheKey) -> ExtractionCacheEntry | None:
        """Return a cached entry, or ``None`` on a miss.  Never raises on a missing key."""
        key = cache_key.cache_key()
        path = self._entry_path(dataset_name, key)
        if not path.exists():
            self._misses += 1
            return None
        try:
            entry = ExtractionCacheEntry.model_validate_json(path.read_text(encoding="utf-8"))
            self._hits += 1
            return entry
        except Exception:
            self._misses += 1
            return None

    def put(
        self,
        dataset_name: str,
        cache_key: ExtractionCacheKey,
        entry: ExtractionCacheEntry,
    ) -> None:
        """Write an entry atomically (temp-file then rename)."""
        key = cache_key.cache_key()
        path = self._entry_path(dataset_name, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = entry.model_dump_json()
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            os.write(fd, payload.encode("utf-8"))
            os.close(fd)
            os.replace(tmp_path, path)
        except Exception:
            with contextlib.suppress(OSError):
                os.close(fd)
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    def stats(self) -> dict[str, int]:
        """Return hit/miss/total counts accumulated since construction."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_entries": self._hits,  # entries seen this session
        }
