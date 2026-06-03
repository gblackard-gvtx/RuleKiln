"""Unit tests for the extraction cache module."""

from __future__ import annotations

from pathlib import Path

import pytest

from rulekiln.pipeline.extraction_cache import (
    ExtractionCache,
    ExtractionCacheEntry,
    ExtractionCacheKey,
)
from rulekiln.schemas.pipeline import MicroRuleSchema


def _key(
    *,
    input_data: dict[str, object] | None = None,
    gold_label: str = "travel",
    provider: str = "fake",
    model: str = "fake-model",
    prompt_version: str = "v1",
) -> ExtractionCacheKey:
    return ExtractionCacheKey.build(
        input_data=input_data or {"text": "sample"},
        gold_label=gold_label,
        provider=provider,
        model=model,
        prompt_version=prompt_version,
    )


def _entry(key: ExtractionCacheKey) -> ExtractionCacheEntry:
    rule = MicroRuleSchema(
        topic="greeting",
        condition="input starts with 'hi'",
        expected_outcome="travel",
        output_path="label",
    )
    return ExtractionCacheEntry(
        cache_key=key.cache_key(),
        model_id="fake/fake-model",
        prompt_version="v1",
        micro_rule=rule,
    )


# ── ExtractionCacheKey ───────────────────────────────────────────────────────


def test_cache_key_is_deterministic() -> None:
    k1 = _key()
    k2 = _key()
    assert k1.cache_key() == k2.cache_key()


def test_different_gold_label_produces_different_key() -> None:
    k1 = _key(gold_label="travel")
    k2 = _key(gold_label="billing")
    assert k1.cache_key() != k2.cache_key()


def test_different_model_produces_different_key() -> None:
    k1 = _key(model="model-a")
    k2 = _key(model="model-b")
    assert k1.cache_key() != k2.cache_key()


def test_different_prompt_version_produces_different_key() -> None:
    k1 = _key(prompt_version="v1")
    k2 = _key(prompt_version="v2")
    assert k1.cache_key() != k2.cache_key()


def test_different_input_produces_different_key() -> None:
    k1 = _key(input_data={"text": "book a flight"})
    k2 = _key(input_data={"text": "cancel my order"})
    assert k1.cache_key() != k2.cache_key()


def test_model_id_normalized_to_lowercase() -> None:
    k = _key(provider="Fake", model="Fake-Model")
    assert k.model_id == "fake/fake-model"


# ── ExtractionCache ──────────────────────────────────────────────────────────


def test_cache_miss_returns_none(tmp_path: Path) -> None:
    cache = ExtractionCache(tmp_path)
    k = _key()
    result = cache.get("dataset_a", k)
    assert result is None


def test_put_then_get_returns_same_entry(tmp_path: Path) -> None:
    cache = ExtractionCache(tmp_path)
    k = _key()
    e = _entry(k)
    cache.put("dataset_a", k, e)
    result = cache.get("dataset_a", k)
    assert result is not None
    assert result.micro_rule.topic == "greeting"
    assert result.cache_key == k.cache_key()


def test_entry_is_byte_identical_after_roundtrip(tmp_path: Path) -> None:
    cache = ExtractionCache(tmp_path)
    k = _key()
    e = _entry(k)
    cache.put("dataset_a", k, e)
    result = cache.get("dataset_a", k)
    assert result is not None
    assert result.model_dump() == e.model_dump()


def test_cache_stats_track_hits_and_misses(tmp_path: Path) -> None:
    cache = ExtractionCache(tmp_path)
    k = _key()
    e = _entry(k)

    cache.get("ds", k)  # miss
    cache.put("ds", k, e)
    cache.get("ds", k)  # hit
    cache.get("ds", k)  # hit

    s = cache.stats()
    assert s["misses"] == 1
    assert s["hits"] == 2


def test_cache_miss_does_not_raise_on_absent_key(tmp_path: Path) -> None:
    cache = ExtractionCache(tmp_path)
    k = _key(input_data={"text": "never stored"})
    result = cache.get("ds", k)
    assert result is None


def test_different_dataset_names_are_isolated(tmp_path: Path) -> None:
    cache = ExtractionCache(tmp_path)
    k = _key()
    e = _entry(k)
    cache.put("ds_a", k, e)
    result = cache.get("ds_b", k)
    assert result is None


def test_entry_written_atomically(tmp_path: Path) -> None:
    """Verify put leaves a valid JSON file (no partial write)."""
    cache = ExtractionCache(tmp_path)
    k = _key()
    e = _entry(k)
    cache.put("ds", k, e)
    entry_path = tmp_path / "ds" / k.cache_key()[:2] / f"{k.cache_key()}.json"
    assert entry_path.exists()
    loaded = ExtractionCacheEntry.model_validate_json(entry_path.read_text())
    assert loaded.cache_key == k.cache_key()


def test_schema_version_is_set() -> None:
    k = _key()
    e = _entry(k)
    assert e.schema_version == "rulekiln.extraction_cache.v1"


@pytest.mark.parametrize(
    "field,value_a,value_b",
    [
        ("gold_label", "travel", "billing"),
        ("model", "model-a", "model-b"),
        ("prompt_version", "v1", "v2"),
    ],
)
def test_property_change_produces_different_key(
    field: str, value_a: str, value_b: str
) -> None:
    k1 = _key(**{field: value_a})  # type: ignore[arg-type]
    k2 = _key(**{field: value_b})  # type: ignore[arg-type]
    assert k1.cache_key() != k2.cache_key()
