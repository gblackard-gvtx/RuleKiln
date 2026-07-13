"""Tests for canonical JSON hashing (reproducibility hash foundation)."""

import json

import pytest

from rulekiln.artifacts.canonical_hash import (
    ALGORITHM,
    CANONICALIZATION_VERSION,
    canonical_hash,
    canonical_json,
    hash_matrix,
    hash_text,
)


def test_algorithm_and_version_are_recorded() -> None:
    assert ALGORITHM == "sha256"
    assert CANONICALIZATION_VERSION == "v1"


def test_same_mapping_different_key_order_produces_same_hash() -> None:
    # M#1: dictionary ordering must not affect the hash.
    a = {"beta": 1, "alpha": 2, "gamma": {"y": 1, "x": 2}}
    b = {"gamma": {"x": 2, "y": 1}, "alpha": 2, "beta": 1}

    assert canonical_hash(a) == canonical_hash(b)


def test_different_ordered_case_id_lists_produce_different_hashes() -> None:
    # M#2: ordered lists remain order-sensitive.
    a = {"case_ids": ["c1", "c2", "c3"]}
    b = {"case_ids": ["c1", "c3", "c2"]}

    assert canonical_hash(a) != canonical_hash(b)


def test_null_value_differs_from_omitted_key() -> None:
    # M#3: an explicit null is distinct from an omitted key.
    with_null = {"eps": None, "min_samples": 5}
    omitted = {"min_samples": 5}

    assert canonical_hash(with_null) != canonical_hash(omitted)
    assert canonical_json(with_null) == '{"eps":null,"min_samples":5}'


def test_int_and_float_are_distinct() -> None:
    assert canonical_hash({"n": 2}) != canonical_hash({"n": 2.0})


def test_float_canonicalization_is_stable() -> None:
    # M#4: repeated hashing of the same float value is stable, and -0.0 == 0.0.
    assert canonical_json(0.1) == "0.1"
    assert canonical_json(-0.0) == "0.0"
    assert canonical_json(0.0) == "0.0"
    assert canonical_hash({"v": 1.5}) == canonical_hash({"v": 1.5})


def test_non_finite_floats_are_rejected() -> None:
    with pytest.raises(ValueError):
        canonical_json(float("nan"))
    with pytest.raises(ValueError):
        canonical_json(float("inf"))


def test_artifact_hash_changes_when_content_changes() -> None:
    # M#5: changing relevant content changes the hash.
    base = {"a": [1, 2, 3], "b": "x"}
    changed = {"a": [1, 2, 4], "b": "x"}

    assert canonical_hash(base) != canonical_hash(changed)


def test_formatting_only_json_changes_do_not_affect_hash() -> None:
    # M#6: semantically identical JSON with different formatting hashes identically.
    obj = {"a": 1, "nested": {"z": [1, 2], "y": "text"}}
    compact = json.loads(json.dumps(obj, separators=(",", ":")))
    pretty = json.loads(json.dumps(obj, indent=4, sort_keys=True))

    assert canonical_hash(compact) == canonical_hash(pretty)


def test_hash_text_is_plain_sha256() -> None:
    assert hash_text("hello") == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


def test_hash_matrix_is_shape_sensitive_and_value_sensitive() -> None:
    m1 = [[0.1, 0.2], [0.3, 0.4]]
    m2 = [[0.1, 0.2], [0.3, 0.5]]
    m3 = [[0.1, 0.2, 0.0], [0.3, 0.4, 0.0]]

    assert hash_matrix(m1) == hash_matrix([[0.1, 0.2], [0.3, 0.4]])
    assert hash_matrix(m1) != hash_matrix(m2)
    assert hash_matrix(m1) != hash_matrix(m3)
