"""Unit tests for rule ablation classification logic."""

from __future__ import annotations

import pytest

from rulekiln.workers.distillation_worker import _ablation_classify


@pytest.mark.parametrize(
    "delta, changed, min_changed, expected",
    [
        # Helpful: removing rule worsens score (negative delta)
        (-0.01, 10, 5, "helpful"),
        (-0.006, 5, 5, "helpful"),
        # Harmful: removing rule improves score (positive delta)
        (0.01, 10, 5, "harmful"),
        (0.006, 5, 5, "harmful"),
        # Neutral: small absolute delta
        (0.005, 10, 5, "neutral"),
        (-0.005, 10, 5, "neutral"),
        (0.0, 10, 5, "neutral"),
        # Inconclusive: too few changed cases
        (-0.02, 4, 5, "inconclusive"),
        (0.02, 0, 5, "inconclusive"),
        (-0.02, 3, 5, "inconclusive"),
    ],
)
def test_ablation_classify(delta: float, changed: int, min_changed: int, expected: str) -> None:
    result = _ablation_classify(delta, changed, min_changed)
    assert result == expected


def test_ablation_classify_boundary_helpful() -> None:
    # Delta just below -0.005 threshold → helpful
    assert _ablation_classify(-0.0051, 10, 5) == "helpful"


def test_ablation_classify_boundary_neutral() -> None:
    # Exactly -0.005 → neutral
    assert _ablation_classify(-0.005, 10, 5) == "neutral"
    assert _ablation_classify(0.005, 10, 5) == "neutral"


def test_ablation_classify_zero_changed_is_inconclusive() -> None:
    # Even large delta is inconclusive when not enough cases changed
    assert _ablation_classify(-0.5, 0, 5) == "inconclusive"
    assert _ablation_classify(0.5, 0, 5) == "inconclusive"
