"""Centralized split policy for extraction and evaluation case routing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, NamedTuple

from rulekiln.schemas.task_case import RuleKilnCase

CaseSplit = Literal["train", "validation", "test", "golden"]


class SplitPolicyDecision(NamedTuple):
    """Resolved split routing used by the worker and preview surfaces."""

    extraction_split: CaseSplit
    extraction_cases: list[RuleKilnCase]
    evaluation_split: CaseSplit
    evaluation_cases: list[RuleKilnCase]
    split_counts: dict[str, int]
    fallback_warning: str | None


def _count_cases_by_split(cases: Sequence[RuleKilnCase]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.split] = counts.get(case.split, 0) + 1
    return counts


def resolve_split_policy(cases: Sequence[RuleKilnCase]) -> SplitPolicyDecision:
    """Resolve extraction/evaluation routing from case splits.

    Policy:
    - Extraction uses only ``train`` rows.
    - Evaluation uses ``validation`` when present.
    - If ``validation`` is absent, evaluation falls back to ``train``.
    - If both ``validation`` and ``train`` are absent, evaluation falls back to
      ``test`` and then ``golden`` as a last resort.
    """

    all_cases = list(cases)
    split_counts = _count_cases_by_split(all_cases)

    train_cases = [case for case in all_cases if case.split == "train"]
    validation_cases = [case for case in all_cases if case.split == "validation"]
    test_cases = [case for case in all_cases if case.split == "test"]
    golden_cases = [case for case in all_cases if case.split == "golden"]

    if validation_cases:
        return SplitPolicyDecision(
            extraction_split="train",
            extraction_cases=train_cases,
            evaluation_split="validation",
            evaluation_cases=validation_cases,
            split_counts=split_counts,
            fallback_warning=None,
        )

    if train_cases:
        return SplitPolicyDecision(
            extraction_split="train",
            extraction_cases=train_cases,
            evaluation_split="train",
            evaluation_cases=train_cases,
            split_counts=split_counts,
            fallback_warning=(
                "No validation cases detected. Evaluation fell back to split=train."
            ),
        )

    if test_cases:
        return SplitPolicyDecision(
            extraction_split="train",
            extraction_cases=train_cases,
            evaluation_split="test",
            evaluation_cases=test_cases,
            split_counts=split_counts,
            fallback_warning=(
                "No validation/train cases detected. Evaluation fell back to split=test."
            ),
        )

    return SplitPolicyDecision(
        extraction_split="train",
        extraction_cases=train_cases,
        evaluation_split="golden",
        evaluation_cases=golden_cases,
        split_counts=split_counts,
        fallback_warning=(
            "No validation/train/test cases detected. Evaluation fell back to split=golden."
        ),
    )
