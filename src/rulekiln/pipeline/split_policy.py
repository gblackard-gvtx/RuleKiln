"""Centralized split policy for extraction and evaluation case routing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, NamedTuple

from rulekiln.schemas.task_case import RuleKilnCase

CaseSplit = Literal["train", "validation", "test", "golden"]


class FewShotExemplarLeakageError(RuntimeError):
    """Raised when few-shot exemplars would be drawn from the evaluation split.

    Few-shot exemplars must come from ``train`` (extraction) cases. If no train
    cases are present, falling back to evaluation cases would embed the exact
    cases being scored into the prompt used to score them.
    """


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
            fallback_warning=("No validation cases detected. Evaluation fell back to split=train."),
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


def resolve_few_shot_source_cases(
    extraction_cases: Sequence[RuleKilnCase],
    eval_cases: Sequence[RuleKilnCase],
    *,
    allow_eval_fallback: bool,
) -> list[RuleKilnCase]:
    """Resolve the case pool used for few-shot exemplar selection.

    Fails closed when no ``train`` cases are available rather than silently
    substituting the evaluation cases, unless the caller has explicitly opted
    into that risk via ``RuleKilnTask.allow_few_shot_exemplars_from_eval_split``.
    """
    if extraction_cases:
        return list(extraction_cases)
    if not allow_eval_fallback:
        raise FewShotExemplarLeakageError(
            "No train-split cases are available for few-shot exemplar selection. "
            "Falling back to evaluation cases would leak the cases being scored "
            "into the prompt used to score them. Set "
            "RuleKilnTask.allow_few_shot_exemplars_from_eval_split=True to accept "
            "this risk explicitly, or include train-split cases in the case set."
        )
    return list(eval_cases)
