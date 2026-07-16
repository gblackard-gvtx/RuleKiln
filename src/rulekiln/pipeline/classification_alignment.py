"""Shared classification alignment: declared label vocabulary, case/label columns,
and strict-vs-valid-output statistics.

Per ADR-0002, strict metrics are computed over every assigned case (malformed
predictions counted as incorrect); valid-output metrics are diagnostic-only and
computed over the subset with a well-formed prediction. Both views must share the
same declared label vocabulary so a label's disappearance from the valid-output
subset can't be mistaken for a genuine score change.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from rulekiln.pipeline.statistics import ClassificationStatistics, compute_classification_statistics
from rulekiln.schemas.pipeline import CaseEvalResult
from rulekiln.schemas.task_case import RuleKilnCase, RuleKilnTask


def declared_label_vocabulary(task: RuleKilnTask) -> tuple[str, ...] | None:
    """Return the sorted declared label vocabulary for classification/routing tasks.

    Derived from enum values in the task output schema. Returns None when the task
    mode has no fixed label vocabulary or the schema declares no enums. A malformed
    prediction's placeholder value never matches any label in this vocabulary, so it
    naturally becomes a false negative against the true label with no exclusion logic
    required.
    """
    if task.task_mode not in ("classification", "routing") or not task.output_schema:
        return None
    enums: set[str] = set()
    properties = task.output_schema.get("properties", {})
    for prop_schema in properties.values():
        if isinstance(prop_schema, dict):
            enum_vals = prop_schema.get("enum", [])
            if isinstance(enum_vals, list):
                enums.update(str(v) for v in enum_vals)
    return tuple(sorted(enums)) if enums else None


def _expected_label_from_case(case: RuleKilnCase) -> str | None:
    if case.expected is None:
        return None
    expected = case.expected
    expected_raw = expected.get("label", "") if isinstance(expected, dict) else expected
    return expected_raw if isinstance(expected_raw, str) else str(expected_raw or "")


def _predicted_label_from_case_result(case_result: CaseEvalResult) -> str:
    if case_result.malformed or case_result.actual_output is None:
        return ""
    actual = case_result.actual_output
    actual_raw = actual.get("label", "") if isinstance(actual, dict) else actual
    return actual_raw if isinstance(actual_raw, str) else str(actual_raw)


def build_classification_label_columns(
    cases: Sequence[RuleKilnCase],
    case_results: Sequence[CaseEvalResult],
) -> tuple[list[str], list[str], list[str], list[bool]]:
    """Align cases with their case results into parallel label columns.

    Returns ``(case_ids, expected_labels, predicted_labels, malformed_flags)``. A case
    is included whenever it has both a declared expected label and a case result —
    malformed cases are included (predicted label ``""``, ``malformed=True``) so strict
    metrics can count them as incorrect against the true label.
    """
    case_results_by_id = {result.case_id: result for result in case_results}
    case_ids: list[str] = []
    expected_labels: list[str] = []
    predicted_labels: list[str] = []
    malformed_flags: list[bool] = []

    for case in cases:
        expected_label = _expected_label_from_case(case)
        if expected_label is None:
            continue
        case_result = case_results_by_id.get(case.id)
        if case_result is None:
            continue
        case_ids.append(case.id)
        expected_labels.append(expected_label)
        predicted_labels.append(_predicted_label_from_case_result(case_result))
        malformed_flags.append(case_result.malformed)

    return case_ids, expected_labels, predicted_labels, malformed_flags


class StrictAndValidOutputStatistics(BaseModel):
    """Strict (all cases) and valid-output (malformed filtered) classification statistics."""

    strict: ClassificationStatistics
    valid_output: ClassificationStatistics | None


def compute_strict_and_valid_output_statistics(
    *,
    case_ids: Sequence[str],
    expected_labels: Sequence[str],
    predicted_labels: Sequence[str],
    malformed_flags: Sequence[bool],
    label_vocabulary: Sequence[str] | None,
    bootstrap_enabled: bool,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> StrictAndValidOutputStatistics:
    """Compute strict and valid-output statistics using the same label vocabulary.

    ``valid_output`` is ``None`` when zero valid-output cases remain (never a
    misleadingly computed 0.0).
    """
    strict = compute_classification_statistics(
        actual_labels=expected_labels,
        predicted_labels=predicted_labels,
        case_ids=case_ids,
        bootstrap_enabled=bootstrap_enabled,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
        label_vocabulary=label_vocabulary,
    )

    valid_indices = [i for i, malformed in enumerate(malformed_flags) if not malformed]
    valid_output: ClassificationStatistics | None = None
    if valid_indices:
        valid_output = compute_classification_statistics(
            actual_labels=[expected_labels[i] for i in valid_indices],
            predicted_labels=[predicted_labels[i] for i in valid_indices],
            case_ids=[case_ids[i] for i in valid_indices],
            bootstrap_enabled=bootstrap_enabled,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
            label_vocabulary=label_vocabulary,
        )

    return StrictAndValidOutputStatistics(strict=strict, valid_output=valid_output)
