"""Row-to-RuleKilnCase conversion."""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import TypeAdapter

from rulekiln.importers.column_mapping import CsvColumnMapping, CsvImportMapping
from rulekiln.importers.type_inference import JsonValue, infer_value
from rulekiln.schemas.task_case import (
    AssertionType,
    EvaluationAssertion,
    EvaluationSpec,
    RuleKilnCase,
    TaskMode,
)

type SplitName = Literal["train", "validation", "test", "golden"]

type CaseFieldValue = JsonValue

_VALID_SPLITS = frozenset({"train", "validation", "test", "golden"})
_TASK_MODE_ADAPTER = TypeAdapter(TaskMode)
_ASSERTION_TYPE_ADAPTER = TypeAdapter(AssertionType)


def _assign_split(case_id: str) -> SplitName:
    digest = int(hashlib.sha256(case_id.encode("utf-8")).hexdigest(), 16)
    return "train" if (digest % 100) < 80 else "validation"


def _set_nested_value(target: dict[str, object], path: str, value: CaseFieldValue) -> None:
    parts = [part.strip() for part in path.split(".") if part.strip()]
    if not parts:
        return
    cursor = target
    for part in parts[:-1]:
        existing = cursor.get(part)
        if not isinstance(existing, dict):
            existing = {}
            cursor[part] = existing
        cursor = existing
    cursor[parts[-1]] = value


def generate_case(
    row: dict[str, str],
    mapping: CsvImportMapping,
    row_index: int,
) -> RuleKilnCase:
    """Generate a single RuleKilnCase from one CSV row."""
    input_data: dict[str, object] = {}
    expected_data: dict[str, object] = {}
    metadata: dict[str, object] = {}
    assertions: list[EvaluationAssertion] = []
    case_id: str | None = None
    split: str | None = None

    column_map: dict[str, CsvColumnMapping] = {
        column.column_name: column for column in mapping.columns
    }
    task_mode = _TASK_MODE_ADAPTER.validate_python(mapping.task_mode)

    for column_name, raw_value in row.items():
        column_config = column_map.get(column_name)
        if column_config is None or column_config.role == "ignore":
            continue

        inferred = infer_value(raw_value)
        key = column_config.path or column_name

        if column_config.role == "case_id":
            case_id = raw_value.strip() or None
        elif column_config.role == "split":
            split = raw_value.strip().lower()
        elif column_config.role == "input":
            _set_nested_value(input_data, key, inferred)
        elif column_config.role == "expected":
            _set_nested_value(expected_data, key, inferred)
            if column_config.create_assertion and inferred is not None:
                assertions.append(
                    EvaluationAssertion(
                        type="must_equal",
                        path=key,
                        value=inferred,
                    )
                )
        elif column_config.role == "metadata":
            _set_nested_value(metadata, key, inferred)
        elif (
            column_config.role == "assertion"
            and column_config.assertion_type is not None
            and raw_value.strip()
        ):
            assertion_type = _ASSERTION_TYPE_ADAPTER.validate_python(column_config.assertion_type)
            assertions.append(
                EvaluationAssertion(
                    type=assertion_type,
                    path=column_config.path,
                    value=inferred,
                )
            )

    resolved_case_id = case_id if case_id else f"case_{row_index:06d}"
    resolved_split: SplitName = _assign_split(resolved_case_id)
    if split is not None:
        resolved_split = split if split in _VALID_SPLITS else "train"

    resolved_expected: dict[str, object] | str | None = expected_data if expected_data else None

    return RuleKilnCase(
        id=resolved_case_id,
        split=resolved_split,
        task_mode=task_mode,
        input=input_data,
        expected=resolved_expected,
        evaluation=EvaluationSpec(assertions=assertions),
        metadata=metadata,
    )


def generate_cases(
    rows: list[dict[str, str]],
    mapping: CsvImportMapping,
) -> tuple[list[RuleKilnCase], list[str]]:
    """Generate cases from rows. Returns (cases, errors)."""
    errors: list[str] = []
    cases: list[RuleKilnCase] = []
    seen_ids: set[str] = set()

    for index, row in enumerate(rows, start=1):
        case = generate_case(row, mapping, index)
        if case.id in seen_ids:
            errors.append(f"Duplicate case ID '{case.id}' at row {index}.")
            continue
        seen_ids.add(case.id)
        cases.append(case)

    return cases, errors
