"""Import-specific validation."""

from __future__ import annotations

from rulekiln.importers.column_mapping import CsvImportMapping

_VALID_SPLITS = frozenset({"train", "validation", "test", "golden"})


def validate_import_mapping(
    mapping: CsvImportMapping,
    rows: list[dict[str, str]],
) -> tuple[list[str], list[str]]:
    """Returns (errors, warnings). Errors are blocking; warnings are not."""
    errors: list[str] = []
    warnings: list[str] = []

    input_columns = [column for column in mapping.columns if column.role == "input"]
    expected_columns = [
        column for column in mapping.columns if column.role in ("expected", "assertion")
    ]

    if not input_columns:
        errors.append("No input columns mapped. At least one column must have role 'input'.")

    if not expected_columns:
        errors.append(
            "No expected or assertion columns mapped. "
            "At least one column must have role 'expected' or 'assertion'."
        )

    if not rows:
        errors.append("CSV contains zero rows.")

    split_columns = [column for column in mapping.columns if column.role == "split"]
    if split_columns:
        split_column_name = split_columns[0].column_name
        invalid_splits: set[str] = set()
        for row in rows:
            value = row.get(split_column_name, "").strip().lower()
            if value and value not in _VALID_SPLITS:
                invalid_splits.add(value)
        for split_name in sorted(invalid_splits):
            errors.append(
                f"Invalid split value '{split_name}' in column '{split_column_name}'. "
                f"Allowed: {sorted(_VALID_SPLITS)!r}."
            )
    else:
        warnings.append(
            "No split column mapped — splits will be assigned deterministically "
            "(80% train / 20% validation)."
        )

    id_columns = [column for column in mapping.columns if column.role == "case_id"]
    if id_columns:
        id_column_name = id_columns[0].column_name
        seen_ids: set[str] = set()
        for row_index, row in enumerate(rows, start=1):
            value = row.get(id_column_name, "").strip()
            if not value:
                continue
            if value in seen_ids:
                errors.append(
                    f"Duplicate case ID '{value}' in column '{id_column_name}' at row {row_index}."
                )
            seen_ids.add(value)

    has_validation = False
    if split_columns:
        split_column_name = split_columns[0].column_name
        has_validation = any(
            row.get(split_column_name, "").strip().lower() == "validation" for row in rows
        )
    if not has_validation and not split_columns:
        warnings.append(
            "No validation split present — 20% of cases will be auto-assigned to validation."
        )

    return errors, warnings
