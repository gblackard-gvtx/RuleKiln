"""CSV parsing and preview generation."""

from __future__ import annotations

import csv
import io

from rulekiln.importers.column_mapping import CsvColumnMapping, CsvImportPreview
from rulekiln.importers.type_inference import infer_column_type

_CASE_ID_NAMES = frozenset({"id", "case_id", "case id", "caseid"})
_SPLIT_NAMES = frozenset({"split", "partition", "fold"})
_EXPECTED_NAMES = frozenset({"expected", "label", "output", "target", "ground_truth", "answer"})
_METADATA_NAMES = frozenset({"metadata", "meta", "info", "context"})


def _normalize_row(row: dict[str | None, str | None]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in row.items():
        if key is None:
            continue
        normalized[key] = "" if value is None else value
    return normalized


def suggest_column_roles(
    columns: list[str], inferred_types: dict[str, str]
) -> list[CsvColumnMapping]:
    """Suggest a best-effort role for each CSV column."""
    _ = inferred_types
    suggestions: list[CsvColumnMapping] = []
    for column in columns:
        lower = column.strip().lower()
        if lower in _CASE_ID_NAMES:
            role = "case_id"
        elif lower in _SPLIT_NAMES:
            role = "split"
        elif lower in _EXPECTED_NAMES:
            role = "expected"
        elif lower in _METADATA_NAMES:
            role = "metadata"
        else:
            role = "input"
        suggestions.append(CsvColumnMapping(column_name=column, role=role))
    return suggestions


def parse_csv_preview(content: bytes, file_name: str, max_sample_rows: int = 5) -> CsvImportPreview:
    """Parse uploaded CSV bytes and build a UI preview payload."""
    errors: list[str] = []
    warnings: list[str] = []
    try:
        text = content.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames is None:
            return CsvImportPreview(
                file_name=file_name,
                row_count=0,
                columns=[],
                errors=["CSV is missing a header row."],
            )
        rows: list[dict[str, str]] = [_normalize_row(row) for row in reader]
    except (UnicodeDecodeError, csv.Error) as exc:
        return CsvImportPreview(
            file_name=file_name,
            row_count=0,
            columns=[],
            errors=[f"CSV parse failure: {exc}"],
        )

    if not rows:
        return CsvImportPreview(
            file_name=file_name,
            row_count=0,
            columns=[],
            errors=["CSV contains zero rows."],
        )

    columns = [column for column in reader.fieldnames if column is not None]
    sample_rows = rows[:max_sample_rows]
    inferred_types = {
        column: infer_column_type([row.get(column, "") for row in sample_rows])
        for column in columns
    }
    suggested_mappings = suggest_column_roles(columns, inferred_types)

    return CsvImportPreview(
        file_name=file_name,
        row_count=len(rows),
        columns=columns,
        sample_rows=sample_rows,
        inferred_types=inferred_types,
        suggested_mappings=suggested_mappings,
        warnings=warnings,
        errors=errors,
    )
