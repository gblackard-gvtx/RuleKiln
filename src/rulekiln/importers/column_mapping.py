"""Pydantic models for CSV import column mapping."""

from typing import Literal

from pydantic import BaseModel, Field

ColumnRole = Literal["case_id", "split", "input", "expected", "metadata", "assertion", "ignore"]


class CsvColumnMapping(BaseModel):
    """Mapping for a single CSV column."""

    column_name: str
    role: ColumnRole
    path: str | None = None
    assertion_type: str | None = None
    create_assertion: bool = False


class CsvImportMapping(BaseModel):
    """User-specified import configuration for one CSV file."""

    task_id: str
    task_name: str
    task_mode: str
    description: str
    columns: list[CsvColumnMapping] = Field(default_factory=list)


class CsvImportPreview(BaseModel):
    """Persisted CSV preview metadata for the UI workflow."""

    file_name: str
    row_count: int
    columns: list[str]
    sample_rows: list[dict[str, str]] = Field(default_factory=list)
    inferred_types: dict[str, str] = Field(default_factory=dict)
    suggested_mappings: list[CsvColumnMapping] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
