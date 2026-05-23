"""Generate RuleKilnTask from CSV import mapping."""

from __future__ import annotations

from pydantic import BaseModel, Field, TypeAdapter

from rulekiln.importers.column_mapping import CsvColumnMapping, CsvImportMapping
from rulekiln.schemas.task_case import RuleKilnTask, TaskMode

_TASK_MODE_ADAPTER = TypeAdapter(TaskMode)


class OutputSchemaNode(BaseModel):
    """Minimal JSON Schema node used for generated task output schemas."""

    type: str
    properties: dict[str, OutputSchemaNode] = Field(default_factory=dict)


class ImportQualityGates(BaseModel):
    """Default quality gates used for generated import tasks."""

    min_metric_delta: float = 0.0
    max_golden_failures: int = 0
    max_malformed_output_rate: float = 0.01


def _add_schema_property(schema: OutputSchemaNode, path: str) -> None:
    cursor = schema
    parts = [part.strip() for part in path.split(".") if part.strip()]
    if not parts:
        return

    for part in parts[:-1]:
        cursor = cursor.properties.setdefault(part, OutputSchemaNode(type="object"))

    cursor.properties[parts[-1]] = OutputSchemaNode(type="string")


def generate_task(
    mapping: CsvImportMapping,
    input_columns: list[CsvColumnMapping],
    expected_columns: list[CsvColumnMapping],
) -> RuleKilnTask:
    """Generate a task definition from an import mapping."""
    input_template_parts = [
        f"{{{{{column.path or column.column_name}}}}}" for column in input_columns
    ]
    input_template = "\n".join(input_template_parts) if input_template_parts else "{{input}}"

    output_schema = OutputSchemaNode(type="object")
    for column in expected_columns:
        _add_schema_property(output_schema, column.path or column.column_name)

    quality_gates = ImportQualityGates()

    return RuleKilnTask(
        task_id=mapping.task_id,
        task_name=mapping.task_name,
        task_mode=_TASK_MODE_ADAPTER.validate_python(mapping.task_mode),
        description=mapping.description,
        input_template=input_template,
        output_schema=output_schema.model_dump(mode="json", exclude_defaults=True),
        quality_gates=quality_gates.model_dump(mode="json"),
    )
