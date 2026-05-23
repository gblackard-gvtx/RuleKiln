"""Offline tests for the CSV importer helpers."""

from rulekiln.importers.case_generator import generate_case, generate_cases
from rulekiln.importers.column_mapping import CsvColumnMapping, CsvImportMapping
from rulekiln.importers.csv_importer import parse_csv_preview
from rulekiln.importers.task_generator import generate_task
from rulekiln.importers.type_inference import infer_column_type, infer_value
from rulekiln.importers.validator import validate_import_mapping


def _mapping(columns: list[CsvColumnMapping]) -> CsvImportMapping:
    return CsvImportMapping(
        task_id="sentiment",
        task_name="Sentiment",
        task_mode="classification",
        description="Import test",
        columns=columns,
    )


def test_infer_value_blank_returns_none() -> None:
    assert infer_value("   ") is None


def test_infer_value_boolean_returns_bool() -> None:
    assert infer_value("true") is True
    assert infer_value("FALSE") is False


def test_infer_value_numeric_returns_number_types() -> None:
    assert infer_value("42") == 42
    assert infer_value("3.14") == 3.14


def test_infer_value_json_returns_objects_and_arrays() -> None:
    assert infer_value('{"label": "positive"}') == {"label": "positive"}
    assert infer_value('["a", "b"]') == ["a", "b"]


def test_infer_column_type_uses_majority_type() -> None:
    assert infer_column_type(["1", "2", "hello"]) == "integer"


def test_parse_csv_preview_extracts_columns_rows_and_suggestions() -> None:
    content = b"case_id,text,label\n1,Great product,positive\n2,Terrible,negative\n"

    preview = parse_csv_preview(content, file_name="examples.csv")

    assert preview.file_name == "examples.csv"
    assert preview.row_count == 2
    assert preview.columns == ["case_id", "text", "label"]
    assert preview.inferred_types["case_id"] == "integer"
    assert preview.sample_rows[0]["text"] == "Great product"
    assert [mapping.role for mapping in preview.suggested_mappings] == [
        "case_id",
        "input",
        "expected",
    ]


def test_parse_csv_preview_rejects_empty_csv() -> None:
    preview = parse_csv_preview(b"case_id,text\n", file_name="empty.csv")

    assert preview.row_count == 0
    assert preview.errors == ["CSV contains zero rows."]


def test_validate_import_mapping_requires_input_and_expected() -> None:
    mapping = _mapping(
        [
            CsvColumnMapping(column_name="case_id", role="case_id"),
            CsvColumnMapping(column_name="split", role="split"),
        ]
    )

    errors, warnings = validate_import_mapping(mapping, [{"case_id": "1", "split": "train"}])

    assert len(errors) == 2
    assert warnings == []


def test_validate_import_mapping_warns_when_split_missing() -> None:
    mapping = _mapping(
        [
            CsvColumnMapping(column_name="text", role="input"),
            CsvColumnMapping(column_name="label", role="expected"),
        ]
    )

    errors, warnings = validate_import_mapping(mapping, [{"text": "hello", "label": "positive"}])

    assert errors == []
    assert len(warnings) == 2
    assert "No split column mapped" in warnings[0]


def test_generate_case_builds_nested_fields_and_assertion() -> None:
    mapping = _mapping(
        [
            CsvColumnMapping(column_name="case_id", role="case_id"),
            CsvColumnMapping(column_name="split", role="split"),
            CsvColumnMapping(column_name="text", role="input", path="message.text"),
            CsvColumnMapping(
                column_name="label",
                role="expected",
                path="labels.sentiment",
                create_assertion=True,
            ),
            CsvColumnMapping(column_name="source", role="metadata", path="origin.name"),
        ]
    )

    case = generate_case(
        {
            "case_id": "row-1",
            "split": "golden",
            "text": "Great service",
            "label": "positive",
            "source": "support",
        },
        mapping,
        1,
    )

    assert case.id == "row-1"
    assert case.split == "golden"
    assert case.input == {"message": {"text": "Great service"}}
    assert case.expected == {"labels": {"sentiment": "positive"}}
    assert case.metadata == {"origin": {"name": "support"}}
    assert case.evaluation.assertions[0].path == "labels.sentiment"
    assert case.evaluation.assertions[0].value == "positive"


def test_generate_case_assigns_deterministic_split_and_default_id() -> None:
    mapping = _mapping(
        [
            CsvColumnMapping(column_name="text", role="input"),
            CsvColumnMapping(column_name="label", role="expected"),
        ]
    )

    first = generate_case({"text": "Hello", "label": "positive"}, mapping, 7)
    second = generate_case({"text": "Hello", "label": "positive"}, mapping, 7)

    assert first.id == "case_000007"
    assert first.split in {"train", "validation"}
    assert second.split == first.split


def test_generate_cases_rejects_duplicate_case_ids() -> None:
    mapping = _mapping(
        [
            CsvColumnMapping(column_name="case_id", role="case_id"),
            CsvColumnMapping(column_name="text", role="input"),
            CsvColumnMapping(column_name="label", role="expected"),
        ]
    )

    cases, errors = generate_cases(
        [
            {"case_id": "dup", "text": "one", "label": "positive"},
            {"case_id": "dup", "text": "two", "label": "negative"},
        ],
        mapping,
    )

    assert len(cases) == 1
    assert errors == ["Duplicate case ID 'dup' at row 2."]


def test_generate_task_builds_template_and_nested_output_schema() -> None:
    mapping = _mapping([])
    input_columns = [CsvColumnMapping(column_name="text", role="input", path="message.text")]
    expected_columns = [
        CsvColumnMapping(column_name="label", role="expected", path="labels.sentiment")
    ]

    task = generate_task(mapping, input_columns, expected_columns)

    assert task.input_template == "{{message.text}}"
    assert task.output_schema == {
        "type": "object",
        "properties": {
            "labels": {
                "type": "object",
                "properties": {"sentiment": {"type": "string"}},
            }
        },
    }
