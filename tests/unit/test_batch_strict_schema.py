"""Unit tests for OpenAI strict structured-output schema construction.

Regression coverage for the batch extraction failure where the request was sent
with ``strict: true`` but the Pydantic schema lacked ``additionalProperties:
false`` and did not mark every property ``required`` — OpenAI rejected all items
with ``invalid_json_schema`` (``additionalProperties is required to be ... false``).
"""

from __future__ import annotations

import json

from rulekiln.agents.rule_extraction import build_extraction_batch_item
from rulekiln.providers.chat.openai_chat import (
    _build_batch_jsonl,
    _to_strict_json_schema,
)
from rulekiln.schemas.pipeline import ExtractionOutput
from rulekiln.schemas.task_case import RuleKilnCase, RuleKilnTask


def _all_object_nodes(node: object) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    if isinstance(node, dict):
        if "properties" in node:
            objects.append(node)
        for value in node.values():
            objects.extend(_all_object_nodes(value))
    elif isinstance(node, list):
        for item in node:
            objects.extend(_all_object_nodes(item))
    return objects


def test_strict_schema_sets_additional_properties_false_on_every_object() -> None:
    strict = _to_strict_json_schema(ExtractionOutput.model_json_schema())
    objects = _all_object_nodes(strict)
    assert objects, "expected at least one object node"
    for obj in objects:
        assert obj["additionalProperties"] is False


def test_strict_schema_requires_all_properties() -> None:
    strict = _to_strict_json_schema(ExtractionOutput.model_json_schema())
    for obj in _all_object_nodes(strict):
        props = obj["properties"]
        assert isinstance(props, dict)
        assert set(obj["required"]) == set(props.keys())  # type: ignore[arg-type]


def test_strict_schema_does_not_mutate_input() -> None:
    original = ExtractionOutput.model_json_schema()
    snapshot = json.dumps(original, sort_keys=True)
    _to_strict_json_schema(original)
    assert json.dumps(original, sort_keys=True) == snapshot


def test_strict_schema_preserves_nullable_unions() -> None:
    # output_path is `str | None`; the nullable union must survive the transform
    # (strict mode expresses optional fields as required + nullable).
    strict = _to_strict_json_schema(ExtractionOutput.model_json_schema())
    micro = strict["$defs"]["MicroRuleSchema"]  # type: ignore[index]
    output_path = micro["properties"]["output_path"]  # type: ignore[index]
    any_of = output_path["anyOf"]  # type: ignore[index]
    types = {branch["type"] for branch in any_of}  # type: ignore[index]
    assert types == {"string", "null"}


def test_build_batch_jsonl_emits_strict_compliant_schema() -> None:
    task = RuleKilnTask(
        task_id="t",
        task_name="T",
        task_mode="classification",
        description="d",
        input_template="{{input}}",
    )
    case = RuleKilnCase(
        id="case-1",
        task_mode="classification",
        split="train",
        input={"text": "hello"},
        expected="positive",
    )
    item = build_extraction_batch_item(task, case)
    raw = _build_batch_jsonl([item], model="gpt-test")
    request = json.loads(raw.decode())

    fmt = request["body"]["text"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["strict"] is True
    schema = fmt["schema"]
    # Root and every nested object must satisfy strict mode.
    for obj in _all_object_nodes(schema):
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"].keys())  # type: ignore[union-attr]
