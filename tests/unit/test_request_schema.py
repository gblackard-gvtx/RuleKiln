"""Unit tests for DistillationRequest schema validation."""

import pytest

from rulekiln.schemas.job import DistillationRequest
from rulekiln.schemas.task_case import ModelRoute, RuleKilnCase, RuleKilnTask


def _base_task() -> dict:
    return {
        "task_id": "test-task",
        "task_name": "Test Task",
        "task_mode": "classification",
        "description": "A test task",
        "input_template": "{{input}}",
    }


def _base_case() -> dict:
    return {
        "id": "case-1",
        "task_mode": "classification",
        "input": {"text": "hello"},
        "expected": {"label": "positive"},
    }


def _base_payload() -> dict:
    return {
        "task": _base_task(),
        "cases": [_base_case()],
        "teacher": {"provider_profile": "teacher_profile", "model": "gpt-4o"},
        "student": {"provider_profile": "student_profile", "model": "gpt-4o-mini"},
        "embedding": {"provider_profile": "embedding_profile", "model": "text-embedding-3-small"},
    }


def test_valid_request_parses() -> None:
    req = DistillationRequest.model_validate(_base_payload())
    assert req.task.task_id == "test-task"
    assert len(req.cases) == 1


def test_legacy_field_task_name_rejected() -> None:
    payload = {**_base_payload(), "task_name": "legacy"}
    with pytest.raises(Exception):
        DistillationRequest.model_validate(payload)


def test_legacy_field_examples_rejected() -> None:
    payload = {**_base_payload(), "examples": []}
    with pytest.raises(Exception):
        DistillationRequest.model_validate(payload)


def test_extra_unknown_field_rejected() -> None:
    payload = {**_base_payload(), "unknown_field": "value"}
    with pytest.raises(Exception):
        DistillationRequest.model_validate(payload)
