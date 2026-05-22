"""Unit tests for request shape validator."""

import pytest

from rulekiln.api.validators.request_shape import (
    RequestValidationError,
    validate_distillation_request,
)
from rulekiln.config.settings import AppSettings, ProviderProfile
from rulekiln.schemas.job import DistillationRequest


def _settings_with_profiles(*profile_names: str) -> AppSettings:
    profiles = {
        name: ProviderProfile(provider="fake", supports_chat=True, supports_embeddings=True)
        for name in profile_names
    }
    return AppSettings(
        DATABASE_URL="postgresql+asyncpg://u:p@localhost/db",
        MLFLOW_TRACKING_URI="http://localhost:5000",
        provider_profiles=profiles,
    )


def _payload(teacher: str = "t", student: str = "s", embedding: str = "e") -> DistillationRequest:
    return DistillationRequest.model_validate(
        {
            "task": {
                "task_id": "t1",
                "task_name": "T1",
                "task_mode": "classification",
                "description": "desc",
                "input_template": "{{input}}",
            },
            "cases": [{"id": "c1", "task_mode": "classification", "input": {"x": 1}}],
            "teacher": {"provider_profile": teacher, "model": "m1"},
            "student": {"provider_profile": student, "model": "m2"},
            "embedding": {"provider_profile": embedding, "model": "m3"},
        }
    )


def test_valid_request_passes() -> None:
    settings = _settings_with_profiles("t", "s", "e")
    validate_distillation_request(_payload(), settings)  # should not raise


def test_unknown_teacher_profile_raises() -> None:
    settings = _settings_with_profiles("s", "e")
    with pytest.raises(RequestValidationError, match="teacher"):
        validate_distillation_request(_payload(), settings)


def test_empty_cases_raises() -> None:
    settings = _settings_with_profiles("t", "s", "e")
    payload = _payload()
    payload.cases.clear()
    with pytest.raises(RequestValidationError, match="cases"):
        validate_distillation_request(payload, settings)
