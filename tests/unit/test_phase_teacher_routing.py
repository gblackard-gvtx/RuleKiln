"""Integration-style tests for per-phase teacher routing in the pipeline request schema."""

from __future__ import annotations

from rulekiln.schemas.classroom import (
    ClassroomConfig,
    PhaseTeacherConfig,
    StudentConfig,
    TeacherConfig,
)
from rulekiln.schemas.job import DistillationRequest
from rulekiln.schemas.task_case import (
    EvaluationAssertion,
    EvaluationSpec,
    ModelRoute,
    RuleKilnCase,
    RuleKilnTask,
)


def _task() -> RuleKilnTask:
    return RuleKilnTask(
        schema_version="rulekiln.task.v1",
        task_id="t",
        task_name="T",
        task_mode="classification",
        description="d",
        input_template="{{input.text}}",
    )


def _case(case_id: str) -> RuleKilnCase:
    return RuleKilnCase(
        id=case_id,
        task_mode="classification",
        input={"text": "hello"},
        expected={"label": "travel"},
        evaluation=EvaluationSpec(
            assertions=[
                EvaluationAssertion(
                    type="must_equal",  # type: ignore[arg-type]
                    path="label",
                    value="travel",
                    weight=1.0,
                )
            ]
        ),
    )


def _route(profile: str = "fake", model: str = "m") -> ModelRoute:
    return ModelRoute(provider_profile=profile, model=model)


# ── teacher_config field on DistillationRequest ──────────────────────────────


def test_request_accepts_teacher_config_with_phase_overrides() -> None:
    """DistillationRequest can carry a TeacherConfig with phase-specific overrides."""
    tc = TeacherConfig(
        default=PhaseTeacherConfig(provider="fake", model="default-model"),
        instruction_extraction=PhaseTeacherConfig(provider="fake", model="cheap-model"),
        conflict_resolution=PhaseTeacherConfig(provider="fake", model="strong-model"),
    )
    req = DistillationRequest(
        task=_task(),
        cases=[_case("c1")],
        teacher=_route(),
        student=_route(),
        embedding=_route(),
        teacher_config=tc,
    )
    assert req.teacher_config is not None
    assert req.teacher_config.for_phase("instruction_extraction").model == "cheap-model"
    assert req.teacher_config.for_phase("conflict_resolution").model == "strong-model"
    assert req.teacher_config.for_phase("cluster_consolidation").model == "default-model"


def test_request_without_teacher_config_is_backward_compat() -> None:
    """Existing requests without teacher_config still deserialize normally."""
    req = DistillationRequest(
        task=_task(),
        cases=[_case("c1")],
        teacher=_route(),
        student=_route(),
        embedding=_route(),
    )
    assert req.teacher_config is None


def test_flat_teacher_config_migration_in_request() -> None:
    """Flat {provider, model} in teacher_config wraps to TeacherConfig.default."""
    req = DistillationRequest.model_validate(
        {
            "task": _task().model_dump(),
            "cases": [_case("c1").model_dump()],
            "teacher": {"provider_profile": "fake", "model": "m"},
            "student": {"provider_profile": "fake", "model": "m"},
            "embedding": {"provider_profile": "fake", "model": "m"},
            "teacher_config": {"provider": "fake", "model": "tiered-m"},
        }
    )
    assert req.teacher_config is not None
    assert req.teacher_config.default.model == "tiered-m"


# ── classroom_config field on DistillationRequest ────────────────────────────


def test_request_accepts_three_student_classroom_config() -> None:
    """DistillationRequest can carry a 3-student ClassroomConfig."""
    cc = ClassroomConfig(
        students=[
            StudentConfig(id="s1", provider="fake", model="m1", is_anchor=True),
            StudentConfig(id="s2", provider="fake", model="m2"),
            StudentConfig(id="s3", provider="fake", model="m3"),
        ],
        anchor_student_id="s1",
    )
    req = DistillationRequest(
        task=_task(),
        cases=[_case("c1")],
        teacher=_route(),
        student=_route(),
        embedding=_route(),
        classroom_config=cc,
    )
    assert req.classroom_config is not None
    assert len(req.classroom_config.students) == 3
    assert req.classroom_config.anchor_student.id == "s1"


def test_request_without_classroom_config_is_backward_compat() -> None:
    req = DistillationRequest(
        task=_task(),
        cases=[_case("c1")],
        teacher=_route(),
        student=_route(),
        embedding=_route(),
    )
    assert req.classroom_config is None


def test_request_with_both_configs() -> None:
    """teacher_config and classroom_config can coexist."""
    tc = TeacherConfig.from_provider_model("fake", "m")
    cc = ClassroomConfig.from_provider_model("fake", "m")
    req = DistillationRequest(
        task=_task(),
        cases=[_case("c1")],
        teacher=_route(),
        student=_route(),
        embedding=_route(),
        teacher_config=tc,
        classroom_config=cc,
    )
    assert req.teacher_config is not None
    assert req.classroom_config is not None
