"""Unit tests for TeacherConfig, ClassroomConfig, and related schemas."""

from __future__ import annotations

import pytest

from rulekiln.schemas.classroom import (
    ClassroomConfig,
    PhaseTeacherConfig,
    StudentConfig,
    TeacherConfig,
)

# ── TeacherConfig ────────────────────────────────────────────────────────────


def test_teacher_config_schema_version() -> None:
    tc = TeacherConfig(default=PhaseTeacherConfig(provider="fake", model="m1"))
    assert tc.schema_version == "rulekiln.teacher_config.v1"


def test_phase_teacher_config_schema_version() -> None:
    ptc = PhaseTeacherConfig(provider="fake", model="m1")
    assert ptc.schema_version == "rulekiln.phase_teacher_config.v1"


def test_for_phase_returns_override_when_set() -> None:
    tc = TeacherConfig(
        default=PhaseTeacherConfig(provider="default_prof", model="default_model"),
        instruction_extraction=PhaseTeacherConfig(provider="cheap_prof", model="cheap_model"),
    )
    result = tc.for_phase("instruction_extraction")
    assert result.provider == "cheap_prof"
    assert result.model == "cheap_model"


def test_for_phase_falls_back_to_default_when_override_absent() -> None:
    tc = TeacherConfig(
        default=PhaseTeacherConfig(provider="default_prof", model="default_model"),
    )
    for phase in ("instruction_extraction", "cluster_consolidation", "conflict_resolution"):
        result = tc.for_phase(phase)  # type: ignore[arg-type]
        assert result.provider == "default_prof"
        assert result.model == "default_model"


def test_for_phase_returns_specific_override_for_each_phase() -> None:
    tc = TeacherConfig(
        default=PhaseTeacherConfig(provider="d", model="dm"),
        instruction_extraction=PhaseTeacherConfig(provider="e", model="em"),
        cluster_consolidation=PhaseTeacherConfig(provider="c", model="cm"),
        conflict_resolution=PhaseTeacherConfig(provider="r", model="rm"),
    )
    assert tc.for_phase("instruction_extraction").provider == "e"
    assert tc.for_phase("cluster_consolidation").provider == "c"
    assert tc.for_phase("conflict_resolution").provider == "r"


def test_flat_config_migrated_to_teacher_config() -> None:
    """A flat {provider, model} dict is promoted to TeacherConfig.default."""
    tc = TeacherConfig.model_validate({"provider": "fake", "model": "my-model"})
    assert tc.default.provider == "fake"
    assert tc.default.model == "my-model"
    assert tc.instruction_extraction is None
    assert tc.cluster_consolidation is None
    assert tc.conflict_resolution is None


def test_from_provider_model_classmethod() -> None:
    tc = TeacherConfig.from_provider_model("fake", "model-x")
    assert tc.default.provider == "fake"
    assert tc.default.model == "model-x"
    assert tc.for_phase("instruction_extraction").model == "model-x"


def test_extra_params_accepted() -> None:
    ptc = PhaseTeacherConfig(
        provider="anthropic",
        model="claude-opus",
        extra_params={"thinking_budget": 1024, "temperature": 0},
    )
    assert ptc.extra_params["thinking_budget"] == 1024


# ── ClassroomConfig ──────────────────────────────────────────────────────────


def test_classroom_config_schema_version() -> None:
    cc = ClassroomConfig(
        students=[StudentConfig(id="s1", provider="fake", model="m", is_anchor=True)],
        anchor_student_id="s1",
    )
    assert cc.schema_version == "rulekiln.classroom_config.v1"


def test_anchor_student_property_returns_correct_student() -> None:
    cc = ClassroomConfig(
        students=[
            StudentConfig(id="s1", provider="p1", model="m1"),
            StudentConfig(id="s2", provider="p2", model="m2", is_anchor=True),
        ],
        anchor_student_id="s2",
    )
    assert cc.anchor_student.id == "s2"
    assert cc.anchor_student.provider == "p2"


def test_invalid_anchor_id_raises_at_load_time() -> None:
    """Bad anchor_student_id raises ValueError at model construction, not eval time."""
    with pytest.raises(ValueError, match="anchor_student_id"):
        ClassroomConfig(
            students=[StudentConfig(id="s1", provider="fake", model="m")],
            anchor_student_id="does_not_exist",
        )


def test_single_student_flat_migration() -> None:
    """Flat {provider, model} wraps as a single 'default' student with is_anchor=True."""
    cc = ClassroomConfig.model_validate({"provider": "fake", "model": "my-model"})
    assert len(cc.students) == 1
    assert cc.students[0].id == "default"
    assert cc.students[0].is_anchor is True
    assert cc.anchor_student_id == "default"
    assert cc.anchor_student.model == "my-model"


def test_classroom_from_provider_model_classmethod() -> None:
    cc = ClassroomConfig.from_provider_model("fake", "m")
    assert cc.anchor_student.provider == "fake"
    assert cc.anchor_student.model == "m"


def test_multi_student_ordered_evaluation() -> None:
    """Students list is preserved in definition order."""
    cc = ClassroomConfig(
        students=[
            StudentConfig(id="a", provider="p", model="m1", is_anchor=True),
            StudentConfig(id="b", provider="p", model="m2"),
            StudentConfig(id="c", provider="p", model="m3"),
        ],
        anchor_student_id="a",
    )
    assert [s.id for s in cc.students] == ["a", "b", "c"]


def test_empty_students_list_no_anchor_validation_error() -> None:
    """Empty students list is allowed (no anchor to validate)."""
    cc = ClassroomConfig(students=[], anchor_student_id="default")
    assert cc.students == []
