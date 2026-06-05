"""Tiered teacher config and multi-student classroom schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# ── Per-phase teacher config ────────────────────────────────────────────────


class PhaseTeacherConfig(BaseModel):
    """Model config for one pipeline phase."""

    schema_version: Literal["rulekiln.phase_teacher_config.v1"] = (
        "rulekiln.phase_teacher_config.v1"
    )
    provider: str  # provider_profile name (e.g. "fake", "openai_default")
    model: str
    extra_params: dict[str, str | int | float | bool] = Field(default_factory=dict)
    batch_enabled: bool = False
    batch_min_items: int = 10


_TeacherPhase = Literal["instruction_extraction", "cluster_consolidation", "conflict_resolution"]


class TeacherConfig(BaseModel):
    """Per-phase teacher routing config.

    Each of the three teacher-intensive pipeline stages can be assigned a
    different model. The ``default`` config is the fallback when a phase override
    is absent.  A flat ``{provider, model}`` input is promoted to a
    ``TeacherConfig`` with only ``default`` set (all phase overrides ``None``).
    """

    schema_version: Literal["rulekiln.teacher_config.v1"] = "rulekiln.teacher_config.v1"
    default: PhaseTeacherConfig
    instruction_extraction: PhaseTeacherConfig | None = None
    cluster_consolidation: PhaseTeacherConfig | None = None
    conflict_resolution: PhaseTeacherConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_flat_config(
        cls, data: dict[str, object]
    ) -> dict[str, object]:
        """Accept flat {provider, model} and wrap it as TeacherConfig.default."""
        if isinstance(data, dict) and "provider" in data and "default" not in data:
            return {"default": data}
        return data

    def for_phase(self, phase: _TeacherPhase) -> PhaseTeacherConfig:
        """Return the phase-specific config or fall back to default."""
        override: PhaseTeacherConfig | None = getattr(self, phase, None)
        return override if override is not None else self.default

    @classmethod
    def from_provider_model(cls, provider: str, model: str) -> TeacherConfig:
        """Build a TeacherConfig from a flat provider+model pair."""
        return cls(default=PhaseTeacherConfig(provider=provider, model=model))


# ── Multi-student classroom config ─────────────────────────────────────────


class StudentConfig(BaseModel):
    """Config for a single evaluation student."""

    schema_version: Literal["rulekiln.student_config.v1"] = "rulekiln.student_config.v1"
    id: str
    display_name: str = ""
    provider: str  # provider_profile name
    model: str
    is_anchor: bool = False


class ClassroomConfig(BaseModel):
    """Multi-student evaluation config.

    The anchor student drives the closed-loop conflict resolution.  Non-anchor
    students are evaluated once at the final iteration only.

    ``anchor_student`` is validated at model load time: an unknown ID raises
    ``ValueError`` immediately rather than silently using the wrong student at
    eval time.

    A flat ``{provider, model}`` input is wrapped as a single-element list with
    ``id="default"`` and ``is_anchor=True``.
    """

    schema_version: Literal["rulekiln.classroom_config.v1"] = "rulekiln.classroom_config.v1"
    students: list[StudentConfig] = Field(default_factory=list)
    anchor_student_id: str = "default"

    @model_validator(mode="before")
    @classmethod
    def _migrate_flat_student(
        cls, data: dict[str, object]
    ) -> dict[str, object]:
        """Accept flat {provider, model} and wrap as a single 'default' student."""
        if isinstance(data, dict) and "provider" in data and "students" not in data:
            return {
                "students": [
                    {
                        "id": "default",
                        "provider": data["provider"],
                        "model": data["model"],
                        "is_anchor": True,
                    }
                ],
                "anchor_student_id": "default",
            }
        return data

    @model_validator(mode="after")
    def _validate_anchor_id(self) -> ClassroomConfig:
        """Ensure anchor_student_id references a known student ID."""
        ids = {s.id for s in self.students}
        if self.students and self.anchor_student_id not in ids:
            raise ValueError(
                f"anchor_student_id '{self.anchor_student_id}' not found in students "
                f"(known IDs: {sorted(ids)})"
            )
        return self

    @property
    def anchor_student(self) -> StudentConfig:
        """Return the anchor student config.

        Guaranteed to exist because anchor_student_id is validated at load time.
        """
        for student in self.students:
            if student.id == self.anchor_student_id:
                return student
        raise ValueError(  # unreachable after model_validator, but satisfies type checker
            f"anchor_student_id '{self.anchor_student_id}' not found"
        )

    @classmethod
    def from_provider_model(cls, provider: str, model: str) -> ClassroomConfig:
        """Build a single-student ClassroomConfig from a flat provider+model pair."""
        return cls(
            students=[
                StudentConfig(
                    id="default",
                    provider=provider,
                    model=model,
                    is_anchor=True,
                )
            ],
            anchor_student_id="default",
        )
