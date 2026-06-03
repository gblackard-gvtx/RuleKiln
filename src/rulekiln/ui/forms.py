"""Multipart form parsers for the operator UI."""

from typing import Annotated

from fastapi import File, Form, UploadFile

from rulekiln.schemas.classroom import PhaseTeacherConfig, TeacherConfig
from rulekiln.ui.view_models import TeacherPhaseView, TeacherRoutingView

# Human-readable label for each phase, in display order.
_PHASE_META: list[tuple[str, str]] = [
    ("instruction_extraction", "Extraction"),
    ("cluster_consolidation", "Synthesis"),
    ("conflict_resolution", "Conflict resolution"),
]


class NewJobForm:
    """Dependency class that parses the new-job multipart form submission."""

    def __init__(
        self,
        task_file: Annotated[UploadFile, File(description="task.yaml")],
        cases_file: Annotated[UploadFile, File(description="cases.jsonl")],
        teacher_profile: Annotated[str, Form()],
        teacher_model: Annotated[str, Form()],
        student_profile: Annotated[str, Form()],
        student_model: Annotated[str, Form()],
        embedding_profile: Annotated[str, Form()],
        embedding_model: Annotated[str, Form()],
        judge_profile: Annotated[str | None, Form()] = None,
        judge_model: Annotated[str | None, Form()] = None,
        baseline_prompt: Annotated[str | None, Form()] = None,
        # ── Optional per-phase teacher overrides ─────────────────────────
        extraction_teacher_profile: Annotated[str | None, Form()] = None,
        extraction_teacher_model: Annotated[str | None, Form()] = None,
        synthesis_teacher_profile: Annotated[str | None, Form()] = None,
        synthesis_teacher_model: Annotated[str | None, Form()] = None,
        conflict_resolution_teacher_profile: Annotated[str | None, Form()] = None,
        conflict_resolution_teacher_model: Annotated[str | None, Form()] = None,
    ) -> None:
        self.task_file = task_file
        self.cases_file = cases_file
        self.teacher_profile = teacher_profile
        self.teacher_model = teacher_model
        self.student_profile = student_profile
        self.student_model = student_model
        self.embedding_profile = embedding_profile
        self.embedding_model = embedding_model
        self.judge_profile = judge_profile
        self.judge_model = judge_model
        self.baseline_prompt = baseline_prompt
        # Normalise empty strings → None so callers can use simple truthiness checks.
        self.extraction_teacher_profile = extraction_teacher_profile or None
        self.extraction_teacher_model = extraction_teacher_model or None
        self.synthesis_teacher_profile = synthesis_teacher_profile or None
        self.synthesis_teacher_model = synthesis_teacher_model or None
        self.conflict_resolution_teacher_profile = (
            conflict_resolution_teacher_profile or None
        )
        self.conflict_resolution_teacher_model = conflict_resolution_teacher_model or None

    # ── Per-phase override helpers ────────────────────────────────────────

    def _phase_fields(self) -> list[tuple[str, str | None, str | None]]:
        """Return (phase_key, profile, model) for each phase, in definition order."""
        return [
            (
                "instruction_extraction",
                self.extraction_teacher_profile,
                self.extraction_teacher_model,
            ),
            (
                "cluster_consolidation",
                self.synthesis_teacher_profile,
                self.synthesis_teacher_model,
            ),
            (
                "conflict_resolution",
                self.conflict_resolution_teacher_profile,
                self.conflict_resolution_teacher_model,
            ),
        ]

    def validate_phase_overrides(self) -> list[str]:
        """Return form-level errors for incomplete phase override pairs.

        Rule: for each phase, both profile and model must either both be blank
        (inherit default) or both be provided (use override).  Supplying one
        without the other is an error.
        """
        errors: list[str] = []
        labels = {
            "instruction_extraction": "Extraction teacher",
            "cluster_consolidation": "Synthesis teacher",
            "conflict_resolution": "Conflict-resolution teacher",
        }
        for phase_key, profile, model in self._phase_fields():
            has_profile = bool(profile)
            has_model = bool(model)
            if has_profile != has_model:
                label = labels[phase_key]
                missing = "model" if has_profile else "profile"
                errors.append(
                    f"{label}: both profile and model must be provided together "
                    f"(missing: {missing})."
                )
        return errors

    def build_teacher_config(self) -> TeacherConfig | None:
        """Build a TeacherConfig from the form data, or return None if no overrides.

        Returns None when all phase override fields are blank so that the
        DistillationRequest falls back to the flat teacher ModelRoute — this
        keeps single-teacher job creation exactly as before.
        """
        phases = self._phase_fields()
        if not any(profile for _, profile, _ in phases):
            return None

        default = PhaseTeacherConfig(
            provider=self.teacher_profile,
            model=self.teacher_model,
        )
        overrides: dict[str, PhaseTeacherConfig | None] = {
            "instruction_extraction": None,
            "cluster_consolidation": None,
            "conflict_resolution": None,
        }
        for phase_key, profile, model in phases:
            if profile and model:
                overrides[phase_key] = PhaseTeacherConfig(provider=profile, model=model)

        return TeacherConfig(
            default=default,
            instruction_extraction=overrides["instruction_extraction"],
            cluster_consolidation=overrides["cluster_consolidation"],
            conflict_resolution=overrides["conflict_resolution"],
        )

    def build_teacher_routing_view(self) -> TeacherRoutingView:
        """Build a TeacherRoutingView for the preview page.

        Always populated — even when no overrides are configured — so the preview
        can show which effective model each phase will use.
        """
        phase_views: list[TeacherPhaseView] = []
        has_any_override = False
        for phase_key, label in _PHASE_META:
            # Find override for this phase
            override_profile: str | None = None
            override_model: str | None = None
            for pk, profile, model in self._phase_fields():
                if pk == phase_key and profile and model:
                    override_profile = profile
                    override_model = model
                    break

            if override_profile and override_model:
                phase_views.append(
                    TeacherPhaseView(
                        phase_label=label,
                        profile_name=override_profile,
                        model_id=override_model,
                        is_override=True,
                    )
                )
                has_any_override = True
            else:
                phase_views.append(
                    TeacherPhaseView(
                        phase_label=label,
                        profile_name=self.teacher_profile,
                        model_id=self.teacher_model,
                        is_override=False,
                    )
                )

        return TeacherRoutingView(
            default_profile=self.teacher_profile,
            default_model=self.teacher_model,
            phases=phase_views,
            has_any_override=has_any_override,
        )
