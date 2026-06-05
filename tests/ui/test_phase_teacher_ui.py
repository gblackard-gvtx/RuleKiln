"""Tests: phase-specific teacher routing in the new-job UI."""

from __future__ import annotations

from httpx import AsyncClient

from tests.ui.conftest import VALID_CASE_LINE, VALID_TASK_YAML


def _base_form() -> dict[str, str]:
    return {
        "teacher_profile": "fake_chat",
        "teacher_model": "model-default",
        "student_profile": "fake_chat",
        "student_model": "model-student",
        "embedding_profile": "fake_embed",
        "embedding_model": "model-embed",
    }


def _files() -> dict[str, tuple[str, bytes, str]]:
    return {
        "task_file": ("task.yaml", VALID_TASK_YAML, "application/yaml"),
        "cases_file": ("cases.jsonl", VALID_CASE_LINE, "application/x-ndjson"),
    }


class TestNewJobFormPhaseTeacher:
    async def test_advanced_section_present_in_form(self, client: AsyncClient) -> None:
        response = await client.get("/ui/jobs/new")
        assert response.status_code == 200
        assert "Advanced teacher routing" in response.text
        assert "extraction_teacher_profile" in response.text
        assert "synthesis_teacher_profile" in response.text
        assert "conflict_resolution_teacher_profile" in response.text

    async def test_multi_student_ui_present(self, client: AsyncClient) -> None:
        response = await client.get("/ui/jobs/new")
        assert "Add another student" in response.text
        assert "student_profile" in response.text
        assert "student_model" in response.text

    async def test_inherit_copy_present(self, client: AsyncClient) -> None:
        response = await client.get("/ui/jobs/new")
        assert "inherit" in response.text.lower()


class TestPreviewPhaseTeacher:
    async def test_no_overrides_still_shows_teacher_routing_table(
        self, client: AsyncClient
    ) -> None:
        """Preview always shows the teacher routing card, even with no overrides."""
        response = await client.post(
            "/ui/jobs/preview",
            files=_files(),
            data=_base_form(),
        )
        assert response.status_code == 200
        assert "Teacher Routing" in response.text
        assert "Extraction" in response.text
        assert "Synthesis" in response.text
        assert "Conflict resolution" in response.text

    async def test_no_overrides_all_phases_show_inherits(self, client: AsyncClient) -> None:
        response = await client.post(
            "/ui/jobs/preview",
            files=_files(),
            data=_base_form(),
        )
        assert response.status_code == 200
        assert "inherits default" in response.text

    async def test_single_phase_override_shown_as_override(self, client: AsyncClient) -> None:
        data = {
            **_base_form(),
            "extraction_teacher_profile": "fake_chat",
            "extraction_teacher_model": "model-cheap",
        }
        response = await client.post(
            "/ui/jobs/preview",
            files=_files(),
            data=data,
        )
        assert response.status_code == 200
        assert "override" in response.text
        assert "model-cheap" in response.text

    async def test_all_three_overrides_all_shown_as_override(self, client: AsyncClient) -> None:
        data = {
            **_base_form(),
            "extraction_teacher_profile": "fake_chat",
            "extraction_teacher_model": "model-extract",
            "synthesis_teacher_profile": "fake_chat",
            "synthesis_teacher_model": "model-synth",
            "conflict_resolution_teacher_profile": "fake_chat",
            "conflict_resolution_teacher_model": "model-cr",
        }
        response = await client.post(
            "/ui/jobs/preview",
            files=_files(),
            data=data,
        )
        assert response.status_code == 200
        assert response.text.count("override") >= 3
        assert "model-extract" in response.text
        assert "model-synth" in response.text
        assert "model-cr" in response.text

    async def test_partial_override_profile_only_returns_error(self, client: AsyncClient) -> None:
        """Profile without model → validation error."""
        data = {
            **_base_form(),
            "extraction_teacher_profile": "fake_chat",
            # extraction_teacher_model intentionally absent
        }
        response = await client.post(
            "/ui/jobs/preview",
            files=_files(),
            data=data,
        )
        assert response.status_code == 422
        assert "model" in response.text.lower()

    async def test_partial_override_model_only_returns_error(self, client: AsyncClient) -> None:
        """Model without profile → validation error."""
        data = {
            **_base_form(),
            "extraction_teacher_model": "model-cheap",
            # extraction_teacher_profile intentionally absent
        }
        response = await client.post(
            "/ui/jobs/preview",
            files=_files(),
            data=data,
        )
        assert response.status_code == 422
        assert "profile" in response.text.lower()

    async def test_existing_single_teacher_path_unchanged(self, client: AsyncClient) -> None:
        """Submitting with no phase overrides creates the same request as before."""
        response = await client.post(
            "/ui/jobs/preview",
            files=_files(),
            data=_base_form(),
        )
        assert response.status_code == 200
        assert "Test Task" in response.text
        # Draft job ID is present so the form can be submitted
        assert 'name="draft_job_id"' in response.text

    async def test_preview_default_teacher_shown(self, client: AsyncClient) -> None:
        response = await client.post(
            "/ui/jobs/preview",
            files=_files(),
            data=_base_form(),
        )
        assert response.status_code == 200
        assert "model-default" in response.text


class TestFormValidation:
    """Unit-level tests on NewJobForm validation methods (no HTTP needed)."""

    def _form(self, **kwargs: str | None) -> object:
        """Build a minimal NewJobForm-like namespace for testing the validation methods."""
        from unittest.mock import MagicMock

        from rulekiln.ui.forms import NewJobForm

        mock_file = MagicMock()
        return NewJobForm(
            task_file=mock_file,
            cases_file=mock_file,
            teacher_profile="fake_chat",
            teacher_model="model-default",
            student_profile="fake_chat",
            student_model="model-s",
            embedding_profile="fake_embed",
            embedding_model="model-e",
            **kwargs,
        )

    def test_no_overrides_returns_no_errors(self) -> None:
        form = self._form()
        assert form.validate_phase_overrides() == []  # type: ignore[union-attr]

    def test_no_overrides_build_teacher_config_returns_none(self) -> None:
        form = self._form()
        assert form.build_teacher_config() is None  # type: ignore[union-attr]

    def test_both_extraction_fields_provided_no_error(self) -> None:
        form = self._form(
            extraction_teacher_profile="fake_chat",
            extraction_teacher_model="cheap-model",
        )
        assert form.validate_phase_overrides() == []  # type: ignore[union-attr]

    def test_profile_only_returns_error(self) -> None:
        form = self._form(extraction_teacher_profile="fake_chat")
        errors = form.validate_phase_overrides()  # type: ignore[union-attr]
        assert len(errors) == 1
        assert "model" in errors[0]

    def test_model_only_returns_error(self) -> None:
        form = self._form(extraction_teacher_model="cheap-model")
        errors = form.validate_phase_overrides()  # type: ignore[union-attr]
        assert len(errors) == 1
        assert "profile" in errors[0]

    def test_all_three_overrides_build_teacher_config(self) -> None:
        form = self._form(
            extraction_teacher_profile="fake_chat",
            extraction_teacher_model="extract-m",
            synthesis_teacher_profile="fake_chat",
            synthesis_teacher_model="synth-m",
            conflict_resolution_teacher_profile="fake_chat",
            conflict_resolution_teacher_model="cr-m",
        )
        tc = form.build_teacher_config()  # type: ignore[union-attr]
        assert tc is not None
        assert tc.for_phase("instruction_extraction").model == "extract-m"
        assert tc.for_phase("cluster_consolidation").model == "synth-m"
        assert tc.for_phase("conflict_resolution").model == "cr-m"

    def test_single_override_inherits_default_for_other_phases(self) -> None:
        form = self._form(
            extraction_teacher_profile="fake_chat",
            extraction_teacher_model="cheap-model",
        )
        tc = form.build_teacher_config()  # type: ignore[union-attr]
        assert tc is not None
        assert tc.for_phase("instruction_extraction").model == "cheap-model"
        # Other phases fall back to default
        assert tc.for_phase("cluster_consolidation").model == "model-default"
        assert tc.for_phase("conflict_resolution").model == "model-default"

    def test_routing_view_has_any_override_false_when_no_overrides(self) -> None:
        form = self._form()
        view = form.build_teacher_routing_view()  # type: ignore[union-attr]
        assert not view.has_any_override
        assert all(not p.is_override for p in view.phases)

    def test_routing_view_has_any_override_true_when_one_set(self) -> None:
        form = self._form(
            extraction_teacher_profile="fake_chat",
            extraction_teacher_model="cheap-model",
        )
        view = form.build_teacher_routing_view()  # type: ignore[union-attr]
        assert view.has_any_override
        extraction = next(p for p in view.phases if p.phase_label == "Extraction")
        assert extraction.is_override
        synthesis = next(p for p in view.phases if p.phase_label == "Synthesis")
        assert not synthesis.is_override
