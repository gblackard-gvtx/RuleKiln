"""Tests: POST /ui/jobs/preview — task/case validation and preview rendering."""

from httpx import AsyncClient

from tests.ui.conftest import VALID_CASE_LINE, VALID_TASK_YAML


_TRAIN_FALLBACK_WARNING = "No validation cases detected. Evaluation fell back to split=train."


def _form_data() -> dict[str, str]:
    """Default provider form fields."""
    return {
        "teacher_profile": "fake_chat",
        "teacher_model": "model-a",
        "student_profile": "fake_chat",
        "student_model": "model-b",
        "embedding_profile": "fake_embed",
        "embedding_model": "model-c",
    }


class TestPreview:
    async def test_valid_upload_renders_preview(self, client: AsyncClient) -> None:
        response = await client.post(
            "/ui/jobs/preview",
            files={
                "task_file": ("task.yaml", VALID_TASK_YAML, "application/yaml"),
                "cases_file": ("cases.jsonl", VALID_CASE_LINE, "application/x-ndjson"),
            },
            data=_form_data(),
        )
        assert response.status_code == 200
        assert "Test Task" in response.text

    async def test_invalid_extension_returns_error(self, client: AsyncClient) -> None:
        response = await client.post(
            "/ui/jobs/preview",
            files={
                "task_file": ("task.txt", VALID_TASK_YAML, "text/plain"),
                "cases_file": ("cases.jsonl", VALID_CASE_LINE, "application/x-ndjson"),
            },
            data=_form_data(),
        )
        assert response.status_code == 422
        assert "extension" in response.text.lower()

    async def test_malformed_yaml_returns_error(self, client: AsyncClient) -> None:
        bad_yaml = b": not: valid: yaml\n  broken"
        response = await client.post(
            "/ui/jobs/preview",
            files={
                "task_file": ("task.yaml", bad_yaml, "application/yaml"),
                "cases_file": ("cases.jsonl", VALID_CASE_LINE, "application/x-ndjson"),
            },
            data=_form_data(),
        )
        assert response.status_code == 422

    async def test_malformed_jsonl_returns_error(self, client: AsyncClient) -> None:
        bad_jsonl = b"this is not json\n"
        response = await client.post(
            "/ui/jobs/preview",
            files={
                "task_file": ("task.yaml", VALID_TASK_YAML, "application/yaml"),
                "cases_file": ("cases.jsonl", bad_jsonl, "application/x-ndjson"),
            },
            data=_form_data(),
        )
        assert response.status_code == 422

    async def test_preview_shows_case_count(self, client: AsyncClient) -> None:
        two_cases = VALID_CASE_LINE + VALID_CASE_LINE.replace(b'"id":"c1"', b'"id":"c2"')
        response = await client.post(
            "/ui/jobs/preview",
            files={
                "task_file": ("task.yaml", VALID_TASK_YAML, "application/yaml"),
                "cases_file": ("cases.jsonl", two_cases, "application/x-ndjson"),
            },
            data=_form_data(),
        )
        assert response.status_code == 200
        assert "2" in response.text

    async def test_preview_warns_when_validation_missing(self, client: AsyncClient) -> None:
        response = await client.post(
            "/ui/jobs/preview",
            files={
                "task_file": ("task.yaml", VALID_TASK_YAML, "application/yaml"),
                "cases_file": ("cases.jsonl", VALID_CASE_LINE, "application/x-ndjson"),
            },
            data=_form_data(),
        )
        assert response.status_code == 200
        assert _TRAIN_FALLBACK_WARNING in response.text

    async def test_preview_train_validation_upload_has_no_train_fallback_warning(
        self, client: AsyncClient
    ) -> None:
        validation_case = (
            VALID_CASE_LINE.replace(b'"id":"c1"', b'"id":"c2"').replace(
                b'"split":"train"', b'"split":"validation"'
            )
        )
        combined_cases = VALID_CASE_LINE + validation_case

        response = await client.post(
            "/ui/jobs/preview",
            files={
                "task_file": ("task.yaml", VALID_TASK_YAML, "application/yaml"),
                "cases_file": ("cases.jsonl", combined_cases, "application/x-ndjson"),
            },
            data=_form_data(),
        )
        assert response.status_code == 200
        assert _TRAIN_FALLBACK_WARNING not in response.text

    async def test_unknown_provider_profile_returns_error(self, client: AsyncClient) -> None:
        data = _form_data()
        data["teacher_profile"] = "nonexistent_profile"
        response = await client.post(
            "/ui/jobs/preview",
            files={
                "task_file": ("task.yaml", VALID_TASK_YAML, "application/yaml"),
                "cases_file": ("cases.jsonl", VALID_CASE_LINE, "application/x-ndjson"),
            },
            data=data,
        )
        assert response.status_code == 422

    async def test_valid_preview_creates_draft_job(
        self, client: AsyncClient, db_session_factory
    ) -> None:
        response = await client.post(
            "/ui/jobs/preview",
            files={
                "task_file": ("task.yaml", VALID_TASK_YAML, "application/yaml"),
                "cases_file": ("cases.jsonl", VALID_CASE_LINE, "application/x-ndjson"),
            },
            data=_form_data(),
        )
        assert response.status_code == 200
        # A hidden draft_job_id input should be present
        assert 'name="draft_job_id"' in response.text
        # Run Pipeline button should be present
        assert "Run Pipeline" in response.text
