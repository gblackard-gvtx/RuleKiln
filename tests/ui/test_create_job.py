"""Tests: POST /ui/jobs — create job from draft."""

import uuid

from httpx import AsyncClient

from tests.ui.conftest import VALID_CASE_LINE, VALID_TASK_YAML


def _form_data() -> dict[str, str]:
    return {
        "teacher_profile": "fake_chat",
        "teacher_model": "model-a",
        "student_profile": "fake_chat",
        "student_model": "model-b",
        "embedding_profile": "fake_embed",
        "embedding_model": "model-c",
    }


class TestCreateJob:
    async def _get_draft_job_id(self, client: AsyncClient) -> str:
        response = await client.post(
            "/ui/jobs/preview",
            files={
                "task_file": ("task.yaml", VALID_TASK_YAML, "application/yaml"),
                "cases_file": ("cases.jsonl", VALID_CASE_LINE, "application/x-ndjson"),
            },
            data=_form_data(),
        )
        assert response.status_code == 200
        # Extract hidden draft_job_id value
        text = response.text
        start = text.index('name="draft_job_id"')
        val_start = text.index('value="', start) + len('value="')
        val_end = text.index('"', val_start)
        return text[val_start:val_end]

    async def test_submit_draft_redirects_to_detail(self, client: AsyncClient) -> None:
        draft_id = await self._get_draft_job_id(client)
        response = await client.post(
            "/ui/jobs",
            data={"draft_job_id": draft_id},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert f"/ui/jobs/{draft_id}" in response.headers["location"]

    async def test_submit_missing_draft_id_returns_error(self, client: AsyncClient) -> None:
        # Missing form field should return 422
        response = await client.post(
            "/ui/jobs",
            data={},
            follow_redirects=False,
        )
        assert response.status_code == 422

    async def test_submit_unknown_draft_id_returns_404(self, client: AsyncClient) -> None:
        response = await client.post(
            "/ui/jobs",
            data={"draft_job_id": str(uuid.uuid4())},
            follow_redirects=False,
        )
        assert response.status_code == 404
