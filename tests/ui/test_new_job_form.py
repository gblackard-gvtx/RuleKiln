"""Tests: GET /ui/jobs/new — new job form."""

from httpx import AsyncClient


class TestNewJobForm:
    async def test_form_renders_200(self, client: AsyncClient) -> None:
        response = await client.get("/ui/jobs/new")
        assert response.status_code == 200

    async def test_form_contains_file_inputs(self, client: AsyncClient) -> None:
        response = await client.get("/ui/jobs/new")
        assert "task_file" in response.text
        assert "cases_file" in response.text

    async def test_profile_names_populated(self, client: AsyncClient) -> None:
        response = await client.get("/ui/jobs/new")
        # conftest provides "fake_chat" and "fake_embed" profiles
        assert "fake_chat" in response.text
        assert "fake_embed" in response.text

    async def test_form_has_multipart_action(self, client: AsyncClient) -> None:
        response = await client.get("/ui/jobs/new")
        assert "multipart/form-data" in response.text
        assert "/ui/jobs/preview" in response.text

    async def test_form_contains_open_source_setup_help(self, client: AsyncClient) -> None:
        response = await client.get("/ui/jobs/new")
        assert "Before you start" in response.text
        assert "PROVIDER_PROFILES__FAKE__" in response.text
