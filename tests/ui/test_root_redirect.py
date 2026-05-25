"""Tests: GET / redirects to the UI job list."""


class TestRootRedirect:
    async def test_root_redirects_to_ui_jobs(self, client) -> None:
        response = await client.get("/", follow_redirects=False)
        assert response.status_code in {301, 302, 307, 308}
        assert response.headers["location"] == "/ui/jobs"
