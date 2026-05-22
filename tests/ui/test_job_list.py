"""Tests: GET /ui/jobs — job list page."""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from rulekiln.db.models import DistillationJob


async def _insert_job(factory, **kwargs) -> str:
    import uuid
    from datetime import datetime, timezone

    defaults = dict(
        id=str(uuid.uuid4()),
        task_id="t1",
        task_name="Test Task",
        task_mode="classification",
        status="completed",
        stage=None,
        request_json={},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    defaults.update(kwargs)
    async with factory() as session:
        session.add(DistillationJob(**defaults))
        await session.commit()
    return defaults["id"]


class TestJobList:
    async def test_empty_list_returns_200(self, client: AsyncClient) -> None:
        response = await client.get("/ui/jobs")
        assert response.status_code == 200
        assert "No jobs yet" in response.text

    async def test_jobs_appear_in_list(
        self, client: AsyncClient, db_session_factory
    ) -> None:
        await _insert_job(db_session_factory, task_name="Alpha Job")
        response = await client.get("/ui/jobs")
        assert response.status_code == 200
        assert "Alpha Job" in response.text

    async def test_root_redirects_to_jobs(self, client: AsyncClient) -> None:
        response = await client.get("/ui/", follow_redirects=False)
        assert response.status_code == 302
        assert "/ui/jobs" in response.headers["location"]

    async def test_status_badge_rendered(
        self, client: AsyncClient, db_session_factory
    ) -> None:
        await _insert_job(db_session_factory, status="failed")
        response = await client.get("/ui/jobs")
        assert response.status_code == 200
        assert "badge-failed" in response.text
