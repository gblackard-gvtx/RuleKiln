"""Tests: GET /ui/jobs/{job_id} — job detail page."""

import uuid
from datetime import UTC, datetime

from httpx import AsyncClient

from rulekiln.db.models import DistillationJob


async def _insert_job(factory, **kwargs) -> str:
    job_id = str(uuid.uuid4())
    defaults = {
        "id": job_id,
        "task_id": "t1",
        "task_name": "Detail Task",
        "task_mode": "classification",
        "status": "completed",
        "stage": None,
        "request_json": {},
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
    }
    defaults.update(kwargs)
    async with factory() as session:
        session.add(DistillationJob(**defaults))
        await session.commit()
    return defaults["id"]


class TestJobDetail:
    async def test_unknown_job_returns_404(self, client: AsyncClient) -> None:
        response = await client.get(f"/ui/jobs/{uuid.uuid4()}")
        assert response.status_code == 404

    async def test_completed_job_renders_200(self, client: AsyncClient, db_session_factory) -> None:
        job_id = await _insert_job(db_session_factory, status="completed")
        response = await client.get(f"/ui/jobs/{job_id}")
        assert response.status_code == 200
        assert "Detail Task" in response.text

    async def test_running_job_has_htmx_polling(
        self, client: AsyncClient, db_session_factory
    ) -> None:
        job_id = await _insert_job(db_session_factory, status="running")
        response = await client.get(f"/ui/jobs/{job_id}")
        assert response.status_code == 200
        assert "hx-get" in response.text
        assert "every 2s" in response.text

    async def test_completed_job_has_no_polling(
        self, client: AsyncClient, db_session_factory
    ) -> None:
        job_id = await _insert_job(db_session_factory, status="completed")
        response = await client.get(f"/ui/jobs/{job_id}")
        assert response.status_code == 200
        assert "every 2s" not in response.text

    async def test_failed_job_shows_error_message(
        self, client: AsyncClient, db_session_factory
    ) -> None:
        job_id = await _insert_job(
            db_session_factory,
            status="failed",
            error_message="Something went wrong",
        )
        response = await client.get(f"/ui/jobs/{job_id}")
        assert response.status_code == 200
        assert "Something went wrong" in response.text

    async def test_status_fragment_polling(self, client: AsyncClient, db_session_factory) -> None:
        job_id = await _insert_job(db_session_factory, status="running")
        response = await client.get(f"/ui/jobs/{job_id}/status-fragment")
        assert response.status_code == 200
        assert "hx-get" in response.text

    async def test_status_fragment_waiting_for_retry_has_polling(
        self, client: AsyncClient, db_session_factory
    ) -> None:
        job_id = await _insert_job(db_session_factory, status="waiting_for_retry")
        response = await client.get(f"/ui/jobs/{job_id}/status-fragment")
        assert response.status_code == 200
        assert "hx-get" in response.text

    async def test_status_fragment_completed_no_polling(
        self, client: AsyncClient, db_session_factory
    ) -> None:
        job_id = await _insert_job(db_session_factory, status="completed")
        response = await client.get(f"/ui/jobs/{job_id}/status-fragment")
        assert response.status_code == 200
        assert "hx-get" not in response.text

    async def test_running_job_shows_cancel_button(
        self,
        client: AsyncClient,
        db_session_factory,
    ) -> None:
        job_id = await _insert_job(db_session_factory, status="running")
        response = await client.get(f"/ui/jobs/{job_id}")
        assert response.status_code == 200
        assert 'id="cancel-pipeline-btn"' in response.text

    async def test_completed_job_hides_cancel_button(
        self,
        client: AsyncClient,
        db_session_factory,
    ) -> None:
        job_id = await _insert_job(db_session_factory, status="completed")
        response = await client.get(f"/ui/jobs/{job_id}")
        assert response.status_code == 200
        assert 'id="cancel-pipeline-btn"' not in response.text

    async def test_failed_retryable_job_shows_retry_button(
        self,
        client: AsyncClient,
        db_session_factory,
    ) -> None:
        job_id = await _insert_job(db_session_factory, status="failed_retryable")
        response = await client.get(f"/ui/jobs/{job_id}")
        assert response.status_code == 200
        assert 'id="retry-pipeline-btn"' in response.text

    async def test_running_job_hides_retry_button(
        self,
        client: AsyncClient,
        db_session_factory,
    ) -> None:
        job_id = await _insert_job(db_session_factory, status="running")
        response = await client.get(f"/ui/jobs/{job_id}")
        assert response.status_code == 200
        assert 'id="retry-pipeline-btn"' not in response.text
