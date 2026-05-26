"""Tests: POST /ui/jobs/{job_id}/cancel — cancel pipeline from UI."""

import uuid
from datetime import UTC, datetime

from httpx import AsyncClient
from sqlalchemy import select

from rulekiln.db.models import DistillationJob


def _build_job(**kwargs: object) -> DistillationJob:
    defaults: dict[str, object] = {
        "id": str(uuid.uuid4()),
        "task_id": "t1",
        "task_name": "Cancelable Task",
        "task_mode": "classification",
        "status": "pending",
        "queue_status": "pending",
        "stage": None,
        "request_json": {},
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
    }
    defaults.update(kwargs)
    return DistillationJob(**defaults)


async def _insert_job(factory, **kwargs: object) -> str:
    job = _build_job(**kwargs)
    async with factory() as session:
        session.add(job)
        await session.commit()
    return job.id


class _FakeDBOS:
    cancelled_workflow_ids: list[str] = []

    @classmethod
    def get_workflow_status(cls, workflow_id: str) -> object:
        _ = workflow_id
        return object()

    @classmethod
    def cancel_workflow(cls, workflow_id: str) -> None:
        cls.cancelled_workflow_ids.append(workflow_id)


class TestCancelJob:
    async def test_unknown_job_returns_404(self, client: AsyncClient) -> None:
        response = await client.post(f"/ui/jobs/{uuid.uuid4()}/cancel", follow_redirects=False)
        assert response.status_code == 404

    async def test_cancel_pending_job_marks_terminal(
        self,
        client: AsyncClient,
        db_session_factory,
    ) -> None:
        job_id = await _insert_job(db_session_factory, status="pending", queue_status="pending")

        response = await client.post(f"/ui/jobs/{job_id}/cancel", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == f"/ui/jobs/{job_id}"

        async with db_session_factory() as session:
            result = await session.execute(
                select(DistillationJob).where(DistillationJob.id == job_id)
            )
            job = result.scalar_one()

        assert job.status == "failed_terminal"
        assert job.queue_status == "failed"
        assert job.error_message == "Cancelled by operator."

    async def test_cancel_completed_job_is_noop(
        self,
        client: AsyncClient,
        db_session_factory,
    ) -> None:
        job_id = await _insert_job(
            db_session_factory,
            status="completed",
            queue_status="completed",
            error_message=None,
        )

        response = await client.post(f"/ui/jobs/{job_id}/cancel", follow_redirects=False)
        assert response.status_code == 303

        async with db_session_factory() as session:
            result = await session.execute(
                select(DistillationJob).where(DistillationJob.id == job_id)
            )
            job = result.scalar_one()

        assert job.status == "completed"
        assert job.queue_status == "completed"
        assert job.error_message is None

    async def test_dbos_cancel_attempts_workflow_cancellation(
        self,
        client: AsyncClient,
        db_session_factory,
        test_settings,
        monkeypatch,
    ) -> None:
        test_settings.execution_backend = "dbos"
        _FakeDBOS.cancelled_workflow_ids = []

        monkeypatch.setattr(
            "rulekiln.ui.routes.ensure_dbos_runtime_launched",
            lambda _settings: None,
        )
        monkeypatch.setattr("dbos.DBOS", _FakeDBOS)

        job_id = await _insert_job(db_session_factory, status="running", queue_status="running")

        response = await client.post(f"/ui/jobs/{job_id}/cancel", follow_redirects=False)
        assert response.status_code == 303

        assert _FakeDBOS.cancelled_workflow_ids == [f"rulekiln-job-{job_id}"]

        async with db_session_factory() as session:
            result = await session.execute(
                select(DistillationJob).where(DistillationJob.id == job_id)
            )
            job = result.scalar_one()

        assert job.status == "failed_terminal"
        assert job.queue_status == "failed"
