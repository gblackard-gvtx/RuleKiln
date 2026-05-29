"""Tests: POST /ui/jobs/{job_id}/retry — retry pipeline from UI."""

import uuid
from datetime import UTC, datetime

from httpx import AsyncClient
from sqlalchemy import select

from rulekiln.db.models import DistillationJob
from rulekiln.schemas.job import DistillationRequest


def _valid_request_json() -> dict[str, object]:
    return {
        "task": {
            "task_id": "t1",
            "task_name": "Retry Task",
            "task_mode": "classification",
            "description": "retry me",
            "input_template": "{{input}}",
        },
        "cases": [{"id": "c1", "task_mode": "classification", "input": {"x": 1}}],
        "teacher": {"provider_profile": "fake_chat", "model": "model-a"},
        "student": {"provider_profile": "fake_chat", "model": "model-b"},
        "embedding": {"provider_profile": "fake_embed", "model": "model-c"},
    }


def _build_job(**kwargs: object) -> DistillationJob:
    defaults: dict[str, object] = {
        "id": str(uuid.uuid4()),
        "task_id": "t1",
        "task_name": "Retryable Task",
        "task_mode": "classification",
        "status": "failed_retryable",
        "queue_status": "failed",
        "stage": "extracting_rules",
        "request_json": _valid_request_json(),
        "error_message": "previous timeout",
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


class TestRetryJob:
    async def test_unknown_job_returns_404(self, client: AsyncClient) -> None:
        response = await client.post(f"/ui/jobs/{uuid.uuid4()}/retry", follow_redirects=False)
        assert response.status_code == 404

    async def test_retry_failed_job_requeues_for_worker_backend(
        self,
        client: AsyncClient,
        db_session_factory,
        test_settings,
    ) -> None:
        test_settings.execution_backend = "postgres_queue"
        job_id = await _insert_job(
            db_session_factory,
            status="failed_retryable",
            queue_status="failed",
            stage="extracting_rules",
            error_message="ReadTimeout",
            locked_by="worker-1",
            locked_at=datetime.now(UTC),
            lease_expires_at=datetime.now(UTC),
        )

        response = await client.post(f"/ui/jobs/{job_id}/retry", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == f"/ui/jobs/{job_id}"

        async with db_session_factory() as session:
            result = await session.execute(
                select(DistillationJob).where(DistillationJob.id == job_id)
            )
            job = result.scalar_one()

        assert job.status == "waiting_for_retry"
        assert job.queue_status == "pending"
        assert job.stage == "extracting_rules"
        assert job.error_message is None
        assert job.locked_by is None
        assert job.locked_at is None
        assert job.lease_expires_at is None

    async def test_retry_completed_job_is_noop(
        self,
        client: AsyncClient,
        db_session_factory,
        test_settings,
    ) -> None:
        test_settings.execution_backend = "postgres_queue"
        job_id = await _insert_job(
            db_session_factory,
            status="completed",
            queue_status="completed",
            stage="completed",
            error_message=None,
        )

        response = await client.post(f"/ui/jobs/{job_id}/retry", follow_redirects=False)
        assert response.status_code == 303

        async with db_session_factory() as session:
            result = await session.execute(
                select(DistillationJob).where(DistillationJob.id == job_id)
            )
            job = result.scalar_one()

        assert job.status == "completed"
        assert job.queue_status == "completed"
        assert job.stage == "completed"
        assert job.error_message is None

    async def test_retry_failed_job_background_restarts_pipeline(
        self,
        client: AsyncClient,
        db_session_factory,
        test_settings,
        monkeypatch,
    ) -> None:
        test_settings.execution_backend = "background_tasks"

        observed_job_ids: list[str] = []

        async def _capture_pipeline_call(job_id: str, payload: DistillationRequest) -> None:
            _ = payload
            observed_job_ids.append(job_id)

        monkeypatch.setattr(
            "rulekiln.ui.routes.run_distillation_pipeline",
            _capture_pipeline_call,
        )

        job_id = await _insert_job(
            db_session_factory,
            status="failed_terminal",
            queue_status="failed",
            error_message="Cancelled by operator.",
        )

        response = await client.post(f"/ui/jobs/{job_id}/retry", follow_redirects=False)
        assert response.status_code == 303
        assert observed_job_ids == [job_id]

        async with db_session_factory() as session:
            result = await session.execute(
                select(DistillationJob).where(DistillationJob.id == job_id)
            )
            job = result.scalar_one()

        assert job.status == "created"
        assert job.queue_status == "created"
        assert job.error_message is None
