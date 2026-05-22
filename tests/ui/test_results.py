"""Tests: GET /ui/jobs/{job_id}/results — results summary page."""

import uuid
from datetime import datetime, timezone

import pytest
from httpx import AsyncClient

from rulekiln.db.models import DistillationJob, EvalRun, PromptVersion


async def _insert_job(factory, **kwargs) -> str:
    job_id = str(uuid.uuid4())
    defaults = dict(
        id=job_id,
        task_id="t1",
        task_name="Results Task",
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


async def _insert_eval_run(factory, job_id: str, strategy: str, split: str) -> None:
    async with factory() as session:
        session.add(
            EvalRun(
                id=str(uuid.uuid4()),
                job_id=job_id,
                prompt_version_id=str(uuid.uuid4()),
                strategy=strategy,
                model="test-model",
                split=split,
                accuracy=0.85,
                macro_f1=0.83,
                weighted_case_score=0.84,
                malformed_output_rate=0.01,
                created_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()


class TestResults:
    async def test_unknown_job_returns_404(self, client: AsyncClient) -> None:
        response = await client.get(f"/ui/jobs/{uuid.uuid4()}/results")
        assert response.status_code == 404

    async def test_no_eval_runs_renders_empty(
        self, client: AsyncClient, db_session_factory
    ) -> None:
        job_id = await _insert_job(db_session_factory)
        response = await client.get(f"/ui/jobs/{job_id}/results")
        assert response.status_code == 200

    async def test_eval_runs_populate_metrics(
        self, client: AsyncClient, db_session_factory
    ) -> None:
        job_id = await _insert_job(db_session_factory)
        await _insert_eval_run(db_session_factory, job_id, "dbscan", "validation")
        await _insert_eval_run(db_session_factory, job_id, "hdbscan", "validation")
        response = await client.get(f"/ui/jobs/{job_id}/results")
        assert response.status_code == 200
        # Should display metric values
        assert "0.84" in response.text
