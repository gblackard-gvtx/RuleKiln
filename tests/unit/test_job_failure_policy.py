"""Unit tests for retryable vs terminal failure policy transitions."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rulekiln.db.models import Base, DistillationJob
from rulekiln.db.repositories.jobs import apply_job_failure_policy, get_job

_IN_MEMORY_URL = "sqlite+aiosqlite://"


@pytest.fixture()
async def db_session_factory():
    engine = create_async_engine(_IN_MEMORY_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    yield factory
    await engine.dispose()


async def _insert_running_job(
    factory: async_sessionmaker[AsyncSession],
    *,
    attempt_count: int,
    max_attempts: int,
) -> str:
    job_id = str(uuid.uuid4())
    async with factory() as session:
        session.add(
            DistillationJob(
                id=job_id,
                task_id="task-1",
                task_name="Task",
                task_mode="classification",
                status="running",
                stage="evaluating_baseline",
                request_json={},
                queue_status="running",
                attempt_count=attempt_count,
                max_attempts=max_attempts,
                locked_by="worker-1",
                locked_at=datetime.now(tz=UTC),
                lease_expires_at=datetime.now(tz=UTC),
            )
        )
        await session.commit()
    return job_id


@pytest.mark.asyncio
async def test_retryable_failure_with_budget_moves_to_waiting_for_retry(db_session_factory) -> None:
    job_id = await _insert_running_job(db_session_factory, attempt_count=1, max_attempts=3)

    async with db_session_factory() as session:
        status = await apply_job_failure_policy(
            session,
            job_id,
            error_message="connection reset",
            retryable=True,
            retry_backoff_seconds=30,
        )

    async with db_session_factory() as session:
        job = await get_job(session, job_id)

    assert status == "waiting_for_retry"
    assert job is not None
    assert job.status == "waiting_for_retry"
    assert job.queue_status == "pending"
    assert job.locked_by is None


@pytest.mark.asyncio
async def test_retryable_failure_without_budget_moves_to_failed_retryable(db_session_factory) -> None:
    job_id = await _insert_running_job(db_session_factory, attempt_count=3, max_attempts=3)

    async with db_session_factory() as session:
        status = await apply_job_failure_policy(
            session,
            job_id,
            error_message="connection reset",
            retryable=True,
            retry_backoff_seconds=30,
        )

    async with db_session_factory() as session:
        job = await get_job(session, job_id)

    assert status == "failed_retryable"
    assert job is not None
    assert job.status == "failed_retryable"
    assert job.queue_status == "failed"


@pytest.mark.asyncio
async def test_terminal_failure_moves_to_failed_terminal(db_session_factory) -> None:
    job_id = await _insert_running_job(db_session_factory, attempt_count=1, max_attempts=3)

    async with db_session_factory() as session:
        status = await apply_job_failure_policy(
            session,
            job_id,
            error_message="invalid task config",
            retryable=False,
            retry_backoff_seconds=30,
        )

    async with db_session_factory() as session:
        job = await get_job(session, job_id)

    assert status == "failed_terminal"
    assert job is not None
    assert job.status == "failed_terminal"
    assert job.queue_status == "failed"
