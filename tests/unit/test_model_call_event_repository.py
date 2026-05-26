"""Unit tests for model_call_events repository deduplication behavior."""

from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rulekiln.db.models import Base, DistillationJob, ModelCallEvent
from rulekiln.db.repositories.model_calls import bulk_insert_model_call_events
from rulekiln.schemas.usage import ModelCallCost, ModelCallRecord, ModelUsage


_IN_MEMORY_URL = "sqlite+aiosqlite://"


@pytest.fixture()
async def db_session_factory():
    engine = create_async_engine(_IN_MEMORY_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    yield factory
    await engine.dispose()


async def _insert_job(session: AsyncSession, job_id: str) -> None:
    session.add(
        DistillationJob(
            id=job_id,
            task_id="task-1",
            task_name="Task",
            task_mode="classification",
            status="created",
            request_json={},
            queue_status="pending",
        )
    )
    await session.commit()


def _record(job_id: str, idempotency_key: str) -> ModelCallRecord:
    return ModelCallRecord(
        job_id=job_id,
        stage="evaluating_baseline",
        role="student",
        provider_profile="local",
        provider="openai_compatible",
        model="qwen-4b",
        student_id="qwen-4b",
        strategy="baseline",
        case_id="case-1",
        usage=ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15, estimated=False),
        cost=ModelCallCost(
            input_cost_usd=Decimal("0.001"),
            output_cost_usd=Decimal("0.002"),
            total_cost_usd=Decimal("0.003"),
            pricing_source="test",
            estimated=False,
        ),
        latency_ms=50,
        status="success",
        idempotency_key=idempotency_key,
    )


@pytest.mark.asyncio
async def test_bulk_insert_model_call_events_skips_duplicate_idempotency_keys(
    db_session_factory,
) -> None:
    job_id = "job-1"
    async with db_session_factory() as session:
        await _insert_job(session, job_id)

    async with db_session_factory() as session:
        records = [_record(job_id, "key-1"), _record(job_id, "key-1")]
        await bulk_insert_model_call_events(session, job_id, records)
        await session.commit()

    async with db_session_factory() as session:
        result = await session.execute(select(ModelCallEvent).where(ModelCallEvent.job_id == job_id))
        events = list(result.scalars().all())

    assert len(events) == 1
    assert events[0].idempotency_key == "key-1"
