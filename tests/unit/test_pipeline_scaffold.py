"""Unit tests for PR 1 pipeline package scaffold.

Covers PhasePlan.from_phase and StageGuard.run_once.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rulekiln.db.models import Base, DistillationJob
from rulekiln.db.repositories.jobs import create_job
from rulekiln.workers.distillation_worker import PipelinePhase
from rulekiln.workers.pipeline.phase_plan import PhasePlan
from rulekiln.workers.pipeline.stage_guard import StageGuard

_IN_MEMORY_URL = "sqlite+aiosqlite://"


# ── PhasePlan.from_phase ──────────────────────────────────────────────────────


def test_phase_plan_full_runs_all() -> None:
    plan = PhasePlan.from_phase("full")
    assert plan.run_validate
    assert plan.run_compile_phase
    assert plan.run_post_extraction
    assert plan.run_baseline_eval
    assert plan.run_aggregate


def test_phase_plan_validate_only() -> None:
    plan = PhasePlan.from_phase("validate_project")
    assert plan.run_validate
    assert not plan.run_compile_phase
    assert not plan.run_post_extraction
    assert not plan.run_baseline_eval
    assert not plan.run_aggregate


def test_phase_plan_compile_prompts() -> None:
    plan = PhasePlan.from_phase("compile_prompts")
    assert plan.run_validate
    assert plan.run_compile_phase
    assert plan.run_post_extraction
    assert not plan.run_baseline_eval


def test_phase_plan_batch_submit_no_post_extraction() -> None:
    plan = PhasePlan.from_phase("extraction_batch_submit")
    assert not plan.run_validate
    assert plan.run_compile_phase
    assert not plan.run_post_extraction


def test_phase_plan_batch_collect_runs_post_extraction() -> None:
    plan = PhasePlan.from_phase("extraction_batch_collect")
    assert not plan.run_validate
    assert plan.run_compile_phase
    assert plan.run_post_extraction


def test_phase_plan_evaluate_baseline() -> None:
    plan = PhasePlan.from_phase("evaluate_baseline")
    assert not plan.run_compile_phase
    assert plan.run_baseline_eval
    assert not plan.run_aggregate


def test_phase_plan_aggregate() -> None:
    plan = PhasePlan.from_phase("aggregate_evaluation_report")
    assert not plan.run_compile_phase
    assert not plan.run_baseline_eval
    assert plan.run_aggregate


def test_phase_plan_covers_all_phase_values() -> None:
    """Every value in the PipelinePhase Literal must be handled without raising."""
    all_phases: list[PipelinePhase] = [
        "full",
        "validate_project",
        "compile_prompts",
        "extraction_batch_submit",
        "extraction_batch_collect",
        "evaluate_baseline",
        "evaluate_dbscan",
        "evaluate_hdbscan",
        "aggregate_evaluation_report",
    ]
    for phase in all_phases:
        plan = PhasePlan.from_phase(phase)
        assert isinstance(plan, PhasePlan)


# ── StageGuard.run_once ───────────────────────────────────────────────────────


@pytest.fixture()
async def db_session():
    engine = create_async_engine(_IN_MEMORY_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        yield session
    await engine.dispose()


async def _make_job(session: AsyncSession, job_id: str) -> None:
    await create_job(
        session,
        DistillationJob(
            id=job_id,
            task_id="test-task",
            task_name="Test",
            task_mode="classification",
            status="created",
            stage=None,
            request_json={},
        ),
    )


@pytest.mark.asyncio
async def test_stage_guard_run_once_calls_fn(db_session: AsyncSession) -> None:
    job_id = "guard-test-0001"
    await _make_job(db_session, job_id)
    guard = StageGuard(db_session, job_id)

    calls: list[int] = []

    async def fn() -> int:
        calls.append(1)
        return 42

    result = await guard.run_once("test_stage", fn=fn)
    assert result == 42
    assert calls == [1]


@pytest.mark.asyncio
async def test_stage_guard_run_once_is_idempotent(db_session: AsyncSession) -> None:
    job_id = "guard-test-0002"
    await _make_job(db_session, job_id)
    guard = StageGuard(db_session, job_id)

    calls: list[int] = []

    async def fn() -> int:
        calls.append(1)
        return 99

    await guard.run_once("idempotent_stage", fn=fn)
    second = await guard.run_once("idempotent_stage", fn=fn)

    assert second is None
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_stage_guard_strategy_scoped(db_session: AsyncSession) -> None:
    """Different strategy values are independent stage markers."""
    job_id = "guard-test-0003"
    await _make_job(db_session, job_id)
    guard = StageGuard(db_session, job_id)

    calls: list[str] = []

    async def fn_a() -> None:
        calls.append("a")

    async def fn_b() -> None:
        calls.append("b")

    await guard.run_once("eval_stage", strategy="dbscan", fn=fn_a)
    await guard.run_once("eval_stage", strategy="hdbscan", fn=fn_b)
    # Re-running same strategy should be skipped
    await guard.run_once("eval_stage", strategy="dbscan", fn=fn_a)

    assert calls == ["a", "b"]


@pytest.mark.asyncio
async def test_stage_guard_is_complete_before_and_after(db_session: AsyncSession) -> None:
    job_id = "guard-test-0004"
    await _make_job(db_session, job_id)
    guard = StageGuard(db_session, job_id)

    assert not await guard.is_complete("some_stage")

    async def fn() -> None:
        pass

    await guard.run_once("some_stage", fn=fn)
    assert await guard.is_complete("some_stage")
