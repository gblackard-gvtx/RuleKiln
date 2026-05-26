"""Integration tests for DBOS stage workflow orchestration."""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rulekiln.config.settings import AppSettings, ProviderProfile
from rulekiln.db.models import Base, DistillationJob, StageMarker
from rulekiln.db.repositories.jobs import create_job, get_eval_runs_for_job, get_job
from rulekiln.db.session import override_session_factory
from rulekiln.schemas.job import DistillationRequest, ModelRoute
from rulekiln.schemas.task_case import EvaluationSpec, RuleKilnCase, RuleKilnTask
from rulekiln.workers import distillation_worker as distillation_worker_module
from rulekiln.workers.distillation_worker import PipelineStage
from rulekiln.workers.dbos_workflow import run_dbos_stage_workflow

_IN_MEMORY_URL = "sqlite+aiosqlite://"


@pytest.fixture()
async def db_factory():
    engine = create_async_engine(_IN_MEMORY_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    override_session_factory(factory)
    yield factory
    await engine.dispose()


@pytest.fixture()
def fake_settings() -> AppSettings:
    return AppSettings(
        DATABASE_URL=_IN_MEMORY_URL,
        MLFLOW_TRACKING_URI="file:///tmp/mlflow-dbos-stage-test",
        provider_profiles={
            "fake": ProviderProfile(
                provider="fake",
                supports_chat=True,
                supports_embeddings=True,
            ),
        },
    )


def _build_payload() -> DistillationRequest:
    task = RuleKilnTask(
        task_id="spike-task",
        task_name="Spike Task",
        task_mode="classification",
        description="Spike test task",
        input_template="{{input}}",
    )
    cases = [
        RuleKilnCase(
            id=f"case-{i}",
            task_mode="classification",
            split="train",
            input={"text": f"input {i}"},
            expected="positive" if i % 2 == 0 else "negative",
            evaluation=EvaluationSpec(assertions=[]),
        )
        for i in range(3)
    ]
    route = ModelRoute(provider_profile="fake", model="fake-model")
    return DistillationRequest(
        task=task,
        cases=cases,
        teacher=route,
        student=route,
        embedding=route,
    )


@pytest.mark.asyncio
async def test_dbos_spike_workflow_completes_and_is_idempotent(
    db_factory,
    fake_settings,
    monkeypatch,
) -> None:
    monkeypatch.setattr("rulekiln.workers.distillation_worker.get_settings", lambda: fake_settings)
    monkeypatch.setattr("rulekiln.workers.dbos_workflow.DBOS", None)
    monkeypatch.setattr("rulekiln.workers.dbos_workflow.SetWorkflowID", None)

    payload = _build_payload()
    job_id = "aaaaaaaa-1111-0000-0000-0000000000db"

    async with db_factory() as session:
        job = DistillationJob(
            id=job_id,
            task_id=payload.task.task_id,
            task_name=payload.task.task_name,
            task_mode=payload.task.task_mode,
            status="created",
            stage=None,
            request_json=payload.model_dump(mode="json"),
        )
        await create_job(session, job)

    async with db_factory() as session:
        _ = session  # compatibility with earlier test setup
        await run_dbos_stage_workflow(job_id, payload)

    # Re-run to verify stage/idempotency behavior does not duplicate eval runs.
    async with db_factory() as session:
        _ = session
        await run_dbos_stage_workflow(job_id, payload)

    async with db_factory() as session:
        db_job = await get_job(session, job_id)
        eval_runs = await get_eval_runs_for_job(session, job_id)
        marker_rows = await session.execute(
            select(StageMarker.stage, StageMarker.strategy).where(StageMarker.job_id == job_id)
        )
        marker_pairs = {(stage, strategy) for stage, strategy in marker_rows.all()}
        marker_stages = {stage for stage, _ in marker_pairs}

    assert db_job is not None
    assert db_job.status == "completed"
    assert db_job.stage == PipelineStage.COMPLETED

    assert PipelineStage.VALIDATING_PROJECT in marker_stages
    assert PipelineStage.COMPILING_PROMPTS in marker_stages
    assert PipelineStage.EVALUATING_BASELINE in marker_stages
    assert PipelineStage.EVALUATING_DISTILLED in marker_stages
    assert PipelineStage.CHECKING_QUALITY_GATES in marker_stages
    assert PipelineStage.SELECTING_STRATEGY in marker_stages
    assert PipelineStage.ANALYZING_FAILURES in marker_stages
    assert PipelineStage.LOGGING_ARTIFACTS in marker_stages
    assert PipelineStage.EXPORTING_ARTIFACTS in marker_stages

    assert (PipelineStage.COMPILING_PROMPTS, "dbscan") in marker_pairs
    assert (PipelineStage.COMPILING_PROMPTS, "hdbscan") in marker_pairs
    assert (PipelineStage.EVALUATING_DISTILLED, "dbscan") in marker_pairs
    assert (PipelineStage.EVALUATING_DISTILLED, "hdbscan") in marker_pairs

    strategy_counts = {
        strategy: len([run for run in eval_runs if run.strategy == strategy])
        for strategy in ("baseline", "dbscan", "hdbscan")
    }
    assert strategy_counts["baseline"] == 1
    assert strategy_counts["dbscan"] == 1
    assert strategy_counts["hdbscan"] == 1


@pytest.mark.asyncio
async def test_dbos_stage_workflow_resumes_without_rerunning_compile_or_baseline(
    db_factory,
    fake_settings,
    monkeypatch,
) -> None:
    monkeypatch.setattr("rulekiln.workers.distillation_worker.get_settings", lambda: fake_settings)
    monkeypatch.setattr("rulekiln.workers.dbos_workflow.DBOS", None)
    monkeypatch.setattr("rulekiln.workers.dbos_workflow.SetWorkflowID", None)

    original_evaluate_prompt = distillation_worker_module.evaluate_prompt
    failed_once = {"dbscan": False}

    async def _evaluate_prompt_fail_once(*args: object, **kwargs: object):
        strategy = str(kwargs.get("strategy", ""))
        if strategy == "dbscan" and not failed_once["dbscan"]:
            failed_once["dbscan"] = True
            raise RuntimeError("forced dbscan eval failure")
        return await original_evaluate_prompt(*args, **kwargs)

    monkeypatch.setattr(
        "rulekiln.workers.distillation_worker.evaluate_prompt",
        _evaluate_prompt_fail_once,
    )

    payload = _build_payload()
    job_id = "aaaaaaaa-1111-0000-0000-0000000000dc"

    async with db_factory() as session:
        job = DistillationJob(
            id=job_id,
            task_id=payload.task.task_id,
            task_name=payload.task.task_name,
            task_mode=payload.task.task_mode,
            status="created",
            stage=None,
            request_json=payload.model_dump(mode="json"),
        )
        await create_job(session, job)

    with pytest.raises(RuntimeError, match="forced dbscan eval failure"):
        await run_dbos_stage_workflow(job_id, payload)

    await run_dbos_stage_workflow(job_id, payload)

    async with db_factory() as session:
        db_job = await get_job(session, job_id)
        eval_runs = await get_eval_runs_for_job(session, job_id)
        marker_rows = await session.execute(
            select(StageMarker.stage, StageMarker.strategy).where(StageMarker.job_id == job_id)
        )
        marker_pairs = [(stage, strategy) for stage, strategy in marker_rows.all()]

    assert db_job is not None
    assert db_job.status == "completed"
    assert db_job.stage == PipelineStage.COMPLETED

    compile_dbscan_count = len(
        [
            pair
            for pair in marker_pairs
            if pair == (PipelineStage.COMPILING_PROMPTS, "dbscan")
        ]
    )
    compile_hdbscan_count = len(
        [
            pair
            for pair in marker_pairs
            if pair == (PipelineStage.COMPILING_PROMPTS, "hdbscan")
        ]
    )
    baseline_stage_count = len(
        [
            pair
            for pair in marker_pairs
            if pair == (PipelineStage.EVALUATING_BASELINE, None)
        ]
    )

    assert compile_dbscan_count == 1
    assert compile_hdbscan_count == 1
    assert baseline_stage_count == 1

    strategy_counts = {
        strategy: len([run for run in eval_runs if run.strategy == strategy])
        for strategy in ("baseline", "dbscan", "hdbscan")
    }
    assert strategy_counts["baseline"] == 1
    assert strategy_counts["dbscan"] == 1
    assert strategy_counts["hdbscan"] == 1


class _FakeHandle:
    def get_result(self) -> None:
        return None


class _FakeStatus:
    def __init__(self, status: str) -> None:
        self.status = status


class _FakeSetWorkflowID:
    def __init__(self, workflow_id: str) -> None:
        self.workflow_id = workflow_id

    def __enter__(self) -> _FakeSetWorkflowID:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        _ = (exc_type, exc_val, exc_tb)
        return False


class _FakeDBOSErrorStatus:
    def __init__(self) -> None:
        self.start_calls: list[str] = []
        self.resume_calls: list[str] = []

    async def get_workflow_status_async(self, workflow_id: str) -> _FakeStatus:
        _ = workflow_id
        return _FakeStatus("ERROR")

    async def start_workflow_async(self, *args: object) -> _FakeHandle:
        _ = args
        self.start_calls.append("start")
        return _FakeHandle()

    async def resume_workflow_async(self, workflow_id: str) -> _FakeHandle:
        self.resume_calls.append(workflow_id)
        return _FakeHandle()


class _FakeDBOSNoStatus:
    def __init__(self) -> None:
        self.start_calls: list[str] = []
        self.resume_calls: list[str] = []

    async def get_workflow_status_async(self, workflow_id: str) -> None:
        _ = workflow_id
        return None

    async def start_workflow_async(self, *args: object) -> _FakeHandle:
        _ = args
        self.start_calls.append("start")
        return _FakeHandle()

    async def resume_workflow_async(self, workflow_id: str) -> _FakeHandle:
        self.resume_calls.append(workflow_id)
        return _FakeHandle()


@pytest.mark.asyncio
async def test_dbos_stage_workflow_uses_resume_on_error_status(monkeypatch) -> None:
    fake_dbos = _FakeDBOSErrorStatus()
    monkeypatch.setattr("rulekiln.workers.dbos_workflow.DBOS", fake_dbos)
    monkeypatch.setattr("rulekiln.workers.dbos_workflow.SetWorkflowID", _FakeSetWorkflowID)

    await run_dbos_stage_workflow("job-resume", _build_payload())

    assert fake_dbos.start_calls == []
    assert fake_dbos.resume_calls == ["rulekiln-job-job-resume"]


@pytest.mark.asyncio
async def test_dbos_stage_workflow_starts_when_status_missing(monkeypatch) -> None:
    fake_dbos = _FakeDBOSNoStatus()
    monkeypatch.setattr("rulekiln.workers.dbos_workflow.DBOS", fake_dbos)
    monkeypatch.setattr("rulekiln.workers.dbos_workflow.SetWorkflowID", _FakeSetWorkflowID)

    await run_dbos_stage_workflow("job-start", _build_payload())

    assert fake_dbos.start_calls == ["start"]
    assert fake_dbos.resume_calls == []
