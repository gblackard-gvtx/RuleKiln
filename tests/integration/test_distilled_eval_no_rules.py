"""Regression tests: distilled strategies skipped when extraction produces zero rules.

Covers:
- evaluate_dbscan / evaluate_hdbscan phases must not create eval case results when
  get_selected_synthesized_rules_for_job returns an empty list.
- EVALUATING_DISTILLED stage must NOT be marked complete in that scenario.
- Full _run_stage_sequence with fake providers (which return zero rules by default)
  must complete successfully without evaluating distilled strategies.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rulekiln.config.settings import AppSettings, ProviderProfile
from rulekiln.db.models import Base, DistillationJob, EvalCaseResultRecord, StageMarker
from rulekiln.db.repositories.jobs import create_job, get_eval_runs_for_job
from rulekiln.db.session import override_session_factory
from rulekiln.schemas.job import DistillationRequest, ModelRoute
from rulekiln.schemas.task_case import EvaluationSpec, RuleKilnCase, RuleKilnTask
from rulekiln.workers.dbos_workflow import _run_stage_sequence
from rulekiln.workers.distillation_worker import PipelineStage, run_pipeline_phase

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
def fake_settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        DATABASE_URL=_IN_MEMORY_URL,
        MLFLOW_TRACKING_URI=f"file://{tmp_path / 'mlflow-no-rules-test'}",
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
        task_id="no-rules-task",
        task_name="No Rules Task",
        task_mode="classification",
        description="Classification task used to verify empty-rules guard",
        input_template="{{input}}",
    )
    cases = [
        RuleKilnCase(
            id=f"case-{i}",
            task_mode="classification",
            split="train" if i < 3 else "validation",
            input={"text": f"input {i}"},
            expected="positive" if i % 2 == 0 else "negative",
            evaluation=EvaluationSpec(assertions=[]),
        )
        for i in range(6)
    ]
    route = ModelRoute(provider_profile="fake", model="fake-model")
    return DistillationRequest(
        task=task,
        cases=cases,
        teacher=route,
        student=route,
        embedding=route,
    )


async def _create_job(db_factory, job_id: str, payload: DistillationRequest) -> None:
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


@pytest.mark.asyncio
async def test_evaluate_dbscan_skips_when_no_rules(
    db_factory,
    fake_settings,
    monkeypatch,
) -> None:
    """evaluate_dbscan phase must not produce eval results when selected_rules is empty."""
    monkeypatch.setattr("rulekiln.workers.distillation_worker.get_settings", lambda: fake_settings)

    payload = _build_payload()
    job_id = str(uuid.uuid4())
    await _create_job(db_factory, job_id, payload)

    async with db_factory() as session:
        # compile phase: inserts cases, runs extraction (returns 0 rules), synthesis, pruning
        await run_pipeline_phase(session, job_id, payload, phase="compile_prompts")

    async with db_factory() as session:
        await run_pipeline_phase(session, job_id, payload, phase="evaluate_baseline")

    async with db_factory() as session:
        await run_pipeline_phase(session, job_id, payload, phase="evaluate_dbscan")

    async with db_factory() as session:
        marker_rows = await session.execute(
            select(StageMarker.stage, StageMarker.strategy).where(
                StageMarker.job_id == job_id,
                StageMarker.stage == PipelineStage.EVALUATING_DISTILLED,
            )
        )
        distilled_markers = {(row[0], row[1]) for row in marker_rows.all()}

        result_rows = await session.execute(
            select(EvalCaseResultRecord.strategy).where(
                EvalCaseResultRecord.job_id == job_id,
                EvalCaseResultRecord.strategy == "dbscan",
            )
        )
        dbscan_result_count = len(result_rows.all())

    assert not distilled_markers, (
        "EVALUATING_DISTILLED must not be marked when selected_rules is empty; "
        f"got {distilled_markers}"
    )
    assert dbscan_result_count == 0, (
        f"No eval case results expected for dbscan with 0 rules; got {dbscan_result_count}"
    )


@pytest.mark.asyncio
async def test_evaluate_hdbscan_skips_when_no_rules(
    db_factory,
    fake_settings,
    monkeypatch,
) -> None:
    """evaluate_hdbscan phase must not produce eval results when selected_rules is empty."""
    monkeypatch.setattr("rulekiln.workers.distillation_worker.get_settings", lambda: fake_settings)

    payload = _build_payload()
    job_id = str(uuid.uuid4())
    await _create_job(db_factory, job_id, payload)

    async with db_factory() as session:
        await run_pipeline_phase(session, job_id, payload, phase="compile_prompts")

    async with db_factory() as session:
        await run_pipeline_phase(session, job_id, payload, phase="evaluate_baseline")

    async with db_factory() as session:
        await run_pipeline_phase(session, job_id, payload, phase="evaluate_hdbscan")

    async with db_factory() as session:
        result_rows = await session.execute(
            select(EvalCaseResultRecord.strategy).where(
                EvalCaseResultRecord.job_id == job_id,
                EvalCaseResultRecord.strategy == "hdbscan",
            )
        )
        hdbscan_result_count = len(result_rows.all())

        marker_rows = await session.execute(
            select(StageMarker.strategy).where(
                StageMarker.job_id == job_id,
                StageMarker.stage == PipelineStage.EVALUATING_DISTILLED,
                StageMarker.strategy == "hdbscan",
            )
        )
        hdbscan_marker_count = len(marker_rows.all())

    assert hdbscan_result_count == 0
    assert hdbscan_marker_count == 0


@pytest.mark.asyncio
async def test_full_pipeline_no_rules_completes_without_distilled_eval(
    db_factory,
    fake_settings,
    monkeypatch,
) -> None:
    """Full stage sequence with fake providers (zero extraction rules) must complete
    successfully without marking EVALUATING_DISTILLED or creating distilled eval results."""
    monkeypatch.setattr("rulekiln.workers.distillation_worker.get_settings", lambda: fake_settings)
    monkeypatch.setattr("rulekiln.workers.dbos_workflow.DBOS", None)
    monkeypatch.setattr("rulekiln.workers.dbos_workflow.SetWorkflowID", None)

    payload = _build_payload()
    job_id = str(uuid.uuid4())
    await _create_job(db_factory, job_id, payload)

    payload_json = payload.model_dump(mode="json")
    await _run_stage_sequence(job_id, payload_json)

    async with db_factory() as session:
        marker_rows = await session.execute(
            select(StageMarker.stage).where(StageMarker.job_id == job_id)
        )
        marker_stages = {row[0] for row in marker_rows.all()}

        result_rows = await session.execute(
            select(EvalCaseResultRecord.strategy).where(
                EvalCaseResultRecord.job_id == job_id,
                EvalCaseResultRecord.strategy.in_(["dbscan", "hdbscan"]),
            )
        )
        distilled_result_count = len(result_rows.all())

        eval_runs = await get_eval_runs_for_job(session, job_id)

    assert PipelineStage.VALIDATING_PROJECT in marker_stages
    assert PipelineStage.EVALUATING_BASELINE in marker_stages
    assert PipelineStage.EVALUATING_DISTILLED not in marker_stages, (
        "EVALUATING_DISTILLED must not be marked when extraction produces zero rules"
    )
    assert distilled_result_count == 0, (
        f"Expected 0 distilled eval case results, got {distilled_result_count}"
    )

    baseline_runs = [r for r in eval_runs if r.strategy == "baseline"]
    assert len(baseline_runs) == 1, "Baseline evaluation must still run even when rules are empty"
