"""Offline end-to-end integration test using fake providers (T044).

Exercises the full pipeline orchestration in-process with SQLite + fake chat/embedding.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rulekiln.config.settings import AppSettings, ProviderProfile
from rulekiln.db.models import Base
from rulekiln.db.session import override_session_factory
from rulekiln.schemas.job import DistillationRequest, ModelRoute
from rulekiln.schemas.task_case import EvaluationSpec, RuleKilnCase, RuleKilnTask
from rulekiln.workers.distillation_worker import PipelineStage, run_distillation_pipeline

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
        MLFLOW_TRACKING_URI="file:///tmp/mlflow-e2e-test",
        provider_profiles={
            "fake": ProviderProfile(
                provider="fake",
                supports_chat=True,
                supports_embeddings=True,
            ),
        },
    )


def _build_payload(baseline: bool = False) -> DistillationRequest:
    task = RuleKilnTask(
        task_id="e2e-task",
        task_name="E2E Task",
        task_mode="classification",
        description="Test classification task",
        input_template="{{input}}",
    )
    cases = [
        RuleKilnCase(
            id=f"case-{i}",
            task_mode="classification",
            split="train" if i < 4 else "test",
            input={"text": f"input {i}"},
            expected="positive" if i % 2 == 0 else "negative",
            evaluation=EvaluationSpec(assertions=[]),
        )
        for i in range(6)
    ]
    route = ModelRoute(provider_profile="fake", model="fake-model")
    payload = DistillationRequest(
        task=task,
        cases=cases,
        teacher=route,
        student=route,
        embedding=route,
        baseline_prompt="You are a baseline assistant." if baseline else None,
    )
    return payload


@pytest.mark.asyncio
async def test_pipeline_runs_to_completion(db_factory, fake_settings, monkeypatch) -> None:
    """Full pipeline should reach COMPLETED status without raising."""
    # Patch get_settings in the worker module (direct import reference)
    monkeypatch.setattr("rulekiln.workers.distillation_worker.get_settings", lambda: fake_settings)

    # Patch resolve_provider_config in the worker to always use fake_settings
    from rulekiln.providers import resolver as _resolver

    def _patched_resolve(profile_name: str, model: str, *, role: str, settings: AppSettings):  # type: ignore[no-untyped-def]
        return _resolver.resolve_provider_config(
            profile_name, model, role=role, settings=fake_settings
        )

    monkeypatch.setattr(
        "rulekiln.workers.distillation_worker.resolve_provider_config", _patched_resolve
    )

    payload = _build_payload()
    job_id = "aaaaaaaa-1111-0000-0000-000000000001"

    from rulekiln.db.models import DistillationJob
    from rulekiln.db.repositories.jobs import create_job

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

    await run_distillation_pipeline(job_id, payload)

    from rulekiln.db.repositories.jobs import get_job

    async with db_factory() as session:
        db_job = await get_job(session, job_id)

    assert db_job is not None
    assert db_job.status == "completed"
    assert db_job.stage == PipelineStage.COMPLETED


@pytest.mark.asyncio
async def test_pipeline_with_baseline_runs(db_factory, fake_settings, monkeypatch) -> None:
    monkeypatch.setattr("rulekiln.workers.distillation_worker.get_settings", lambda: fake_settings)

    from rulekiln.providers import resolver as _resolver

    def _patched_resolve(profile_name: str, model: str, *, role: str, settings: AppSettings):  # type: ignore[no-untyped-def]
        return _resolver.resolve_provider_config(
            profile_name, model, role=role, settings=fake_settings
        )

    monkeypatch.setattr(
        "rulekiln.workers.distillation_worker.resolve_provider_config", _patched_resolve
    )

    payload = _build_payload(baseline=True)
    job_id = "aaaaaaaa-1111-0000-0000-000000000002"

    from rulekiln.db.models import DistillationJob
    from rulekiln.db.repositories.jobs import create_job

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

    await run_distillation_pipeline(job_id, payload)

    from rulekiln.db.repositories.jobs import get_job

    async with db_factory() as session:
        db_job = await get_job(session, job_id)

    assert db_job is not None
    assert db_job.status == "completed"
