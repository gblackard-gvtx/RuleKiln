"""Contract tests for output route shapes: GET /prompt, /rules, /eval-report (T043)."""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rulekiln.api.app import create_app
from rulekiln.config.settings import AppSettings, ProviderProfile, get_settings
from rulekiln.db.models import Base, DistillationJob, EvalRun, PromptVersion, SynthesizedRule
from rulekiln.db.session import get_db_session, override_session_factory

_IN_MEMORY_URL = "sqlite+aiosqlite://"


@pytest.fixture()
async def db_factory():
    engine = create_async_engine(_IN_MEMORY_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    yield factory
    await engine.dispose()


@pytest.fixture()
def test_settings() -> AppSettings:
    return AppSettings(
        DATABASE_URL=_IN_MEMORY_URL,
        MLFLOW_TRACKING_URI="file:///tmp/mlflow-test",
        provider_profiles={
            "teacher": ProviderProfile(provider="fake", supports_chat=True),
            "student": ProviderProfile(provider="fake", supports_chat=True),
            "embedding": ProviderProfile(provider="fake", supports_embeddings=True),
        },
    )


@pytest.fixture()
async def client(db_factory, test_settings):
    override_session_factory(db_factory)
    app = create_app()

    async def _session():
        async with db_factory() as s:
            yield s

    app.dependency_overrides[get_db_session] = _session
    app.dependency_overrides[get_settings] = lambda: test_settings

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


import uuid

# Use UUIDs with hex letters (a-f) to prevent SQLite NUMERIC affinity coercing
# all-decimal 32-char hex strings to integers, breaking UUID result processors.
_JOB_1 = str(uuid.UUID("aaaaaaaa-0000-0000-0000-000000000001"))
_JOB_2 = str(uuid.UUID("aaaaaaaa-0000-0000-0000-000000000002"))
_JOB_3 = str(uuid.UUID("aaaaaaaa-0000-0000-0000-000000000003"))


async def _seed_completed_job(factory, job_id: str) -> None:
    """Insert a minimal completed job with one selected prompt and one eval run."""

    async with factory() as session:
        job = DistillationJob(
            id=job_id,
            task_id="t1",
            task_name="T",
            task_mode="classification",
            status="completed",
            stage="completed",
            request_json={},
        )
        session.add(job)

        pv = PromptVersion(
            id=str(uuid.uuid4()),
            job_id=job_id,
            task_id="t1",
            task_name="T",
            strategy="hdbscan",
            version="v1",
            system_prompt="You are a helpful assistant.",
            prompt_hash="abc123",
            is_selected=True,
        )
        session.add(pv)

        sr = SynthesizedRule(
            id=str(uuid.uuid4()),
            job_id=job_id,
            strategy="hdbscan",
            topic="greeting",
            applies_when=["when user greets"],
            outcome_conditions={},
            tie_breakers=[],
            priority=1,
            source_case_ids=["c1"],
            source_micro_rule_ids=["r1"],
        )
        session.add(sr)

        er = EvalRun(
            id=str(uuid.uuid4()),
            job_id=job_id,
            strategy="hdbscan",
            model="fake",
            split="train",
            accuracy=0.9,
            macro_f1=0.88,
            weighted_case_score=0.9,
            malformed_output_rate=0.0,
            per_outcome_precision={},
            per_outcome_recall={},
            confusion_matrix={},
        )
        session.add(er)
        await session.commit()


async def test_get_prompt_returns_correct_shape(client, db_factory) -> None:
    await _seed_completed_job(db_factory, _JOB_1)
    resp = await client.get(f"/v1/jobs/{_JOB_1}/prompt")
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == _JOB_1
    assert body["strategy"] == "hdbscan"
    assert "system_prompt" in body
    assert "prompt_hash" in body


async def test_get_prompt_missing_job_returns_404(client) -> None:
    resp = await client.get(f"/v1/jobs/{str(uuid.uuid4())}/prompt")
    assert resp.status_code == 404


async def test_get_rules_returns_list(client, db_factory) -> None:
    await _seed_completed_job(db_factory, _JOB_2)
    resp = await client.get(f"/v1/jobs/{_JOB_2}/rules")
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == _JOB_2
    assert isinstance(body["rules"], list)
    assert len(body["rules"]) >= 1


async def test_get_eval_report_returns_runs(client, db_factory) -> None:
    await _seed_completed_job(db_factory, _JOB_3)
    resp = await client.get(f"/v1/jobs/{_JOB_3}/eval-report")
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == _JOB_3
    assert len(body["eval_runs"]) >= 1
    run = body["eval_runs"][0]
    assert "accuracy" in run
    assert "macro_f1" in run
