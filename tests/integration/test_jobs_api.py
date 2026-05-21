"""Integration tests for the distillation jobs HTTP API.

Runs fully in-process against a SQLite in-memory database (no real provider calls).
The pipeline background task is mocked so these tests only exercise the API layer.
"""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rulekiln.api.app import create_app
from rulekiln.config.settings import AppSettings, ProviderProfile, get_settings
from rulekiln.db.models import Base
from rulekiln.db.session import get_db_session, override_session_factory


_IN_MEMORY_URL = "sqlite+aiosqlite://"


@pytest.fixture()
async def db_session_factory():
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
            "teacher": ProviderProfile(
                provider="fake", supports_chat=True, supports_embeddings=False
            ),
            "student": ProviderProfile(
                provider="fake", supports_chat=True, supports_embeddings=False
            ),
            "embedding": ProviderProfile(
                provider="fake", supports_chat=False, supports_embeddings=True
            ),
        },
    )


@pytest.fixture()
async def client(db_session_factory, test_settings, monkeypatch):
    override_session_factory(db_session_factory)
    app = create_app()

    async def _override_session():
        async with db_session_factory() as session:
            yield session

    # Stub out the background pipeline task — these tests only verify the HTTP layer.
    async def _noop_pipeline(job_id: str, payload) -> None:  # type: ignore[no-untyped-def]
        pass

    monkeypatch.setattr(
        "rulekiln.api.routes.distillation_jobs.run_distillation_pipeline",
        _noop_pipeline,
    )

    app.dependency_overrides[get_db_session] = _override_session
    app.dependency_overrides[get_settings] = lambda: test_settings

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _valid_payload() -> dict:
    return {
        "task": {
            "task_id": "t1",
            "task_name": "T1",
            "task_mode": "classification",
            "description": "desc",
            "input_template": "{{input}}",
        },
        "cases": [{"id": "c1", "task_mode": "classification", "input": {"x": 1}}],
        "teacher": {"provider_profile": "teacher", "model": "fake-model"},
        "student": {"provider_profile": "student", "model": "fake-model"},
        "embedding": {"provider_profile": "embedding", "model": "fake-embedding"},
    }


async def test_create_job_returns_202(client) -> None:
    resp = await client.post("/v1/jobs/", json=_valid_payload())
    assert resp.status_code == 202
    body = resp.json()
    assert "job_id" in body
    assert body["status"] == "created"


async def test_get_job_returns_status(client) -> None:
    create_resp = await client.post("/v1/jobs/", json=_valid_payload())
    job_id = create_resp.json()["job_id"]
    get_resp = await client.get(f"/v1/jobs/{job_id}")
    assert get_resp.status_code == 200
    body = get_resp.json()
    assert body["job_id"] == job_id


async def test_get_unknown_job_returns_404(client) -> None:
    resp = await client.get("/v1/jobs/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


async def test_create_job_with_legacy_field_returns_422(client) -> None:
    payload = {**_valid_payload(), "task_name": "legacy"}
    resp = await client.post("/v1/jobs/", json=payload)
    assert resp.status_code == 422
