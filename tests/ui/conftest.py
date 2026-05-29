"""Shared fixtures for UI integration tests."""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rulekiln.api.app import create_app
from rulekiln.config.settings import AppSettings, ProviderProfile, get_settings
from rulekiln.db.models import Base
from rulekiln.db.session import get_db_session, override_session_factory

_IN_MEMORY_URL = "sqlite+aiosqlite://"

VALID_TASK_YAML = b"""schema_version: rulekiln.task.v1
task_id: t1
task_name: Test Task
task_mode: classification
description: A test task
input_template: "{{input}}"
"""

VALID_CASE_LINE = (
    b'{"schema_version":"rulekiln.case.v1","id":"c1","split":"train",'
    b'"task_mode":"classification","input":{"text":"hello"}}\n'
)


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
        EXECUTION_BACKEND="background_tasks",
        provider_profiles={
            "fake_chat": ProviderProfile(
                provider="fake", supports_chat=True, supports_embeddings=False
            ),
            "fake_embed": ProviderProfile(
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

    async def _noop_pipeline(job_id: str, payload) -> None:  # type: ignore[no-untyped-def]
        pass

    monkeypatch.setattr(
        "rulekiln.ui.routes.run_distillation_pipeline",
        _noop_pipeline,
    )

    app.dependency_overrides[get_db_session] = _override_session
    app.dependency_overrides[get_settings] = lambda: test_settings

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
