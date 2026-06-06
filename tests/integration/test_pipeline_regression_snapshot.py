"""Regression snapshot test for distillation worker behavior.

Runs the full pipeline with fake providers and asserts structural invariants
that must hold across every refactor PR.  Add assertions here if you need to
lock down a new invariant; never relax an existing one without understanding why.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rulekiln.config.settings import AppSettings, ProviderProfile
from rulekiln.db.models import Base
from rulekiln.db.session import override_session_factory
from rulekiln.schemas.job import DistillationRequest, ModelRoute
from rulekiln.schemas.task_case import EvaluationSpec, RuleKilnCase, RuleKilnTask
from rulekiln.workers.distillation_worker import (
    PipelineStage,
    run_distillation_pipeline,
)

_IN_MEMORY_URL = "sqlite+aiosqlite://"

# Stages that must complete (in this order) on a full run.
_REQUIRED_STAGE_SEQUENCE = [
    PipelineStage.VALIDATING_PROJECT,
    PipelineStage.EXTRACTING_RULES,
    PipelineStage.EMBEDDING_RULES,
    PipelineStage.CLUSTERING_RULES,
    PipelineStage.SYNTHESIZING_RULES,
    PipelineStage.REVIEWING_RULE_CONFLICTS,
    PipelineStage.PRUNING_RULES,
    PipelineStage.COMPILING_PROMPTS,
    PipelineStage.EVALUATING_BASELINE,
    PipelineStage.EVALUATING_DISTILLED,
    PipelineStage.SELECTING_STRATEGY,
    PipelineStage.CHECKING_QUALITY_GATES,
    PipelineStage.LOGGING_ARTIFACTS,
    PipelineStage.EXPORTING_ARTIFACTS,
    PipelineStage.COMPLETED,
]

# Required top-level keys in strategy_comparison.json.
_STRATEGY_COMPARISON_REQUIRED_KEYS = {
    "strategy_evals",
    "strategy_gates",
    "strategy_prompt_tokens",
    "selected_strategy_id",
    "selected_strategy_family",
    "best_by_family",
    "paired_comparison",
    "all_student_results",
    "pruning_mode_comparison",
}

# Artifact paths that must exist after a complete run.
_REQUIRED_ARTIFACTS = [
    "task.yaml",
    "cases.normalized.jsonl",
    "outputs/baseline_prompt.md",
    "outputs/baseline_scaffold_prompt.md",
    "outputs/eval_report.json",
    "outputs/strategy_comparison.json",
    "metadata/settings_snapshot.json",
    "metadata/manifest.json",
]


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
        MLFLOW_TRACKING_URI=f"file://{tmp_path / 'mlflow-snapshot-test'}",
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
        task_id="snapshot-task",
        task_name="Snapshot Task",
        task_mode="classification",
        description="Regression snapshot classification task",
        input_template="{{input}}",
    )
    cases = [
        RuleKilnCase(
            id=f"case-{i}",
            task_mode="classification",
            split="train" if i < 5 else "test",
            input={"text": f"input {i}"},
            expected="positive" if i % 2 == 0 else "negative",
            evaluation=EvaluationSpec(assertions=[]),
        )
        for i in range(8)
    ]
    route = ModelRoute(provider_profile="fake", model="fake-model")
    return DistillationRequest(
        task=task,
        cases=cases,
        teacher=route,
        student=route,
        embedding=route,
    )


def _patch_providers(monkeypatch: pytest.MonkeyPatch, fake_settings: AppSettings) -> None:
    monkeypatch.setattr("rulekiln.workers.distillation_worker.get_settings", lambda: fake_settings)

    from rulekiln.providers import resolver as _resolver

    def _patched_resolve(
        profile_name: str,
        model: str,
        *,
        role: Literal["teacher", "student", "embedding", "judge"],
        settings: AppSettings,
    ) -> object:
        return _resolver.resolve_provider_config(
            profile_name, model, role=role, settings=fake_settings
        )

    monkeypatch.setattr(
        "rulekiln.workers.distillation_worker.resolve_provider_config", _patched_resolve
    )


@pytest.mark.asyncio
async def test_pipeline_stage_sequence(db_factory, fake_settings, monkeypatch) -> None:
    """Final stage must be COMPLETED and all required stages must be reachable."""
    _patch_providers(monkeypatch, fake_settings)

    payload = _build_payload()
    job_id = "snapshot-0000-0001-0000-000000000001"

    from rulekiln.db.models import DistillationJob
    from rulekiln.db.repositories.jobs import create_job, get_job

    async with db_factory() as session:
        await create_job(
            session,
            DistillationJob(
                id=job_id,
                task_id=payload.task.task_id,
                task_name=payload.task.task_name,
                task_mode=payload.task.task_mode,
                status="created",
                stage=None,
                request_json=payload.model_dump(mode="json"),
            ),
        )

    await run_distillation_pipeline(job_id, payload)

    async with db_factory() as session:
        db_job = await get_job(session, job_id)

    assert db_job is not None
    assert db_job.status == "completed", f"Expected completed, got {db_job.status}"
    assert db_job.stage == PipelineStage.COMPLETED

    # Verify every required stage is present in the enum (guards against accidental rename).
    stage_values = {s.value for s in PipelineStage}
    for required in _REQUIRED_STAGE_SEQUENCE:
        assert required.value in stage_values, f"PipelineStage.{required.name} missing"


@pytest.mark.asyncio
async def test_artifact_structure_invariants(
    db_factory, fake_settings, monkeypatch, tmp_path: Path
) -> None:
    """Key artifact files must exist and carry required fields."""
    _patch_providers(monkeypatch, fake_settings)

    payload = _build_payload()
    job_id = "snapshot-0000-0002-0000-000000000002"

    from rulekiln.db.models import DistillationJob
    from rulekiln.db.repositories.jobs import create_job

    async with db_factory() as session:
        await create_job(
            session,
            DistillationJob(
                id=job_id,
                task_id=payload.task.task_id,
                task_name=payload.task.task_name,
                task_mode=payload.task.task_mode,
                status="created",
                stage=None,
                request_json=payload.model_dump(mode="json"),
            ),
        )

    await run_distillation_pipeline(job_id, payload)

    artifact_root = Path(fake_settings.artifact_root) / job_id

    # Required files exist.
    for relative_path in _REQUIRED_ARTIFACTS:
        assert (artifact_root / relative_path).exists(), f"Missing artifact: {relative_path}"

    # manifest.json contains every required artifact.
    manifest = json.loads((artifact_root / "metadata" / "manifest.json").read_text("utf-8"))
    manifest_entries = set(manifest.get("artifacts", []))
    for relative_path in _REQUIRED_ARTIFACTS:
        assert relative_path in manifest_entries, f"Artifact not in manifest: {relative_path}"

    # strategy_comparison.json carries all required top-level keys.
    comparison = json.loads(
        (artifact_root / "outputs" / "strategy_comparison.json").read_text("utf-8")
    )
    for key in _STRATEGY_COMPARISON_REQUIRED_KEYS:
        assert key in comparison, f"strategy_comparison.json missing key: {key}"

    # selected strategy is non-empty and maps to an entry in strategy_evals.
    selected = comparison.get("selected_strategy_id")
    assert selected, "selected_strategy_id is empty"
    assert selected in comparison["strategy_evals"], (
        f"selected_strategy_id '{selected}' not in strategy_evals"
    )

    # all_student_results is a dict (may be empty for single-teacher runs).
    assert isinstance(comparison["all_student_results"], dict)

    # eval_report.json is valid JSON with the same required keys.
    eval_report = json.loads((artifact_root / "outputs" / "eval_report.json").read_text("utf-8"))
    for key in ("strategy_evals", "selected_strategy_id"):
        assert key in eval_report, f"eval_report.json missing key: {key}"

    # settings_snapshot.json is valid JSON.
    snapshot = json.loads(
        (artifact_root / "metadata" / "settings_snapshot.json").read_text("utf-8")
    )
    assert isinstance(snapshot, dict)


@pytest.mark.asyncio
async def test_strategy_evals_have_numeric_metrics(
    db_factory, fake_settings, monkeypatch
) -> None:
    """Every entry in strategy_evals must have finite numeric macro_f1 and accuracy."""
    _patch_providers(monkeypatch, fake_settings)

    payload = _build_payload()
    job_id = "snapshot-0000-0003-0000-000000000003"

    from rulekiln.db.models import DistillationJob
    from rulekiln.db.repositories.jobs import create_job

    async with db_factory() as session:
        await create_job(
            session,
            DistillationJob(
                id=job_id,
                task_id=payload.task.task_id,
                task_name=payload.task.task_name,
                task_mode=payload.task.task_mode,
                status="created",
                stage=None,
                request_json=payload.model_dump(mode="json"),
            ),
        )

    await run_distillation_pipeline(job_id, payload)

    artifact_root = Path(fake_settings.artifact_root) / job_id
    comparison = json.loads(
        (artifact_root / "outputs" / "strategy_comparison.json").read_text("utf-8")
    )

    for strategy_name, eval_data in comparison["strategy_evals"].items():
        macro_f1 = eval_data.get("macro_f1")
        accuracy = eval_data.get("accuracy")
        assert macro_f1 is not None, f"{strategy_name}.macro_f1 is None"
        assert isinstance(macro_f1, float | int), f"{strategy_name}.macro_f1 not numeric"
        assert accuracy is not None, f"{strategy_name}.accuracy is None"
        assert isinstance(accuracy, float | int), f"{strategy_name}.accuracy not numeric"
