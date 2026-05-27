"""Offline end-to-end integration test using fake providers (T044).

Exercises the full pipeline orchestration in-process with SQLite + fake chat/embedding.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rulekiln.config.settings import AppSettings, ProviderProfile
from rulekiln.db.models import Base
from rulekiln.db.session import override_session_factory
from rulekiln.providers.contracts import ChatModelClient, ProviderConfig
from rulekiln.schemas.job import DistillationRequest, ModelRoute
from rulekiln.schemas.pipeline import (
    ExtractionOutput,
    MicroRuleSchema,
    OutcomeCondition,
    RuleClusterSchema,
    RuleConflictReview,
    SynthesisOutput,
    SynthesizedRuleSchema,
)
from rulekiln.schemas.task_case import EvaluationSpec, RuleKilnCase, RuleKilnTask
from rulekiln.workers.distillation_worker import (
    PipelineStage,
    run_distillation_pipeline,
    run_pipeline_phase,
)

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

    def _patched_resolve(
        profile_name: str,
        model: str,
        *,
        role: str,
        settings: AppSettings,
    ) -> ProviderConfig:
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

    def _patched_resolve(
        profile_name: str,
        model: str,
        *,
        role: str,
        settings: AppSettings,
    ) -> ProviderConfig:
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


@pytest.mark.asyncio
async def test_compile_phase_resumes_extraction_by_case(db_factory, fake_settings, monkeypatch) -> None:
    monkeypatch.setattr("rulekiln.workers.distillation_worker.get_settings", lambda: fake_settings)

    from rulekiln.providers import resolver as _resolver

    def _patched_resolve(
        profile_name: str,
        model: str,
        *,
        role: str,
        settings: AppSettings,
    ) -> ProviderConfig:
        return _resolver.resolve_provider_config(
            profile_name, model, role=role, settings=fake_settings
        )

    monkeypatch.setattr(
        "rulekiln.workers.distillation_worker.resolve_provider_config", _patched_resolve
    )

    payload = _build_payload()
    job_id = "aaaaaaaa-1111-0000-0000-000000000003"

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

    call_case_ids: list[str] = []
    first_case_failure_raised = False

    async def _flaky_extract(
        task: RuleKilnTask,
        case: RuleKilnCase,
        chat_client: ChatModelClient,
        config: ProviderConfig,
    ) -> ExtractionOutput:
        nonlocal first_case_failure_raised
        del task
        del chat_client
        del config

        call_case_ids.append(case.id)
        if case.id == "case-1" and not first_case_failure_raised:
            first_case_failure_raised = True
            raise TimeoutError("transient extraction timeout")

        return ExtractionOutput(
            rules=[
                MicroRuleSchema(
                    topic=f"topic-{case.id}",
                    condition="if input is present",
                    expected_outcome="emit expected label",
                )
            ]
        )

    monkeypatch.setattr("rulekiln.workers.distillation_worker.extract_rules_for_case", _flaky_extract)

    async with db_factory() as session:
        with pytest.raises(TimeoutError):
            await run_pipeline_phase(session, job_id, payload, phase="compile_prompts")

    async with db_factory() as session:
        await run_pipeline_phase(session, job_id, payload, phase="compile_prompts")

    assert call_case_ids.count("case-0") == 1
    assert call_case_ids.count("case-1") == 2
    assert call_case_ids.count("case-2") == 1
    assert call_case_ids.count("case-3") == 1


@pytest.mark.asyncio
async def test_compile_phase_resumes_synthesis_by_cluster(
    db_factory,
    fake_settings,
    monkeypatch,
) -> None:
    monkeypatch.setattr("rulekiln.workers.distillation_worker.get_settings", lambda: fake_settings)

    from rulekiln.providers import resolver as _resolver

    def _patched_resolve(
        profile_name: str,
        model: str,
        *,
        role: str,
        settings: AppSettings,
    ) -> ProviderConfig:
        return _resolver.resolve_provider_config(
            profile_name, model, role=role, settings=fake_settings
        )

    monkeypatch.setattr(
        "rulekiln.workers.distillation_worker.resolve_provider_config", _patched_resolve
    )

    payload = _build_payload()
    job_id = "aaaaaaaa-1111-0000-0000-000000000004"

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

    async def _simple_extract(
        task: RuleKilnTask,
        case: RuleKilnCase,
        chat_client: ChatModelClient,
        config: ProviderConfig,
    ) -> ExtractionOutput:
        del task
        del chat_client
        del config
        return ExtractionOutput(
            rules=[
                MicroRuleSchema(
                    topic=f"topic-{case.id}",
                    condition="if input is present",
                    expected_outcome="emit expected label",
                )
            ]
        )

    def _split_dbscan_clusters(
        rule_ids: list[str],
        embeddings: list[list[float]],
        eps: float = 0.3,
        min_samples: int = 2,
    ) -> list[RuleClusterSchema]:
        del embeddings
        del eps
        del min_samples
        halfway = max(1, len(rule_ids) // 2)
        first = sorted(rule_ids)[:halfway]
        second = sorted(rule_ids)[halfway:]
        clusters: list[RuleClusterSchema] = [
            RuleClusterSchema(strategy="dbscan", algorithm="dbscan", topic="first", rule_ids=first)
        ]
        if second:
            clusters.append(
                RuleClusterSchema(
                    strategy="dbscan",
                    algorithm="dbscan",
                    topic="second",
                    rule_ids=second,
                )
            )
        return clusters

    def _empty_hdbscan_clusters(
        rule_ids: list[str],
        embeddings: list[list[float]],
        min_cluster_size: int = 2,
    ) -> list[RuleClusterSchema]:
        del rule_ids
        del embeddings
        del min_cluster_size
        return []

    synth_call_markers: list[str] = []
    second_cluster_failed_once = False

    async def _flaky_synthesize(
        task: RuleKilnTask,
        cluster_topic: str | None,
        micro_rules: list[MicroRuleSchema],
        source_case_ids: list[str],
        source_micro_rule_ids: list[str],
        chat_client: ChatModelClient,
        config: ProviderConfig,
    ) -> SynthesisOutput:
        nonlocal second_cluster_failed_once
        del task
        del cluster_topic
        del micro_rules
        del chat_client
        del config

        marker = ",".join(sorted(source_micro_rule_ids))
        synth_call_markers.append(marker)
        sorted_markers = sorted({m for m in synth_call_markers if m})
        target_marker = sorted_markers[-1] if sorted_markers else marker
        if marker == target_marker and not second_cluster_failed_once:
            second_cluster_failed_once = True
            raise TimeoutError("transient synthesis timeout")

        return SynthesisOutput(
            rules=[
                SynthesizedRuleSchema(
                    topic=f"synth-{len(source_micro_rule_ids)}",
                    applies_when=["input present"],
                    outcome_conditions={
                        "label": OutcomeCondition(outcome="positive", when=["input present"])
                    },
                    source_case_ids=list(source_case_ids),
                    source_micro_rule_ids=list(source_micro_rule_ids),
                )
            ]
        )

    async def _no_conflict_review(
        task: RuleKilnTask,
        rule: SynthesizedRuleSchema,
        micro_rules: list[MicroRuleSchema],
        chat_client: ChatModelClient,
        config: ProviderConfig,
    ) -> RuleConflictReview:
        del task
        del micro_rules
        del chat_client
        del config
        return RuleConflictReview(
            synthesized_rule_id=rule.id,
            has_conflicts=False,
            resolution="keep",
        )

    monkeypatch.setattr("rulekiln.workers.distillation_worker.extract_rules_for_case", _simple_extract)
    monkeypatch.setattr("rulekiln.workers.distillation_worker.cluster_dbscan", _split_dbscan_clusters)
    monkeypatch.setattr("rulekiln.workers.distillation_worker.cluster_hdbscan", _empty_hdbscan_clusters)
    monkeypatch.setattr("rulekiln.workers.distillation_worker.synthesize_cluster", _flaky_synthesize)
    monkeypatch.setattr(
        "rulekiln.workers.distillation_worker.review_rule_for_conflicts",
        _no_conflict_review,
    )

    async with db_factory() as session:
        with pytest.raises(TimeoutError):
            await run_pipeline_phase(session, job_id, payload, phase="compile_prompts")

    async with db_factory() as session:
        await run_pipeline_phase(session, job_id, payload, phase="compile_prompts")

    marker_counts: dict[str, int] = {}
    for marker in synth_call_markers:
        marker_counts[marker] = marker_counts.get(marker, 0) + 1

    assert len(marker_counts) == 2
    assert sorted(marker_counts.values()) == [1, 2]


@pytest.mark.asyncio
async def test_compile_phase_resumes_conflict_review_by_rule(
    db_factory,
    fake_settings,
    monkeypatch,
) -> None:
    monkeypatch.setattr("rulekiln.workers.distillation_worker.get_settings", lambda: fake_settings)

    from rulekiln.providers import resolver as _resolver

    def _patched_resolve(
        profile_name: str,
        model: str,
        *,
        role: str,
        settings: AppSettings,
    ) -> ProviderConfig:
        return _resolver.resolve_provider_config(
            profile_name, model, role=role, settings=fake_settings
        )

    monkeypatch.setattr(
        "rulekiln.workers.distillation_worker.resolve_provider_config", _patched_resolve
    )

    payload = _build_payload()
    job_id = "aaaaaaaa-1111-0000-0000-000000000005"

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

    async def _simple_extract(
        task: RuleKilnTask,
        case: RuleKilnCase,
        chat_client: ChatModelClient,
        config: ProviderConfig,
    ) -> ExtractionOutput:
        del task
        del chat_client
        del config
        return ExtractionOutput(
            rules=[
                MicroRuleSchema(
                    topic=f"topic-{case.id}",
                    condition="if input is present",
                    expected_outcome="emit expected label",
                )
            ]
        )

    def _single_dbscan_cluster(
        rule_ids: list[str],
        embeddings: list[list[float]],
        eps: float = 0.3,
        min_samples: int = 2,
    ) -> list[RuleClusterSchema]:
        del embeddings
        del eps
        del min_samples
        return [
            RuleClusterSchema(
                strategy="dbscan",
                algorithm="dbscan",
                topic="all-rules",
                rule_ids=sorted(rule_ids),
            )
        ]

    def _empty_hdbscan_clusters(
        rule_ids: list[str],
        embeddings: list[list[float]],
        min_cluster_size: int = 2,
    ) -> list[RuleClusterSchema]:
        del rule_ids
        del embeddings
        del min_cluster_size
        return []

    async def _synthesize_two_rules(
        task: RuleKilnTask,
        cluster_topic: str | None,
        micro_rules: list[MicroRuleSchema],
        source_case_ids: list[str],
        source_micro_rule_ids: list[str],
        chat_client: ChatModelClient,
        config: ProviderConfig,
    ) -> SynthesisOutput:
        del task
        del cluster_topic
        del micro_rules
        del chat_client
        del config

        return SynthesisOutput(
            rules=[
                SynthesizedRuleSchema(
                    topic="synth-keep",
                    applies_when=["input present"],
                    outcome_conditions={
                        "label": OutcomeCondition(outcome="positive", when=["input present"])
                    },
                    source_case_ids=list(source_case_ids),
                    source_micro_rule_ids=list(source_micro_rule_ids),
                ),
                SynthesizedRuleSchema(
                    topic="synth-flaky",
                    applies_when=["input present"],
                    outcome_conditions={
                        "label": OutcomeCondition(outcome="negative", when=["input present"])
                    },
                    source_case_ids=list(source_case_ids),
                    source_micro_rule_ids=list(source_micro_rule_ids),
                ),
            ]
        )

    review_topics: list[str] = []
    flaky_topic_failed_once = False

    async def _flaky_conflict_review(
        task: RuleKilnTask,
        rule: SynthesizedRuleSchema,
        micro_rules: list[MicroRuleSchema],
        chat_client: ChatModelClient,
        config: ProviderConfig,
    ) -> RuleConflictReview:
        nonlocal flaky_topic_failed_once
        del task
        del micro_rules
        del chat_client
        del config

        review_topics.append(rule.topic)
        if rule.topic == "synth-flaky" and not flaky_topic_failed_once:
            flaky_topic_failed_once = True
            raise TimeoutError("transient conflict-review timeout")

        return RuleConflictReview(
            synthesized_rule_id=rule.id,
            has_conflicts=False,
            resolution="keep",
        )

    monkeypatch.setattr("rulekiln.workers.distillation_worker.extract_rules_for_case", _simple_extract)
    monkeypatch.setattr("rulekiln.workers.distillation_worker.cluster_dbscan", _single_dbscan_cluster)
    monkeypatch.setattr("rulekiln.workers.distillation_worker.cluster_hdbscan", _empty_hdbscan_clusters)
    monkeypatch.setattr(
        "rulekiln.workers.distillation_worker.synthesize_cluster",
        _synthesize_two_rules,
    )
    monkeypatch.setattr(
        "rulekiln.workers.distillation_worker.review_rule_for_conflicts",
        _flaky_conflict_review,
    )

    async with db_factory() as session:
        with pytest.raises(TimeoutError):
            await run_pipeline_phase(session, job_id, payload, phase="compile_prompts")

    async with db_factory() as session:
        await run_pipeline_phase(session, job_id, payload, phase="compile_prompts")

    assert review_topics.count("synth-keep") == 1
    assert review_topics.count("synth-flaky") == 2
