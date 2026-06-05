"""Integration regression tests for worker split routing policy."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Literal

import pytest
from pytest import MonkeyPatch
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rulekiln.config.settings import AppSettings, ProviderProfile
from rulekiln.db.models import Base, DistillationJob
from rulekiln.db.repositories.jobs import create_job
from rulekiln.db.session import override_session_factory
from rulekiln.providers.contracts import ChatModelClient, ProviderConfig
from rulekiln.schemas.job import DistillationRequest, ModelRoute
from rulekiln.schemas.pipeline import (
    CaseEvalResult,
    EvalResult,
    ExtractionOutput,
    MicroRuleSchema,
    OutcomeCondition,
    RuleClusterSchema,
    RuleConflictReview,
    SynthesisOutput,
    SynthesizedRuleSchema,
)
from rulekiln.schemas.task_case import EvaluationSpec, RuleKilnCase, RuleKilnTask
from rulekiln.workers.distillation_worker import run_distillation_pipeline

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
        MLFLOW_TRACKING_URI=f"file://{tmp_path / 'mlflow-split-policy-test'}",
        provider_profiles={
            "fake": ProviderProfile(
                provider="fake",
                supports_chat=True,
                supports_embeddings=True,
            ),
        },
    )


def _build_payload(
    splits: list[Literal["train", "validation", "test", "golden"]],
) -> DistillationRequest:
    task = RuleKilnTask(
        task_id="split-policy-task",
        task_name="Split Policy Task",
        task_mode="classification",
        description="Split policy integration test",
        input_template="{{input}}",
    )
    cases = [
        RuleKilnCase(
            id=f"case-{index}",
            task_mode="classification",
            split=split,
            input={"text": f"input {index}"},
            expected="positive",
            evaluation=EvaluationSpec(assertions=[]),
        )
        for index, split in enumerate(splits)
    ]
    route = ModelRoute(provider_profile="fake", model="fake-model")
    return DistillationRequest(
        task=task,
        cases=cases,
        teacher=route,
        student=route,
        embedding=route,
    )


def _patch_worker_dependencies(monkeypatch: MonkeyPatch, fake_settings: AppSettings) -> None:
    monkeypatch.setattr("rulekiln.workers.distillation_worker.get_settings", lambda: fake_settings)

    from rulekiln.providers import resolver as _resolver

    def _patched_resolve(
        profile_name: str,
        model: str,
        *,
        role: Literal["teacher", "student", "embedding", "judge"],
        settings: AppSettings,
    ) -> ProviderConfig:
        return _resolver.resolve_provider_config(
            profile_name,
            model,
            role=role,
            settings=fake_settings,
        )

    monkeypatch.setattr(
        "rulekiln.workers.distillation_worker.resolve_provider_config",
        _patched_resolve,
    )


async def _seed_job(
    db_factory: async_sessionmaker[AsyncSession],
    job_id: str,
    payload: DistillationRequest,
) -> None:
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


def _single_cluster(
    strategy: str,
    rule_ids: list[str],
) -> list[RuleClusterSchema]:
    if not rule_ids:
        return []
    return [
        RuleClusterSchema(
            strategy=strategy,
            algorithm=strategy,
            topic=f"{strategy}-cluster",
            rule_ids=sorted(rule_ids),
        )
    ]


@pytest.mark.asyncio
async def test_split_policy_uses_train_for_extraction_and_validation_for_eval(
    db_factory,
    fake_settings,
    monkeypatch: MonkeyPatch,
) -> None:
    _patch_worker_dependencies(monkeypatch, fake_settings)

    payload = _build_payload(["train", "train", "validation", "validation"])
    job_id = "bbbbbbbb-1111-0000-0000-000000000101"
    await _seed_job(db_factory, job_id, payload)

    extraction_case_ids: list[str] = []
    eval_invocations: list[tuple[str, str, list[str]]] = []

    async def _extract_stub(
        task: RuleKilnTask,
        case: RuleKilnCase,
        chat_client: ChatModelClient,
        config: ProviderConfig,
    ) -> ExtractionOutput:
        del task
        del chat_client
        del config
        extraction_case_ids.append(case.id)
        return ExtractionOutput(
            rules=[
                MicroRuleSchema(
                    topic=f"topic-{case.id}",
                    condition="input present",
                    expected_outcome="emit positive",
                )
            ]
        )

    async def _synthesize_stub(
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
                    topic="synth-topic",
                    applies_when=["always"],
                    outcome_conditions={
                        "label": OutcomeCondition(outcome="positive", when=["always"])
                    },
                    source_case_ids=list(source_case_ids),
                    source_micro_rule_ids=list(source_micro_rule_ids),
                )
            ]
        )

    async def _review_stub(
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

    async def _evaluate_stub(
        system_prompt: str,
        cases: list[RuleKilnCase],
        task: RuleKilnTask,
        chat_client: ChatModelClient,
        config: ProviderConfig,
        strategy: str,
        split: str = "train",
        completed_case_results: Mapping[str, CaseEvalResult] | None = None,
        on_case_result: Callable[[CaseEvalResult], Awaitable[None]] | None = None,
        *,
        bootstrap_enabled: bool = True,
        bootstrap_iterations: int = 1000,
        bootstrap_seed: int = 1729,
    ) -> EvalResult:
        del system_prompt
        del task
        del chat_client
        del completed_case_results

        case_ids = [case.id for case in cases]
        eval_invocations.append((strategy, split, case_ids))
        case_results = [
            CaseEvalResult(
                case_id=case.id,
                score=1.0,
                passed=True,
                malformed=False,
                assertion_scores={},
                actual_output={"label": "positive"},
            )
            for case in cases
        ]

        if on_case_result is not None:
            for case_result in case_results:
                await on_case_result(case_result)

        return EvalResult(
            strategy=strategy,
            model=config.model,
            split=split,
            accuracy=1.0,
            macro_f1=1.0,
            weighted_case_score=1.0,
            malformed_output_rate=0.0,
            per_outcome_precision={"positive": 1.0},
            per_outcome_recall={"positive": 1.0},
            confusion_matrix={"positive": {"positive": len(cases)}},
            case_results=case_results,
        )

    monkeypatch.setattr(
        "rulekiln.workers.distillation_worker.extract_rules_for_case", _extract_stub
    )
    monkeypatch.setattr("rulekiln.workers.distillation_worker.synthesize_cluster", _synthesize_stub)
    monkeypatch.setattr(
        "rulekiln.workers.distillation_worker.review_rule_for_conflicts",
        _review_stub,
    )
    monkeypatch.setattr("rulekiln.workers.distillation_worker.evaluate_prompt", _evaluate_stub)
    monkeypatch.setattr(
        "rulekiln.workers.distillation_worker.cluster_dbscan",
        lambda rule_ids, embeddings, eps=0.3, min_samples=2: _single_cluster("dbscan", rule_ids),
    )
    monkeypatch.setattr(
        "rulekiln.workers.distillation_worker.cluster_hdbscan",
        lambda rule_ids, embeddings, min_cluster_size=2: _single_cluster("hdbscan", rule_ids),
    )

    await run_distillation_pipeline(job_id, payload)

    assert extraction_case_ids == ["case-0", "case-1"]
    invoked_strategies = {strategy for strategy, _, _ in eval_invocations}
    assert {"baseline", "dbscan", "hdbscan"}.issubset(invoked_strategies)
    assert {"baseline_few_shot_k3", "baseline_few_shot_k5"}.issubset(invoked_strategies)
    assert "retrieval_few_shot_k5" in invoked_strategies
    expected_eval_case_ids = ["case-2", "case-3"]
    for strategy, split, case_ids in eval_invocations:
        assert split == "validation"
        if strategy == "retrieval_few_shot_k5":
            assert len(case_ids) == 1
            assert case_ids[0] in expected_eval_case_ids
        else:
            assert case_ids == expected_eval_case_ids

    artifact_root = Path(fake_settings.artifact_root) / job_id
    normalized_path = artifact_root / "cases.normalized.jsonl"
    normalized_counts = Counter(
        json.loads(line)["split"]
        for line in normalized_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    assert normalized_counts == Counter({"train": 2, "validation": 2})

    eval_report_path = artifact_root / "outputs" / "eval_report.json"
    eval_report = json.loads(eval_report_path.read_text(encoding="utf-8"))
    assert eval_report["baseline_eval"]["split"] == "validation"
    assert eval_report["dbscan_eval"]["split"] == "validation"
    assert eval_report["hdbscan_eval"]["split"] == "validation"
    assert eval_report.get("evaluation_split_warning") is None


@pytest.mark.asyncio
async def test_split_policy_falls_back_to_train_with_warning(
    db_factory,
    fake_settings,
    monkeypatch: MonkeyPatch,
) -> None:
    _patch_worker_dependencies(monkeypatch, fake_settings)

    payload = _build_payload(["train", "train", "train"])
    job_id = "bbbbbbbb-1111-0000-0000-000000000102"
    await _seed_job(db_factory, job_id, payload)

    extraction_case_ids: list[str] = []
    eval_invocations: list[tuple[str, str, list[str]]] = []

    async def _extract_stub(
        task: RuleKilnTask,
        case: RuleKilnCase,
        chat_client: ChatModelClient,
        config: ProviderConfig,
    ) -> ExtractionOutput:
        del task
        del chat_client
        del config
        extraction_case_ids.append(case.id)
        return ExtractionOutput(
            rules=[
                MicroRuleSchema(
                    topic=f"topic-{case.id}",
                    condition="input present",
                    expected_outcome="emit positive",
                )
            ]
        )

    async def _synthesize_stub(
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
                    topic="synth-topic",
                    applies_when=["always"],
                    outcome_conditions={
                        "label": OutcomeCondition(outcome="positive", when=["always"])
                    },
                    source_case_ids=list(source_case_ids),
                    source_micro_rule_ids=list(source_micro_rule_ids),
                )
            ]
        )

    async def _review_stub(
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

    async def _evaluate_stub(
        system_prompt: str,
        cases: list[RuleKilnCase],
        task: RuleKilnTask,
        chat_client: ChatModelClient,
        config: ProviderConfig,
        strategy: str,
        split: str = "train",
        completed_case_results: Mapping[str, CaseEvalResult] | None = None,
        on_case_result: Callable[[CaseEvalResult], Awaitable[None]] | None = None,
        *,
        bootstrap_enabled: bool = True,
        bootstrap_iterations: int = 1000,
        bootstrap_seed: int = 1729,
    ) -> EvalResult:
        del system_prompt
        del task
        del chat_client
        del completed_case_results

        case_ids = [case.id for case in cases]
        eval_invocations.append((strategy, split, case_ids))
        case_results = [
            CaseEvalResult(
                case_id=case.id,
                score=1.0,
                passed=True,
                malformed=False,
                assertion_scores={},
                actual_output={"label": "positive"},
            )
            for case in cases
        ]

        if on_case_result is not None:
            for case_result in case_results:
                await on_case_result(case_result)

        return EvalResult(
            strategy=strategy,
            model=config.model,
            split=split,
            accuracy=1.0,
            macro_f1=1.0,
            weighted_case_score=1.0,
            malformed_output_rate=0.0,
            per_outcome_precision={"positive": 1.0},
            per_outcome_recall={"positive": 1.0},
            confusion_matrix={"positive": {"positive": len(cases)}},
            case_results=case_results,
        )

    monkeypatch.setattr(
        "rulekiln.workers.distillation_worker.extract_rules_for_case", _extract_stub
    )
    monkeypatch.setattr("rulekiln.workers.distillation_worker.synthesize_cluster", _synthesize_stub)
    monkeypatch.setattr(
        "rulekiln.workers.distillation_worker.review_rule_for_conflicts",
        _review_stub,
    )
    monkeypatch.setattr("rulekiln.workers.distillation_worker.evaluate_prompt", _evaluate_stub)
    monkeypatch.setattr(
        "rulekiln.workers.distillation_worker.cluster_dbscan",
        lambda rule_ids, embeddings, eps=0.3, min_samples=2: _single_cluster("dbscan", rule_ids),
    )
    monkeypatch.setattr(
        "rulekiln.workers.distillation_worker.cluster_hdbscan",
        lambda rule_ids, embeddings, min_cluster_size=2: _single_cluster("hdbscan", rule_ids),
    )

    await run_distillation_pipeline(job_id, payload)

    assert extraction_case_ids == ["case-0", "case-1", "case-2"]
    invoked_strategies = {strategy for strategy, _, _ in eval_invocations}
    assert {"baseline", "dbscan", "hdbscan"}.issubset(invoked_strategies)
    assert {"baseline_few_shot_k3", "baseline_few_shot_k5"}.issubset(invoked_strategies)
    assert "retrieval_few_shot_k5" in invoked_strategies
    expected_eval_case_ids = ["case-0", "case-1", "case-2"]
    for strategy, split, case_ids in eval_invocations:
        assert split == "train"
        if strategy == "retrieval_few_shot_k5":
            assert len(case_ids) == 1
            assert case_ids[0] in expected_eval_case_ids
        else:
            assert case_ids == expected_eval_case_ids

    artifact_root = Path(fake_settings.artifact_root) / job_id
    eval_report_path = artifact_root / "outputs" / "eval_report.json"
    eval_report = json.loads(eval_report_path.read_text(encoding="utf-8"))

    assert eval_report["baseline_eval"]["split"] == "train"
    assert eval_report["dbscan_eval"]["split"] == "train"
    assert eval_report["hdbscan_eval"]["split"] == "train"

    warning_text = eval_report.get("evaluation_split_warning")
    assert isinstance(warning_text, str)
    assert "split=train" in warning_text
