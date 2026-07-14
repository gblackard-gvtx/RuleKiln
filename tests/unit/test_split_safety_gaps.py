"""Targeted tests for the six confirmed train/validation/test split-safety gaps.

Each test below is scoped to one gap:

1. Few-shot exemplars must not silently fall back to eval-split cases.
2. RuleKilnCase.split is required (no silent default to "train").
3. evaluate_prompt asserts split purity between its cases and its split param.
4. Cluster/rule provenance persists source_split end to end.
5. Refinement iterations record refined_using_split/validation_tuned.

(The sixth gap - a dedicated leakage-validation pass - is covered in
tests/unit/test_split_leakage_validation.py.)
"""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import ValidationError

from rulekiln.pipeline.clustering_provenance import build_cluster_assignments
from rulekiln.pipeline.evaluator import evaluate_prompt
from rulekiln.pipeline.rule_provenance import build_provenance_records
from rulekiln.pipeline.split_policy import (
    FewShotExemplarLeakageError,
    resolve_few_shot_source_cases,
    resolve_split_policy,
)
from rulekiln.providers.contracts import ChatModelClient, ProviderConfig
from rulekiln.schemas.pipeline import (
    RefinementIterationArtifact,
    RuleClusterSchema,
    SynthesizedRuleSchema,
)
from rulekiln.schemas.task_case import RuleKilnCase, RuleKilnTask
from rulekiln.schemas.usage import ChatCompletionResult, ModelUsage


def _case(case_id: str, split: Literal["train", "validation", "test", "golden"]) -> RuleKilnCase:
    return RuleKilnCase(
        id=case_id,
        split=split,
        task_mode="classification",
        input={"text": f"input for {case_id}"},
        expected={"label": "entailment"},
    )


# ── Gap 1: few-shot exemplar leak (fail-closed unless explicitly opted in) ────


def test_resolve_few_shot_source_cases_prefers_train_cases() -> None:
    train_cases = [_case("t1", "train")]
    eval_cases = [_case("v1", "validation")]

    result = resolve_few_shot_source_cases(train_cases, eval_cases, allow_eval_fallback=False)

    assert result == train_cases


def test_resolve_few_shot_source_cases_fails_closed_without_opt_in() -> None:
    with pytest.raises(FewShotExemplarLeakageError):
        resolve_few_shot_source_cases([], [_case("v1", "validation")], allow_eval_fallback=False)


def test_resolve_few_shot_source_cases_allows_eval_fallback_with_explicit_opt_in() -> None:
    eval_cases = [_case("v1", "validation")]

    result = resolve_few_shot_source_cases([], eval_cases, allow_eval_fallback=True)

    assert result == eval_cases


def test_task_few_shot_eval_fallback_flag_defaults_to_false() -> None:
    task = RuleKilnTask(
        task_id="t1",
        task_name="test",
        task_mode="classification",
        description="test task",
        input_template="{{ text }}",
    )
    assert task.allow_few_shot_exemplars_from_eval_split is False


# ── Gap 2: RuleKilnCase.split is required, no silent "train" default ─────────


def test_case_missing_split_fails_validation() -> None:
    with pytest.raises(ValidationError):
        RuleKilnCase(
            id="c1",
            task_mode="classification",
            input={"text": "hello"},
        )  # type: ignore[call-arg]


def test_case_invalid_split_value_fails_validation() -> None:
    with pytest.raises(ValidationError):
        RuleKilnCase(
            id="c1",
            split="not_a_real_split",  # type: ignore[arg-type]
            task_mode="classification",
            input={"text": "hello"},
        )


def test_resolve_split_policy_still_works_with_explicit_splits() -> None:
    cases = [_case("c1", "train"), _case("c2", "validation")]
    decision = resolve_split_policy(cases)
    assert decision.extraction_split == "train"
    assert decision.evaluation_split == "validation"


# ── Gap 3: evaluate_prompt asserts split purity ───────────────────────────────


class _FakeClient(ChatModelClient):
    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type,
        config: ProviderConfig,
    ) -> ChatCompletionResult:
        return ChatCompletionResult(
            content='{"label": "entailment"}',
            parsed=None,
            usage=ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            raw_model="fake",
        )


def _provider_config() -> ProviderConfig:
    return ProviderConfig(
        profile_name="local", provider="openai_compatible", model="fake", max_retries=0
    )


def _task() -> RuleKilnTask:
    return RuleKilnTask(
        task_id="t1",
        task_name="test",
        task_mode="classification",
        description="test task",
        input_template="{{ text }}",
        output_schema={
            "type": "object",
            "required": ["label"],
            "properties": {"label": {"type": "string", "enum": ["entailment"]}},
        },
    )


@pytest.mark.asyncio
async def test_evaluate_prompt_rejects_cases_from_wrong_split() -> None:
    cases = [_case("c1", "train")]

    with pytest.raises(ValueError, match="different split"):
        await evaluate_prompt(
            "system prompt",
            cases,
            _task(),
            _FakeClient(),
            _provider_config(),
            strategy="baseline_few_shot_k5",
            split="validation",
            bootstrap_enabled=False,
        )


@pytest.mark.asyncio
async def test_evaluate_prompt_accepts_cases_matching_declared_split() -> None:
    cases = [_case("c1", "validation")]

    result = await evaluate_prompt(
        "system prompt",
        cases,
        _task(),
        _FakeClient(),
        _provider_config(),
        strategy="baseline_few_shot_k5",
        split="validation",
        bootstrap_enabled=False,
    )

    assert result.split == "validation"


# ── Gap 4: source_split provenance persists through clustering + rule provenance ──


def test_build_cluster_assignments_propagates_split_by_rule() -> None:
    clusters = [
        RuleClusterSchema(strategy="hdbscan", algorithm="hdbscan", rule_ids=["r1"]),
    ]
    records = build_cluster_assignments(
        rule_ids=["r1"],
        rule_texts=["rule one"],
        clusters=clusters,
        label_by_rule={"r1": "entailment"},
        source_case_by_rule={"r1": "c1"},
        hypothesis_by_rule={"r1": None},
        split_by_rule={"r1": "validation"},
    )

    assert records[0].source_split == "validation"


def test_build_cluster_assignments_defaults_split_to_train_when_unspecified() -> None:
    clusters = [
        RuleClusterSchema(strategy="hdbscan", algorithm="hdbscan", rule_ids=["r1"]),
    ]
    records = build_cluster_assignments(
        rule_ids=["r1"],
        rule_texts=["rule one"],
        clusters=clusters,
        label_by_rule={"r1": "entailment"},
        source_case_by_rule={"r1": "c1"},
        hypothesis_by_rule={"r1": None},
    )

    assert records[0].source_split == "train"


def test_build_provenance_records_propagates_source_split_from_rule() -> None:
    rule = SynthesizedRuleSchema(
        id="r1",
        topic="entailment",
        source_case_ids=["c1"],
        source_split="validation",
    )

    artifact = build_provenance_records(
        job_id="job-1",
        strategy_id="hdbscan",
        selected_rules=[rule],
    )

    assert artifact.rules[0].source_split == "validation"


def test_build_provenance_records_defaults_source_split_to_train() -> None:
    rule = SynthesizedRuleSchema(id="r1", topic="entailment", source_case_ids=["c1"])

    artifact = build_provenance_records(
        job_id="job-1",
        strategy_id="hdbscan",
        selected_rules=[rule],
    )

    assert artifact.rules[0].source_split == "train"


# ── Gap 5: refinement iterations mark validation-tuned prompts explicitly ────


def test_refinement_iteration_artifact_defaults_are_train_and_untuned() -> None:
    artifact = RefinementIterationArtifact(
        job_id="job-1",
        iteration=1,
        strategy_id="hdbscan",
        prior_metric=0.5,
        new_metric=0.6,
        improvement=0.1,
    )
    assert artifact.refined_using_split == "train"
    assert artifact.validation_tuned is False


def test_refinement_iteration_artifact_marks_validation_tuned_explicitly() -> None:
    artifact = RefinementIterationArtifact(
        job_id="job-1",
        iteration=1,
        strategy_id="hdbscan",
        prior_metric=0.5,
        new_metric=0.6,
        improvement=0.1,
        refined_using_split="validation",
        validation_tuned=True,
    )
    assert artifact.refined_using_split == "validation"
    assert artifact.validation_tuned is True
