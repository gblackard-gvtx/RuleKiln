"""Unit tests for the rule refinement module (Task 2: empirical teacher call)."""

from __future__ import annotations

import pytest

from rulekiln.pipeline.failure_analysis import (
    FailureAnalysisResult,
    analyze_failures,
)
from rulekiln.pipeline.rule_refinement import (
    RefinementResult,
    RevisedRuleEntry,
    _build_refinement_prompt,
    apply_refinements,
    refine_rules_with_teacher,
)
from rulekiln.providers.chat.fake import FakeChatClient
from rulekiln.providers.contracts import ProviderConfig
from rulekiln.schemas.pipeline import (
    CaseEvalResult,
    CaseEvaluationFailure,
    EvalResult,
    OutcomeCondition,
    SynthesizedRuleSchema,
)
from rulekiln.schemas.task_case import EvaluationAssertion, EvaluationSpec, RuleKilnCase

# ── Fixtures ────────────────────────────────────────────────────────────────


def _rule(rule_id: str, outcome_label: str) -> SynthesizedRuleSchema:
    return SynthesizedRuleSchema(
        id=rule_id,
        topic=f"rule_topic_{outcome_label}",
        applies_when=["some condition"],
        outcome_conditions={
            outcome_label: OutcomeCondition(
                outcome=outcome_label,
                when=["some condition"],
                confidence="high",
            )
        },
        priority=1,
        source_case_ids=["c1"],
        source_micro_rule_ids=["m1"],
        support_count=3,
    )


def _case(case_id: str, expected_label: str) -> RuleKilnCase:
    return RuleKilnCase(
        id=case_id,
        task_mode="classification",
        input={"text": "sample text"},
        expected={"label": expected_label},
        evaluation=EvaluationSpec(
            assertions=[
                EvaluationAssertion(
                    type="must_equal",  # type: ignore[arg-type]
                    path="label",
                    value=expected_label,
                    weight=1.0,
                )
            ]
        ),
    )


def _case_result(case_id: str, passed: bool) -> CaseEvalResult:
    return CaseEvalResult(
        case_id=case_id,
        passed=passed,
        score=1.0 if passed else 0.0,
        malformed=False,
        assertion_scores={"assertion_0": 1.0 if passed else 0.0},
    )


def _eval_result(case_results: list[CaseEvalResult]) -> EvalResult:
    return EvalResult(
        strategy="dbscan",
        model="fake",
        split="test",
        macro_f1=0.7,
        weighted_case_score=0.7,
        per_outcome_precision={},
        per_outcome_recall={},
        malformed_output_rate=0.0,
        confusion_matrix={},
        case_results=case_results,
    )


def _fake_config() -> ProviderConfig:
    return ProviderConfig(profile_name="fake", model="fake-model", provider="fake")


def _failure_analysis_with_broken_rule_x() -> tuple[FailureAnalysisResult, list[RuleKilnCase]]:
    """Fixture: one broken case attributed to rule_X, one fixed case attributed to rule_X."""
    rule = _rule("rule_X", "entailment")
    case_broken = _case("c_broken", "entailment")
    case_fixed = _case("c_fixed", "entailment")
    baseline = _eval_result(
        [_case_result("c_broken", passed=True), _case_result("c_fixed", passed=False)]
    )
    distilled = _eval_result(
        [_case_result("c_broken", passed=False), _case_result("c_fixed", passed=True)]
    )
    result = analyze_failures(
        baseline, distilled, selected_rules=[rule], cases=[case_broken, case_fixed]
    )
    return result, [case_broken, case_fixed]


# ── Tests ───────────────────────────────────────────────────────────────────


def test_refinement_result_schema_version() -> None:
    """RefinementResult has correct schema_version."""
    r = RefinementResult()
    assert r.schema_version == "rulekiln.refinement_result.v1"


def test_apply_refinements_replaces_rule_by_id() -> None:
    """apply_refinements replaces the rule matching entry.rule_id, preserving ID."""
    original = _rule("rule_X", "entailment")
    revised_body = _rule("rule_X_revised", "entailment")  # different topic
    revised_body.topic = "revised_topic"
    entry = RevisedRuleEntry(rule_id="rule_X", revised_rule=revised_body, rationale="test")
    refinement = RefinementResult(revised_rules=[entry])

    result = apply_refinements([original], refinement)
    assert len(result) == 1
    assert result[0].id == "rule_X"  # ID preserved
    assert result[0].topic == "revised_topic"  # body updated


def test_apply_refinements_leaves_unaffected_rules() -> None:
    """apply_refinements does not touch rules not in the refinement."""
    rule_a = _rule("rule_A", "entailment")
    rule_b = _rule("rule_B", "contradiction")
    revised_b = _rule("rule_B_v2", "contradiction")
    revised_b.topic = "revised_b"
    entry = RevisedRuleEntry(rule_id="rule_B", revised_rule=revised_b, rationale="fix B")
    refinement = RefinementResult(revised_rules=[entry])

    result = apply_refinements([rule_a, rule_b], refinement)
    assert result[0].id == "rule_A"
    assert result[0].topic == rule_a.topic  # unchanged
    assert result[1].id == "rule_B"
    assert result[1].topic == "revised_b"


def test_apply_refinements_empty_refinement_is_noop() -> None:
    """apply_refinements with no revised_rules returns the original rules unchanged."""
    rules = [_rule("rule_A", "entailment"), _rule("rule_B", "contradiction")]
    result = apply_refinements(rules, RefinementResult())
    assert [r.id for r in result] == ["rule_A", "rule_B"]


def test_build_prompt_contains_success_cases() -> None:
    """Success cases appear in the constructed prompt."""
    rule = _rule("rule_X", "entailment")
    sf_failure = CaseEvaluationFailure(
        case_id="c_broken",
        split="",
        failure_class="broken",
        violated_rule_ids=["rule_X"],
        failed_assertion_paths=["assertion_0"],
        failed_assertion_types=["must_equal"],
    )
    case_success = _case("c_fixed", "entailment")
    prompt = _build_refinement_prompt(
        implicated_rules=[rule],
        failure_pairs=[(sf_failure, None)],
        success_pairs=[(None, case_success)],
    )
    assert "c_fixed" in prompt
    assert "Success Cases" in prompt


def test_build_prompt_contains_failure_info() -> None:
    """Failure cases and violated rule IDs appear in the constructed prompt."""
    rule = _rule("rule_X", "entailment")
    sf = CaseEvaluationFailure(
        case_id="c_broken",
        split="",
        failure_class="broken",
        violated_rule_ids=["rule_X"],
        failed_assertion_paths=["assertion_0"],
        failed_assertion_types=["must_equal"],
    )
    prompt = _build_refinement_prompt(
        implicated_rules=[rule],
        failure_pairs=[(sf, None)],
        success_pairs=[],
    )
    assert "rule_X" in prompt
    assert "c_broken" in prompt
    assert "Failure Cases" in prompt


@pytest.mark.asyncio
async def test_refine_rules_with_teacher_offline_fake_provider() -> None:
    """Works offline with the fake provider (no external calls)."""
    failure_analysis, cases = _failure_analysis_with_broken_rule_x()
    rule = _rule("rule_X", "entailment")
    case_map = {c.id: c for c in cases}

    client = FakeChatClient()
    config = _fake_config()

    result = await refine_rules_with_teacher(
        current_rules=[rule],
        failure_analysis_result=failure_analysis,
        case_map=case_map,
        chat_client=client,
        config=config,
        seed=1729,
    )

    assert isinstance(result, RefinementResult)
    assert result.schema_version == "rulekiln.refinement_result.v1"


@pytest.mark.asyncio
async def test_refine_rules_implicated_rule_requested() -> None:
    """The prompt includes rule_X because it appears in failure violated_rule_ids."""
    failure_analysis, cases = _failure_analysis_with_broken_rule_x()
    rule_x = _rule("rule_X", "entailment")
    rule_unrelated = _rule("rule_Y", "contradiction")
    case_map = {c.id: c for c in cases}

    # Capture what prompt is built by monkey-patching via a custom client
    captured_prompts: list[str] = []

    class CapturingClient(FakeChatClient):
        async def complete_structured(self, *, system_prompt: str, user_prompt: str, **kwargs):  # type: ignore[override]
            captured_prompts.append(user_prompt)
            return await super().complete_structured(
                system_prompt=system_prompt, user_prompt=user_prompt, **kwargs
            )

    client = CapturingClient()
    config = _fake_config()

    await refine_rules_with_teacher(
        current_rules=[rule_x, rule_unrelated],
        failure_analysis_result=failure_analysis,
        case_map=case_map,
        chat_client=client,
        config=config,
        seed=1729,
    )

    assert captured_prompts, "prompt should have been captured"
    prompt = captured_prompts[0]
    assert "rule_X" in prompt


@pytest.mark.asyncio
async def test_refine_rules_success_cases_in_prompt() -> None:
    """Success cases (fixed) appear in the teacher prompt."""
    failure_analysis, cases = _failure_analysis_with_broken_rule_x()
    rule = _rule("rule_X", "entailment")
    case_map = {c.id: c for c in cases}

    captured_prompts: list[str] = []

    class CapturingClient(FakeChatClient):
        async def complete_structured(self, *, system_prompt: str, user_prompt: str, **kwargs):  # type: ignore[override]
            captured_prompts.append(user_prompt)
            return await super().complete_structured(
                system_prompt=system_prompt, user_prompt=user_prompt, **kwargs
            )

    await refine_rules_with_teacher(
        current_rules=[rule],
        failure_analysis_result=failure_analysis,
        case_map=case_map,
        chat_client=CapturingClient(),
        config=_fake_config(),
        seed=1729,
    )

    assert captured_prompts
    # The fixed case (c_fixed) should appear under Success Cases
    assert "c_fixed" in captured_prompts[0]


def test_refinement_result_seed_determinism() -> None:
    """Same seed produces the same sampled failure/success cases (determinism guarantee)."""
    failure_analysis, cases = _failure_analysis_with_broken_rule_x()
    rule = _rule("rule_X", "entailment")
    case_map = {c.id: c for c in cases}

    captured_1: list[str] = []
    captured_2: list[str] = []

    import asyncio

    class Cap1(FakeChatClient):
        async def complete_structured(self, *, system_prompt: str, user_prompt: str, **kwargs):  # type: ignore[override]
            captured_1.append(user_prompt)
            return await super().complete_structured(
                system_prompt=system_prompt, user_prompt=user_prompt, **kwargs
            )

    class Cap2(FakeChatClient):
        async def complete_structured(self, *, system_prompt: str, user_prompt: str, **kwargs):  # type: ignore[override]
            captured_2.append(user_prompt)
            return await super().complete_structured(
                system_prompt=system_prompt, user_prompt=user_prompt, **kwargs
            )

    asyncio.get_event_loop().run_until_complete(
        refine_rules_with_teacher(
            current_rules=[rule],
            failure_analysis_result=failure_analysis,
            case_map=case_map,
            chat_client=Cap1(),
            config=_fake_config(),
            seed=42,
        )
    )
    asyncio.get_event_loop().run_until_complete(
        refine_rules_with_teacher(
            current_rules=[rule],
            failure_analysis_result=failure_analysis,
            case_map=case_map,
            chat_client=Cap2(),
            config=_fake_config(),
            seed=42,
        )
    )

    assert captured_1 == captured_2
