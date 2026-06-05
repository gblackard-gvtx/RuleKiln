"""Unit tests for the rule refinement module (empirical teacher call)."""

from __future__ import annotations

import pytest

from rulekiln.pipeline.failure_analysis import (
    FailureAnalysisResult,
    analyze_failures,
)
from rulekiln.pipeline.rule_refinement import (
    RefinementResult,
    RevisedRuleEntry,
    RuleRefinementAction,
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


# ── Schema version ───────────────────────────────────────────────────────────


def test_refinement_result_schema_version() -> None:
    r = RefinementResult()
    assert r.schema_version == "rulekiln.refinement_result.v2"


# ── apply_refinements: keep action ───────────────────────────────────────────


def test_apply_refinements_keep_preserves_rule() -> None:
    rule = _rule("rule_A", "entailment")
    action = RuleRefinementAction(action="keep", rule_id="rule_A")
    refinement = RefinementResult(actions=[action])

    result = apply_refinements([rule], refinement)
    assert len(result) == 1
    assert result[0].id == "rule_A"
    assert result[0].topic == rule.topic


def test_apply_refinements_no_action_preserves_rule() -> None:
    rule = _rule("rule_A", "entailment")
    result = apply_refinements([rule], RefinementResult())
    assert len(result) == 1
    assert result[0].id == "rule_A"


# ── apply_refinements: modify action ────────────────────────────────────────


def test_apply_refinements_modify_replaces_content_preserves_id() -> None:
    original = _rule("rule_X", "entailment")
    revised_body = _rule("rule_X_v2", "entailment")
    revised_body.topic = "revised_topic"
    action = RuleRefinementAction(
        action="modify", rule_id="rule_X", revised_rules=[revised_body], rationale="fix"
    )
    refinement = RefinementResult(actions=[action])

    result = apply_refinements([original], refinement)
    assert len(result) == 1
    assert result[0].id == "rule_X"  # ID preserved
    assert result[0].topic == "revised_topic"  # body updated


def test_apply_refinements_modify_invalid_zero_revised_rules_keeps_original(
    caplog: pytest.LogCaptureFixture,
) -> None:
    original = _rule("rule_X", "entailment")
    action = RuleRefinementAction(action="modify", rule_id="rule_X", revised_rules=[])
    refinement = RefinementResult(actions=[action])

    result = apply_refinements([original], refinement)
    assert len(result) == 1
    assert result[0].id == "rule_X"
    assert result[0].topic == original.topic


def test_apply_refinements_modify_invalid_multiple_revised_rules_keeps_original() -> None:
    original = _rule("rule_X", "entailment")
    action = RuleRefinementAction(
        action="modify",
        rule_id="rule_X",
        revised_rules=[_rule("r1", "entailment"), _rule("r2", "entailment")],
    )
    refinement = RefinementResult(actions=[action])

    result = apply_refinements([original], refinement)
    assert len(result) == 1
    assert result[0].id == "rule_X"


# ── apply_refinements: split action ─────────────────────────────────────────


def test_apply_refinements_split_creates_deterministic_child_ids() -> None:
    original = _rule("rule_X", "entailment")
    child_a = _rule("child_a", "entailment")
    child_b = _rule("child_b", "contradiction")
    action = RuleRefinementAction(
        action="split", rule_id="rule_X", revised_rules=[child_a, child_b]
    )
    refinement = RefinementResult(actions=[action])

    result = apply_refinements([original], refinement)
    assert len(result) == 2
    assert result[0].id == "rule_X__refined_1"
    assert result[1].id == "rule_X__refined_2"


def test_apply_refinements_split_invalid_empty_keeps_original() -> None:
    original = _rule("rule_X", "entailment")
    action = RuleRefinementAction(action="split", rule_id="rule_X", revised_rules=[])
    refinement = RefinementResult(actions=[action])

    result = apply_refinements([original], refinement)
    assert len(result) == 1
    assert result[0].id == "rule_X"


# ── apply_refinements: discard action ───────────────────────────────────────


def test_apply_refinements_discard_removes_rule() -> None:
    rule_a = _rule("rule_A", "entailment")
    rule_b = _rule("rule_B", "contradiction")
    action = RuleRefinementAction(action="discard", rule_id="rule_A")
    refinement = RefinementResult(actions=[action])

    result = apply_refinements([rule_a, rule_b], refinement)
    assert len(result) == 1
    assert result[0].id == "rule_B"


# ── apply_refinements: unknown rule IDs ignored ──────────────────────────────


def test_apply_refinements_unknown_rule_id_ignored() -> None:
    rule = _rule("rule_A", "entailment")
    action = RuleRefinementAction(action="discard", rule_id="nonexistent_rule")
    refinement = RefinementResult(actions=[action])

    result = apply_refinements([rule], refinement)
    assert len(result) == 1
    assert result[0].id == "rule_A"


# ── apply_refinements: v1 backward-compat ───────────────────────────────────


def test_apply_refinements_v1_revised_rules_backward_compat() -> None:
    original = _rule("rule_X", "entailment")
    revised_body = _rule("rule_X_revised", "entailment")
    revised_body.topic = "revised_topic"
    entry = RevisedRuleEntry(rule_id="rule_X", revised_rule=revised_body, rationale="test")
    refinement = RefinementResult(revised_rules=[entry])

    result = apply_refinements([original], refinement)
    assert len(result) == 1
    assert result[0].id == "rule_X"
    assert result[0].topic == "revised_topic"


def test_apply_refinements_leaves_unaffected_rules() -> None:
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
    rules = [_rule("rule_A", "entailment"), _rule("rule_B", "contradiction")]
    result = apply_refinements(rules, RefinementResult())
    assert [r.id for r in result] == ["rule_A", "rule_B"]


# ── refine_rules_with_teacher: no-op on no attributable failures ─────────────


@pytest.mark.asyncio
async def test_refine_rules_no_attributable_failures_returns_noop() -> None:
    """No teacher call is made when there are no attributable failures."""
    call_count = 0

    class CountingClient(FakeChatClient):
        async def complete_structured(self, *, system_prompt: str, user_prompt: str, **kwargs):  # type: ignore[override]
            nonlocal call_count
            call_count += 1
            return await super().complete_structured(
                system_prompt=system_prompt, user_prompt=user_prompt, **kwargs
            )

    # Failure analysis with zero failures
    empty_analysis = FailureAnalysisResult()
    rule = _rule("rule_X", "entailment")

    result = await refine_rules_with_teacher(
        current_rules=[rule],
        failure_analysis_result=empty_analysis,
        case_map={},
        chat_client=CountingClient(),
        config=_fake_config(),
        seed=1729,
    )

    assert isinstance(result, RefinementResult)
    assert result.actions == []
    assert call_count == 0


# ── refine_rules_with_teacher: only real attributions sent to teacher ─────────


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
    assert result.schema_version == "rulekiln.refinement_result.v2"


@pytest.mark.asyncio
async def test_refine_rules_implicated_rule_requested() -> None:
    """The prompt includes rule_X because it appears in failure violated_rule_ids."""
    failure_analysis, cases = _failure_analysis_with_broken_rule_x()
    rule_x = _rule("rule_X", "entailment")
    rule_unrelated = _rule("rule_Y", "contradiction")
    case_map = {c.id: c for c in cases}

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


# ── refine_rules_with_teacher: teacher output filtered to implicated rules ────


@pytest.mark.asyncio
async def test_refine_rules_rejects_non_implicated_teacher_actions() -> None:
    """Teacher-returned actions for non-implicated rule IDs are rejected."""
    failure_analysis, cases = _failure_analysis_with_broken_rule_x()
    rule_x = _rule("rule_X", "entailment")
    rule_y = _rule("rule_Y", "contradiction")
    case_map = {c.id: c for c in cases}

    # Return an action for both rule_X (implicated) and rule_Y (not implicated)
    from rulekiln.providers.estimation import build_usage_from_provider
    from rulekiln.schemas.usage import ChatCompletionResult

    class InjectingClient(FakeChatClient):
        async def complete_structured(  # type: ignore[override]
            self,
            *,
            system_prompt: str,
            user_prompt: str,
            output_schema: type,
            config: ProviderConfig,
        ) -> ChatCompletionResult:
            parsed = RefinementResult(
                actions=[
                    RuleRefinementAction(action="modify", rule_id="rule_X", revised_rules=[rule_x]),
                    RuleRefinementAction(action="discard", rule_id="rule_Y"),  # not implicated
                ]
            )
            usage = build_usage_from_provider(input_tokens=0, output_tokens=0, total_tokens=0)
            return ChatCompletionResult(content="", parsed=parsed, usage=usage, raw_model="fake")

    result = await refine_rules_with_teacher(
        current_rules=[rule_x, rule_y],
        failure_analysis_result=failure_analysis,
        case_map=case_map,
        chat_client=InjectingClient(),
        config=_fake_config(),
        seed=1729,
    )

    # Only rule_X action accepted; rule_Y rejected
    assert len(result.actions) == 1
    assert result.actions[0].rule_id == "rule_X"


# ── refine_rules_with_teacher: success cases include fixed + unchanged_correct ─


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


@pytest.mark.asyncio
async def test_refine_rules_unchanged_correct_included_in_success_cases() -> None:
    """unchanged_correct cases are included as success cases alongside fixed cases."""
    rule = _rule("rule_X", "entailment")
    case_passing = _case("c_passing", "entailment")
    case_broken = _case("c_broken", "entailment")

    # No baseline: c_passing is unchanged_correct, c_broken is unchanged_wrong
    distilled = _eval_result(
        [_case_result("c_passing", passed=True), _case_result("c_broken", passed=False)]
    )
    analysis = analyze_failures(
        None, distilled, selected_rules=[rule], cases=[case_passing, case_broken]
    )

    # Verify unchanged_correct is in structured_failures
    unchanged_correct = [
        sf for sf in analysis.structured_failures if sf.failure_class == "unchanged_correct"
    ]
    assert len(unchanged_correct) == 1
    assert unchanged_correct[0].case_id == "c_passing"

    captured_prompts: list[str] = []

    class CapturingClient(FakeChatClient):
        async def complete_structured(self, *, system_prompt: str, user_prompt: str, **kwargs):  # type: ignore[override]
            captured_prompts.append(user_prompt)
            return await super().complete_structured(
                system_prompt=system_prompt, user_prompt=user_prompt, **kwargs
            )

    await refine_rules_with_teacher(
        current_rules=[rule],
        failure_analysis_result=analysis,
        case_map={case_passing.id: case_passing, case_broken.id: case_broken},
        chat_client=CapturingClient(),
        config=_fake_config(),
        seed=1729,
    )

    assert captured_prompts
    # c_passing (unchanged_correct) should appear in the success cases section
    assert "c_passing" in captured_prompts[0]


# ── Prompt content tests ─────────────────────────────────────────────────────


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


# ── Determinism ──────────────────────────────────────────────────────────────


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
