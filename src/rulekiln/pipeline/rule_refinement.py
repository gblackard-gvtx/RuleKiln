"""Rule refinement: empirical teacher interaction to revise rules based on observed failures.

This is the empirical, case-outcome-based component of closed-loop conflict resolution
(paper Phase 3, §3.3). It is distinct from review_rule_for_conflicts, which is a
static linguistic consistency check that runs before evaluation.
"""

from __future__ import annotations

import json
import random
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from rulekiln.observability.logging import get_logger
from rulekiln.providers.contracts import ChatModelClient, ProviderConfig
from rulekiln.schemas.pipeline import CaseEvaluationFailure, SynthesizedRuleSchema
from rulekiln.schemas.task_case import RuleKilnCase

if TYPE_CHECKING:
    from rulekiln.pipeline.failure_analysis import FailureAnalysisResult

logger = get_logger(__name__)


class RevisedRuleEntry(BaseModel):
    """v1 compat: a revised rule and its rationale. Superseded by RuleRefinementAction."""

    rule_id: str
    revised_rule: SynthesizedRuleSchema
    rationale: str = ""


class RuleRefinementAction(BaseModel):
    """One teacher-directed action on a single implicated rule."""

    action: Literal["keep", "modify", "split", "discard"]
    rule_id: str
    revised_rules: list[SynthesizedRuleSchema] = Field(default_factory=list)
    rationale: str = ""


class RefinementResult(BaseModel):
    """Output from a single refinement teacher call.

    v2 uses ``actions`` (keep/modify/split/discard per rule).
    ``revised_rules`` is retained for loading v1 artifacts only.
    schema_version follows the Phase 8 artifact-versioning convention.
    """

    schema_version: Literal["rulekiln.refinement_result.v2"] = "rulekiln.refinement_result.v2"
    actions: list[RuleRefinementAction] = Field(default_factory=list)
    # v1 backward-compatibility field — not populated by the teacher in v2
    revised_rules: list[RevisedRuleEntry] = Field(default_factory=list)
    reasoning: str | None = None


_REFINEMENT_SYSTEM_PROMPT = """\
You are a task-policy refinement expert performing closed-loop conflict resolution.

You receive:
1. The rules implicated in recent evaluation failures.
2. FAILURE CASES: cases where the student followed the rules but still produced the wrong answer.
3. SUCCESS CASES: cases where the student answered correctly (must not regress).

Your task:
- Diagnose the root cause of failures.
- Revise ONLY the implicated rules to fix the root cause.
- Your revisions MUST NOT break the provided success cases.
- Leave all other rules untouched.

Return a RefinementResult with an actions list. For each implicated rule, include one action:
- keep: rule is correct as-is, no change needed
- modify: replace the rule with exactly one improved rule (revised_rules must have exactly 1 entry)
- split: split the rule into multiple non-conflicting rules (revised_rules has 2+ entries)
- discard: rule is harmful or irredeemably conflicted, remove it

Always include rule_id and a rationale.
"""


def _build_refinement_prompt(
    implicated_rules: list[SynthesizedRuleSchema],
    failure_pairs: list[tuple[CaseEvaluationFailure, RuleKilnCase | None]],
    success_pairs: list[tuple[CaseEvaluationFailure | None, RuleKilnCase | None]],
) -> str:
    parts: list[str] = []

    parts.append("## Implicated Rules\n")
    for rule in implicated_rules:
        parts.append(f"Rule ID: {rule.id}")
        parts.append(f"Topic: {rule.topic}")
        if rule.applies_when:
            parts.append("Applies when:\n" + "\n".join(f"  - {c}" for c in rule.applies_when))
        if rule.outcome_conditions:
            parts.append(
                "Outcomes:\n"
                + "\n".join(
                    f"  {name}: outcome={oc.outcome}, when={oc.when}"
                    for name, oc in rule.outcome_conditions.items()
                )
            )
        parts.append("")

    parts.append("## Failure Cases (student answered incorrectly)\n")
    for sf, case in failure_pairs:
        parts.append(f"Case ID: {sf.case_id}")
        parts.append(f"Failure class: {sf.failure_class}")
        parts.append(f"Violated rule IDs: {sf.violated_rule_ids}")
        parts.append(f"Failed assertion paths: {sf.failed_assertion_paths}")
        parts.append(f"Failed assertion types: {sf.failed_assertion_types}")
        if case is not None:
            parts.append(f"Input: {json.dumps(case.input)}")
            parts.append(f"Expected: {json.dumps(case.expected)}")
        parts.append("")

    parts.append("## Success Cases (revisions MUST NOT break these)\n")
    for _sf, case in success_pairs:
        if case is not None:
            parts.append(f"Case ID: {case.id}")
            parts.append(f"Input: {json.dumps(case.input)}")
            parts.append(f"Expected: {json.dumps(case.expected)}")
            parts.append("")

    parts.append(
        "Revise ONLY the rules whose IDs appear in the failure list above. "
        "Return a RefinementResult with the actions list."
    )
    return "\n".join(parts)


async def refine_rules_with_teacher(
    *,
    current_rules: list[SynthesizedRuleSchema],
    failure_analysis_result: FailureAnalysisResult,
    case_map: dict[str, RuleKilnCase],
    chat_client: ChatModelClient,
    config: ProviderConfig,
    seed: int = 1729,
    max_failure_cases: int = 20,
    max_success_cases: int = 20,
) -> RefinementResult:
    """Call the teacher to diagnose root causes and emit rule refinement actions.

    This is the empirical refinement step (paper Phase 3, §3.3).
    Works offline with the fake provider (deterministic stub revisions).

    Sampling is deterministic given seed. Only failure cases with real rule
    attribution (not the UNATTRIBUTED sentinel) are sent to the teacher.
    Success cases (fixed + unchanged_correct) are included as regression guards.

    Returns a no-op RefinementResult (empty actions) when there are no attributable
    failures, without making a provider call.
    """
    from rulekiln.pipeline.failure_analysis import UNATTRIBUTED_RULE_ID

    rng = random.Random(seed)  # noqa: S311

    attributable_failures = [
        sf
        for sf in failure_analysis_result.structured_failures
        if sf.failure_class in ("broken", "unchanged_wrong")
        and sf.violated_rule_ids
        and sf.violated_rule_ids != [UNATTRIBUTED_RULE_ID]
    ]
    sampled_failures = rng.sample(
        attributable_failures, min(max_failure_cases, len(attributable_failures))
    )
    failure_pairs: list[tuple[CaseEvaluationFailure, RuleKilnCase | None]] = [
        (sf, case_map.get(sf.case_id)) for sf in sampled_failures
    ]

    success_sfs = [
        sf
        for sf in failure_analysis_result.structured_failures
        if sf.failure_class in ("fixed", "unchanged_correct")
    ]
    sampled_success_sfs = rng.sample(success_sfs, min(max_success_cases, len(success_sfs)))
    success_pairs: list[tuple[CaseEvaluationFailure | None, RuleKilnCase | None]] = [
        (sf, case_map.get(sf.case_id)) for sf in sampled_success_sfs
    ]

    implicated_ids = {
        rule_id
        for sf in sampled_failures
        for rule_id in sf.violated_rule_ids
        if rule_id != UNATTRIBUTED_RULE_ID
    }
    implicated_rules = [r for r in current_rules if r.id in implicated_ids]

    if not implicated_rules:
        return RefinementResult(
            reasoning="No attributable rule failures found; no refinement applied.",
            actions=[],
        )

    user_prompt = _build_refinement_prompt(implicated_rules, failure_pairs, success_pairs)

    result = await chat_client.complete_structured(
        system_prompt=_REFINEMENT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        output_schema=RefinementResult,
        config=config,
    )
    parsed = result.parsed
    if not isinstance(parsed, RefinementResult):
        parsed = RefinementResult.model_validate(parsed.model_dump() if parsed else {})

    # Reject teacher actions for rule IDs not in the implicated set
    allowed_ids = {rule.id for rule in implicated_rules}
    accepted_actions: list[RuleRefinementAction] = []
    rejected_rule_ids: list[str] = []
    for action in parsed.actions:
        if action.rule_id in allowed_ids:
            accepted_actions.append(action)
        else:
            rejected_rule_ids.append(action.rule_id)
    if rejected_rule_ids:
        logger.warning(
            "refinement_rejected_non_implicated_rules",
            rejected_rule_ids=rejected_rule_ids,
            allowed_rule_ids=sorted(allowed_ids),
        )
    parsed = parsed.model_copy(update={"actions": accepted_actions})

    return parsed


def apply_refinements(
    current_rules: list[SynthesizedRuleSchema],
    refinement: RefinementResult,
) -> list[SynthesizedRuleSchema]:
    """Apply a RefinementResult to the current rule set.

    Uses v2 ``actions`` when present. Falls back to v1 ``revised_rules`` for
    loading legacy artifacts produced before the v2 schema was introduced.
    """
    if refinement.actions:
        return _apply_refinement_actions(current_rules, refinement.actions)
    if refinement.revised_rules:
        # v1 backward-compat path: plain replacement by rule ID
        revised_by_id = {entry.rule_id: entry.revised_rule for entry in refinement.revised_rules}
        result: list[SynthesizedRuleSchema] = []
        for rule in current_rules:
            revised = revised_by_id.get(rule.id)
            if revised is not None:
                result.append(revised.model_copy(update={"id": rule.id}))
            else:
                result.append(rule)
        return result
    return list(current_rules)


def _apply_refinement_actions(
    current_rules: list[SynthesizedRuleSchema],
    actions: list[RuleRefinementAction],
) -> list[SynthesizedRuleSchema]:
    """Apply action-based refinements (v2 schema)."""
    actions_by_id = {entry.rule_id: entry for entry in actions}
    result: list[SynthesizedRuleSchema] = []

    for rule in current_rules:
        action = actions_by_id.get(rule.id)

        if action is None or action.action == "keep":
            result.append(rule)
            continue

        if action.action == "modify":
            if len(action.revised_rules) != 1:
                logger.warning(
                    "refinement_invalid_modify",
                    rule_id=rule.id,
                    revised_rule_count=len(action.revised_rules),
                )
                result.append(rule)
            else:
                result.append(action.revised_rules[0].model_copy(update={"id": rule.id}))
            continue

        if action.action == "split":
            if not action.revised_rules:
                logger.warning("refinement_invalid_split_empty", rule_id=rule.id)
                result.append(rule)
            else:
                for idx, revised_rule in enumerate(action.revised_rules, start=1):
                    result.append(
                        revised_rule.model_copy(update={"id": f"{rule.id}__refined_{idx}"})
                    )
            continue

        if action.action == "discard":
            continue

    return result
