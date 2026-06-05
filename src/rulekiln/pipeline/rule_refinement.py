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

from rulekiln.providers.contracts import ChatModelClient, ProviderConfig
from rulekiln.schemas.pipeline import CaseEvaluationFailure, SynthesizedRuleSchema
from rulekiln.schemas.task_case import RuleKilnCase

if TYPE_CHECKING:
    from rulekiln.pipeline.failure_analysis import FailureAnalysisResult


class RevisedRuleEntry(BaseModel):
    """A revised rule and its rationale."""

    rule_id: str
    revised_rule: SynthesizedRuleSchema
    rationale: str = ""


class RefinementResult(BaseModel):
    """Output from a single refinement teacher call.

    Contains revised SynthesizedRuleSchema objects keyed by the rule ID they replace,
    plus a per-rule rationale. schema_version follows the Phase 8 artifact-versioning
    convention.
    """

    schema_version: Literal["rulekiln.refinement_result.v1"] = "rulekiln.refinement_result.v1"
    revised_rules: list[RevisedRuleEntry] = Field(default_factory=list)
    reasoning: str | None = None


_REFINEMENT_SYSTEM_PROMPT = """\
You are a task-policy refinement expert performing closed-loop conflict resolution.

You receive:
1. The rules implicated in recent evaluation failures.
2. FAILURE CASES: cases where the student followed the rules but still produced the wrong answer.
3. SUCCESS CASES: cases where the student answered correctly.

Your task:
- Diagnose the root cause of failures.
- Revise ONLY the implicated rules to fix the root cause.
- Your revisions MUST NOT break the provided success cases.
- Leave all other rules untouched.

Return a RefinementResult with a revised_rules list. For each revised rule, include:
- rule_id: the ID of the rule being replaced
- revised_rule: the updated SynthesizedRuleSchema
- rationale: a brief explanation of what you changed and why
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
        "Return a RefinementResult with the revised_rules list."
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
    """Call the teacher to diagnose root causes and emit revised rules.

    This is the empirical refinement step (paper Phase 3, §3.3).
    Works offline with the fake provider (deterministic stub revisions).

    Sampling is deterministic given seed. Only failure cases with real rule
    attribution (not the UNATTRIBUTED sentinel) are sent to the teacher.
    Success cases (fixed) are always included to prevent regressions.
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
        sf for sf in failure_analysis_result.structured_failures if sf.failure_class == "fixed"
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
    return parsed


def apply_refinements(
    current_rules: list[SynthesizedRuleSchema],
    refinement: RefinementResult,
) -> list[SynthesizedRuleSchema]:
    """Apply revised rules from a RefinementResult, replacing rules by ID.

    Unaffected rules are kept unchanged. The rule ID is always preserved so
    downstream indexes remain valid.
    """
    revised_by_id = {entry.rule_id: entry.revised_rule for entry in refinement.revised_rules}
    result: list[SynthesizedRuleSchema] = []
    for rule in current_rules:
        revised = revised_by_id.get(rule.id)
        if revised is not None:
            result.append(revised.model_copy(update={"id": rule.id}))
        else:
            result.append(rule)
    return result
