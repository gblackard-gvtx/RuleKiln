"""Rule conflict review agent: detects and resolves contradictions in synthesized rules."""

from rulekiln.providers.contracts import ChatModelClient, ProviderConfig
from rulekiln.schemas.pipeline import MicroRuleSchema, RuleConflictReview, SynthesizedRuleSchema
from rulekiln.schemas.task_case import RuleKilnTask

_CONFLICT_SYSTEM_PROMPT = """\
You review synthesized task-policy rules for contradictions.

Given:
- a synthesized rule
- supporting micro-rules
- task output schema
- task mode

Determine whether the synthesized rule contains conflicts.

A conflict exists when:
- similar conditions map to incompatible outcomes
- the rule contains mutually exclusive instructions
- exceptions contradict the main condition
- the output path is ambiguous
- the rule would cause inconsistent student behavior

Return one of:
- keep: no material conflict — rule is correct as-is
- modify: rewrite into one clean resolved rule
- split: split into multiple clean non-conflicting rules
- discard: conflicts cannot be safely resolved

If resolution is "modify" or "split", include the resolved rule(s) in resolved_rules.
If resolution is "discard", leave resolved_rules empty.
"""


def _build_conflict_review_prompt(
    task: RuleKilnTask,
    rule: SynthesizedRuleSchema,
    micro_rules: list[MicroRuleSchema],
) -> str:
    applies_when = "\n".join(f"  - {c}" for c in rule.applies_when)
    outcomes = "\n".join(
        f"  {name}: {', '.join(oc.when) or 'default'}"
        for name, oc in rule.outcome_conditions.items()
    )
    tie_breakers = "\n".join(f"  - {tb}" for tb in rule.tie_breakers)

    micros_text = "\n\n".join(
        f"Micro-rule {i + 1}:\n"
        f"  topic: {r.topic}\n"
        f"  condition: {r.condition}\n"
        f"  expected_outcome: {r.expected_outcome}"
        for i, r in enumerate(micro_rules)
    )

    return (
        f"Task: {task.task_name} ({task.task_mode})\n"
        f"Output schema: {task.output_schema}\n\n"
        f"Synthesized rule ID: {rule.id}\n"
        f"Topic: {rule.topic}\n"
        f"Applies when:\n{applies_when}\n"
        f"Outcomes:\n{outcomes}\n"
        f"Tie-breakers:\n{tie_breakers}\n\n"
        f"Supporting micro-rules ({len(micro_rules)}):\n\n{micros_text}\n\n"
        "Does this synthesized rule contain conflicts? Respond with a RuleConflictReview."
    )


async def review_rule_for_conflicts(
    task: RuleKilnTask,
    rule: SynthesizedRuleSchema,
    micro_rules: list[MicroRuleSchema],
    chat_client: ChatModelClient,
    config: ProviderConfig,
) -> RuleConflictReview:
    """Call the teacher model to review a synthesized rule for conflicts."""
    user_prompt = _build_conflict_review_prompt(task, rule, micro_rules)
    result = await chat_client.complete_structured(
        system_prompt=_CONFLICT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        output_schema=RuleConflictReview,
        config=config,
    )
    parsed = result.parsed
    if not isinstance(parsed, RuleConflictReview):
        review = RuleConflictReview.model_validate(parsed.model_dump() if parsed else {})
    else:
        review = parsed
    # Ensure the rule ID is always set on the result
    return review.model_copy(update={"synthesized_rule_id": rule.id})
