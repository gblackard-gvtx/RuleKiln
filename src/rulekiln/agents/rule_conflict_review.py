"""Rule conflict review agent: detects and resolves contradictions in synthesized rules."""

from rulekiln.observability.logging import get_logger
from rulekiln.providers.contracts import ChatModelClient, ProviderConfig
from rulekiln.schemas.pipeline import MicroRuleSchema, RuleConflictReview, SynthesizedRuleSchema
from rulekiln.schemas.task_case import RuleKilnTask

logger = get_logger(__name__)

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
    try:
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
    except Exception as exc:
        if _is_output_validation_retry_exhausted(exc):
            logger.warning(
                "conflict_review_fallback_applied",
                synthesized_rule_id=rule.id,
                error=str(exc),
            )
            # Fallback to a conservative keep decision when the provider repeatedly
            # fails to emit schema-valid structured output.
            return RuleConflictReview(
                synthesized_rule_id=rule.id,
                has_conflicts=False,
                conflict_summary=(
                    "Conflict review fallback: provider exhausted output validation retries."
                ),
                conflicting_micro_rule_ids=[],
                resolution="keep",
                resolved_rules=[],
            )
        raise

    # Ensure the rule ID is always set on the result
    return review.model_copy(update={"synthesized_rule_id": rule.id})


def _is_output_validation_retry_exhausted(exc: Exception) -> bool:
    for message in _collect_exception_messages(exc):
        has_retry_exhaustion = "exceeded maximum retries" in message
        has_validation_failure = (
            "output validation" in message or "result validation" in message
        )
        if has_retry_exhaustion and has_validation_failure:
            return True
    return False


def _collect_exception_messages(exc: BaseException) -> list[str]:
    messages: list[str] = []
    seen: set[int] = set()
    pending: list[BaseException] = [exc]

    while pending:
        current = pending.pop()
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)

        message = str(current).strip().lower()
        if message:
            messages.append(message)

        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)

        if current.__cause__ is not None:
            pending.append(current.__cause__)

        if current.__context__ is not None and current.__context__ is not current.__cause__:
            pending.append(current.__context__)

    return messages
