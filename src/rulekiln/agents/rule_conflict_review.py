"""Static rule review agent: detects and resolves linguistic contradictions in synthesized rules.

This is a pre-evaluation hygiene pass that checks each rule for internal inconsistencies
before student evaluation runs. It is NOT the paper's Phase 3 closed-loop conflict
resolution — that iterative, case-outcome-based process is implemented in rule_refinement.py.

``apply_conflict_reviews`` is provided as a standalone function but is NOT called in the
main distillation pipeline — static reviews are advisory only. The pipeline records conflict
metadata on each synthesized rule (has_conflicts, conflict_summary) and uses that signal
during pruning, but does not apply keep/modify/split/discard at synthesis time.
"""

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
    """Static rule review: call the teacher to check a synthesized rule for linguistic conflicts.

    Performs no student inference and does not inspect case outcomes. This is purely
    a pre-evaluation hygiene check. For the empirical, case-outcome-based refinement
    (closed-loop conflict resolution), see rule_refinement.refine_rules_with_teacher.
    """
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
            # review_status="fallback_validation_failed" distinguishes this from a
            # real "no conflict" review so callers can treat it differently.
            return RuleConflictReview(
                synthesized_rule_id=rule.id,
                has_conflicts=False,
                conflict_summary=(
                    "Conflict review fallback: provider exhausted output validation retries."
                ),
                conflicting_micro_rule_ids=[],
                resolution="keep",
                resolved_rules=[],
                review_status="fallback_validation_failed",
            )
        raise

    # Ensure the rule ID is always set on the result
    return review.model_copy(update={"synthesized_rule_id": rule.id})


def _is_output_validation_retry_exhausted(exc: Exception) -> bool:
    for message in _collect_exception_messages(exc):
        has_retry_exhaustion = "exceeded maximum retries" in message
        has_validation_failure = "output validation" in message or "result validation" in message
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


def apply_conflict_reviews(
    current_rules: list[SynthesizedRuleSchema],
    reviews: list[RuleConflictReview],
) -> list[SynthesizedRuleSchema]:
    """Apply static conflict review decisions to a rule set.

    Mirrors the semantics of ``apply_refinements`` in rule_refinement.py.

    NOTE: this function is NOT called by the main distillation pipeline. Static
    conflict reviews are advisory only — the pipeline records conflict metadata
    (has_conflicts, conflict_summary) on each rule but does not apply
    keep/modify/split/discard transformations at synthesis time. Call this
    function explicitly if you want to materialise the review decisions.

    Actions:
    - keep:    rule is kept unchanged.
    - modify:  rule is replaced by the single entry in resolved_rules, preserving ID.
    - split:   rule is replaced by all entries in resolved_rules with deterministic IDs.
    - discard: rule is removed.
    """
    reviews_by_id = {r.synthesized_rule_id: r for r in reviews}
    result: list[SynthesizedRuleSchema] = []

    for rule in current_rules:
        review = reviews_by_id.get(rule.id)

        if review is None or review.resolution == "keep":
            result.append(rule)
            continue

        if review.resolution == "modify":
            if len(review.resolved_rules) != 1:
                logger.warning(
                    "conflict_review_invalid_modify",
                    rule_id=rule.id,
                    resolved_rule_count=len(review.resolved_rules),
                )
                result.append(rule)
            else:
                result.append(review.resolved_rules[0].model_copy(update={"id": rule.id}))
            continue

        if review.resolution == "split":
            if not review.resolved_rules:
                logger.warning("conflict_review_invalid_split_empty", rule_id=rule.id)
                result.append(rule)
            else:
                for idx, resolved_rule in enumerate(review.resolved_rules, start=1):
                    result.append(
                        resolved_rule.model_copy(update={"id": f"{rule.id}__split_{idx}"})
                    )
            continue

        if review.resolution == "discard":
            continue

    return result
