"""Pydantic AI rule synthesis agent wrapper."""

from rulekiln.providers.contracts import ChatModelClient, ProviderConfig
from rulekiln.schemas.pipeline import MicroRuleSchema, SynthesisOutput
from rulekiln.schemas.task_case import RuleKilnTask

_SYNTHESIS_SYSTEM_PROMPT = """\
You are a rule synthesis specialist. Given a cluster of related micro-rules extracted from \
multiple cases, synthesize them into a single coherent rule that generalizes across all \
cases in the cluster.

Produce:
- topic: a concise label for the synthesized rule's domain
- applies_when: list of conditions under which this rule applies
- outcome_conditions: mapping from outcome name to its triggering conditions
- tie_breakers: ordered list of tie-breaking conditions when outcomes are ambiguous
- priority: integer (lower = higher priority); default 100

The synthesized rule should be deterministic, minimal, and correct. \
Do not introduce conditions not supported by the input micro-rules.
"""


def _build_synthesis_prompt(
    task: RuleKilnTask,
    cluster_topic: str | None,
    rules: list[MicroRuleSchema],
) -> str:
    rules_text = "\n\n".join(
        f"Rule {i + 1}:\n"
        f"  topic: {r.topic}\n"
        f"  condition: {r.condition}\n"
        f"  expected_outcome: {r.expected_outcome}\n"
        f"  positive_cues: {r.positive_cues}\n"
        f"  negative_cues: {r.negative_cues}"
        for i, r in enumerate(rules)
    )
    return (
        f"Task: {task.task_name}\n"
        f"Cluster topic: {cluster_topic or 'unknown'}\n\n"
        f"Micro-rules in this cluster:\n\n{rules_text}\n\n"
        "Synthesize these into one generalized rule."
    )


async def synthesize_cluster(
    task: RuleKilnTask,
    cluster_topic: str | None,
    micro_rules: list[MicroRuleSchema],
    source_case_ids: list[str],
    source_micro_rule_ids: list[str],
    chat_client: ChatModelClient,
    config: ProviderConfig,
) -> SynthesisOutput:
    """Call the teacher model to synthesize a cluster of micro-rules into one rule."""
    user_prompt = _build_synthesis_prompt(task, cluster_topic, micro_rules)
    result = await chat_client.complete_structured(
        system_prompt=_SYNTHESIS_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        output_schema=SynthesisOutput,
        config=config,
    )
    parsed = result.parsed
    if not isinstance(parsed, SynthesisOutput):
        output = SynthesisOutput.model_validate(parsed.model_dump() if parsed else {})
    else:
        output = parsed
    # Inject provenance into each rule
    for rule in output.rules:
        rule.source_case_ids = source_case_ids
        rule.source_micro_rule_ids = source_micro_rule_ids
    return output
