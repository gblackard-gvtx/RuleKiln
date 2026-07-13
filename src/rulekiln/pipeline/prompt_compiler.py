"""Deterministic prompt compiler: task definition → baseline prompt, rules → hardened prompt."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from rulekiln.observability.logging import get_logger
from rulekiln.schemas.pipeline import SynthesizedRuleSchema
from rulekiln.schemas.task_case import RuleKilnTask

logger = get_logger(__name__)

_MAX_CUES = 5  # max positive/negative cues listed per rule

# Synonyms the teacher/synthesis model occasionally emits in OutcomeCondition.outcome
# instead of the task's canonical enum value.
_OUTCOME_SYNONYMS: dict[str, str] = {
    "entails": "entailment",
    "entailed": "entailment",
    "contradicts": "contradiction",
    "contradicted": "contradiction",
}

# Phrases that indicate a synthesized rule is instructing the model about *response
# format* rather than describing classification/decision conditions. Rules must never
# alter the output schema.
_FORBIDDEN_INSTRUCTION_PATTERNS = [
    re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
    for phrase in (
        "explain",
        "justify",
        "analyze",
        "compare",
        "state why",
        "return json",
        "provide reasoning",
        "answer with",
    )
]


def _canonical_outcome_label(value: str, allowed_labels: frozenset[str] | None) -> str | None:
    """Resolve a rule outcome's free-text value to one of the task's allowed output labels.

    Returns None when the value can't be resolved — the caller should drop that outcome
    line rather than guess a mapping, since injecting a wrong label into the compiled
    prompt is worse than omitting the (already partially redundant) guidance.
    """
    if allowed_labels is None:
        return value
    normalized = value.strip().lower()
    if normalized in allowed_labels:
        return normalized
    mapped = _OUTCOME_SYNONYMS.get(normalized)
    if mapped is not None and mapped in allowed_labels:
        return mapped
    return None


def _contains_forbidden_instruction(text: str) -> str | None:
    """Return the matched forbidden phrase if `text` contains output-format/meta instructions."""
    for pattern in _FORBIDDEN_INSTRUCTION_PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    return None


def _normalize_compound_label_mentions(text: str, allowed_labels: frozenset[str] | None) -> str:
    """Rewrite compound mnemonic labels (e.g. 'entailment_verbal_included') found inside
    free-text guidance down to their canonical label ('entailment').

    The synthesis model sometimes writes these mnemonics directly into applies_when,
    tie_breakers, or outcome `when` prose — not just as outcome_conditions dict keys.
    Left as-is, that prose leaks the compound string into the compiled prompt, and the
    student model echoes it back verbatim as its final answer (the same failure mode
    _canonical_outcome_label guards against for the dict key, applied here to prose).
    """
    if not allowed_labels or not text:
        return text
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(label) for label in allowed_labels) + r")(?:_\w+)+\b"
    )
    return pattern.sub(lambda m: m.group(1), text)


def _sanitize_rule(
    rule: SynthesizedRuleSchema, allowed_labels: frozenset[str] | None = None
) -> SynthesizedRuleSchema:
    """Strip forbidden instruction-like entries and normalize compound label mentions
    in a rule's free-text condition fields.

    Rules must guide classification reasoning only, never alter the output schema. This
    is defense-in-depth against a synthesis call still emitting phrases like "explain
    your reasoning" or "return JSON", or leaking a compound mnemonic label into prose,
    despite the system prompt forbidding both. Only the offending list entries are
    dropped, not the whole rule. `topic` is a single required string (used as the rule
    header) and can't be filtered the same way; a match there is only logged.
    """
    if matched := _contains_forbidden_instruction(rule.topic):
        logger.warning(
            "rule_topic_forbidden_instruction",
            rule_id=rule.id,
            rule_topic=rule.topic,
            matched_phrase=matched,
        )

    applies_when: list[str] = []
    for raw_cond in rule.applies_when:
        cond = _normalize_compound_label_mentions(raw_cond, allowed_labels)
        if matched := _contains_forbidden_instruction(cond):
            logger.warning(
                "rule_applies_when_forbidden_instruction",
                rule_id=rule.id,
                rule_topic=rule.topic,
                matched_phrase=matched,
                condition=cond,
            )
            continue
        applies_when.append(cond)

    tie_breakers: list[str] = []
    for raw_tb in rule.tie_breakers:
        tb = _normalize_compound_label_mentions(raw_tb, allowed_labels)
        if matched := _contains_forbidden_instruction(tb):
            logger.warning(
                "rule_tie_breaker_forbidden_instruction",
                rule_id=rule.id,
                rule_topic=rule.topic,
                matched_phrase=matched,
                tie_breaker=tb,
            )
            continue
        tie_breakers.append(tb)

    outcome_conditions = {}
    for outcome_name, oc in rule.outcome_conditions.items():
        when: list[str] = []
        for raw_cond in oc.when:
            cond = _normalize_compound_label_mentions(raw_cond, allowed_labels)
            if matched := _contains_forbidden_instruction(cond):
                logger.warning(
                    "rule_outcome_when_forbidden_instruction",
                    rule_id=rule.id,
                    rule_topic=rule.topic,
                    outcome_name=outcome_name,
                    matched_phrase=matched,
                    condition=cond,
                )
                continue
            when.append(cond)
        outcome_conditions[outcome_name] = oc.model_copy(update={"when": when})

    return rule.model_copy(
        update={
            "applies_when": applies_when,
            "tie_breakers": tie_breakers,
            "outcome_conditions": outcome_conditions,
        }
    )


def _render_rule(
    index: int,
    rule: SynthesizedRuleSchema,
    allowed_labels: frozenset[str] | None = None,
) -> str:
    lines: list[str] = [f"## Rule {index}: {rule.topic}"]

    if rule.applies_when:
        lines.append("**Applies when:**")
        for cond in rule.applies_when:
            lines.append(f"- {cond}")

    if rule.outcome_conditions:
        lines.append("**Outcomes:**")
        for outcome_name, oc in rule.outcome_conditions.items():
            label = _canonical_outcome_label(oc.outcome, allowed_labels)
            if label is None:
                logger.warning(
                    "rule_outcome_label_unresolvable",
                    rule_id=rule.id,
                    rule_topic=rule.topic,
                    outcome_key=outcome_name,
                    outcome_value=oc.outcome,
                    allowed_labels=sorted(allowed_labels) if allowed_labels else None,
                )
                continue
            lines.append(f"- **{label}**: {', '.join(oc.when) if oc.when else 'default'}")

    if rule.tie_breakers:
        lines.append("**Tie-breakers (in order):**")
        for tb in rule.tie_breakers:
            lines.append(f"- {tb}")

    return "\n".join(lines)


def _extract_enum_values(output_schema: dict[str, Any]) -> list[str]:
    """Collect all enum string values from the output schema's property definitions."""
    enums: list[str] = []
    properties = output_schema.get("properties", {})
    for prop_schema in properties.values():
        if isinstance(prop_schema, dict):
            enum_vals = prop_schema.get("enum", [])
            if isinstance(enum_vals, list):
                enums.extend(str(v) for v in enum_vals)
    return enums


def _render_output_schema_example(output_schema: dict[str, Any]) -> str:
    """Render a concise JSON placeholder from the output schema's required fields."""
    required: list[str] = output_schema.get("required", [])
    properties: dict[str, Any] = output_schema.get("properties", {})
    example: dict[str, str] = {}
    for field_name in required:
        prop_schema = properties.get(field_name, {})
        if isinstance(prop_schema, dict) and prop_schema.get("enum"):
            example[field_name] = "<one allowed value>"
        else:
            example[field_name] = f"<{field_name}>"
    return json.dumps(example, indent=2)


def compile_baseline_prompt(task: RuleKilnTask) -> str:
    """Compile a deterministic baseline prompt from task definition fields.

    Contains no distilled rules — only the task scaffold, description, input template,
    output schema, allowed values, and prompt-injection boundary.
    """
    policy = task.baseline_prompt_policy
    scaffold = task.prompt_scaffold
    sections: list[str] = []

    # Role
    role: str = scaffold.get("role", "") or ""
    if policy.include_role and role:
        sections.append(f"# Role\n\n{role.strip()}")

    # Task description
    if policy.include_task_description and task.description:
        sections.append(f"# Task\n\n{task.description.strip()}")

    # Input template
    if policy.include_input_template and task.input_template:
        sections.append(f"# Input\n\n{task.input_template.strip()}")

    # Output format
    if policy.include_output_schema and task.output_schema:
        example = _render_output_schema_example(task.output_schema)
        sections.append(
            "# Output Format\n\n"
            "Return only valid JSON matching this schema:\n\n"
            f"```json\n{example}\n```"
        )

    # Allowed values (enum list)
    if policy.include_allowed_values and task.output_schema:
        enum_vals = _extract_enum_values(task.output_schema)
        if enum_vals:
            label_list = "\n".join(f"- {v}" for v in enum_vals)
            sections.append(
                f"# Allowed Values\n\nThe output field must be exactly one of:\n\n{label_list}"
            )

    # Rules: task_scope + non_scope combined
    if policy.include_prompt_scaffold:
        task_scope: list[str] = scaffold.get("task_scope") or []
        non_scope: list[str] = scaffold.get("non_scope") or []
        all_rules = [str(r) for r in task_scope] + [str(r) for r in non_scope]
        if all_rules:
            rule_lines = "\n".join(f"- {r}" for r in all_rules)
            sections.append(f"# Rules\n\n{rule_lines}")

    # Input boundary
    if policy.include_input_boundary:
        boundary: list[str] = scaffold.get("prompt_injection_boundary") or []
        if boundary:
            boundary_lines = "\n".join(f"- {b}" for b in boundary)
            sections.append(f"# Input Boundary\n\n{boundary_lines}")

    return "\n\n".join(sections)


def compile_prompt(
    task: RuleKilnTask,
    synthesized_rules: list[SynthesizedRuleSchema],
    strategy: str,
    version: str = "v1",
    additional_instructions: str | None = None,
) -> tuple[str, str]:
    """Compile a deterministic distilled system prompt from synthesized rules.

    The prompt is the baseline scaffold plus a distilled rule policy section.
    Returns (system_prompt, sha256_hex_hash).
    """
    allowed_labels: frozenset[str] | None = None
    if task.output_schema:
        allowed_labels = frozenset(_extract_enum_values(task.output_schema)) or None

    # Sort rules deterministically by (priority, topic) so output is stable
    sorted_rules = sorted(synthesized_rules, key=lambda r: (r.priority, r.topic))
    sanitized_rules = [_sanitize_rule(r, allowed_labels) for r in sorted_rules]

    sections: list[str] = [compile_baseline_prompt(task)]

    if sanitized_rules:
        total_rules = len(sanitized_rules)
        rule_header = (
            f"# Distilled Rule Policy (strategy: {strategy}, version: {version})\n\n"
            f"The following {total_rules} rule(s) are distilled from observed examples. "
            "Apply them strictly and in order of priority."
        )
        rule_blocks = [rule_header] + [
            _render_rule(i, r, allowed_labels) for i, r in enumerate(sanitized_rules, 1)
        ]
        sections.append("\n\n".join(rule_blocks))

    if additional_instructions:
        sections.append(f"# Additional Instructions\n\n{additional_instructions}")

    # Reinforce the exact same output contract as compile_baseline_prompt's "# Output
    # Format" / "# Allowed Values" sections, repeated here so it stays close to the end
    # of the prompt after a potentially long distilled-rules block. Same contract, not a
    # new one — recency, not a format change. Gated on the same policy flags as the
    # baseline sections so we never add contract text a task deliberately opted out of.
    policy = task.baseline_prompt_policy
    if allowed_labels and policy.include_output_schema and policy.include_allowed_values:
        example = _render_output_schema_example(task.output_schema)
        label_list = "\n".join(f"- {v}" for v in sorted(allowed_labels))
        sections.append(
            "# Final Answer Format (Required)\n\n"
            "Return only valid JSON matching this schema:\n\n"
            f"```json\n{example}\n```\n\n"
            f"The output field must be exactly one of:\n\n{label_list}\n\n"
            "Do not include explanations, markdown, or prose outside the JSON object."
        )

    system_prompt = "\n\n".join(sections)
    prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    return system_prompt, prompt_hash


def count_tokens_approx(text: str) -> int:
    """Approximate token count: 1 token ≈ 4 characters (conservative)."""
    return max(1, len(text) // 4)


def estimate_rule_tokens(rule: SynthesizedRuleSchema) -> int:
    """Approximate the token cost of a rule as it will actually be rendered.

    Mirrors _render_rule's output (topic, applies_when, outcome_conditions,
    tie_breakers) so pruning-time estimates match the compiled prompt.
    """
    return count_tokens_approx(_render_rule(0, rule))
