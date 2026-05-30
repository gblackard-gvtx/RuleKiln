"""Deterministic prompt compiler: task definition → baseline prompt, rules → hardened prompt."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from rulekiln.schemas.pipeline import SynthesizedRuleSchema
from rulekiln.schemas.task_case import RuleKilnTask

_MAX_CUES = 5  # max positive/negative cues listed per rule


def _render_rule(index: int, rule: SynthesizedRuleSchema) -> str:
    lines: list[str] = [f"## Rule {index}: {rule.topic}"]

    if rule.applies_when:
        lines.append("**Applies when:**")
        for cond in rule.applies_when:
            lines.append(f"- {cond}")

    if rule.outcome_conditions:
        lines.append("**Outcomes:**")
        for outcome_name, oc in rule.outcome_conditions.items():
            lines.append(f"- **{outcome_name}**: {', '.join(oc.when) if oc.when else 'default'}")

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
    # Sort rules deterministically by (priority, topic) so output is stable
    sorted_rules = sorted(synthesized_rules, key=lambda r: (r.priority, r.topic))

    sections: list[str] = [compile_baseline_prompt(task)]

    if sorted_rules:
        total_rules = len(sorted_rules)
        rule_header = (
            f"# Distilled Rule Policy (strategy: {strategy}, version: {version})\n\n"
            f"The following {total_rules} rule(s) are distilled from observed examples. "
            "Apply them strictly and in order of priority."
        )
        rule_blocks = [rule_header] + [_render_rule(i, r) for i, r in enumerate(sorted_rules, 1)]
        sections.append("\n\n".join(rule_blocks))

    if additional_instructions:
        sections.append(f"# Additional Instructions\n\n{additional_instructions}")

    system_prompt = "\n\n".join(sections)
    prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    return system_prompt, prompt_hash


def count_tokens_approx(text: str) -> int:
    """Approximate token count: 1 token ≈ 4 characters (conservative)."""
    return max(1, len(text) // 4)
