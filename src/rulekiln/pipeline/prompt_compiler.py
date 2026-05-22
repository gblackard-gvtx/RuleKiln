"""Deterministic prompt compiler: synthesized rules → versioned system prompt."""

from __future__ import annotations

import hashlib
import textwrap

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


def compile_prompt(
    task: RuleKilnTask,
    synthesized_rules: list[SynthesizedRuleSchema],
    strategy: str,
    version: str = "v1",
    additional_instructions: str | None = None,
) -> tuple[str, str]:
    """Compile a deterministic system prompt from synthesized rules.

    Returns (system_prompt, sha256_hex_hash).
    """
    # Sort rules deterministically by (priority, topic) so output is stable
    sorted_rules = sorted(synthesized_rules, key=lambda r: (r.priority, r.topic))

    sections: list[str] = []

    # Header
    sections.append(
        textwrap.dedent(f"""\
        # System Prompt
        # Task: {task.task_name}
        # Strategy: {strategy}
        # Version: {version}

        You are a specialized assistant for the task: **{task.task_name}**.

        {task.description}
        """)
    )

    # Output schema hint
    if task.output_schema:
        sections.append(
            "## Output Format\n"
            f"Always respond with valid JSON matching this schema:\n```json\n{task.output_schema}\n```"
        )

    # Rules
    if sorted_rules:
        total_rules = len(sorted_rules)
        sections.append(
            "## Distilled Rule Policy\n"
            f"The following {total_rules} rule(s) are distilled from observed examples. "
            "Apply them strictly and in order of priority.\n"
        )
        for i, rule in enumerate(sorted_rules, start=1):
            sections.append(_render_rule(i, rule))

    # Additional instructions
    if additional_instructions:
        sections.append(f"## Additional Instructions\n{additional_instructions}")

    system_prompt = "\n\n".join(sections)
    prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    return system_prompt, prompt_hash


def count_tokens_approx(text: str) -> int:
    """Approximate token count: 1 token ≈ 4 characters (conservative)."""
    return max(1, len(text) // 4)
