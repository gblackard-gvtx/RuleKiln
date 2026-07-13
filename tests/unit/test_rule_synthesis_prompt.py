"""Tests for the rule synthesis prompt's label-vocabulary and format-instruction
guardrails (item E / I.3) — the source-side half of the defense-in-depth pairing
with prompt_compiler's _sanitize_rule (see test_prompt_compiler_determinism.py).
"""

from __future__ import annotations

from rulekiln.agents.rule_synthesis import (
    _SYNTHESIS_SYSTEM_PROMPT,  # pyright: ignore[reportPrivateUsage]
    _build_synthesis_prompt,  # pyright: ignore[reportPrivateUsage]
)
from rulekiln.schemas.pipeline import MicroRuleSchema
from rulekiln.schemas.task_case import RuleKilnTask


def _task_with_labels() -> RuleKilnTask:
    return RuleKilnTask(
        task_id="contractnli",
        task_name="ContractNLI",
        task_mode="classification",
        description="NLI over contract clauses",
        input_template="{{ premise }} / {{ hypothesis }}",
        output_schema={
            "type": "object",
            "required": ["label"],
            "properties": {
                "label": {"type": "string", "enum": ["entailment", "contradiction", "neutral"]}
            },
        },
    )


def _micro_rule() -> MicroRuleSchema:
    return MicroRuleSchema(
        topic="verbal grant",
        condition="clause grants a license orally",
        expected_outcome="entailment",
        positive_cues=["oral", "verbal"],
        negative_cues=[],
    )


def test_synthesis_system_prompt_forbids_output_format_instructions() -> None:
    assert "response format" in _SYNTHESIS_SYSTEM_PROMPT.lower()
    assert "explain your reasoning" in _SYNTHESIS_SYSTEM_PROMPT.lower()


def test_synthesis_system_prompt_forbids_invented_label_variants() -> None:
    assert "entailment_verbal_included" in _SYNTHESIS_SYSTEM_PROMPT.lower()
    assert "never invent label variants" in _SYNTHESIS_SYSTEM_PROMPT.lower()


def test_synthesis_prompt_enumerates_allowed_labels_when_task_has_output_schema() -> None:
    prompt = _build_synthesis_prompt(_task_with_labels(), "verbal grants", [_micro_rule()])
    assert "Allowed output labels for this task:" in prompt
    for label in ("entailment", "contradiction", "neutral"):
        assert label in prompt


def test_synthesis_prompt_omits_label_line_without_output_schema() -> None:
    task = _task_with_labels().model_copy(update={"output_schema": {}})
    prompt = _build_synthesis_prompt(task, "verbal grants", [_micro_rule()])
    assert "Allowed output labels for this task:" not in prompt
