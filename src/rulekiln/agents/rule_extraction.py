"""Pydantic AI rule extraction agent wrapper."""

from rulekiln.providers.contracts import ChatModelClient, ProviderConfig
from rulekiln.schemas.pipeline import ExtractionOutput
from rulekiln.schemas.task_case import RuleKilnCase, RuleKilnTask

_EXTRACTION_SYSTEM_PROMPT = """\
You are a rule extraction specialist. Given a task definition and a case example with its \
expected output, extract the explicit decision rules that a model must follow \
to produce that output.

For each rule produce:
- topic: a short label for the rule domain (e.g. "date_format", "tone", "label_selection")
- condition: the IF condition that triggers this rule
- expected_outcome: THEN what the model should do or output
- rationale_summary: brief explanation of why this rule applies
- positive_cues: list of signals in the input that confirm this rule applies
- negative_cues: list of signals that indicate this rule does NOT apply
- rule_type: one of "decision", "constraint", "format", "fallback"

Return only rules that are clearly demonstrated by the case. Do not invent rules not \
supported by evidence.
"""


def _build_extraction_prompt(task: RuleKilnTask, case: RuleKilnCase) -> str:
    return (
        f"Task: {task.task_name}\n"
        f"Mode: {task.task_mode}\n"
        f"Description: {task.description}\n\n"
        f"Input:\n{case.input}\n\n"
        f"Expected output:\n{case.expected}\n\n"
        "Extract the rules."
    )


async def extract_rules_for_case(
    task: RuleKilnTask,
    case: RuleKilnCase,
    chat_client: ChatModelClient,
    config: ProviderConfig,
) -> ExtractionOutput:
    """Call the teacher model to extract micro-rules for a single case."""
    user_prompt = _build_extraction_prompt(task, case)
    result = await chat_client.complete_structured(
        system_prompt=_EXTRACTION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        output_schema=ExtractionOutput,
        config=config,
    )
    parsed = result.parsed
    if not isinstance(parsed, ExtractionOutput):
        return ExtractionOutput.model_validate(parsed.model_dump() if parsed else {})
    return parsed
