"""Unit tests for the few-shot k5 transition separator (Run 4 fix).

Guards the strong instruction break inserted between the few-shot examples and
the test-case user message that stops the student from continuing the example
sequence / echoing the input at k=5.
"""

from rulekiln.pipeline.baseline_strategies import (
    render_few_shot_examples,
    render_few_shot_transition,
)
from rulekiln.pipeline.evaluator import (
    FEW_SHOT_ANSWER_CUE,
    build_student_user_message,
)
from rulekiln.schemas.task_case import RuleKilnCase

# Mirrors examples/datasets/contractnli/task.yaml input_template.
_CONTRACTNLI_TEMPLATE = (
    "Contract clause:\n{{ premise }}\n\nLegal statement:\n{{ hypothesis }}"
)
_PREMISE = "All Confidential Information shall be returned on termination."
_HYPOTHESIS = "The Receiving Party may retain some Confidential Information."


def _k5_user_message() -> str:
    return build_student_user_message(
        _CONTRACTNLI_TEMPLATE,
        {"premise": _PREMISE, "hypothesis": _HYPOTHESIS},
        transition_prefix=render_few_shot_transition(5),
    )


def test_separator_appears_exactly_twice() -> None:
    message = _k5_user_message()
    assert message.count("===") == 2


def test_do_not_continue_instruction_present() -> None:
    message = _k5_user_message()
    assert "Do not continue the pattern" in message
    assert "The 5 examples above" in message


def test_test_case_input_appears_after_second_separator() -> None:
    message = _k5_user_message()
    second_separator_end = message.rindex("===") + len("===")
    tail = message[second_separator_end:]
    # The rendered test case (premise/hypothesis) must live after the closing ===,
    # not before it.
    assert _PREMISE in tail
    assert _HYPOTHESIS in tail
    assert _PREMISE not in message[:second_separator_end]


def test_input_uses_template_rendering_not_json_dumps() -> None:
    message = _k5_user_message()
    # Prefixed with the separator, so it starts with the break, never a JSON object.
    assert not message.startswith("{")
    # input_template rendering yields prose ("Contract clause:" / "Legal statement:"),
    # never a top-level JSON object with a "premise" key.
    assert "Contract clause:" in message
    assert '"premise"' not in message


def test_message_ends_with_output_answer_cue() -> None:
    # The examples train the model to emit the label after an `Output:` cue; the
    # test-case message must end with the same cue so the model completes that slot
    # instead of continuing the input prose.
    message = _k5_user_message()
    assert message.rstrip().endswith(FEW_SHOT_ANSWER_CUE)
    # The cue comes after the rendered test input, not before it.
    assert message.index(_HYPOTHESIS) < message.rindex(FEW_SHOT_ANSWER_CUE)


def test_answer_cue_matches_example_output_label() -> None:
    # Guard the invariant that the appended cue matches the literal `Output:` label
    # that render_few_shot_examples ends every example with.
    examples = [
        RuleKilnCase(
            id="ex1",
            split="train",
            task_mode="classification",
            input={"premise": _PREMISE, "hypothesis": _HYPOTHESIS},
            expected={"label": "entailment"},
        )
    ]
    rendered_examples = render_few_shot_examples(examples)
    assert f"{FEW_SHOT_ANSWER_CUE}\n" in rendered_examples


def test_baseline_without_prefix_is_plain_rendered_input() -> None:
    # No transition prefix (baseline / scaffold): user message is the rendered
    # template only, unchanged by this fix (no separator, no answer cue).
    message = build_student_user_message(
        _CONTRACTNLI_TEMPLATE,
        {"premise": _PREMISE, "hypothesis": _HYPOTHESIS},
    )
    assert "===" not in message
    assert not message.rstrip().endswith(FEW_SHOT_ANSWER_CUE)
    assert message.startswith("Contract clause:")
