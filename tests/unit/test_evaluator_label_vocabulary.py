"""Tests for label-vocabulary enforcement in the evaluator (ContractNLI few-shot bug).

Observed: few-shot strategies echoed the input back (e.g. the whole premise
text, or a {"premise": ..., "hypothesis": ...} dict). The parser accepted these
as "valid" predictions with malformed=False, so every unique garbage string
became its own zero-F1 class and macro F1 collapsed to ~0.02 while accuracy
stayed ~0.35 and malformed reported 0%.

Fixed behavior: for classification/routing tasks with an enum label vocabulary,
a parseable output whose label is not in the vocabulary is marked malformed.
Accuracy and macro F1 are then computed from the same normalized prediction
column, over real labels only.
"""

from __future__ import annotations

import pytest

from rulekiln.pipeline.evaluator import (
    _allowed_labels_from_task,  # pyright: ignore[reportPrivateUsage]
    _score_case,  # pyright: ignore[reportPrivateUsage]
    _try_recover_label,  # pyright: ignore[reportPrivateUsage]
    evaluate_prompt,
)
from rulekiln.providers.contracts import ChatModelClient, ProviderConfig
from rulekiln.schemas.task_case import (
    EvaluationAssertion,
    EvaluationSpec,
    RuleKilnCase,
    RuleKilnTask,
)
from rulekiln.schemas.usage import ChatCompletionResult, ModelUsage

_LABELS = frozenset({"entailment", "contradiction", "neutral"})


def _task() -> RuleKilnTask:
    return RuleKilnTask(
        task_id="contractnli",
        task_name="ContractNLI",
        task_mode="classification",
        description="NLI over contract clauses",
        input_template="Contract clause:\n{{ premise }}\n\nLegal statement:\n{{ hypothesis }}",
        output_schema={
            "type": "object",
            "required": ["label"],
            "properties": {
                "label": {"type": "string", "enum": ["entailment", "contradiction", "neutral"]}
            },
        },
    )


def _case(case_id: str, label: str) -> RuleKilnCase:
    return RuleKilnCase(
        id=case_id,
        split="train",
        task_mode="classification",
        input={"premise": f"premise {case_id}", "hypothesis": f"hypothesis {case_id}"},
        expected={"label": label},
        evaluation=EvaluationSpec(
            assertions=[
                EvaluationAssertion(type="must_equal", path="$.label", value=label, weight=1.0)
            ]
        ),
    )


# ── allowed-label extraction ──────────────────────────────────────────────────


def test_allowed_labels_extracted_from_output_schema_enum() -> None:
    assert _allowed_labels_from_task(_task()) == _LABELS


def test_allowed_labels_none_without_enum() -> None:
    task = _task().model_copy(update={"output_schema": {"properties": {"label": {}}}})
    assert _allowed_labels_from_task(task) is None


def test_allowed_labels_none_for_non_classification_mode() -> None:
    task = _task().model_copy(update={"task_mode": "summarization"})
    assert _allowed_labels_from_task(task) is None


# ── _score_case vocabulary enforcement ────────────────────────────────────────


def test_valid_wrong_label_is_not_malformed() -> None:
    result = _score_case(
        _case("c1", "entailment"), {"label": "neutral"}, malformed=False, allowed_labels=_LABELS
    )
    assert result.malformed is False
    assert result.passed is False


def test_echoed_input_dict_is_malformed() -> None:
    echoed: dict[str, str | int | float | bool | None] = {
        "premise": "Confidential Information ...",
        "hypothesis": "Receiving Party may ...",
    }
    result = _score_case(
        _case("c1", "entailment"), echoed, malformed=False, allowed_labels=_LABELS
    )
    assert result.malformed is True
    assert result.error == "prediction_not_in_allowed_labels"


def test_echoed_prose_string_is_malformed() -> None:
    result = _score_case(
        _case("c1", "entailment"),
        "Some obligations of Agreement may survive termination.",
        malformed=False,
        allowed_labels=_LABELS,
    )
    assert result.malformed is True


def test_bare_valid_label_string_is_accepted() -> None:
    result = _score_case(
        _case("c1", "entailment"), "entailment", malformed=False, allowed_labels=_LABELS
    )
    assert result.malformed is False


def test_no_vocabulary_keeps_previous_behavior() -> None:
    result = _score_case(
        _case("c1", "entailment"), "arbitrary text", malformed=False, allowed_labels=None
    )
    assert result.malformed is False


# ── parser recovery of a single unambiguous label (items I.4, I.5) ────────────


def test_recover_exact_valid_label() -> None:
    assert _try_recover_label("entailment", _LABELS) == "entailment"


def test_recover_label_with_trailing_punctuation() -> None:
    assert _try_recover_label("entailment.", _LABELS) == "entailment"


def test_recover_label_with_different_casing() -> None:
    assert _try_recover_label("Entailment", _LABELS) == "entailment"


def test_recover_label_with_surrounding_whitespace() -> None:
    assert _try_recover_label("  neutral  \n", _LABELS) == "neutral"


def test_recover_label_wrapped_in_explanation() -> None:
    text = "Based on the clause, the correct answer is contradiction."
    assert _try_recover_label(text, _LABELS) == "contradiction"


def test_recover_returns_none_for_multiple_distinct_labels() -> None:
    """Two different allowed labels both present → ambiguous, must stay malformed."""
    text = "This could be entailment or neutral depending on interpretation."
    assert _try_recover_label(text, _LABELS) is None


def test_recover_returns_none_for_no_label() -> None:
    assert _try_recover_label("I cannot determine the relationship.", _LABELS) is None


def test_recover_does_not_match_compound_leaked_label() -> None:
    """Regression test tied directly to the root-cause bug: a compiled prompt that
    leaked 'entailment_verbal_included' as the rendered label must NOT recover to
    'entailment' via substring matching — '_' is a regex word character, so there is
    no word boundary between 'entailment' and the following '_verbal_included'."""
    assert _try_recover_label("entailment_verbal_included", _LABELS) is None


def test_score_case_recovers_single_valid_label_with_extra_text() -> None:
    result = _score_case(
        _case("c1", "entailment"),
        "entailment.",
        malformed=False,
        allowed_labels=_LABELS,
        raw_content="entailment.",
    )
    assert result.malformed is False
    assert result.recovered is True
    assert result.passed is True


def test_score_case_does_not_recover_multiple_conflicting_labels() -> None:
    result = _score_case(
        _case("c1", "entailment"),
        "entailment or contradiction",
        malformed=False,
        allowed_labels=_LABELS,
        raw_content="entailment or contradiction",
    )
    assert result.malformed is True
    assert result.recovered is False


# ── end to end: accuracy and macro F1 from the same normalized predictions ────


class _ScriptedClient(ChatModelClient):
    """Returns a scripted raw-JSON response per call, in case order."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._index = 0

    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type,
        config: ProviderConfig,
    ) -> ChatCompletionResult:
        text = self._responses[self._index]
        self._index += 1
        return ChatCompletionResult(
            content=text,
            parsed=None,
            usage=ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            raw_model="fake",
        )


def _config() -> ProviderConfig:
    return ProviderConfig(
        profile_name="local", provider="openai_compatible", model="fake", max_retries=0
    )


@pytest.mark.asyncio
async def test_garbage_predictions_raise_malformed_rate_not_spurious_classes() -> None:
    """3 valid predictions + 2 echoed non-labels: malformed rate reflects the echoes,
    and macro F1 is averaged over real classes only (same predictions as accuracy)."""
    cases = [
        _case("c1", "entailment"),
        _case("c2", "entailment"),
        _case("c3", "neutral"),
        _case("c4", "contradiction"),
        _case("c5", "neutral"),
    ]
    client = _ScriptedClient(
        [
            '{"label": "entailment"}',  # correct
            '{"label": "neutral"}',  # wrong but valid
            '{"label": "neutral"}',  # correct
            '{"premise": "echoed clause text", "hypothesis": "echoed statement"}',  # echo
            '"Receiving Party may retain one copy..."',  # echoed prose string
        ]
    )
    result = await evaluate_prompt(
        "system prompt",
        cases,
        _task(),
        client,
        _config(),
        strategy="baseline_few_shot_k5",
        split="train",
        bootstrap_enabled=False,
    )

    assert result.malformed_output_rate == pytest.approx(0.4)
    # Metrics computed over the 3 non-malformed predictions only
    assert result.accuracy == pytest.approx(2 / 3)
    # Label space is the union of gold + predicted VALID labels — no spurious
    # garbage classes diluting the mean.
    observed_labels = {row.label for row in result.per_label_metrics}
    assert observed_labels <= _LABELS
