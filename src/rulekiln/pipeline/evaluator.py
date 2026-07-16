"""Evaluator: run student model against cases and compute metrics."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from pydantic import BaseModel

from rulekiln.observability.logging import get_logger
from rulekiln.pipeline.classification_alignment import (
    build_classification_label_columns,
    compute_strict_and_valid_output_statistics,
    declared_label_vocabulary,
)
from rulekiln.providers.contracts import ChatModelClient, ProviderConfig
from rulekiln.schemas.pipeline import CaseEvalResult, EvalResult
from rulekiln.schemas.task_case import AssertionType, RuleKilnCase, RuleKilnTask, TaskMode

type CaseResultPersistFn = Callable[[CaseEvalResult], Awaitable[None]]
type CancellationCheckFn = Callable[[], Awaitable[bool]]

_log = get_logger(__name__)


class JobCancelledError(Exception):
    """Raised by evaluate_prompt when check_cancelled reports the job was cancelled."""

# Primary metric by task mode (C005)
_PRIMARY_METRIC: dict[TaskMode, str] = {
    "classification": "macro_f1",
    "routing": "macro_f1",
    "tool_use": "weighted_case_score",
    "extraction": "weighted_case_score",
    "summarization": "weighted_case_score",
    "rubric_review": "weighted_case_score",
    "freeform_generation": "weighted_case_score",
    "agent_behavior": "weighted_case_score",
}


def get_primary_metric(task_mode: TaskMode) -> str:
    return _PRIMARY_METRIC.get(task_mode, "weighted_case_score")


class _StudentOutputSchema(BaseModel):
    """Generic passthrough for structured student output."""

    raw: dict[str, str | int | float | bool | None] | str | None = None


def _render_input_from_template(template: str, input_fields: dict[str, Any]) -> str:
    """Substitute {{ key }} placeholders in template with values from input_fields."""

    def _replace(m: re.Match[str]) -> str:
        value = input_fields.get(m.group(1).strip())
        return str(value) if value is not None else ""

    return re.sub(r"\{\{\s*(\w+)\s*\}\}", _replace, template).strip()


# Answer cue appended after the test-case input for few-shot strategies. Must
# match the `Output:` label that every rendered few-shot example ends with (see
# render_few_shot_examples in baseline_strategies.py): the examples train the
# model to emit the label only after this cue, so presenting it on the test case
# lets the model complete the trained answer slot instead of continuing the input
# prose (the k5 echo failure).
FEW_SHOT_ANSWER_CUE = "Output:"


def build_student_user_message(
    input_template: str,
    input_fields: dict[str, Any],
    *,
    transition_prefix: str | None = None,
) -> str:
    """Build the student user message: optional transition block + rendered input.

    The test case is always rendered via ``input_template`` (not json.dumps) so the
    model sees a fresh natural-language case rather than another example to echo.
    When ``transition_prefix`` is supplied (few-shot strategies) the transition block
    is placed ahead of the rendered input to break example-continuation momentum, and
    the ``Output:`` answer cue is appended so the model completes the trained answer
    slot the examples establish.
    """
    rendered = _render_input_from_template(input_template, input_fields)
    if transition_prefix:
        return f"{transition_prefix}\n\n{rendered}\n\n{FEW_SHOT_ANSWER_CUE}"
    return rendered


def _extract_json_from_content(
    content: str,
) -> dict[str, str | int | float | bool | None] | str | None:
    """Extract a JSON object from raw model output text.

    Handles plain JSON, markdown-fenced JSON, and JSON embedded in prose.
    Filters the result to primitive-valued fields only (matching the type of
    _StudentOutputSchema.raw).  Returns None when no parseable object is found.
    """
    if not content:
        return None

    text = content.strip()

    # Prefer content inside a markdown code fence when present.
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    candidates: list[str] = []
    if fence_match:
        candidates.append(fence_match.group(1).strip())
    candidates.append(text)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return {
                    k: v
                    for k, v in data.items()
                    if isinstance(v, (str, int, float, bool, type(None)))
                }
            if isinstance(data, str):
                return data
        except json.JSONDecodeError:
            pass

    # Last resort: find the first flat {...} block in the text.
    brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if brace_match:
        try:
            data = json.loads(brace_match.group(0))
            if isinstance(data, dict):
                return {
                    k: v
                    for k, v in data.items()
                    if isinstance(v, (str, int, float, bool, type(None)))
                }
        except json.JSONDecodeError:
            pass

    return None


async def _call_student(
    system_prompt: str,
    case: RuleKilnCase,
    chat_client: ChatModelClient,
    config: ProviderConfig,
    *,
    input_template: str,
    transition_prefix: str | None = None,
) -> tuple[dict[str, str | int | float | bool | None] | str | None, bool, str | None]:
    """Return (parsed_output, is_malformed, raw_content)."""
    user_prompt = build_student_user_message(
        input_template, case.input, transition_prefix=transition_prefix
    )
    try:
        result = await chat_client.complete_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=_StudentOutputSchema,
            config=config,
        )
        parsed = result.parsed
        output = getattr(parsed, "raw", None) if parsed is not None else None

        # Fallback for providers that return plain JSON text rather than enforcing
        # _StudentOutputSchema via tool-calling (e.g. openai_compatible with local models).
        if output is None and result.content:
            output = _extract_json_from_content(result.content)

        if output is None:
            _log.warning(
                "malformed_output",
                case_id=case.id,
                raw_content=result.content,
                parsed=repr(parsed),
            )

        return output, False, result.content
    except Exception as exc:
        _log.warning(
            "malformed_output_exception",
            case_id=case.id,
            exc_type=type(exc).__name__,
            exc_msg=str(exc),
        )
        return None, True, None


def _score_assertion(
    assertion_type: AssertionType,
    value: object,
    actual: dict[str, str | int | float | bool | None] | str | None,
    path: str | None,
) -> float:
    """Score a single assertion: 1.0 pass, 0.0 fail."""
    if actual is None:
        return 0.0

    # Resolve path within actual if applicable.
    # Strip the JSONPath root selector ("$.") if present — "$" is not a literal key.
    target: object = actual
    if path and isinstance(actual, dict):
        clean_path = path[2:] if path.startswith("$.") else path.lstrip("$")
        for part in clean_path.split(".") if clean_path else []:
            if isinstance(target, dict) and part in target:
                target = target[part]
            else:
                return 0.0

    match assertion_type:
        case "must_equal":
            return 1.0 if target == value else 0.0
        case "must_include":
            return 1.0 if (str(value) in str(target)) else 0.0
        case "must_not_include":
            return 1.0 if (str(value) not in str(target)) else 0.0
        case "must_match_regex":
            return 1.0 if re.search(str(value), str(target)) is not None else 0.0
        case _:
            # Unsupported assertion type: skip (neutral)
            return 1.0


def _extract_predicted_label(
    actual: dict[str, str | int | float | bool | None] | str,
) -> str:
    """Extract the predicted label string the same way _compute_metrics does."""
    raw = actual.get("label", "") if isinstance(actual, dict) else actual
    return raw if isinstance(raw, str) else str(raw)


def _try_recover_label(text: str, allowed_labels: frozenset[str]) -> str | None:
    """Best-effort recovery of a single unambiguous allowed label from `text`.

    Returns None (stays malformed) when zero, or more than one distinct, allowed labels
    are found — recovery must never launder a genuinely ambiguous or invalid response.
    Word-boundary matching means compound/underscore-joined strings such as
    "entailment_verbal_included" do NOT match "entailment" (`_` is a regex word
    character, so there is no boundary between "entailment" and the following "_").
    """
    normalized = text.strip().rstrip(".!?,;:").lower()
    if normalized in allowed_labels:
        return normalized

    lowered = text.lower()
    found = {label for label in allowed_labels if re.search(rf"\b{re.escape(label)}\b", lowered)}
    if len(found) == 1:
        return next(iter(found))
    return None


def _score_case(
    case: RuleKilnCase,
    actual: dict[str, str | int | float | bool | None] | str | None,
    malformed: bool,
    allowed_labels: frozenset[str] | None = None,
    raw_content: str | None = None,
) -> CaseEvalResult:
    if malformed or actual is None:
        return CaseEvalResult(
            case_id=case.id,
            score=0.0,
            passed=False,
            malformed=True,
            actual_output=actual,
            raw_content=raw_content,
        )

    # A parseable output whose label is not in the task's allowed vocabulary is a
    # contract violation, not a wrong-but-valid prediction. Counting it as a valid
    # prediction creates one spurious zero-F1 class per unique garbage string and
    # silently dilutes macro F1 (observed: few-shot echo outputs → macro_f1 ~0.02
    # with 0% malformed). Mark it malformed so the malformed-rate gate sees it —
    # unless a single unambiguous label can be recovered from harmless extra text.
    recovered = False
    if allowed_labels is not None:
        predicted_label = _extract_predicted_label(actual)
        if predicted_label not in allowed_labels:
            recovered_label = _try_recover_label(predicted_label, allowed_labels)
            if recovered_label is None and raw_content:
                recovered_label = _try_recover_label(raw_content, allowed_labels)
            if recovered_label is None:
                return CaseEvalResult(
                    case_id=case.id,
                    score=0.0,
                    passed=False,
                    malformed=True,
                    actual_output=actual,
                    error="prediction_not_in_allowed_labels",
                    raw_content=raw_content,
                )
            actual = (
                {**actual, "label": recovered_label}
                if isinstance(actual, dict)
                else recovered_label
            )
            recovered = True

    assertion_scores: dict[str, float] = {}
    total_weight = 0.0
    weighted_sum = 0.0

    for i, assertion in enumerate(case.evaluation.assertions):
        score = _score_assertion(assertion.type, assertion.value, actual, assertion.path)
        key = f"assertion_{i}"
        assertion_scores[key] = score
        weighted_sum += score * assertion.weight
        total_weight += assertion.weight

    final_score = (weighted_sum / total_weight) if total_weight > 0 else 1.0
    return CaseEvalResult(
        case_id=case.id,
        score=final_score,
        passed=final_score >= 1.0,
        malformed=False,
        assertion_scores=assertion_scores,
        actual_output=actual,
        raw_content=raw_content,
        recovered=recovered,
    )


def _compute_metrics(
    case_results: list[CaseEvalResult],
    cases: list[RuleKilnCase],
    task: RuleKilnTask,
    strategy: str,
    model: str,
    split: str,
    *,
    bootstrap_enabled: bool,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> EvalResult:
    n = len(case_results)
    if n == 0:
        return EvalResult(strategy=strategy, model=model, split=split)

    malformed_count = sum(1 for r in case_results if r.malformed)
    malformed_output_rate = malformed_count / n

    # Weighted case score
    weight_map = {c.id: c.weight for c in cases}
    total_weight = sum(weight_map.get(r.case_id, 1.0) for r in case_results)
    weighted_sum = sum(r.score * weight_map.get(r.case_id, 1.0) for r in case_results)
    weighted_case_score = weighted_sum / total_weight if total_weight > 0 else 0.0

    result = EvalResult(
        strategy=strategy,
        model=model,
        split=split,
        weighted_case_score=weighted_case_score,
        malformed_output_rate=malformed_output_rate,
        case_results=case_results,
    )

    if task.task_mode not in ("classification", "routing"):
        return result

    case_ids, expected_labels, predicted_labels, malformed_flags = (
        build_classification_label_columns(cases, case_results)
    )
    if not case_ids:
        return result

    vocabulary = declared_label_vocabulary(task)
    stats = compute_strict_and_valid_output_statistics(
        case_ids=case_ids,
        expected_labels=expected_labels,
        predicted_labels=predicted_labels,
        malformed_flags=malformed_flags,
        label_vocabulary=vocabulary,
        bootstrap_enabled=bootstrap_enabled,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    strict = stats.strict
    valid_output = stats.valid_output

    return result.model_copy(
        update={
            "accuracy": strict.accuracy,
            "accuracy_ci_95": strict.accuracy_ci_95,
            "macro_f1": strict.macro_f1,
            "macro_f1_ci_95": strict.macro_f1_ci_95,
            "valid_output_accuracy": valid_output.accuracy if valid_output else None,
            "valid_output_accuracy_ci_95": valid_output.accuracy_ci_95 if valid_output else None,
            "valid_output_macro_f1": valid_output.macro_f1 if valid_output else None,
            "valid_output_macro_f1_ci_95": valid_output.macro_f1_ci_95 if valid_output else None,
            "per_outcome_precision": strict.per_outcome_precision,
            "per_outcome_recall": strict.per_outcome_recall,
            "per_label_metrics": strict.per_label_metrics,
            "confusion_matrix": strict.confusion_matrix,
            "top_confusions": strict.top_confusions,
            "declared_label_vocabulary": list(vocabulary) if vocabulary else None,
        }
    )


async def evaluate_prompt(
    system_prompt: str,
    cases: list[RuleKilnCase],
    task: RuleKilnTask,
    chat_client: ChatModelClient,
    config: ProviderConfig,
    strategy: str,
    split: str = "train",
    completed_case_results: Mapping[str, CaseEvalResult] | None = None,
    on_case_result: CaseResultPersistFn | None = None,
    *,
    user_message_prefix: str | None = None,
    bootstrap_enabled: bool = True,
    bootstrap_iterations: int = 1000,
    bootstrap_seed: int = 1729,
    check_cancelled: CancellationCheckFn | None = None,
) -> EvalResult:
    """Run the student model against every case and return aggregate metrics."""
    mismatched_ids = [case.id for case in cases if case.split != split]
    if mismatched_ids:
        shown = mismatched_ids[:5]
        remainder = len(mismatched_ids) - len(shown)
        suffix = f" and {remainder} more" if remainder > 0 else ""
        raise ValueError(
            f"evaluate_prompt called with split={split!r} (strategy={strategy!r}) but "
            f"received cases from a different split: {shown}{suffix}"
        )

    case_result_by_id: dict[str, CaseEvalResult] = {}
    if completed_case_results:
        case_result_by_id.update(completed_case_results)

    vocabulary = declared_label_vocabulary(task)
    allowed_labels = frozenset(vocabulary) if vocabulary is not None else None
    for case in cases:
        if case.id in case_result_by_id:
            continue

        if check_cancelled is not None and await check_cancelled():
            raise JobCancelledError(
                f"Job cancelled during evaluation (strategy={strategy}, split={split})"
            )

        actual, malformed, raw_content = await _call_student(
            system_prompt,
            case,
            chat_client,
            config,
            input_template=task.input_template,
            transition_prefix=user_message_prefix,
        )
        result = _score_case(
            case, actual, malformed, allowed_labels=allowed_labels, raw_content=raw_content
        )
        case_result_by_id[case.id] = result

        if on_case_result is not None:
            await on_case_result(result)

    case_results = [case_result_by_id[case.id] for case in cases if case.id in case_result_by_id]

    return _compute_metrics(
        case_results=case_results,
        cases=cases,
        task=task,
        strategy=strategy,
        model=config.model,
        split=split,
        bootstrap_enabled=bootstrap_enabled,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )


def build_case_result_from_label_prediction(
    case: RuleKilnCase,
    predicted_label: str,
    *,
    error: str | None = None,
) -> CaseEvalResult:
    """Build a CaseEvalResult from a predicted classification/routing label."""
    actual_output: dict[str, str | int | float | bool | None] = {"label": predicted_label}
    result = _score_case(case, actual_output, malformed=False)
    result.error = error
    return result


def build_eval_result_from_case_results(
    *,
    strategy: str,
    model: str,
    split: str,
    task: RuleKilnTask,
    cases: list[RuleKilnCase],
    case_results: list[CaseEvalResult],
    prompt_token_count: int | None = None,
    retrieval_failure_count: int = 0,
    model_failure_count: int = 0,
    bootstrap_enabled: bool = True,
    bootstrap_iterations: int = 1000,
    bootstrap_seed: int = 1729,
) -> EvalResult:
    """Aggregate precomputed case results into the standard EvalResult shape."""
    result = _compute_metrics(
        case_results=case_results,
        cases=cases,
        task=task,
        strategy=strategy,
        model=model,
        split=split,
        bootstrap_enabled=bootstrap_enabled,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    result.prompt_token_count = prompt_token_count
    result.retrieval_failure_count = retrieval_failure_count
    result.model_failure_count = model_failure_count
    return result
