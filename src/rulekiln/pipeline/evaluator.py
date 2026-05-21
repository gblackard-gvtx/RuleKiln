"""Evaluator: run student model against cases and compute metrics."""

from __future__ import annotations

import json
import re
from collections import defaultdict

from pydantic import BaseModel

from rulekiln.providers.contracts import ChatModelClient, ProviderConfig
from rulekiln.schemas.pipeline import CaseEvalResult, EvalResult
from rulekiln.schemas.task_case import AssertionType, RuleKilnCase, RuleKilnTask, TaskMode

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


async def _call_student(
    system_prompt: str,
    case: RuleKilnCase,
    chat_client: ChatModelClient,
    config: ProviderConfig,
) -> tuple[dict[str, str | int | float | bool | None] | str | None, bool]:
    """Return (parsed_output, is_malformed)."""
    user_prompt = json.dumps(case.input)
    try:
        result = await chat_client.complete_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=_StudentOutputSchema,
            config=config,
        )
        output = getattr(result, "raw", None)
        return output, False
    except Exception:
        return None, True


def _score_assertion(
    assertion_type: AssertionType,
    value: object,
    actual: dict[str, str | int | float | bool | None] | str | None,
    path: str | None,
) -> float:
    """Score a single assertion: 1.0 pass, 0.0 fail."""
    if actual is None:
        return 0.0

    # Resolve path within actual if applicable
    target: object = actual
    if path and isinstance(actual, dict):
        for part in path.split("."):
            if isinstance(target, dict) and part in target:
                target = target[part]
            else:
                return 0.0

    match assertion_type:
        case "must_equal":
            return 1.0 if target == value else 0.0
        case "must_include":
            return 1.0 if (value in str(target)) else 0.0
        case "must_not_include":
            return 1.0 if (value not in str(target)) else 0.0
        case "must_match_regex":
            return 1.0 if re.search(str(value), str(target)) is not None else 0.0
        case _:
            # Unsupported assertion type: skip (neutral)
            return 1.0


def _score_case(
    case: RuleKilnCase,
    actual: dict[str, str | int | float | bool | None] | str | None,
    malformed: bool,
) -> CaseEvalResult:
    if malformed or actual is None:
        return CaseEvalResult(
            case_id=case.id,
            score=0.0,
            passed=False,
            malformed=True,
            actual_output=actual,
        )

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
    )


def _compute_metrics(
    case_results: list[CaseEvalResult],
    cases: list[RuleKilnCase],
    task_mode: TaskMode,
    strategy: str,
    model: str,
    split: str,
) -> EvalResult:
    n = len(case_results)
    if n == 0:
        return EvalResult(strategy=strategy, model=model, split=split)

    malformed_count = sum(1 for r in case_results if r.malformed)
    malformed_output_rate = malformed_count / n

    # Weighted case score
    weight_map = {c.id: c.weight for c in cases}
    total_weight = sum(weight_map.get(r.case_id, 1.0) for r in case_results)
    weighted_sum = sum(
        r.score * weight_map.get(r.case_id, 1.0) for r in case_results
    )
    weighted_case_score = weighted_sum / total_weight if total_weight > 0 else 0.0

    # For classification/routing: compute accuracy + macro F1 from expected labels
    accuracy: float | None = None
    macro_f1: float | None = None
    per_outcome_precision: dict[str, float] = {}
    per_outcome_recall: dict[str, float] = {}
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    if task_mode in ("classification", "routing"):
        case_map = {c.id: c for c in cases}
        correct = 0
        tp: dict[str, int] = defaultdict(int)
        fp: dict[str, int] = defaultdict(int)
        fn: dict[str, int] = defaultdict(int)
        labels: set[str] = set()

        for res in case_results:
            case = case_map.get(res.case_id)
            if case is None or case.expected is None:
                continue
            expected_label = (
                case.expected.get("label", "") if isinstance(case.expected, dict) else str(case.expected)
            )
            actual_label = (
                res.actual_output.get("label", "") if isinstance(res.actual_output, dict) else str(res.actual_output or "")
            )
            labels.add(expected_label)
            labels.add(actual_label)
            confusion[expected_label][actual_label] += 1
            if expected_label == actual_label:
                correct += 1
                tp[expected_label] += 1
            else:
                fp[actual_label] += 1
                fn[expected_label] += 1

        valid = len([r for r in case_results if not r.malformed])
        accuracy = correct / valid if valid > 0 else 0.0
        f1_scores: list[float] = []
        for label in labels:
            prec = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
            rec = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1_scores.append(f1)
            per_outcome_precision[label] = prec
            per_outcome_recall[label] = rec
        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return EvalResult(
        strategy=strategy,
        model=model,
        split=split,
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_case_score=weighted_case_score,
        malformed_output_rate=malformed_output_rate,
        per_outcome_precision=per_outcome_precision,
        per_outcome_recall=per_outcome_recall,
        confusion_matrix={k: dict(v) for k, v in confusion.items()},
        case_results=case_results,
    )


async def evaluate_prompt(
    system_prompt: str,
    cases: list[RuleKilnCase],
    task: RuleKilnTask,
    chat_client: ChatModelClient,
    config: ProviderConfig,
    strategy: str,
    split: str = "train",
) -> EvalResult:
    """Run the student model against every case and return aggregate metrics."""
    case_results: list[CaseEvalResult] = []
    for case in cases:
        actual, malformed = await _call_student(system_prompt, case, chat_client, config)
        result = _score_case(case, actual, malformed)
        case_results.append(result)

    return _compute_metrics(
        case_results=case_results,
        cases=cases,
        task_mode=task.task_mode,
        strategy=strategy,
        model=config.model,
        split=split,
    )
