"""Classification metric helpers: label extraction, column alignment, statistics refresh."""

from __future__ import annotations

import json

from rulekiln.pipeline.baseline_strategies import expected_label, resolve_distance_metric
from rulekiln.pipeline.split_policy import resolve_few_shot_source_cases
from rulekiln.pipeline.statistics import (
    PairedComparisonArtifacts,
    compute_classification_statistics,
    compute_paired_comparison,
    compute_regressed_labels,
)
from rulekiln.schemas.pipeline import CaseEvalResult, EvalResult, RegressedLabelRow
from rulekiln.schemas.task_case import RuleKilnCase, RuleKilnTask


def _is_classification_task_mode(task_mode: str) -> bool:
    return task_mode in {"classification", "routing"}


def _prediction_label_from_case_result(case_result: CaseEvalResult | None) -> str:
    if case_result is None or case_result.actual_output is None:
        return ""
    if isinstance(case_result.actual_output, dict):
        label_value = case_result.actual_output.get("label")
        if isinstance(label_value, str):
            return label_value.strip()
        return ""
    return case_result.actual_output.strip()


def _case_input_text(case: RuleKilnCase) -> str:
    utterance = case.input.get("utterance")
    if isinstance(utterance, str):
        return utterance

    text_value = case.input.get("text")
    if isinstance(text_value, str):
        return text_value

    return json.dumps(case.input, sort_keys=True, ensure_ascii=False)


def _aligned_classification_columns(
    *,
    eval_cases: list[RuleKilnCase],
    eval_result: EvalResult,
) -> tuple[list[str], list[str], list[str], list[str]]:
    case_results_by_id = {
        case_result.case_id: case_result for case_result in eval_result.case_results
    }

    case_ids: list[str] = []
    expected_labels: list[str] = []
    predicted_labels: list[str] = []
    input_texts: list[str] = []

    for case in eval_cases:
        exp = expected_label(case)
        if exp is None:
            continue
        case_ids.append(case.id)
        expected_labels.append(exp)
        predicted_labels.append(_prediction_label_from_case_result(case_results_by_id.get(case.id)))
        input_texts.append(_case_input_text(case))

    return case_ids, expected_labels, predicted_labels, input_texts


def _refresh_eval_classification_statistics(
    *,
    eval_result: EvalResult,
    eval_cases: list[RuleKilnCase],
    bootstrap_enabled: bool,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> EvalResult:
    case_ids, expected_labels, predicted_labels, _ = _aligned_classification_columns(
        eval_cases=eval_cases,
        eval_result=eval_result,
    )
    if not case_ids:
        return eval_result

    classification_stats = compute_classification_statistics(
        actual_labels=expected_labels,
        predicted_labels=predicted_labels,
        case_ids=case_ids,
        bootstrap_enabled=bootstrap_enabled,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )

    return eval_result.model_copy(
        update={
            "accuracy": classification_stats.accuracy,
            "accuracy_ci_95": classification_stats.accuracy_ci_95,
            "macro_f1": classification_stats.macro_f1,
            "macro_f1_ci_95": classification_stats.macro_f1_ci_95,
            "per_label_metrics": classification_stats.per_label_metrics,
            "per_outcome_precision": classification_stats.per_outcome_precision,
            "per_outcome_recall": classification_stats.per_outcome_recall,
            "confusion_matrix": classification_stats.confusion_matrix,
            "top_confusions": classification_stats.top_confusions,
        }
    )


def _build_runtime_paired_comparison(
    *,
    eval_cases: list[RuleKilnCase],
    baseline_eval: EvalResult,
    candidate_eval: EvalResult,
    baseline_strategy_id: str,
    candidate_strategy_id: str,
) -> tuple[PairedComparisonArtifacts, list[RegressedLabelRow]]:
    (
        case_ids,
        expected_labels,
        baseline_predictions,
        input_texts,
    ) = _aligned_classification_columns(
        eval_cases=eval_cases,
        eval_result=baseline_eval,
    )
    _, _, candidate_predictions, _ = _aligned_classification_columns(
        eval_cases=eval_cases,
        eval_result=candidate_eval,
    )

    paired_artifacts = compute_paired_comparison(
        case_ids=case_ids,
        actual_labels=expected_labels,
        baseline_predictions=baseline_predictions,
        candidate_predictions=candidate_predictions,
        input_texts=input_texts,
        baseline_strategy_id=baseline_strategy_id,
        candidate_strategy_id=candidate_strategy_id,
    )
    regressed_labels = compute_regressed_labels(
        case_ids=case_ids,
        actual_labels=expected_labels,
        baseline_predictions=baseline_predictions,
        candidate_predictions=candidate_predictions,
    )
    return paired_artifacts, regressed_labels


def _resolve_distance_metric_from_task(task: RuleKilnTask) -> str:
    raw_metric = task.prompt_scaffold.get("embedding_distance_metric")
    if isinstance(raw_metric, str):
        return resolve_distance_metric(raw_metric)
    return resolve_distance_metric(None)


def _resolve_training_cases(
    extraction_cases: list[RuleKilnCase],
    eval_cases: list[RuleKilnCase],
    *,
    allow_eval_fallback: bool,
) -> list[RuleKilnCase]:
    return resolve_few_shot_source_cases(
        extraction_cases, eval_cases, allow_eval_fallback=allow_eval_fallback
    )
