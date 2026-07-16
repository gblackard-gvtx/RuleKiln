"""Classification metric helpers: label extraction, column alignment, statistics refresh."""

from __future__ import annotations

import json

from rulekiln.pipeline.baseline_strategies import expected_label, resolve_distance_metric
from rulekiln.pipeline.classification_alignment import (
    build_classification_label_columns,
    compute_strict_and_valid_output_statistics,
    declared_label_vocabulary,
)
from rulekiln.pipeline.split_policy import resolve_few_shot_source_cases
from rulekiln.pipeline.statistics import (
    PairedComparisonArtifacts,
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
    task: RuleKilnTask,
    bootstrap_enabled: bool,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> EvalResult:
    """Rebuild classification-derived fields from the always-authoritative ``case_results``.

    A DB-loaded row with ``metric_semantics_version < 2`` predates strict/valid-output
    metric semantics (Work Package 1). When ``case_results`` is available it is
    recomputed and stamped ``recomputed_legacy``; when it is empty (data predates the
    ``EvalCaseResultRecord`` table itself), the row is left unmodified but stamped
    ``incomplete_legacy`` so callers don't display stale or fabricated numbers as current.
    """
    is_legacy = eval_result.metric_semantics_version < 2
    case_ids, expected_labels, predicted_labels, malformed_flags = (
        build_classification_label_columns(eval_cases, eval_result.case_results)
    )
    if not case_ids:
        if is_legacy:
            return eval_result.model_copy(
                update={"metric_semantics_status": "incomplete_legacy"}
            )
        return eval_result

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

    return eval_result.model_copy(
        update={
            "accuracy": strict.accuracy,
            "accuracy_ci_95": strict.accuracy_ci_95,
            "macro_f1": strict.macro_f1,
            "macro_f1_ci_95": strict.macro_f1_ci_95,
            "valid_output_accuracy": valid_output.accuracy if valid_output else None,
            "valid_output_accuracy_ci_95": valid_output.accuracy_ci_95 if valid_output else None,
            "valid_output_macro_f1": valid_output.macro_f1 if valid_output else None,
            "valid_output_macro_f1_ci_95": (
                valid_output.macro_f1_ci_95 if valid_output else None
            ),
            "per_label_metrics": strict.per_label_metrics,
            "per_outcome_precision": strict.per_outcome_precision,
            "per_outcome_recall": strict.per_outcome_recall,
            "confusion_matrix": strict.confusion_matrix,
            "top_confusions": strict.top_confusions,
            "declared_label_vocabulary": list(vocabulary) if vocabulary else None,
            "metric_semantics_version": 2,
            "metric_semantics_status": "recomputed_legacy" if is_legacy else "current",
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
