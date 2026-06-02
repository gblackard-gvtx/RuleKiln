"""Shared deterministic classification and paired-comparison statistics utilities."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from random import Random

from pydantic import BaseModel, Field

from rulekiln.schemas.pipeline import (
    MetricConfidenceInterval,
    PairedComparisonExample,
    PairedComparisonSummary,
    PerLabelMetricsRow,
    RegressedLabelRow,
    TopConfusionRow,
)


class ClassificationStatistics(BaseModel):
    """Aggregate classification statistics computed from actual/predicted labels."""

    accuracy: float
    accuracy_ci_95: MetricConfidenceInterval | None = None
    macro_f1: float
    macro_f1_ci_95: MetricConfidenceInterval | None = None
    per_label_metrics: list[PerLabelMetricsRow] = Field(default_factory=list)
    per_outcome_precision: dict[str, float] = Field(default_factory=dict)
    per_outcome_recall: dict[str, float] = Field(default_factory=dict)
    confusion_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)
    top_confusions: list[TopConfusionRow] = Field(default_factory=list)


class PairedComparisonArtifacts(BaseModel):
    """Structured paired-comparison outputs for JSONL artifacts and summary."""

    fixed_examples: list[PairedComparisonExample] = Field(default_factory=list)
    broken_examples: list[PairedComparisonExample] = Field(default_factory=list)
    unchanged_examples: list[PairedComparisonExample] = Field(default_factory=list)
    summary: PairedComparisonSummary


def compute_accuracy(actual_labels: Sequence[str], predicted_labels: Sequence[str]) -> float:
    """Compute exact-match accuracy from aligned labels."""
    _validate_aligned_inputs(actual_labels, predicted_labels)
    if not actual_labels:
        return 0.0

    correct = sum(
        1
        for actual_label, predicted_label in zip(actual_labels, predicted_labels, strict=True)
        if actual_label == predicted_label
    )
    return correct / len(actual_labels)


def compute_per_label_metrics(
    actual_labels: Sequence[str],
    predicted_labels: Sequence[str],
) -> list[PerLabelMetricsRow]:
    """Compute deterministic per-label precision, recall, and F1 rows."""
    _validate_aligned_inputs(actual_labels, predicted_labels)

    label_set = sorted(set(actual_labels) | set(predicted_labels))
    rows: list[PerLabelMetricsRow] = []

    for label in label_set:
        true_positive = 0
        false_positive = 0
        false_negative = 0
        support = 0

        for actual_label, predicted_label in zip(actual_labels, predicted_labels, strict=True):
            if actual_label == label:
                support += 1
            if actual_label == label and predicted_label == label:
                true_positive += 1
            elif actual_label != label and predicted_label == label:
                false_positive += 1
            elif actual_label == label and predicted_label != label:
                false_negative += 1

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = true_positive / precision_denominator if precision_denominator > 0 else 0.0
        recall = true_positive / recall_denominator if recall_denominator > 0 else 0.0
        f1_denominator = precision + recall
        f1 = (2 * precision * recall / f1_denominator) if f1_denominator > 0 else 0.0

        rows.append(
            PerLabelMetricsRow(
                label=label,
                support=support,
                true_positive=true_positive,
                false_positive=false_positive,
                false_negative=false_negative,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )

    return rows


def compute_macro_f1(per_label_metrics: Sequence[PerLabelMetricsRow]) -> float:
    """Compute macro F1 from per-label rows."""
    if not per_label_metrics:
        return 0.0
    return sum(row.f1 for row in per_label_metrics) / len(per_label_metrics)


def compute_confusion_matrix(
    actual_labels: Sequence[str],
    predicted_labels: Sequence[str],
) -> dict[str, dict[str, int]]:
    """Build deterministic confusion matrix keyed by actual_label then predicted_label."""
    _validate_aligned_inputs(actual_labels, predicted_labels)

    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for actual_label, predicted_label in zip(actual_labels, predicted_labels, strict=True):
        matrix[actual_label][predicted_label] += 1

    serialized: dict[str, dict[str, int]] = {}
    for actual_label in sorted(matrix.keys()):
        row_counts = matrix[actual_label]
        serialized[actual_label] = {
            predicted_label: row_counts[predicted_label]
            for predicted_label in sorted(row_counts.keys())
        }

    return serialized


def extract_top_confusions(
    actual_labels: Sequence[str],
    predicted_labels: Sequence[str],
    case_ids: Sequence[str],
    *,
    limit: int = 20,
) -> list[TopConfusionRow]:
    """Return top non-diagonal confusions with deterministic ordering and example IDs."""
    _validate_aligned_inputs(actual_labels, predicted_labels)
    _validate_aligned_inputs(actual_labels, case_ids)

    confusion_examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    for case_id, actual_label, predicted_label in zip(
        case_ids, actual_labels, predicted_labels, strict=True
    ):
        if actual_label == predicted_label:
            continue
        confusion_examples[(actual_label, predicted_label)].append(case_id)

    sorted_pairs = sorted(
        confusion_examples.items(),
        key=lambda item: (-len(item[1]), item[0][0], item[0][1]),
    )

    rows: list[TopConfusionRow] = []
    for (actual_label, predicted_label), example_case_ids in sorted_pairs[:limit]:
        rows.append(
            TopConfusionRow(
                actual_label=actual_label,
                predicted_label=predicted_label,
                count=len(example_case_ids),
                example_case_ids=example_case_ids,
            )
        )

    return rows


def compute_classification_statistics(
    *,
    actual_labels: Sequence[str],
    predicted_labels: Sequence[str],
    case_ids: Sequence[str],
    bootstrap_enabled: bool,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    top_confusion_limit: int = 20,
) -> ClassificationStatistics:
    """Compute all shared classification statistics for artifacts and summaries."""
    _validate_aligned_inputs(actual_labels, predicted_labels)
    _validate_aligned_inputs(actual_labels, case_ids)

    per_label_metrics = compute_per_label_metrics(actual_labels, predicted_labels)
    accuracy = compute_accuracy(actual_labels, predicted_labels)
    macro_f1 = compute_macro_f1(per_label_metrics)

    accuracy_ci_95: MetricConfidenceInterval | None = None
    macro_f1_ci_95: MetricConfidenceInterval | None = None
    if bootstrap_enabled:
        accuracy_ci_95, macro_f1_ci_95 = _bootstrap_confidence_intervals(
            actual_labels=actual_labels,
            predicted_labels=predicted_labels,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        )

    return ClassificationStatistics(
        accuracy=accuracy,
        accuracy_ci_95=accuracy_ci_95,
        macro_f1=macro_f1,
        macro_f1_ci_95=macro_f1_ci_95,
        per_label_metrics=per_label_metrics,
        per_outcome_precision={row.label: row.precision for row in per_label_metrics},
        per_outcome_recall={row.label: row.recall for row in per_label_metrics},
        confusion_matrix=compute_confusion_matrix(actual_labels, predicted_labels),
        top_confusions=extract_top_confusions(
            actual_labels,
            predicted_labels,
            case_ids,
            limit=top_confusion_limit,
        ),
    )


def compute_paired_comparison(
    *,
    case_ids: Sequence[str],
    actual_labels: Sequence[str],
    baseline_predictions: Sequence[str],
    candidate_predictions: Sequence[str],
    input_texts: Sequence[str],
    baseline_strategy_id: str,
    candidate_strategy_id: str,
) -> PairedComparisonArtifacts:
    """Compute paired fixed/broken/unchanged artifacts and summary."""
    _validate_aligned_inputs(case_ids, actual_labels)
    _validate_aligned_inputs(actual_labels, baseline_predictions)
    _validate_aligned_inputs(actual_labels, candidate_predictions)
    _validate_aligned_inputs(actual_labels, input_texts)

    fixed_examples: list[PairedComparisonExample] = []
    broken_examples: list[PairedComparisonExample] = []
    unchanged_examples: list[PairedComparisonExample] = []

    for case_id, input_text, expected_label, baseline_prediction, candidate_prediction in zip(
        case_ids,
        input_texts,
        actual_labels,
        baseline_predictions,
        candidate_predictions,
        strict=True,
    ):
        baseline_correct = baseline_prediction == expected_label
        candidate_correct = candidate_prediction == expected_label

        change_class: str
        unchanged_status: str | None = None
        if (not baseline_correct) and candidate_correct:
            change_class = "fixed"
        elif baseline_correct and (not candidate_correct):
            change_class = "broken"
        else:
            change_class = "unchanged"
            unchanged_status = "both_correct" if baseline_correct else "both_wrong"

        row = PairedComparisonExample(
            case_id=case_id,
            change_class=change_class,
            unchanged_status=unchanged_status,
            baseline_strategy_id=baseline_strategy_id,
            candidate_strategy_id=candidate_strategy_id,
            expected_label=expected_label,
            baseline_prediction=baseline_prediction,
            candidate_prediction=candidate_prediction,
            baseline_correct=baseline_correct,
            candidate_correct=candidate_correct,
            input_text=input_text,
            notes=[],
        )

        if change_class == "fixed":
            fixed_examples.append(row)
        elif change_class == "broken":
            broken_examples.append(row)
        else:
            unchanged_examples.append(row)

    fixed_count = len(fixed_examples)
    broken_count = len(broken_examples)
    unchanged_correct_count = sum(
        1 for row in unchanged_examples if row.unchanged_status == "both_correct"
    )
    unchanged_wrong_count = sum(
        1 for row in unchanged_examples if row.unchanged_status == "both_wrong"
    )
    total_cases = len(case_ids)

    changed_outcomes = fixed_count + broken_count
    if changed_outcomes == 0:
        net_fix_rate: float | None = None
        net_fix_rate_status = "no_changed_outcomes"
    else:
        net_fix_rate = (fixed_count - broken_count) / changed_outcomes
        net_fix_rate_status = "ok"

    overall_net_fix_rate = (
        (fixed_count - broken_count) / total_cases if total_cases > 0 else 0.0
    )

    summary = PairedComparisonSummary(
        baseline_strategy_id=baseline_strategy_id,
        candidate_strategy_id=candidate_strategy_id,
        fixed_count=fixed_count,
        broken_count=broken_count,
        unchanged_correct_count=unchanged_correct_count,
        unchanged_wrong_count=unchanged_wrong_count,
        total_cases=total_cases,
        net_fix_rate=net_fix_rate,
        net_fix_rate_status=net_fix_rate_status,
        overall_net_fix_rate=overall_net_fix_rate,
    )

    return PairedComparisonArtifacts(
        fixed_examples=fixed_examples,
        broken_examples=broken_examples,
        unchanged_examples=unchanged_examples,
        summary=summary,
    )


def compute_regressed_labels(
    *,
    case_ids: Sequence[str],
    actual_labels: Sequence[str],
    baseline_predictions: Sequence[str],
    candidate_predictions: Sequence[str],
) -> list[RegressedLabelRow]:
    """Compute label-level regressions using recall delta as primary criterion."""
    _validate_aligned_inputs(case_ids, actual_labels)
    _validate_aligned_inputs(actual_labels, baseline_predictions)
    _validate_aligned_inputs(actual_labels, candidate_predictions)

    baseline_rows = compute_per_label_metrics(actual_labels, baseline_predictions)
    candidate_rows = compute_per_label_metrics(actual_labels, candidate_predictions)

    baseline_by_label = {row.label: row for row in baseline_rows}
    candidate_by_label = {row.label: row for row in candidate_rows}

    wrong_prediction_labels: dict[str, Counter[str]] = defaultdict(Counter)
    wrong_prediction_case_ids: dict[str, list[str]] = defaultdict(list)
    for case_id, actual_label, candidate_prediction in zip(
        case_ids, actual_labels, candidate_predictions, strict=True
    ):
        if actual_label == candidate_prediction:
            continue
        wrong_prediction_labels[actual_label][candidate_prediction] += 1
        wrong_prediction_case_ids[actual_label].append(case_id)

    regressed_rows: list[RegressedLabelRow] = []
    for label in sorted(set(baseline_by_label.keys()) | set(candidate_by_label.keys())):
        baseline_row = baseline_by_label.get(label)
        candidate_row = candidate_by_label.get(label)
        if baseline_row is None or candidate_row is None:
            continue

        recall_delta = candidate_row.recall - baseline_row.recall
        if recall_delta >= 0:
            continue

        top_wrong_labels = sorted(
            wrong_prediction_labels[label].items(),
            key=lambda item: (-item[1], item[0]),
        )
        top_predicted_wrong_labels = [
            f"{predicted_label} ({count})" for predicted_label, count in top_wrong_labels[:5]
        ]

        regressed_rows.append(
            RegressedLabelRow(
                label=label,
                support=candidate_row.support,
                baseline_recall=baseline_row.recall,
                candidate_recall=candidate_row.recall,
                recall_delta=recall_delta,
                baseline_f1=baseline_row.f1,
                candidate_f1=candidate_row.f1,
                f1_delta=candidate_row.f1 - baseline_row.f1,
                new_false_negatives=max(
                    0,
                    candidate_row.false_negative - baseline_row.false_negative,
                ),
                top_predicted_wrong_labels=top_predicted_wrong_labels,
                example_case_ids=wrong_prediction_case_ids[label][:10],
            )
        )

    regressed_rows.sort(key=lambda row: (row.recall_delta, -row.support, row.label))
    return regressed_rows


def _bootstrap_confidence_intervals(
    *,
    actual_labels: Sequence[str],
    predicted_labels: Sequence[str],
    iterations: int,
    seed: int,
) -> tuple[MetricConfidenceInterval | None, MetricConfidenceInterval | None]:
    """Compute deterministic 95% bootstrap confidence intervals for accuracy and macro F1."""
    _validate_aligned_inputs(actual_labels, predicted_labels)

    if len(actual_labels) < 2:
        return None, None
    if iterations < 1:
        return None, None

    randomizer = Random(seed)  # noqa: S311
    sample_count = len(actual_labels)
    accuracy_samples: list[float] = []
    macro_f1_samples: list[float] = []

    for _ in range(iterations):
        sampled_indices = [randomizer.randrange(sample_count) for _ in range(sample_count)]
        sample_actual = [actual_labels[index] for index in sampled_indices]
        sample_predicted = [predicted_labels[index] for index in sampled_indices]

        sample_rows = compute_per_label_metrics(sample_actual, sample_predicted)
        accuracy_samples.append(compute_accuracy(sample_actual, sample_predicted))
        macro_f1_samples.append(compute_macro_f1(sample_rows))

    accuracy_samples.sort()
    macro_f1_samples.sort()

    accuracy_ci = MetricConfidenceInterval(
        low=_interpolated_quantile(accuracy_samples, 0.025),
        high=_interpolated_quantile(accuracy_samples, 0.975),
        iterations=iterations,
        seed=seed,
    )
    macro_f1_ci = MetricConfidenceInterval(
        low=_interpolated_quantile(macro_f1_samples, 0.025),
        high=_interpolated_quantile(macro_f1_samples, 0.975),
        iterations=iterations,
        seed=seed,
    )

    return accuracy_ci, macro_f1_ci


def _interpolated_quantile(values: Sequence[float], quantile: float) -> float:
    """Return an interpolated quantile from sorted values."""
    if not values:
        raise ValueError("Cannot compute quantile of empty values.")
    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * quantile
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))

    if lower_index == upper_index:
        return values[lower_index]

    lower_value = values[lower_index]
    upper_value = values[upper_index]
    weight = position - lower_index
    return lower_value + (upper_value - lower_value) * weight


def _validate_aligned_inputs(primary: Sequence[str], secondary: Sequence[str]) -> None:
    if len(primary) != len(secondary):
        raise ValueError("Expected aligned inputs with equal lengths.")
