"""Unit tests for shared classification and paired-comparison statistics."""

from __future__ import annotations

import pytest

from rulekiln.pipeline.statistics import (
    compute_classification_statistics,
    compute_confusion_matrix,
    compute_paired_comparison,
    compute_per_label_metrics,
    compute_regressed_labels,
    extract_top_confusions,
)


def test_classification_statistics_bootstrap_is_deterministic() -> None:
    actual_labels = ["a", "a", "a", "b", "b", "b", "c", "c", "c", "c"]
    predicted_labels = ["a", "b", "a", "b", "b", "a", "c", "a", "c", "b"]
    case_ids = [f"case_{index:02d}" for index in range(len(actual_labels))]

    first = compute_classification_statistics(
        actual_labels=actual_labels,
        predicted_labels=predicted_labels,
        case_ids=case_ids,
        bootstrap_enabled=True,
        bootstrap_iterations=300,
        bootstrap_seed=12345,
    )
    second = compute_classification_statistics(
        actual_labels=actual_labels,
        predicted_labels=predicted_labels,
        case_ids=case_ids,
        bootstrap_enabled=True,
        bootstrap_iterations=300,
        bootstrap_seed=12345,
    )

    assert first.accuracy == second.accuracy
    assert first.macro_f1 == second.macro_f1
    assert first.accuracy_ci_95 is not None
    assert second.accuracy_ci_95 is not None
    assert first.accuracy_ci_95.low == second.accuracy_ci_95.low
    assert first.accuracy_ci_95.high == second.accuracy_ci_95.high
    assert first.accuracy_ci_95.method == "bootstrap"
    assert first.accuracy_ci_95.iterations == 300
    assert first.accuracy_ci_95.seed == 12345

    assert first.macro_f1_ci_95 is not None
    assert second.macro_f1_ci_95 is not None
    assert first.macro_f1_ci_95.low == second.macro_f1_ci_95.low
    assert first.macro_f1_ci_95.high == second.macro_f1_ci_95.high


def test_classification_statistics_ci_is_none_for_single_case() -> None:
    stats = compute_classification_statistics(
        actual_labels=["a"],
        predicted_labels=["a"],
        case_ids=["case_01"],
        bootstrap_enabled=True,
        bootstrap_iterations=1000,
        bootstrap_seed=17,
    )

    assert stats.accuracy_ci_95 is None
    assert stats.macro_f1_ci_95 is None


def test_paired_comparison_counts_and_rates() -> None:
    paired = compute_paired_comparison(
        case_ids=["c1", "c2", "c3", "c4"],
        input_texts=["i1", "i2", "i3", "i4"],
        actual_labels=["a", "a", "b", "b"],
        baseline_predictions=["a", "b", "b", "a"],
        candidate_predictions=["a", "a", "a", "b"],
        baseline_strategy_id="baseline",
        candidate_strategy_id="candidate",
    )

    assert paired.summary.fixed_count == 2
    assert paired.summary.broken_count == 1
    assert paired.summary.unchanged_correct_count == 1
    assert paired.summary.unchanged_wrong_count == 0
    assert paired.summary.total_cases == 4
    assert paired.summary.net_fix_rate is not None
    assert paired.summary.net_fix_rate == pytest.approx(1 / 3)
    assert paired.summary.net_fix_rate_status == "ok"
    assert paired.summary.overall_net_fix_rate == pytest.approx(0.25)

    unchanged = paired.unchanged_examples
    assert len(unchanged) == 1
    assert unchanged[0].unchanged_status == "both_correct"
    for row in paired.fixed_examples + paired.broken_examples:
        assert row.unchanged_status is None


def test_paired_comparison_handles_no_changed_outcomes() -> None:
    paired = compute_paired_comparison(
        case_ids=["c1", "c2"],
        input_texts=["i1", "i2"],
        actual_labels=["a", "b"],
        baseline_predictions=["a", "a"],
        candidate_predictions=["a", "a"],
        baseline_strategy_id="baseline",
        candidate_strategy_id="candidate",
    )

    assert paired.summary.fixed_count == 0
    assert paired.summary.broken_count == 0
    assert paired.summary.net_fix_rate is None
    assert paired.summary.net_fix_rate_status == "no_changed_outcomes"


def test_per_label_and_confusion_are_sorted() -> None:
    actual = ["z", "a", "z", "b", "a"]
    predicted = ["a", "a", "z", "b", "z"]

    per_label = compute_per_label_metrics(actual, predicted)
    assert [row.label for row in per_label] == ["a", "b", "z"]

    confusion = compute_confusion_matrix(actual, predicted)
    assert list(confusion.keys()) == ["a", "b", "z"]
    assert list(confusion["a"].keys()) == ["a", "z"]
    assert list(confusion["b"].keys()) == ["b"]
    assert list(confusion["z"].keys()) == ["a", "z"]


def test_top_confusions_non_diagonal_and_limited() -> None:
    actual = ["a", "a", "a", "b", "b", "c"]
    predicted = ["b", "b", "a", "a", "b", "a"]
    case_ids = ["c1", "c2", "c3", "c4", "c5", "c6"]

    rows = extract_top_confusions(actual, predicted, case_ids, limit=2)

    assert len(rows) == 2
    assert rows[0].actual_label == "a"
    assert rows[0].predicted_label == "b"
    assert rows[0].count == 2
    assert rows[0].example_case_ids == ["c1", "c2"]
    assert rows[1].actual_label != rows[1].predicted_label


def test_per_label_metrics_uses_declared_vocabulary_including_zero_support() -> None:
    actual = ["a", "a", "b"]
    predicted = ["a", "a", "b"]

    per_label = compute_per_label_metrics(actual, predicted, label_vocabulary=["a", "b", "c"])

    assert [row.label for row in per_label] == ["a", "b", "c"]
    zero_support_row = per_label[2]
    assert zero_support_row.label == "c"
    assert zero_support_row.support == 0
    assert zero_support_row.precision == 0.0
    assert zero_support_row.recall == 0.0
    assert zero_support_row.f1 == 0.0


def test_per_label_metrics_declared_vocabulary_ignores_out_of_vocabulary_predictions() -> None:
    """A predicted value outside the declared vocabulary (e.g. a malformed
    placeholder) never becomes its own class — it simply never matches."""
    actual = ["a", "b"]
    predicted = ["a", "garbage-echoed-text"]

    per_label = compute_per_label_metrics(actual, predicted, label_vocabulary=["a", "b"])

    assert [row.label for row in per_label] == ["a", "b"]
    b_row = per_label[1]
    assert b_row.support == 1
    assert b_row.recall == 0.0


def test_classification_statistics_respects_label_vocabulary_on_point_estimate() -> None:
    stats = compute_classification_statistics(
        actual_labels=["a", "a", "b"],
        predicted_labels=["a", "a", "b"],
        case_ids=["c1", "c2", "c3"],
        bootstrap_enabled=False,
        bootstrap_iterations=100,
        bootstrap_seed=1,
        label_vocabulary=["a", "b", "c"],
    )

    assert [row.label for row in stats.per_label_metrics] == ["a", "b", "c"]
    assert stats.per_label_metrics[2].support == 0


def test_classification_statistics_bootstrap_uses_fixed_label_vocabulary() -> None:
    """Every bootstrap resample must score against the same declared vocabulary,
    not whatever subset of labels happens to appear in a given resample."""
    actual_labels = ["a", "a", "a", "b", "b", "b"]
    predicted_labels = ["a", "a", "a", "b", "b", "b"]
    case_ids = [f"case_{i}" for i in range(len(actual_labels))]

    stats = compute_classification_statistics(
        actual_labels=actual_labels,
        predicted_labels=predicted_labels,
        case_ids=case_ids,
        bootstrap_enabled=True,
        bootstrap_iterations=50,
        bootstrap_seed=7,
        label_vocabulary=["a", "b", "c"],
    )

    assert [row.label for row in stats.per_label_metrics] == ["a", "b", "c"]
    assert stats.macro_f1_ci_95 is not None


def test_regressed_labels_use_recall_delta() -> None:
    rows = compute_regressed_labels(
        case_ids=["c1", "c2", "c3", "c4", "c5", "c6"],
        actual_labels=["a", "a", "a", "b", "b", "b"],
        baseline_predictions=["a", "a", "b", "b", "b", "a"],
        candidate_predictions=["a", "b", "b", "b", "a", "a"],
    )

    assert len(rows) == 2
    assert rows[0].recall_delta <= rows[1].recall_delta
    assert all(row.recall_delta < 0 for row in rows)
    assert all(row.f1_delta <= 0 for row in rows)
