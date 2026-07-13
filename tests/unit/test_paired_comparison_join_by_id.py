"""Paired comparison must join baseline and candidate by stable case_id.

Motivated by docs/investigations/contractnli_hdbscan_validation_win.md (§C):
``compute_paired_comparison`` pairs by ``zip`` over parallel arrays — i.e. by
*array position*. That is only safe because the runtime caller
``_build_runtime_paired_comparison`` builds both prediction columns from a shared
``eval_cases`` order via a ``case_id -> result`` dict lookup. These tests lock
that by-id alignment: shuffling one strategy's ``case_results`` must NOT change
the fixed/broken/unchanged counts, and the counts must total the sample size.
"""

from __future__ import annotations

from rulekiln.schemas.pipeline import CaseEvalResult, EvalResult
from rulekiln.schemas.task_case import EvaluationSpec, RuleKilnCase
from rulekiln.workers.pipeline.classification_metrics import (
    _build_runtime_paired_comparison,  # pyright: ignore[reportPrivateUsage]
)


def _case(case_id: str, label: str) -> RuleKilnCase:
    return RuleKilnCase(
        id=case_id,
        split="validation",  # type: ignore[arg-type]
        task_mode="classification",
        input={"text": f"input {case_id}"},
        expected={"label": label},
        evaluation=EvaluationSpec(assertions=[]),
    )


def _cr(case_id: str, label: str) -> CaseEvalResult:
    return CaseEvalResult(case_id=case_id, score=1.0, passed=True, actual_output=label)


# Gold labels
_CASES = [
    _case("a", "entailment"),
    _case("b", "neutral"),
    _case("c", "contradiction"),
    _case("d", "entailment"),
]

# baseline: a✓ b✗ c✓ d✗   candidate: a✓ b✓ c✗ d✓
# -> fixed={b,d}=2, broken={c}=1, unchanged_correct={a}=1, unchanged_wrong={}=0
_BASE = [
    _cr("a", "entailment"),
    _cr("b", "contradiction"),
    _cr("c", "contradiction"),
    _cr("d", "neutral"),
]
_CAND = [
    _cr("a", "entailment"),
    _cr("b", "neutral"),
    _cr("c", "neutral"),
    _cr("d", "entailment"),
]


def _summary(base: list[CaseEvalResult], cand: list[CaseEvalResult]):
    artifacts, _ = _build_runtime_paired_comparison(
        eval_cases=_CASES,
        baseline_eval=EvalResult(
            strategy="baseline", model="m", split="validation", case_results=base
        ),
        candidate_eval=EvalResult(
            strategy="hdbscan", model="m", split="validation", case_results=cand
        ),
        baseline_strategy_id="baseline",
        candidate_strategy_id="hdbscan",
    )
    return artifacts.summary


def test_reference_counts() -> None:
    s = _summary(_BASE, _CAND)
    assert (s.fixed_count, s.broken_count, s.unchanged_correct_count, s.unchanged_wrong_count) == (
        2,
        1,
        1,
        0,
    )


def test_counts_total_sample_size() -> None:
    s = _summary(_BASE, _CAND)
    total = (
        s.fixed_count + s.broken_count + s.unchanged_correct_count + s.unchanged_wrong_count
    )
    assert total == s.total_cases == len(_CASES)


def test_shuffled_candidate_results_yield_identical_counts() -> None:
    """Reordering the candidate's case_results (different array order, same case_ids)
    must not change the paired result — proving the join is by case_id, not position."""
    shuffled = [_CAND[2], _CAND[0], _CAND[3], _CAND[1]]
    assert [r.case_id for r in shuffled] != [r.case_id for r in _CAND]

    baseline_order = _summary(_BASE, _CAND)
    shuffled_order = _summary(_BASE, shuffled)
    assert (
        shuffled_order.fixed_count,
        shuffled_order.broken_count,
        shuffled_order.unchanged_correct_count,
        shuffled_order.unchanged_wrong_count,
    ) == (
        baseline_order.fixed_count,
        baseline_order.broken_count,
        baseline_order.unchanged_correct_count,
        baseline_order.unchanged_wrong_count,
    )


def test_shuffled_baseline_results_yield_identical_counts() -> None:
    shuffled = [_BASE[3], _BASE[1], _BASE[0], _BASE[2]]
    baseline_order = _summary(_BASE, _CAND)
    shuffled_order = _summary(shuffled, _CAND)
    assert (
        shuffled_order.fixed_count,
        shuffled_order.broken_count,
    ) == (baseline_order.fixed_count, baseline_order.broken_count)
