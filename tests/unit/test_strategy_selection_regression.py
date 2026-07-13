"""Regression tests for the ContractNLI strategy-selection failure (job 1521e48b).

Observed: every candidate strategy failed its quality gates, and the selector
then ranked the *failed* pool anyway, promoting embedding_knn_k1 (macro_f1
0.6144) over the baseline scaffold (0.7088) and reporting it under the
"baseline" family. These tests pin the fixed behavior:

1. A passing strategy with higher macro_f1 wins by default.
2. A malformed (gate-failing) strategy is never selected over a passing one.
3. When ALL candidates fail gates, selection falls back to the baseline
   scaffold and the reason names the gate failures.
4. The paired net-improvement gate rejects candidates that break more cases
   than they fix.
5. Strategy families distinguish baseline / embedding / retrieval / distilled.
6. The selection table explains every strategy's outcome (eligible, gate
   failures, rank, selected).
"""

from __future__ import annotations

from rulekiln.config.settings import QualityGateDefaults
from rulekiln.pipeline.quality_gates import check_quality_gates
from rulekiln.pipeline.strategy_families import strategy_family
from rulekiln.pipeline.strategy_selection import (
    build_strategy_comparison,
    select_strategy_generic,
)
from rulekiln.schemas.pipeline import CaseEvalResult, EvalResult, QualityGateResult
from rulekiln.schemas.task_case import EvaluationSpec, RuleKilnCase

BASELINE = "baseline_scaffold"


def _eval(strategy: str, macro_f1: float, malformed_rate: float = 0.0) -> EvalResult:
    return EvalResult(
        strategy=strategy,
        model="fake",
        split="validation",
        macro_f1=macro_f1,
        accuracy=macro_f1,
        malformed_output_rate=malformed_rate,
    )


def _gate(strategy: str, passed: bool, violations: list[str] | None = None) -> QualityGateResult:
    return QualityGateResult(
        strategy=strategy,
        passed=passed,
        violations=violations or ([] if passed else [f"{strategy} failed a gate"]),
    )


# ── 1. higher macro_f1 wins by default ────────────────────────────────────────


def test_higher_macro_f1_passing_strategy_wins() -> None:
    evals = {
        BASELINE: _eval(BASELINE, 0.70),
        "dbscan": _eval("dbscan", 0.80),
        "embedding_knn_k1": _eval("embedding_knn_k1", 0.75),
    }
    gates = {
        "dbscan": _gate("dbscan", passed=True),
        "embedding_knn_k1": _gate("embedding_knn_k1", passed=True),
    }
    selected, reason = select_strategy_generic(
        task_mode="classification",
        strategy_evals=evals,
        strategy_gates=gates,
        strategy_prompt_tokens={},
        baseline_strategy=BASELINE,
    )
    assert selected == "dbscan"
    assert "primary metric" in reason


def test_lower_macro_f1_strategy_cannot_beat_clearly_higher_one() -> None:
    """Beyond the 0.005 tie epsilon, the higher-scoring passing strategy must win."""
    evals = {
        BASELINE: _eval(BASELINE, 0.50),
        "dbscan": _eval("dbscan", 0.60),
        "hdbscan": _eval("hdbscan", 0.75),
    }
    gates = {
        "dbscan": _gate("dbscan", passed=True),
        "hdbscan": _gate("hdbscan", passed=True),
    }
    selected, _ = select_strategy_generic(
        task_mode="classification",
        strategy_evals=evals,
        strategy_gates=gates,
        strategy_prompt_tokens={},
        baseline_strategy=BASELINE,
    )
    assert selected == "hdbscan"


# ── 2. malformed strategy is ineligible ──────────────────────────────────────


def test_malformed_strategy_is_not_selected_over_passing_strategy() -> None:
    evals = {
        BASELINE: _eval(BASELINE, 0.70),
        "hdbscan": _eval("hdbscan", 0.95, malformed_rate=1.0),
        "dbscan": _eval("dbscan", 0.72),
    }
    gates = {
        "hdbscan": _gate(
            "hdbscan", passed=False, violations=["malformed_output_rate 1.0000 > 0.0100"]
        ),
        "dbscan": _gate("dbscan", passed=True),
    }
    selected, _ = select_strategy_generic(
        task_mode="classification",
        strategy_evals=evals,
        strategy_gates=gates,
        strategy_prompt_tokens={},
        baseline_strategy=BASELINE,
    )
    assert selected == "dbscan"


# ── 3. all candidates fail gates → baseline fallback (the observed regression) ─


def test_all_candidates_failing_gates_falls_back_to_baseline() -> None:
    """Reproduces the ContractNLI run: every candidate below baseline, all gates
    failed. The selector must NOT rank the failed pool (which promoted
    embedding_knn_k1); it must fall back to the baseline scaffold."""
    evals = {
        BASELINE: _eval(BASELINE, 0.7088),
        "embedding_knn_k1": _eval("embedding_knn_k1", 0.6144),
        "embedding_knn_k5": _eval("embedding_knn_k5", 0.6190),
        "dbscan": _eval("dbscan", 0.2722),
        "hdbscan": _eval("hdbscan", 0.0, malformed_rate=1.0),
    }
    gates = {
        strategy: _gate(strategy, passed=False, violations=[f"metric_delta < 0 for {strategy}"])
        for strategy in evals
        if strategy != BASELINE
    }
    selected, reason = select_strategy_generic(
        task_mode="classification",
        strategy_evals=evals,
        strategy_gates=gates,
        strategy_prompt_tokens={},
        baseline_strategy=BASELINE,
    )
    assert selected == BASELINE
    assert "failed quality gates" in reason
    assert "embedding_knn_k1" in reason  # gate failures are surfaced


# ── 4. paired net-improvement gate ────────────────────────────────────────────


def _case(case_id: str) -> RuleKilnCase:
    return RuleKilnCase(
        id=case_id,
        split="validation",
        task_mode="classification",
        input={"text": case_id},
        expected="yes",
        evaluation=EvaluationSpec(assertions=[]),
    )


def _case_result(case_id: str, passed: bool) -> CaseEvalResult:
    return CaseEvalResult(case_id=case_id, score=1.0 if passed else 0.0, passed=passed)


def test_paired_broken_exceeding_fixed_fails_gate() -> None:
    """A candidate that breaks more cases than it fixes must fail the paired gate
    even when the other thresholds are relaxed."""
    case_ids = [f"c{i}" for i in range(10)]
    cases = [_case(cid) for cid in case_ids]
    # Baseline passes c0..c4; candidate passes c0..c2 (breaks c3, c4) and fixes c5.
    baseline_eval = EvalResult(
        strategy=BASELINE,
        model="fake",
        split="validation",
        macro_f1=0.5,
        case_results=[_case_result(cid, passed=cid in {"c0", "c1", "c2", "c3", "c4"})
                      for cid in case_ids],
    )
    candidate_eval = EvalResult(
        strategy="dbscan",
        model="fake",
        split="validation",
        macro_f1=0.6,
        case_results=[_case_result(cid, passed=cid in {"c0", "c1", "c2", "c5"})
                      for cid in case_ids],
    )
    result = check_quality_gates(
        strategy="dbscan",
        distilled_eval=candidate_eval,
        baseline_eval=baseline_eval,
        cases=cases,
        task_mode="classification",
        task_gates={"min_metric_delta": -1.0, "max_regression_rate": 1.0},
        settings_defaults=QualityGateDefaults(),
    )
    assert result.paired_fixed == 1
    assert result.paired_broken == 2
    assert result.passed is False
    assert any("paired_broken" in v for v in result.violations)


def test_paired_gate_can_be_disabled_per_task() -> None:
    case_ids = ["c0", "c1", "c2"]
    cases = [_case(cid) for cid in case_ids]
    baseline_eval = EvalResult(
        strategy=BASELINE,
        model="fake",
        split="validation",
        case_results=[_case_result(cid, passed=cid != "c2") for cid in case_ids],
    )
    candidate_eval = EvalResult(
        strategy="dbscan",
        model="fake",
        split="validation",
        case_results=[_case_result(cid, passed=cid == "c2") for cid in case_ids],
    )
    result = check_quality_gates(
        strategy="dbscan",
        distilled_eval=candidate_eval,
        baseline_eval=baseline_eval,
        cases=cases,
        task_mode="classification",
        task_gates={
            "min_metric_delta": -1.0,
            "max_regression_rate": 1.0,
            "require_paired_net_improvement": False,
        },
        settings_defaults=QualityGateDefaults(),
    )
    assert not any("paired_broken" in v for v in result.violations)


# ── 5. family classification ─────────────────────────────────────────────────


def test_strategy_families_are_distinct() -> None:
    assert strategy_family("baseline") == "baseline"
    assert strategy_family("baseline_scaffold") == "baseline"
    assert strategy_family("baseline_few_shot_k3") == "baseline"
    assert strategy_family("embedding_knn_k1") == "embedding_baseline"
    assert strategy_family("embedding_centroid") == "embedding_baseline"
    assert strategy_family("retrieval_few_shot_k5") == "retrieval_baseline"
    assert strategy_family("dbscan") == "distilled"
    assert strategy_family("hdbscan") == "distilled"


def test_selected_family_matches_selected_strategy_family() -> None:
    """An embedding baseline win must not be reported under the 'baseline' family."""
    comparison = build_strategy_comparison(
        baseline_eval=_eval(BASELINE, 0.70),
        dbscan_eval=None,
        hdbscan_eval=None,
        dbscan_gate=None,
        hdbscan_gate=None,
        task_mode="classification",
        strategy_evals={
            BASELINE: _eval(BASELINE, 0.70),
            "embedding_knn_k1": _eval("embedding_knn_k1", 0.90),
        },
        strategy_gates={"embedding_knn_k1": _gate("embedding_knn_k1", passed=True)},
        baseline_strategy=BASELINE,
    )
    assert comparison.selected_strategy == "embedding_knn_k1"
    selected_rows = [r for r in comparison.selection_table if r.selected]
    assert len(selected_rows) == 1
    assert selected_rows[0].family == "embedding_baseline"


# ── 6. selection table explains every strategy ────────────────────────────────


def test_selection_table_contains_gate_reasons_for_rejected_strategies() -> None:
    comparison = build_strategy_comparison(
        baseline_eval=_eval(BASELINE, 0.7088),
        dbscan_eval=_eval("dbscan", 0.2722),
        hdbscan_eval=_eval("hdbscan", 0.0, malformed_rate=1.0),
        dbscan_gate=_gate("dbscan", passed=False, violations=["metric_delta -0.4366 < 0.0"]),
        hdbscan_gate=_gate(
            "hdbscan",
            passed=False,
            violations=["malformed_output_rate 1.0000 > 0.0100", "prompt_tokens 11006 > 8000"],
        ),
        task_mode="classification",
        strategy_evals={
            BASELINE: _eval(BASELINE, 0.7088),
            "dbscan": _eval("dbscan", 0.2722),
            "hdbscan": _eval("hdbscan", 0.0, malformed_rate=1.0),
            "embedding_knn_k1": _eval("embedding_knn_k1", 0.6144),
        },
        strategy_gates={
            "dbscan": _gate("dbscan", passed=False, violations=["metric_delta -0.4366 < 0.0"]),
            "hdbscan": _gate(
                "hdbscan",
                passed=False,
                violations=["malformed_output_rate 1.0000 > 0.0100"],
            ),
            "embedding_knn_k1": _gate(
                "embedding_knn_k1", passed=False, violations=["regression_rate 0.3289 > 0.1000"]
            ),
        },
        baseline_strategy=BASELINE,
        unavailable_strategies={"hdbscan_skipped_example": "prompt_tokens 11006 exceeds budget"},
    )
    # All candidates failed → baseline selected
    assert comparison.selected_strategy == BASELINE

    rows_by_strategy = {r.strategy: r for r in comparison.selection_table}
    # Every evaluated strategy has a row
    for strategy in (BASELINE, "dbscan", "hdbscan", "embedding_knn_k1"):
        assert strategy in rows_by_strategy

    # Rejected strategies carry their gate failures
    assert rows_by_strategy["hdbscan"].eligible is False
    assert any("malformed" in v for v in rows_by_strategy["hdbscan"].gate_failures)
    assert rows_by_strategy["embedding_knn_k1"].eligible is False
    assert any("regression_rate" in v for v in rows_by_strategy["embedding_knn_k1"].gate_failures)

    # Baseline is always eligible and is the selected row with a reason
    baseline_row = rows_by_strategy[BASELINE]
    assert baseline_row.eligible is True
    assert baseline_row.selected is True
    assert baseline_row.reason is not None and "failed quality gates" in baseline_row.reason

    # Strategies skipped before evaluation appear as ineligible with the skip reason
    skipped_row = rows_by_strategy["hdbscan_skipped_example"]
    assert skipped_row.eligible is False
    assert skipped_row.selection_score is None
    assert any("exceeds budget" in v for v in skipped_row.gate_failures)


# ── HDBSCAN gate promotion criteria (items I.8, I.9, constraint J) ────────────


def _classification_case(case_id: str, label: str) -> RuleKilnCase:
    return RuleKilnCase(
        id=case_id,
        split="validation",
        task_mode="classification",
        input={"premise": case_id, "hypothesis": case_id},
        expected={"label": label},
        evaluation=EvaluationSpec(assertions=[]),
    )


def _hdbscan_gate_check(
    *, malformed_rate: float, distilled_passed: set[str], baseline_passed: set[str]
) -> QualityGateResult:
    case_ids = [f"c{i}" for i in range(20)]
    cases = [_classification_case(cid, "entailment") for cid in case_ids]
    baseline_eval = EvalResult(
        strategy=BASELINE,
        model="fake",
        split="validation",
        macro_f1=0.70,
        case_results=[_case_result(cid, passed=cid in baseline_passed) for cid in case_ids],
    )
    distilled_eval = EvalResult(
        strategy="hdbscan",
        model="fake",
        split="validation",
        macro_f1=0.72,
        malformed_output_rate=malformed_rate,
        case_results=[_case_result(cid, passed=cid in distilled_passed) for cid in case_ids],
    )
    return check_quality_gates(
        strategy="hdbscan",
        distilled_eval=distilled_eval,
        baseline_eval=baseline_eval,
        cases=cases,
        task_mode="classification",
        task_gates={"min_metric_delta": -1.0, "require_paired_net_improvement": False},
        settings_defaults=QualityGateDefaults(),
    )


def test_hdbscan_ineligible_when_malformed_rate_exceeds_gate() -> None:
    """Item I.8: HDBSCAN stays ineligible while malformed_output_rate > max_malformed
    (default 0.01), independent of how well the rest of the run performs."""
    all_ids = {f"c{i}" for i in range(20)}
    result = _hdbscan_gate_check(
        malformed_rate=0.06, distilled_passed=all_ids, baseline_passed=all_ids
    )
    assert result.passed is False
    assert any("malformed" in v for v in result.violations)


def test_hdbscan_ineligible_when_regression_rate_exceeds_gate_even_with_low_malformed() -> None:
    """A low malformed rate alone does not make HDBSCAN eligible if regression_rate
    (baseline-passed, candidate-failed) exceeds the cap."""
    baseline_passed = {f"c{i}" for i in range(20)}  # baseline passes all 20
    # Candidate regresses 4/20 = 0.20 > default 0.10 cap
    distilled_passed = {f"c{i}" for i in range(20) if i >= 4}
    result = _hdbscan_gate_check(
        malformed_rate=0.0, distilled_passed=distilled_passed, baseline_passed=baseline_passed
    )
    assert result.passed is False
    assert any("regression_rate" in v for v in result.violations)


def test_hdbscan_eligible_only_when_both_malformed_and_regression_within_gates() -> None:
    """Item I.9: HDBSCAN becomes eligible once malformed_rate <= 1% AND
    regression_rate <= 10% — the promotion criteria from constraint J."""
    baseline_passed = {f"c{i}" for i in range(20)}
    # Candidate regresses 1/20 = 0.05, within the 0.10 cap.
    distilled_passed = {f"c{i}" for i in range(20) if i != 0}
    result = _hdbscan_gate_check(
        malformed_rate=0.0, distilled_passed=distilled_passed, baseline_passed=baseline_passed
    )
    assert result.passed is True
    assert result.violations == []


def test_eligible_strategies_are_ranked_by_score() -> None:
    comparison = build_strategy_comparison(
        baseline_eval=_eval(BASELINE, 0.70),
        dbscan_eval=None,
        hdbscan_eval=None,
        dbscan_gate=None,
        hdbscan_gate=None,
        task_mode="classification",
        strategy_evals={
            BASELINE: _eval(BASELINE, 0.70),
            "dbscan": _eval("dbscan", 0.80),
            "embedding_knn_k1": _eval("embedding_knn_k1", 0.75),
        },
        strategy_gates={
            "dbscan": _gate("dbscan", passed=True),
            "embedding_knn_k1": _gate("embedding_knn_k1", passed=True),
        },
        baseline_strategy=BASELINE,
    )
    rows_by_strategy = {r.strategy: r for r in comparison.selection_table}
    assert rows_by_strategy["dbscan"].selection_rank == 1
    assert rows_by_strategy["embedding_knn_k1"].selection_rank == 2
    assert rows_by_strategy[BASELINE].selection_rank == 3
