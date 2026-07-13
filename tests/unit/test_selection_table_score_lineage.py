"""Selection-table score must trace to macro-F1, and stale scores must be flagged.

Motivated by docs/investigations/contractnli_hdbscan_validation_win.md (§A, §K.1):
the Strategy Selection "Score" column is read from a persisted strategy_comparison.json
and the evaluation table is read live from the DB. When the persisted artifact is stale
(a diluted baseline macro-F1 of 0.5350 vs the live 0.7142), the two disagree silently.

These tests lock two invariants:
1. ``build_selection_table`` sets ``selection_score`` to the primary metric (macro-F1 for
   classification) — it is NOT an adjusted/penalized score.
2. The UI lineage helper flags strategies whose persisted selection score diverges from the
   live eval macro-F1.
"""

from __future__ import annotations

from rulekiln.pipeline.strategy_selection import build_selection_table
from rulekiln.schemas.pipeline import EvalResult
from rulekiln.ui.routes import (
    _selection_score_lineage_warning,  # pyright: ignore[reportPrivateUsage]
)


def _eval(strategy: str, macro_f1: float) -> EvalResult:
    return EvalResult(
        strategy=strategy,
        model="fake",
        split="validation",
        accuracy=0.75,
        macro_f1=macro_f1,
        weighted_case_score=0.9,
    )


def test_selection_score_equals_macro_f1_for_classification() -> None:
    strategy_evals = {
        "baseline_scaffold": _eval("baseline_scaffold", 0.7142),
        "hdbscan": _eval("hdbscan", 0.7338),
    }
    rows = build_selection_table(
        task_mode="classification",
        strategy_evals=strategy_evals,
        strategy_gates={},
        selected_strategy="hdbscan",
        selection_reason="Selected hdbscan by primary metric (0.7338).",
        baseline_strategy="baseline_scaffold",
    )
    score_by_strategy = {row.strategy: row.selection_score for row in rows}
    assert score_by_strategy["baseline_scaffold"] == 0.7142
    assert score_by_strategy["hdbscan"] == 0.7338


class _FakeEvalRun:
    def __init__(self, strategy: str, macro_f1: float) -> None:
        self.strategy = strategy
        self.split = "validation"
        self.macro_f1 = macro_f1
        self.accuracy = 0.75
        self.weighted_case_score = 0.9


def test_lineage_warning_none_when_scores_agree() -> None:
    selection_table = [
        {"strategy": "baseline_scaffold", "selection_score": 0.7142},
        {"strategy": "hdbscan", "selection_score": 0.7338},
    ]
    eval_runs = [_FakeEvalRun("baseline_scaffold", 0.7142), _FakeEvalRun("hdbscan", 0.7338)]
    assert _selection_score_lineage_warning(selection_table, eval_runs, "macro_f1") is None


def test_lineage_warning_flags_stale_baseline() -> None:
    # Persisted selection score is the stale, diluted 0.5350; live DB row is 0.7142.
    selection_table = [
        {"strategy": "baseline_scaffold", "selection_score": 0.5350},
        {"strategy": "hdbscan", "selection_score": 0.7338},
    ]
    eval_runs = [_FakeEvalRun("baseline_scaffold", 0.7142), _FakeEvalRun("hdbscan", 0.7338)]
    warning = _selection_score_lineage_warning(selection_table, eval_runs, "macro_f1")
    assert warning is not None
    assert "baseline_scaffold" in warning
    assert "hdbscan" not in warning
