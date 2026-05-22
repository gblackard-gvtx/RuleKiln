"""Unit tests for rule pruning service."""

from __future__ import annotations

from rulekiln.pipeline.rule_pruning import prune_rules
from rulekiln.schemas.pipeline import OutcomeCondition, SynthesizedRuleSchema


def _rule(
    rule_id: str = "r1",
    topic: str = "topic",
    priority: int = 1,
    support_count: int = 3,
    support_ratio: float = 0.5,
    has_conflicts: bool = False,
    golden_case_backed: bool = False,
    estimated_token_count: int = 50,
    source_case_ids: list[str] | None = None,
) -> SynthesizedRuleSchema:
    return SynthesizedRuleSchema(
        id=rule_id,
        topic=topic,
        applies_when=["some condition"],
        outcome_conditions={
            "out": OutcomeCondition(outcome="out", when=["cond"], confidence="high")
        },
        tie_breakers=[],
        priority=priority,
        source_case_ids=source_case_ids or ["c1"],
        source_micro_rule_ids=["m1"],
        has_conflicts=has_conflicts,
        support_count=support_count,
        support_ratio=support_ratio,
        golden_case_backed=golden_case_backed,
        estimated_token_count=estimated_token_count,
    )


def test_unresolved_conflicts_pruned() -> None:
    rules = [_rule("r1", has_conflicts=True), _rule("r2")]
    result = prune_rules(rules, max_rules=10, max_prompt_tokens=9999)
    assert len(result.selected) == 1
    assert result.selected[0].id == "r2"
    assert any(r.reason == "unresolved_conflict" for r in result.pruned)


def test_below_min_support_pruned() -> None:
    rules = [_rule("r1", support_count=1), _rule("r2", support_count=5)]
    result = prune_rules(rules, min_rule_support_count=2, max_rules=10, max_prompt_tokens=9999)
    assert len(result.selected) == 1
    assert result.selected[0].id == "r2"
    assert any(r.reason == "below_min_support" for r in result.pruned)


def test_golden_rules_preserved_below_min_support() -> None:
    rules = [
        _rule("r1", support_count=1, golden_case_backed=True),
        _rule("r2", support_count=5),
    ]
    result = prune_rules(rules, min_rule_support_count=2, preserve_golden_rules=True)
    assert len(result.selected) == 2


def test_golden_rules_not_preserved_when_flag_off() -> None:
    rules = [_rule("r1", support_count=1, golden_case_backed=True)]
    result = prune_rules(rules, min_rule_support_count=2, preserve_golden_rules=False)
    assert len(result.selected) == 0


def test_max_rules_cap() -> None:
    rules = [_rule(f"r{i}", priority=i) for i in range(10)]
    result = prune_rules(rules, max_rules=5, max_prompt_tokens=9999, min_rule_support_count=1)
    assert len(result.selected) == 5
    assert all(r.reason == "max_rules_exceeded" for r in result.pruned)


def test_token_budget_cap() -> None:
    # Each rule has estimated_token_count=100; budget=250 → 2 rules fit
    rules = [_rule(f"r{i}", estimated_token_count=100) for i in range(5)]
    result = prune_rules(rules, max_rules=10, max_prompt_tokens=250, min_rule_support_count=1)
    assert len(result.selected) == 2
    assert all(r.reason == "prompt_token_budget_exceeded" for r in result.pruned)


def test_sort_order_priority_then_support() -> None:
    rules = [
        _rule("r1", priority=2, support_count=10),
        _rule("r2", priority=1, support_count=5),
        _rule("r3", priority=1, support_count=8),
    ]
    result = prune_rules(rules, max_rules=3, max_prompt_tokens=9999, min_rule_support_count=1)
    ids = [r.id for r in result.selected]
    # priority=1 rules first (sorted by priority asc), then r3 before r2 (higher support)
    assert ids[0] == "r3"
    assert ids[1] == "r2"
    assert ids[2] == "r1"


def test_empty_input_returns_empty() -> None:
    result = prune_rules([], max_rules=10)
    assert result.selected == []
    assert result.pruned == []


def test_pruning_report_structure() -> None:
    rules = [_rule("r1", has_conflicts=True), _rule("r2"), _rule("r3")]
    result = prune_rules(rules, max_rules=10, max_prompt_tokens=9999)
    report = result.pruning_report()
    assert "selected_count" in report
    assert "pruned_count" in report
    assert report["pruned_count"] == 1
