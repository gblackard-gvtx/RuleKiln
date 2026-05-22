"""Unit tests for conflict review agent output contract."""

from __future__ import annotations

import pytest

from rulekiln.schemas.pipeline import RuleConflictReview, SynthesizedRuleSchema, OutcomeCondition


def _rule(rule_id: str = "r1") -> SynthesizedRuleSchema:
    return SynthesizedRuleSchema(
        id=rule_id,
        topic="Loan approval",
        applies_when=["credit score > 700"],
        outcome_conditions={
            "approve": OutcomeCondition(outcome="approve", when=["score > 700"], confidence="high"),
            "deny": OutcomeCondition(outcome="deny", when=["score < 600"], confidence="high"),
        },
        tie_breakers=[],
        priority=1,
        source_case_ids=["c1"],
        source_micro_rule_ids=["m1"],
    )


def test_rule_conflict_review_valid_keep() -> None:
    review = RuleConflictReview(
        synthesized_rule_id="r1",
        has_conflicts=False,
        conflict_summary=None,
        conflicting_micro_rule_ids=[],
        resolution="keep",
        resolved_rules=[],
    )
    assert review.resolution == "keep"
    assert not review.has_conflicts


def test_rule_conflict_review_valid_discard() -> None:
    review = RuleConflictReview(
        synthesized_rule_id="r1",
        has_conflicts=True,
        conflict_summary="Contradictory conditions",
        conflicting_micro_rule_ids=["m1", "m2"],
        resolution="discard",
        resolved_rules=[],
    )
    assert review.resolution == "discard"
    assert len(review.conflicting_micro_rule_ids) == 2


def test_rule_conflict_review_valid_split() -> None:
    resolved = _rule("r2")
    review = RuleConflictReview(
        synthesized_rule_id="r1",
        has_conflicts=True,
        conflict_summary="Split into two",
        conflicting_micro_rule_ids=["m1"],
        resolution="split",
        resolved_rules=[resolved],
    )
    assert len(review.resolved_rules) == 1


def test_rule_conflict_review_invalid_resolution() -> None:
    with pytest.raises(Exception):
        RuleConflictReview(
            synthesized_rule_id="r1",
            has_conflicts=False,
            conflict_summary=None,
            conflicting_micro_rule_ids=[],
            resolution="invalid_value",  # type: ignore[arg-type]
            resolved_rules=[],
        )
