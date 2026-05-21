"""Unit tests for prompt_compiler determinism (T026)."""

import pytest

from rulekiln.pipeline.prompt_compiler import compile_prompt, count_tokens_approx
from rulekiln.schemas.pipeline import OutcomeCondition, SynthesizedRuleSchema
from rulekiln.schemas.task_case import RuleKilnTask


def _task() -> RuleKilnTask:
    return RuleKilnTask(
        task_id="t1",
        task_name="Test Task",
        task_mode="classification",
        description="A test",
        input_template="{{input}}",
    )


def _rules(n: int = 3) -> list[SynthesizedRuleSchema]:
    return [
        SynthesizedRuleSchema(
            topic=f"topic_{i}",
            applies_when=[f"when_{i}"],
            outcome_conditions={
                f"outcome_{i}": OutcomeCondition(
                    outcome=f"outcome_{i}", when=[f"when_{i}"], confidence="high"
                )
            },
            tie_breakers=[],
            priority=i,
            source_case_ids=[f"case_{i}"],
            source_micro_rule_ids=[f"rule_{i}"],
        )
        for i in range(n)
    ]


def test_same_inputs_yield_same_hash() -> None:
    task = _task()
    rules = _rules()
    _, hash1 = compile_prompt(task, rules, "dbscan")
    _, hash2 = compile_prompt(task, rules, "dbscan")
    assert hash1 == hash2


def test_different_strategy_yields_different_hash() -> None:
    task = _task()
    rules = _rules()
    _, hash_db = compile_prompt(task, rules, "dbscan")
    _, hash_hdb = compile_prompt(task, rules, "hdbscan")
    assert hash_db != hash_hdb


def test_rule_order_does_not_affect_hash() -> None:
    """Rules sorted by (priority, topic) → permuting input order gives same hash."""
    task = _task()
    rules = _rules(4)
    shuffled = [rules[2], rules[0], rules[3], rules[1]]
    _, hash_orig = compile_prompt(task, rules, "dbscan")
    _, hash_shuf = compile_prompt(task, shuffled, "dbscan")
    assert hash_orig == hash_shuf


def test_additional_instructions_change_hash() -> None:
    task = _task()
    rules = _rules()
    _, hash1 = compile_prompt(task, rules, "dbscan", additional_instructions="extra")
    _, hash2 = compile_prompt(task, rules, "dbscan")
    assert hash1 != hash2


def test_prompt_text_is_str() -> None:
    task = _task()
    text, digest = compile_prompt(task, _rules(), "dbscan")
    assert isinstance(text, str)
    assert len(digest) == 64  # SHA-256 hex


def test_count_tokens_approx() -> None:
    assert count_tokens_approx("") == 1  # min=1 by implementation
    assert count_tokens_approx("abcd") == 1
    assert count_tokens_approx("a" * 400) == 100
