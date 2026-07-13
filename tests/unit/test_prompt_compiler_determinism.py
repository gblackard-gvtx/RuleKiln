"""Unit tests for prompt_compiler determinism (T026)."""

from rulekiln.pipeline.prompt_compiler import (
    _render_rule,  # pyright: ignore[reportPrivateUsage]
    _sanitize_rule,  # pyright: ignore[reportPrivateUsage]
    compile_baseline_prompt,
    compile_prompt,
    count_tokens_approx,
)
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


# ── output-contract reinforcement (items I.1, I.2) ─────────────────────────────


def _task_with_labels() -> RuleKilnTask:
    return RuleKilnTask(
        task_id="contractnli",
        task_name="ContractNLI",
        task_mode="classification",
        description="NLI over contract clauses",
        input_template="{{ premise }} / {{ hypothesis }}",
        output_schema={
            "type": "object",
            "required": ["label"],
            "properties": {
                "label": {"type": "string", "enum": ["entailment", "contradiction", "neutral"]}
            },
        },
    )


def _labeled_rules(n: int = 40) -> list[SynthesizedRuleSchema]:
    """A long rule block, mirroring the up-to-40-rule HDBSCAN prompts from job bf6c5492."""
    labels = ["entailment", "contradiction", "neutral"]
    return [
        SynthesizedRuleSchema(
            topic=f"topic_{i}",
            applies_when=[f"when_{i}"],
            outcome_conditions={
                f"{labels[i % 3]}_mnemonic_{i}": OutcomeCondition(
                    outcome=labels[i % 3], when=[f"cue_{i}"], confidence="high"
                )
            },
            tie_breakers=[],
            priority=i,
        )
        for i in range(n)
    ]


def test_distilled_prompt_reinforces_same_contract_as_baseline() -> None:
    """Item I.1: the distilled prompt appends the exact same final answer contract
    (allowed labels) that compile_baseline_prompt states in its Output Format /
    Allowed Values sections."""
    task = _task_with_labels()
    baseline_text = compile_baseline_prompt(task)
    distilled_text, _ = compile_prompt(task, _labeled_rules(), "hdbscan")

    assert "# Final Answer Format (Required)" in distilled_text
    for label in ("entailment", "contradiction", "neutral"):
        assert label in baseline_text
        assert label in distilled_text
    assert "Do not include explanations, markdown, or prose outside the JSON object." in (
        distilled_text
    )


def test_final_answer_contract_appears_after_rules_block_not_before() -> None:
    """Item I.2: the reinforced contract sits after the distilled rules, not before —
    it exists specifically to stay close to the end of a long rules block."""
    task = _task_with_labels()
    distilled_text, _ = compile_prompt(task, _labeled_rules(), "hdbscan")

    rules_index = distilled_text.index("# Distilled Rule Policy")
    contract_index = distilled_text.index("# Final Answer Format (Required)")
    last_rule_index = distilled_text.rindex("## Rule")
    assert rules_index < contract_index
    assert last_rule_index < contract_index


def test_compile_prompt_without_output_schema_omits_final_contract() -> None:
    """No enum labels to reinforce → no contract section is added (unchanged behavior)."""
    task = _task()
    distilled_text, _ = compile_prompt(task, _rules(), "hdbscan")
    assert "# Final Answer Format (Required)" not in distilled_text


# ── rule sanitizer forbids output-format instructions (item I.3) ──────────────


def test_sanitizer_strips_forbidden_instruction_without_dropping_whole_rule() -> None:
    rule = SynthesizedRuleSchema(
        id="r1",
        topic="verbal grant of rights",
        applies_when=["explain your reasoning before answering", "clause grants a license"],
        outcome_conditions={
            "entailment": OutcomeCondition(
                outcome="entailment",
                when=["license is granted", "return JSON with a rationale field"],
            )
        },
        tie_breakers=["state why entailment beats neutral"],
    )
    sanitized = _sanitize_rule(rule)

    assert sanitized.applies_when == ["clause grants a license"]
    assert sanitized.outcome_conditions["entailment"].when == ["license is granted"]
    assert sanitized.tie_breakers == []
    # The rule itself survives — only the offending entries are dropped.
    assert sanitized.topic == "verbal grant of rights"
    assert sanitized.id == "r1"


# ── outcome-label rendering (root-cause regression, item A) ────────────────────


def test_render_rule_uses_outcome_value_not_dict_key() -> None:
    """Regression test for the root-cause bug: the compiled prompt must render
    OutcomeCondition.outcome (canonical), never the free-form dict key the teacher
    model used as a mnemonic (e.g. 'entailment_verbal_included')."""
    rule = SynthesizedRuleSchema(
        topic="verbal grant",
        outcome_conditions={
            "entailment_verbal_included": OutcomeCondition(
                outcome="entailment", when=["oral license granted"]
            )
        },
    )
    allowed = frozenset({"entailment", "contradiction", "neutral"})
    rendered = _render_rule(1, rule, allowed)

    assert "entailment_verbal_included" not in rendered
    assert "**entailment**" in rendered


def test_render_rule_maps_known_synonym_to_canonical_label() -> None:
    rule = SynthesizedRuleSchema(
        topic="synonym check",
        outcome_conditions={"k": OutcomeCondition(outcome="entails", when=["cue"])},
    )
    allowed = frozenset({"entailment", "contradiction", "neutral"})
    rendered = _render_rule(1, rule, allowed)
    assert "**entailment**" in rendered
    assert "entails" not in rendered


def test_compile_prompt_normalizes_compound_label_mentions_in_free_text() -> None:
    """Regression test: the teacher model sometimes writes a compound mnemonic label
    directly into tie_breakers/applies_when/when prose (not just as an
    outcome_conditions key), e.g. 'choose entailment_verbal_included'. That must be
    normalized down to the canonical label rather than leaking into the compiled
    prompt verbatim — discovered by re-rendering job bf6c5492's real HDBSCAN rules."""
    task = _task_with_labels()
    rule = SynthesizedRuleSchema(
        topic="verbal disclosure",
        applies_when=["oral disclosure clause present"],
        outcome_conditions={
            "entailment": OutcomeCondition(outcome="entailment", when=["cue"]),
        },
        tie_breakers=[
            "If premise mentions oral/verbal disclosure, choose entailment_verbal_included."
        ],
    )
    prompt_text, _ = compile_prompt(task, [rule], "hdbscan")

    assert "entailment_verbal_included" not in prompt_text
    assert "choose entailment." in prompt_text


def test_render_rule_drops_unresolvable_outcome_label() -> None:
    """An outcome value that isn't in the allowed set and has no synonym is dropped
    from the rendered rule rather than leaking an arbitrary string as the label."""
    rule = SynthesizedRuleSchema(
        topic="unresolvable",
        outcome_conditions={"k": OutcomeCondition(outcome="maybe_entailment", when=["cue"])},
    )
    allowed = frozenset({"entailment", "contradiction", "neutral"})
    rendered = _render_rule(1, rule, allowed)
    assert "maybe_entailment" not in rendered
    assert "- **" not in rendered
