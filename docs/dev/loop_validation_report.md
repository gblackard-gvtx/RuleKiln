# Loop Validation Report

**Purpose:** Confirm or refute the claims in SPEC_close_conflict_resolution_loop.md before modifying code.

---

## 0.1 — Where is pruning invoked? Is utility_signals ever non-None/non-empty?

**Findings:**

Two call sites in `src/rulekiln/workers/distillation_worker.py`:

**Call site A — line 657** (stage `PRUNING_RULES`):
```python
pruning_result = prune_rules(
    schemas,
    max_rules=task.max_rules,
    max_prompt_tokens=task.max_prompt_tokens,
    min_rule_support_count=task.min_rule_support_count,
    preserve_golden_rules=task.preserve_golden_rules,
)
```
No `ranking_mode`, no `utility_signals`. This is the **main pipeline pruning**, and runs **before** evaluation. Straight-line confirmed: `utility_signals` is `None` here every time.

**Call site B — line 2671** (inside `_run_two_pass_optimizer`, stage `OPTIMIZING_PRUNING`):
```python
pruned_result = prune_rules(
    schemas_all,
    ...
    ranking_mode=mode,
    regression_penalty=regression_penalty,
    utility_signals=utility_signals,
)
```
`utility_signals` IS passed and IS non-empty here. But this is the Phase 4 **pruning mode comparison** experiment, not the main loop. Its `utility_signals` comes from the ablation artifact (line 2600–2617), **not** from `analyze_failures`. The pruning mode optimizer runs post-evaluation but independently of failure attribution.

**Pipeline order (straight line, confirmed):**
```
extracting_rules → ... → reviewing_rule_conflicts → pruning_rules (Call A, utility_signals=None)
→ compiling_prompts → evaluating_baseline → evaluating_distilled → selecting_strategy
→ analyzing_failures (result discarded, line 1582) → ablating_rules → optimizing_pruning
→ checking_quality_gates → logging_artifacts → exporting_artifacts
```

No loop-back exists. The `analyze_failures` result at line 1582 is not stored — the return value is silently dropped.

---

## 0.2 — Is the `outcome_conditions` loop a stub? Is `failed_assertion_types` always empty?

**Findings (failure_analysis.py lines 66–73):**

```python
rule_output_path_index: dict[str, str] = {}
if selected_rules:
    for rule in selected_rules:
        for _oc in rule.outcome_conditions.values():
            # outcome_conditions may have paths embedded in the "when" conditions
            pass                              ← stub confirmed
        if rule.id:
            rule_output_path_index[rule.topic.lower()] = rule.id   ← topic-only index confirmed
```

The `outcome_conditions` loop body is literally `pass`. The index contains only `rule.topic.lower() → rule.id`.

**`failed_assertion_types` (line 131):**
```python
failed_types: list[str] = []
```
This list is never populated. `CaseEvaluationFailure.failed_assertion_types` is always `[]`. Confirmed.

---

## 0.3 — What is the actual shape of `CaseEvalResult.assertion_scores` keys?

**Findings (evaluator.py line 122):**
```python
for i, assertion in enumerate(case.evaluation.assertions):
    score = _score_assertion(assertion.type, assertion.value, actual, assertion.path)
    key = f"assertion_{i}"
    assertion_scores[key] = score
```

**Concrete key format:** `"assertion_0"`, `"assertion_1"`, ..., `"assertion_{n-1}"`  
where `i` is the **0-based index** into `case.evaluation.assertions`.

Example for a single-assertion classification case: `assertion_scores = {"assertion_0": 1.0}` (pass) or `{"assertion_0": 0.0}` (fail).

**Note:** The test fixture in `test_failure_analysis.py` uses `{"path.a": 0.0}` — this is a manually-constructed `CaseEvalResult` that does NOT come from the evaluator. Real evaluator output always uses `assertion_{i}` keys.

---

## 0.4 — What is the structure of `SynthesizedRuleSchema.outcome_conditions`?

**Schema (pipeline.py lines 123–136):**
```python
class OutcomeCondition(BaseModel):
    outcome: str           # the outcome label, e.g. "entailment", "positive"
    when: list[str]        # conditions under which this outcome applies
    confidence: str = "high"

class SynthesizedRuleSchema(BaseModel):
    ...
    outcome_conditions: dict[str, OutcomeCondition]
```

The dict **key** is the outcome name/label (e.g. `"entailment"`, `"contradiction"`). `OutcomeCondition.outcome` holds the same label string.

**The join for attribution:** For a failing `assertion_i`, `case.evaluation.assertions[i].value` is the expected outcome label (e.g. `"entailment"`). A rule is violated if any of its `OutcomeCondition.outcome` values equal that label. Building the index:

```
outcome_label → [rule_ids whose outcome_conditions contain that outcome]
```

This index bridges `assertion_scores` keys (via case lookup + assertion index) to `SynthesizedRuleSchema.outcome_conditions`.

---

## 0.5 — Does `review_rule_for_conflicts` run the student or inspect case outcomes?

**Findings (agents/rule_conflict_review.py lines 72–115):**

`review_rule_for_conflicts` is purely **linguistic**:
- Input: task definition, synthesized rule, supporting micro-rules
- Builds a text prompt describing the rule's conditions and outcomes
- Calls the teacher model to check for internal contradictions
- Returns a `RuleConflictReview` with resolution = `keep | modify | split | discard`

No student model is called. No case outcomes are inspected. Confirmed: this is a pre-evaluation hygiene check, not the paper's Phase 3.

**`discard` path:** `resolution="discard"` leaves `resolved_rules=[]` and sets `has_conflicts=True` on the rule. In `prune_rules` (rule_pruning.py line 127–129), `rule.has_conflicts` causes pruning with reason `"unresolved_conflict"`. The discard path does drop rules.

---

## Claim verification summary

| Claim | Verdict |
|---|---|
| `prune_rules` never called with non-None `utility_signals` in main pipeline | **TRUE** — Call A (PRUNING_RULES) has no utility_signals. Call B (OPTIMIZING_PRUNING) uses ablation-derived signals, not failure analysis. |
| Straight-line, no loop-back | **TRUE** — `analyze_failures` result discarded at line 1582 |
| `outcome_conditions` loop is `pass` | **TRUE** — line 70 |
| `rule_output_path_index` keyed only on `topic.lower()` | **TRUE** — line 73 |
| `failed_assertion_types` always `[]` | **TRUE** — line 131 |
| `assertion_scores` keys are `assertion_{i}` | **TRUE** — evaluator.py line 122 |
| `review_rule_for_conflicts` is purely linguistic | **TRUE** |
| `discard` drops rules with no validation evidence | **TRUE** |

All spec claims hold. No adjustments needed to the implementation plan.
