# ContractNLI HDBSCAN Malformed Output & Regression Investigation

Date: 2026-07-07
Job under investigation: `bf6c5492-f502-4d4d-893b-26790bc81951` (300 validation cases,
task `contractnli_nli_classification`, output labels `contradiction | entailment | neutral`)
Evidence source: Postgres `eval_case_results` (raw, untruncated `raw_output` per case) and
`synthesized_rules` for job `bf6c5492`, both queried directly against the live
`rulekiln-postgres-1` container. This is the authoritative source for a completed job's
case-level results — the on-disk `{strategy}_eval.json` artifact does not retain raw model
output at this granularity, which is why item H below adds a dedicated debug-artifact writer.

## Observed issue

`hdbscan` reported `malformed_output_rate` 0.06 (18/300, cap is 0.01) and `regression_rate`
0.1239 (27/218 baseline-passing cases went from pass → fail, cap is 0.10 —
`src/rulekiln/config/settings.py:63,65`), both over the promotion gates in
`quality_gates.py`, correctly keeping HDBSCAN ineligible for selection. `baseline_scaffold`
malformed rate on the same job was 2/300 (0.67%).

## Root causes

### 1. `_render_rule()` rendered the dict key, not the validated outcome field

`SynthesizedRuleSchema.outcome_conditions` is `dict[str, OutcomeCondition]`. The dict *key*
is a free-form mnemonic the teacher model invents to label a cluster of conditions (e.g.
`entailment_verbal_included`, `entailment_exception_third_party_source`); the *validated*
label lives in `OutcomeCondition.outcome`. `_render_rule()` in
`src/rulekiln/pipeline/prompt_compiler.py` rendered the dict key directly into the compiled
prompt's "**Outcomes:**" section instead of `oc.outcome`:

```python
for outcome_name, oc in rule.outcome_conditions.items():
    lines.append(f"- **{outcome_name}**: ...")   # bug: outcome_name is the free mnemonic key
```

Querying job `bf6c5492`'s 16 non-pruned HDBSCAN rules directly from Postgres confirmed the
teacher populates `oc.outcome` correctly in the large majority of cases (canonical label)
while giving the dict key a descriptive mnemonic — **13 of the rules' outcome
keys did not match their own `oc.outcome` value.** The student model, reading
`- **entailment_verbal_included**: ...` as an instruction, echoed the mnemonic back as its
final answer verbatim.

### 2. The same mnemonics also leak into free-text guidance, not just dict keys

Fixing (1) alone was insufficient. Re-rendering job `bf6c5492`'s real persisted rules
through the patched `_render_rule()` still reproduced 5 of the 13 known leaked strings,
because the teacher model independently writes the identical compound mnemonic into
`tie_breakers` / `applies_when` / `OutcomeCondition.when` prose (e.g. *"If premise mentions
oral/verbal disclosure, choose entailment_verbal_included over any broad/ambiguous
language."*). This is a second, distinct leak path the dict-key fix does not cover.

### 3. Distilled prompts did not reinforce the same final-answer contract as baseline

`compile_baseline_prompt()` appends an explicit JSON/allowed-values contract after the task
description; `compile_prompt()` (the distilled path) did not — the student had to infer the
contract purely from the system/task scaffold, with no repetition after ~26 rules' worth of
intervening text.

## Fixes

| # | File | Change |
|---|------|--------|
| 1 | `src/rulekiln/pipeline/prompt_compiler.py` | `_render_rule()` renders `_canonical_outcome_label(oc.outcome, allowed_labels)` (with synonym mapping, dropping unresolvable values) instead of the dict key. |
| 2 | `src/rulekiln/pipeline/prompt_compiler.py` | New `_normalize_compound_label_mentions()`: rewrites `label(_word)+` mentions found in `applies_when`, `tie_breakers`, and `OutcomeCondition.when` prose down to the bare canonical label, via `_sanitize_rule(rule, allowed_labels)`. Applied before the forbidden-instruction filter. |
| 3 | `src/rulekiln/pipeline/prompt_compiler.py` | `compile_prompt()` appends the same "# Final Answer Format (Required)" JSON/allowed-values contract that `compile_baseline_prompt()` uses, after the rules block, whenever `task.output_schema` defines an enum. |
| 4 | `src/rulekiln/pipeline/prompt_compiler.py` | `_sanitize_rule()` also strips forbidden output-format instructions (`_FORBIDDEN_INSTRUCTION_PATTERNS`) from rule text without dropping the whole rule, so distilled rules can no longer tell the student to change its output format. |
| 5 | `src/rulekiln/agents/rule_synthesis.py` | `_SYNTHESIS_SYSTEM_PROMPT` strengthened: outcome keys/values must be exactly one of the task's defined labels, never invented compound variants; rules must not include response-format instructions. `_build_synthesis_prompt()` now enumerates the task's allowed labels explicitly when `output_schema` is present — source-side defense in depth alongside the compiler-side sanitizer. |
| 6 | `src/rulekiln/pipeline/evaluator.py` | `_call_student()` now returns raw content alongside the parsed result; new `_try_recover_label()` recovers a single unambiguous valid label surrounded by explanatory prose (case/punctuation/whitespace insensitive). Structurally cannot recover multiple distinct labels or compound underscore-joined strings (`\b` does not match before `_`), so it cannot launder the leak this investigation targets or any genuinely ambiguous answer. Recovery is flagged (`recovered=True`), never hidden. |
| 7 | `src/rulekiln/artifacts/writer.py` | New `write_strategy_debug_artifacts()`: per strategy, writes `_raw_outputs.jsonl`, `_parsed_outputs.jsonl`, `_malformed_outputs.jsonl`, `_regressions_vs_baseline.jsonl`, `_prompt_preview.txt`, `_compiled_prompt.txt` (hash-tagged), `_confusion_matrix.json`, `_classification_report.json` under `outputs/debug/`. |
| 8 | `src/rulekiln/workers/pipeline/output_stage_services.py` | `ArtifactExportService.run()` wired to call the above for every strategy in `comparison.strategy_evals` (not just distilled ones), using `state.baseline_scaffold_eval or state.baseline_eval` for the regression comparison. |

No changes were made to `quality_gates.py`, gate thresholds, clustering parameters, or
metric definitions — per constraint J.

## A. Malformed-output bucket analysis (18 HDBSCAN cases)

Every malformed case's full `raw_output` was pulled from Postgres and classified:

| Bucket | Count | % |
|---|---|---|
| Parser bug / normalization mismatch (compound mnemonic leaked as literal label — root causes 1+2) | 15 | 83.3% |
| Explanation with no final label / reasoning-trace dump (long chain-of-thought, no clean parseable terminal label line) | 2 | 11.1% |
| Valid label + extra text (clean terminal label line after explanation — recoverable) | 1 | 5.6% |
| Valid synonym not normalized | 0 | — |
| JSON/schema formatting issue | 0 | — |
| Multiple labels returned | 0 | — |
| Empty/truncated response | 0 | — |
| Truly invalid answer (non-responsive, no attempt at classification) | 0 | — |

The 15 compound-leak cases are an exact string match against one of the 13 mnemonic dict
keys independently confirmed (via direct Postgres inspection of `synthesized_rules`) to be
leaking into the compiled prompt — e.g. `entailment_verbal_included` appears verbatim as the
raw model output in 5 separate cases. This is the dominant failure mode and is exactly what
fixes 1–4 target. The 2 reasoning-trace-dump cases and 1 recoverable case are addressed by
fix 6 (parser recovery) where a clean terminal label exists (`contractnli_validation_000108`,
which ends in `Output: entailment`); the 2 cases with no clean terminal label
(`_000063`, `_000208`) are not recoverable by design and remain correctly malformed.

## B. HDBSCAN malformed cases vs. baseline_scaffold on the same case IDs

| Baseline_scaffold outcome on HDBSCAN's 18 malformed case IDs | Count |
|---|---|
| Baseline passed (correct, well-formed) | 10 |
| Baseline failed but produced a valid label (case is hard regardless of strategy) | 7 |
| Baseline also malformed | 1 |

10 of 18 (56%) are cases baseline_scaffold answered *correctly* — direct evidence the
distilled rules actively caused failures that would not otherwise occur, not that these are
simply hard cases. Only 1 of 18 (`contractnli_validation_000028`) was malformed under both
strategies (baseline echoed a raw contract clause with no label attempt at all — an
unrelated, pre-existing failure mode, not a rules-format interference case). This confirms
the distilled rules interfere with the output contract; it is not shared model confusion.

## G. Regression bucket analysis (27 cases: baseline passed, HDBSCAN did not)

`regression_rate = 27 / 218 = 0.1239`, matching the reported gate violation exactly.

| Bucket | Count | % |
|---|---|---|
| Malformed-only regression (compound-leak/reasoning-dump case whose canonical/terminal label matches expected — i.e., would have passed if not malformed) | 10 | 37.0% |
| Neutral / not-mentioned confusion (expected or predicted `neutral`, confused with either other label) | 15 | 55.6% |
| Contradiction/entailment confusion (expected `entailment`, predicted `contradiction` or vice versa) | 2 | 7.4% |
| Label flip from over-broad distilled rule | 0 | — |
| Prompt-too-long / instruction conflict | 0 | — |
| Parser/normalization issue (distinct from malformed) | 0 | — |
| True model reasoning miss (unclassifiable by the above) | 0 | — |

All 10 malformed-only regressions have a canonical label matching the expected label — e.g.
`contractnli_validation_000121` expects `entailment`, HDBSCAN emitted the malformed
`entailment_verbal_included`, whose canonical root is `entailment`. Fixes 1–4 eliminate the
leak at the source (the mnemonic is no longer compiled into the prompt at all), and fix 6
recovers the reasoning-dump case with a clean terminal label. **Projected impact: removing
these 10 malformed-only regressions would drop the measured regression rate from 12.39% to
17/218 = 7.80%**, under the 10% cap — but this is an offline projection from re-rendering
the prompt, not an empirical re-measurement (see Validation evidence below).

The remaining 17 regressions (15 neutral-confusion + 2 contradiction/entailment-confusion)
are well-formed, syntactically valid predictions that are simply wrong — genuine distilled-rule
quality issues, not a format bug. None of this investigation's fixes address them; per
constraint J ("do not do broad clustering retuning," "do not relax gates just to promote
HDBSCAN") they are out of scope here and would require separate rule-quality work.

## Validation evidence

- Job `bf6c5492`'s actual persisted HDBSCAN rules (16 non-pruned) were re-rendered through
  the patched `compile_prompt()` offline. Before fix 2, 5/13 known-leaked mnemonic strings
  still appeared in the compiled prompt (from free-text prose, not the dict-key path already
  fixed). After fix 2, 12/13 (92%) are eliminated.
- **Known residual limitation**: one string, `not_entailment_or_contradiction`, still appears
  in a `tie_breakers` entry. It is not a valid label to begin with (already correctly dropped
  from the rendered "**Outcomes:**" section as unresolvable by `_canonical_outcome_label`),
  but the tie-breaker prose mention survives because `_normalize_compound_label_mentions`'s
  regex requires a word boundary before the label token — `not_entailment_or_contradiction`
  has no boundary before `entailment` (preceded by `_`), so it isn't matched. This is
  documented rather than patched further with a speculative regex, to avoid overfitting to
  one job's data (per "make the smallest safe change").
- **This report validates the fix by offline prompt re-rendering against real production
  rules and real production eval data, not by re-running the pipeline.** The bucket tallies
  in A/B/G are computed from job `bf6c5492`'s actual historical `eval_case_results`, which
  predate the fix. No new student-model inference has been run against the patched prompt.
  Per constraint J, HDBSCAN should not be promoted until a fresh run of this job (or an
  equivalent) empirically measures `malformed_output_rate ≤ 0.01` and
  `regression_rate ≤ 0.10` under the fixed compiler — the projections above (12/13 leaks
  eliminated, projected regression rate 7.80%) are supporting evidence for expecting that
  outcome, not a substitute for it.
- New/extended tests (all passing): `test_prompt_compiler_determinism.py` (contract
  inheritance and ordering, sanitizer strips forbidden instructions without dropping the
  rule, outcome-value vs dict-key regression test, synonym mapping, unresolvable-label
  dropping, compound-label-in-free-text normalization regression test),
  `test_evaluator_label_vocabulary.py` (parser recovery: exact/punctuation/casing/whitespace/
  explanation-wrapped recover; multiple distinct labels and compound-leak strings stay
  malformed), `test_strategy_debug_artifacts.py` (new — all 8 artifact types, regression
  jsonl only contains baseline-pass/candidate-fail cases), `test_shared_provider_config.py`
  (new — confirms generation config has no per-strategy divergence), `test_rule_synthesis_prompt.py`
  (new — synthesis prompt forbids format instructions and invented label variants, enumerates
  allowed labels), `test_strategy_selection_regression.py` (extended — HDBSCAN stays
  ineligible while malformed/regression rates exceed the *unchanged* gates, becomes eligible
  only once both are within them).
- Full suite: 618 passed (`tests/unit`, `tests/integration`, `tests/ui`, `-m "not external"`).
  `ruff` and `pyright` clean on all touched files.

## F. Generation settings across strategies

Already unified before this investigation: `baseline_eval_service.py` and
`distilled_eval_service.py` both call `evaluate_prompt()` with the same
`bundle.student_config` (`ProviderConfig`) instance — there is no per-strategy temperature/
top_p/max_tokens divergence. `resolve_provider_config()` takes no `strategy` parameter.
Confirmed by `test_shared_provider_config.py`, added as a regression guard rather than a fix.

## Remaining risks / follow-ups

1. **Not empirically re-measured.** As stated above, promotion still requires a fresh
   pipeline run of this job (or equivalent) against the patched compiler before HDBSCAN can
   be considered for selection under the existing, unchanged gates.
2. **17 non-malformed regressions are untouched.** These are distilled-rule quality issues
   (neutral vs. contradiction/entailment confusion), not a format bug; separate follow-up
   work, out of scope for this investigation per constraint J.
3. **One known residual leak string** (`not_entailment_or_contradiction`, an already-invalid
   label) remains in tie-breaker prose; documented above rather than patched with a
   speculative regex.
