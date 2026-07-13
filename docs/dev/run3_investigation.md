# Run 3 Investigation — k5 underperformance & suspected baseline regression

Date: 2026-07-03
Job under investigation: `4dfcd4c2-d254-4931-9e00-dac37e9744f7` (Run 3)

## TL;DR

- **The log file does not contain the evidence the investigation plan assumed.**
  `logs/rulekiln.log` is 99.9% SQLAlchemy query logging plus a handful of
  startup events. There are **no student responses, prompts, per-case results,
  or strategy scores** in it. Steps 1.3–1.6 (extract raw responses / prompts by
  job ID) cannot be answered from any local artifact — nothing persists them to
  disk. So the analysis below is grounded in **the code that actually ran**,
  which is stronger evidence than a log would have been.
- **Question A is based on a false premise.** There is no separate
  "baseline formatting" code path. Baseline, scaffold, k3, and k5 all render the
  user message through the *same* function (`_call_student` →
  `_render_input_from_template`). The Run 3 template change was one line and
  applied uniformly to every strategy. It cannot be "scoped back to few-shot
  only" without inventing a new conditional that would deliberately give the
  baseline a worse format. **Not implemented.**
- **Question B hypothesis #1 ("template not applied to k5") is false** — k5
  provably uses the template (same shared path). Hypothesis #3 ("weak separator
  at k=5") is unlikely — the separator/suffix is byte-for-byte identical for k3
  and k5. **The real, deterministic cause is example-selection label
  imbalance:** for a 3-label task, k3 selects a perfectly balanced 1/1/1 set and
  k5 selects an imbalanced 2/2/1 set that under-represents one label.
- **The YAML scope-colon bug is already fixed** in the checked-out tree
  (`contractnli/task.yaml` quotes its colon items; no other `task.yaml` has the
  problem). Added a CI regression guard so it can't come back.

---

## Step 1 — What the log actually contains

`logs/rulekiln.log` (current) — 13,822 JSON lines:

| logger                     | count  |
|----------------------------|--------|
| `sqlalchemy.engine.Engine` | 13,815 |
| `rulekiln.api.lifespan`    | 5      |
| `__main__`                 | 2      |

The only non-SQL events are `dbos_worker_started`, `dbos_worker_shutdown_signal_received`,
four `provider_profile_loaded`, and one `rulekiln_startup`. The rotated logs
(`rulekiln.log.1`, `.2`) are 100% SQLAlchemy. Filtering by the job ID returns 14
lines, all of which are SQL bind-parameter tuples (e.g.
`('4dfcd4c2-…', 'dbscan', 'validation') [cached since …]`) — no response bodies.

Root cause: `configure_logging` attaches the root logger at INFO with
SQLAlchemy `echo` enabled, and the evaluator only emits two `warning` events
(`malformed_output`, `malformed_output_exception`). Student responses and
rendered prompts are never logged at any level, and there is no local MLflow
store (`mlruns/`) or results DB checked out. **The requested raw-response and
prompt extraction is therefore not possible from local artifacts.** Scores in
the smoke table came from the run's stdout / a remote Postgres, neither of which
is available here.

### Consequence for verification
Because per-case predictions are not persisted locally, the specific claims
"baseline = 0.3000 in Run 3" and "k5 returned continuation text" cannot be
confirmed or refuted from data. They are analyzed below against the code paths
that produced them.

---

## Step 2 — Question A: same cases, and is baseline's format regressed?

### Were Run 2 and Run 3 the same validation cases?
**Almost certainly yes.** Case generation is fully deterministic:
`examples/datasets/contractnli/generate_cases.py` writes `rows[:limit]` in
dataset order with stable IDs (`contractnli_validation_000000`, `_000001`, …).
There is no shuffle and no RNG seed. Given the same generated file and the same
validation limit, every run evaluates the identical cases.

### Was the template change applied "too broadly" to the baseline?
There is **no per-strategy user-message formatting** to scope. All LLM
strategies funnel through one function:

```
evaluate_prompt(system_prompt, cases, task, …)   # src/rulekiln/pipeline/evaluator.py:338
  └─ _call_student(system_prompt, case, …, input_template=task.input_template)   # :362
        user_prompt = _render_input_from_template(input_template, case.input)     # :124
```

The *only* thing that differs between baseline, scaffold, k3, and k5 is the
`system_prompt` argument (few-shot appends an examples block). The Run 3 diff
changed exactly one line in `_call_student`:

```diff
- user_prompt = json.dumps(case.input)
+ user_prompt = _render_input_from_template(input_template, case.input)
```

So the template change is **uniform across all strategies by construction**. The
plan's proposed remediation — "baseline should keep `json.dumps`, few-shot
should use the template" — would require adding a new strategy conditional that
does not exist today, and would force the zero-shot baseline back onto the
inferior JSON format that hurt k3. That is not a bug fix; it is a new,
unmotivated divergence. **Not implemented.**

### So what explains baseline 0.7619 → 0.3000 (if the scores are accurate)?
On **n = 5** validation cases with 3 labels, macro-F1 moves in large quantized
steps — a single case flipping can swing the score by 0.2–0.4. `0.7619` (≈16/21)
and `0.3000` are both attainable by one or two cases changing prediction under
the JSON→template reformat. With no per-case data persisted, this cannot be
distinguished from a genuine regression. **Recommendation:** before treating it
as a regression, re-run with a larger validation limit (≥20) so macro-F1 is not
dominated by single-case quantization, and persist per-case predictions (see
"Follow-ups"). Do **not** special-case the baseline formatter on the strength of
a 5-case swing.

---

## Step 3 — Question B: why does k5 underperform k3?

k3 and k5 are the *same code* with a different `k`
(`baseline_eval_service.py` iterates `BASELINE_FEW_SHOT_STRATEGY_TO_K` and calls
the identical `build_few_shot_prompt_with_budget`). Walking the three plan
hypotheses against the code:

1. **"Template not applied / applied incorrectly to k5"** — **False.** k5 uses
   the same `_call_student` path as every other strategy; the test-case user
   message is template-rendered identically.

2. **"Model returns clean-but-wrong labels because the examples mislead"** —
   **This is the confirmed mechanism, and it is structural, not random.** The
   selector `select_deterministic_few_shot_examples`
   (`baseline_strategies.py:47`) does **label-balanced round-robin** over the
   sorted label set. contractnli has 3 labels (`contradiction`, `entailment`,
   `neutral`):

   | k  | selection (round-robin)                                             | balance |
   |----|---------------------------------------------------------------------|---------|
   | 3  | contradiction, entailment, neutral                                  | 1/1/1 ✅ |
   | 5  | contradiction, entailment, neutral, **contradiction, entailment**   | 2/2/1 ⚠️ |

   k3 shows the model exactly one example of each class. k5 adds a *second*
   contradiction and a *second* entailment but **still only one neutral**,
   biasing the prompt toward two of three labels and under-representing the
   third. This deterministically depresses recall on the under-represented
   label. It also explains why k5 was consistently low across *all three* runs
   (0.25, 0.08, 0.11) while k3 varied and improved — a consistent deficit points
   to a structural cause, not variance. **k is not a multiple of the label
   count, so k5 cannot be balanced for a 3-label task.**

3. **"Suffix too weak to break few-shot momentum at k=5"** — **Unlikely.** The
   closing instruction is identical for k3 and k5:
   `"\n\n---\n\nClassify the input you receive using the pattern above."`
   (`baseline_strategies.py:147`). If it were too weak at k5 it would be equally
   weak at k3. Outputs are also schema-constrained via `_StudentOutputSchema`
   tool-calling, making raw "continuation text" the exception (only possible
   when a local model ignores the schema — which is why the Run 3
   `_extract_json_from_content` fallback was added; that fallback only *recovers*
   output, it never hurts).

### Recommendation for Question B
The fix is **experiment design, not the selector**: for an N-label task, use
few-shot k values that are multiples of N (e.g. k3 and k6 for contractnli) so
every prompt is class-balanced, or make k5 explicitly balanced by capping
per-label counts. Changing round-robin behavior is deliberately **not**
implemented here because it would alter k3/dbscan comparisons and existing
regression snapshots, and the plan says not to over-engineer. This is surfaced
as a follow-up for a decision, not silently changed.

---

## Step 4 — Fixes implemented

### YAML scope-colon guard (implemented)
The described bug — a `task_scope`/`non_scope` list item containing `:` being
parsed by YAML as a single-key dict — is **already fixed** in the tree:
`examples/datasets/contractnli/task.yaml` quotes both colon-bearing items
(lines 37, 41), and a sweep of every `examples/**/task.yaml` found **zero**
dict-valued scope items. To prevent regression, added:

- `tests/unit/test_task_yaml_scope_parsing.py` — parametrized over every
  `examples/**/task.yaml`, asserts each parses to a mapping and that all
  `task_scope`/`non_scope` items are strings (never dict/list). 10 tests, all
  green.

### Deliberately NOT implemented
- **Question A baseline-format scoping** — no separate code path exists; the fix
  as specified is unfounded (see Step 2).
- **Question B k5 template fix** — k5 already uses the template (false premise).
- **Question B separator strengthening** — no evidence of continuation text; the
  separator is identical to k3's. The real cause is label imbalance (Step 3).

Implementing 2–3 would add code and tests asserting behavior the evidence does
not support, per the plan's own rule ("Implement only what the evidence
supports").

---

## Step 5 — Verification status

- New guard test: `10 passed`.
- Full unit suite: `401 passed` (`pytest tests/unit -m "not external"`).
- `ruff check` and `pyright` clean on the new file.
- **Smoke re-run not performed.** A smoke job needs a live Postgres + provider
  credentials + embedding models; none are available in this environment, and
  the plan forbids a full-profile run. The score-stability checks in Step 5
  (baseline near 0.76, k5 ≥ k3, dbscan ≈ 1.0) require running the pipeline and
  reading scores that are not persisted locally. They remain **unverified** and
  should be re-checked by the operator, ideally after (a) raising the validation
  limit so macro-F1 is less quantized, and (b) adding per-case prediction
  persistence.

---

## Follow-ups (for a decision — not done here)

1. **Persist per-case predictions/prompts** (DB or MLflow artifact) so questions
   like this are answerable from data instead of code inference. Today only
   aggregate metrics survive a run.
2. **Use label-count-aligned k** for few-shot on N-label tasks (k3/k6 for
   contractnli), or make k5 selection per-label balanced.
3. **Raise the smoke validation limit** above 5 so single-case flips don't swing
   macro-F1 by 0.3+.

## Run 4 Fix: k5 Separator Strengthening

**What the log/DB showed (Run 4, job bf418815):** all 5 k5 cases scored 0.0.
Every raw student response echoed the test-case input — either as the
`{"hypothesis": ..., "premise": ...}` JSON object or as the `input_template`
prose (`Contract clause: ... Legal statement: ...`), sometimes with a trailing
`Output:` cue. No case produced a usable label. malformed_rate=0.0 (parser fell
back to a default label rather than flagging), macro_f1=0.0.

**What the prior separator looked like (Run 3, weak):** appended to the *system
prompt* after the examples:
```
---

Classify the input you receive using the pattern above.
```

**What the new separator looks like (strong, in the *user* message before the
rendered test input):**
```
===

The 5 examples above are for illustration only. A new case follows.
Do not continue the pattern above.
Do not repeat or echo the input.
Return only valid JSON with a single "label" field.

===
```
Named constant `FEW_SHOT_TRANSITION_SEPARATOR` in `baseline_strategies.py`;
rendered via `render_few_shot_transition(k)` and injected into the test-case
user message via `build_student_user_message()` in `evaluator.py`. The weak
system-prompt suffix was removed. Applied to baseline_few_shot_k3/k5 only
(retrieval/embedding untouched).

**Run 5 result (job a9bdf063, separator live in worker):** k5 macro_f1 still
0.00. Partial effect only:
- 000020 → clean `{"label": "entailment"}` (parseable, wrong class)
- 000022 → echoed input prose, then appended `{"label": "neutral"}` at the end
- 000021, 000024 → echoed the `Contract clause:/Legal statement:` prose, no label
- 000023 → `{` only

**Status: still open — escalated.**

**Diagnosis of why the separator failed.** The separator treats the failure as
pure "sequence momentum," but the root cause is a **format contract mismatch plus
a missing answer cue**:
- Examples (system prompt) render input as a fenced JSON block and end each with
  an explicit `Output:\n```json\n{"label": ...}\n``` cue.
- The test-case user message renders input as *prose* via `input_template`
  (the Run 3 fix) and ends with the raw hypothesis — **no `Output:` cue**.
- The 5 examples train the model to emit a label only after the `Output:` scaffold.
  The test message never presents that scaffold, so under strong k5 momentum the
  model continues/echoes the prose instead of answering.
- Case 000022 is the smoking gun: the model *reconstructed* the input and only
  then emitted `{"label": "neutral"}` — it was rebuilding the `Input → Output`
  scaffold itself before answering, and the parser could not extract the trailing
  label.

**Recommended next step (needs human decision before implementing — do NOT keep
adding separator text):**
- **Option B (preferred):** keep prose `input_template` rendering but append an
  explicit answer cue to the end of the test-case user message
  (e.g. `\n\nOutput (JSON only):`), matching the `Output:` cue the examples train.
- **Option A:** render the test input in the *same* fenced-JSON `Input:/Output:`
  format the examples use (reverts the Run 3 prose change; may regress k3).
- **Option C:** wrap the test case as `### Example 6` with an empty `Output:` to
  force completion of the trained slot.
The separator approach alone cannot close the gap; the missing `Output:` cue is
the strongest single hypothesis and Option B tests it directly.
