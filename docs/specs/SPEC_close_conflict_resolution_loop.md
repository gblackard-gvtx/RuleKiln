# Spec: Close the Conflict-Resolution Loop and Fix Eval-to-Rule Mapping

**Status:** Required before Phase 5 (task diversity).
**Owner:** Coding agent (Claude Code).
**Scope:** Pipeline correctness and paper-alignment. No new architecture beyond what is specified here.

---

## Why this work exists

RuleKiln is an implementation of Prompt-Level Distillation (PLD), arXiv:2602.21103.
The paper's method has four phases. RuleKiln implements phases 1, 2, and a *static*
version of phase 3 — but **not** the paper's actual phase 3, which is a closed
loop, and the load-bearing part of the method on hard tasks.

What the paper's Phase 3 ("Closed-Loop Conflict Resolution", §3.3) actually does:

1. Deploy the student with the *current full instruction set* on the training data.
2. Isolate cases where the student **followed the instructions but still produced
   the wrong label** (empirical failures, not linguistic contradictions).
3. Feed those failures **plus sampled successes** back to the teacher, which
   diagnoses root cause and rewrites the implicated instructions. Including the
   successes is explicitly required by the paper to prevent the rewrite from
   degrading overall accuracy.
4. Repeat until validation error converges (paper observed 1 iteration on
   StereoSet, 2 on Contract-NLI).

What RuleKiln currently does instead:

- `review_rule_for_conflicts` (rule_conflict_review module): a **single-pass,
  per-rule linguistic consistency check**. It never runs the student, never sees a
  case outcome, never iterates. This is useful pre-eval hygiene but is NOT the
  paper's loop.
- `analyze_failures` (failure_analysis module): already runs the distilled student,
  compares against baseline per-case, and isolates `unchanged_failing` /
  `broken` cases with rule attribution. **This is the paper's Phase 3 step 1–2,
  already built.**
- `prune_rules` (rule_pruning module): already accepts
  `utility_signals: dict[rule_id -> (fixed, broken)]` and can rank by `utility`
  / `utility_per_token`. **This is the consumer of the failure signal, already
  built.**

The gap is a missing **feedback edge**: nothing routes `analyze_failures` output
back into rule regeneration and re-evaluation. The pipeline runs as a straight
line and stops after one pass. We are missing one arrow, not a subsystem.

Two latent defects make the straight line worse than it looks:

- **The eval-to-rule mapping is a stub.** In `failure_analysis._maybe_add_structured_failure`,
  `violated_rule_ids` is resolved only via `rule_output_path_index`, which is keyed
  on `rule.topic.lower()`. The loop over `outcome_conditions` that is supposed to
  extract real output paths is literally `pass`. `failed_assertion_types` is always
  `[]`. Result: failures rarely attribute to any rule, so `utility_signals` is
  near-empty, so utility pruning silently degrades to `support_count`, and the
  Phase 4 provenance claim ("every rule traced to cases it fixed/broke") does not
  actually hold.
- **`utility_signals` is dead on the only pass that runs.** Pruning happens *before*
  evaluation in a single forward pass, so `utility_signals` is necessarily `None`
  the only time `prune_rules` is called. The `utility` and `utility_per_token`
  ranking modes are never exercised on real signal. The loop is what makes them live.

---

## Objectives (in dependency order)

1. **Validate** the current wiring and confirm the claims above against the actual
   source. Do not assume; verify and report.
2. **Fix the eval-to-rule mapping** so failures attribute to real rule IDs. This is
   the connective tissue for Phases 3, 4, and the loop simultaneously, so it goes first.
3. **Close the loop**: add an iterative refinement controller that feeds
   failures+successes back to the teacher, re-prunes with real `utility_signals`,
   recompiles, re-evaluates, and stops on validation convergence.
4. **Make the Phase 4 ablation real**: loop-on vs loop-off becomes the paper's own
   ablation. Provenance data (`violated_rule_ids`) must be non-empty.
5. **Do not start Phase 5.** Stop when objectives 1–4 are complete and tests pass.

---

## Task 0 — Validation pass (do this first, produce a written report)

Before changing anything, read the source and confirm or refute each claim. Output
a short markdown report at `docs/dev/loop_validation_report.md` answering:

- **0.1** Where is pruning invoked in the pipeline orchestrator/stage chain? Is
  `prune_rules` ever called with a non-`None`, non-empty `utility_signals`? Trace
  the call sites. Confirm whether the straight-line assumption holds or whether a
  loop-back already exists somewhere not yet seen.
- **0.2** In `failure_analysis._maybe_add_structured_failure`, confirm
  `rule_output_path_index` is built only from `rule.topic.lower()` and that the
  `outcome_conditions` loop body is `pass`. Confirm `failed_assertion_types` is
  always empty.
- **0.3** What is the actual shape of `CaseEvalResult.assertion_scores` keys (the
  "paths")? Find a real example from a benchmark run or a test fixture. This
  determines how the mapping must be rebuilt. Do not guess the key format —
  find a concrete instance.
- **0.4** What is the structure of `SynthesizedRuleSchema.outcome_conditions` and
  how does an outcome condition relate to an assertion path / output label? This
  is the join we need to make attribution work.
- **0.5** Confirm `review_rule_for_conflicts` performs no student inference and no
  case-outcome inspection (i.e. it is purely linguistic). Confirm the `discard`
  path drops rules with no validation evidence.

**Acceptance for Task 0:** `docs/dev/loop_validation_report.md` exists and answers
0.1–0.5 with specific file/line references and at least one concrete example of an
`assertion_scores` key. If any claim in "Why this work exists" is wrong, say so
explicitly and adjust the plan in the report before proceeding.

---

## Task 1 — Fix the eval-to-rule mapping

**Problem:** Failures do not attribute to rules because the mapping is stubbed.

**Required changes (failure_analysis module):**

- Replace the `rule.topic.lower()`-only index with a real mapping from a case's
  failed assertion paths/predicted-vs-expected outcome to the rule(s) whose
  `outcome_conditions` govern that outcome. Use the concrete `assertion_scores`
  key format found in Task 0.3 and the `outcome_conditions` structure from 0.4.
- Populate `failed_assertion_types` with the actual assertion types, not `[]`.
- A single failure may legitimately attribute to multiple rules; keep the list,
  de-duplicated, order-stable.
- If a failure genuinely cannot be attributed to any rule (e.g. malformed output
  with no governing rule), attribute it to a reserved sentinel
  (e.g. `"__unattributed__"`) rather than silently dropping it, so coverage is
  measurable.

**Acceptance criteria:**

- New unit tests in the failure-analysis test module assert that, given a fixture
  with known rules and known failing cases, `violated_rule_ids` is non-empty and
  contains the *expected* rule IDs (not just "any").
- A test asserts the unattributed-coverage metric: the fraction of failures with
  no real rule attribution is reported and is below a stated threshold on the
  fixture (pick a threshold the fixture can meet, document it).
- `violated_rule_summary()` returns non-zero counts on the fixture.
- `failed_assertion_types` is non-empty when assertions of a known type fail.
- Existing tests still pass.

---

## Task 2 — Build the refinement teacher call

This is a **new** teacher interaction, distinct from `review_rule_for_conflicts`.
It is empirical, not linguistic.

**New module (suggested):** `src/rulekiln/pipeline/rule_refinement.py`

**Behavior:**

- Input: the current synthesized rule set, a sample of failure cases
  (`unchanged_failing` + `broken` from `analyze_failures`, with their
  `violated_rule_ids`), and a sample of success cases (`unchanged_passing` +
  `fixed`).
- The teacher prompt must:
  - present the failing cases with the rules they violated,
  - present the success cases (required — state in the prompt that revisions must
    not break these),
  - ask the teacher to diagnose root cause and emit revised rule(s) for the
    implicated rule IDs only, leaving unrelated rules untouched.
- Output: a structured result (new Pydantic schema, versioned) containing revised
  `SynthesizedRuleSchema` objects keyed by the rule ID they replace, plus a
  per-rule rationale.
- Sampling must be deterministic given a seed (reuse the benchmark seed
  convention). Cap the number of failure and success cases sent (configurable;
  default e.g. 20 failures + 20 successes) to control token cost.

**Acceptance criteria:**

- Works offline with the fake provider (deterministic stub revisions).
- Schema has a `schema_version` field (consistent with the Phase 8 artifact-
  versioning convention).
- Unit test: given fixture failures attributed to rule X, the refinement call is
  asked to revise rule X and the returned revised set replaces X by ID.
- Success cases are present in the constructed prompt (assert on prompt contents).

---

## Task 3 — Close the loop (iteration controller)

**Goal:** wrap `compile → evaluate_distilled → analyze_failures → refine → re-prune`
in a bounded iteration that stops on validation convergence.

**New pipeline stage:** `refining_rules`, inserted so the chain becomes:

```
... → compiling_prompts → evaluating_baseline → evaluating_distilled
    → analyzing_failures → refining_rules → (loop back to pruning/compiling)
    → selecting_strategy → ...
```

**Controller logic:**

1. Run the existing forward segment to get `analyze_failures` output and a
   validation metric (macro_f1 on the validation split; reuse the existing
   evaluator and split-selection policy).
2. Build `utility_signals` from `analyze_failures.violated_rule_summary()`:
   `rule_id -> (fixed_count, broken_count)`. This is now non-empty because of Task 1.
3. Call the Task 2 refinement to obtain revised rules.
4. Re-run `prune_rules` **with** the real `utility_signals` and `ranking_mode`
   from config (so `utility` / `utility_per_token` are finally exercised), then
   recompile and re-evaluate on validation.
5. **Convergence/stop conditions** (stop if ANY is true):
   - validation macro_f1 improvement over previous iteration is below a configured
     epsilon (default e.g. 0.005),
   - validation macro_f1 *regressed* vs previous iteration — in which case **roll
     back** to the previous iteration's rule set and stop,
   - max iterations reached (configurable; default 2, matching the paper's
     observed range; hard cap 3).
6. Each iteration is expensive (a full student eval pass); make the whole loop
   **optional via flag/config**, default ON for `standard`/`full` profiles and
   OFF for `smoke`.

**Resume/cost:** reuse the existing DBOS per-stage resume machinery so a crashed
worker resumes mid-loop rather than restarting iteration 0.

**Acceptance criteria:**

- A single config flag toggles the loop on/off without code changes.
- Deterministic given a seed: same inputs + seed → same iteration count and same
  final rule set (byte-identical selected-rule IDs).
- Loop never *lowers* the final validation metric vs the pre-loop baseline: if no
  iteration improves, it returns the original rule set (assert this in a test with
  a fixture where refinement is a no-op or harmful).
- `utility_signals` passed to `prune_rules` on iteration ≥1 is non-empty (assert).
- Each iteration emits an artifact:
  `outputs/refinement_iter_{n}.json` with prior metric, new metric, revised rule
  IDs, stop reason. All artifacts carry `schema_version`.
- Max-iteration and epsilon are configurable and documented.

---

## Task 4 — Make the Phase 4 ablation real

- Add a benchmark/ablation switch that runs the pipeline with the loop OFF and
  with the loop ON over the same seed and split, and emits a comparison:
  `refinement_ablation.json` containing macro_f1 (with CIs if Phase 3 stats are
  available), regression rate, prompt token count, and teacher cost for each arm.
- Mirror the paper's finding format: report whether the loop helped, and by how
  much, per dataset. The paper saw negligible gain on the easy task (StereoSet)
  and a real gain on the hard task (Contract-NLI); your report should be able to
  express that pattern.

**Acceptance criteria:**

- One command produces `refinement_ablation.json` for a smoke-sized run.
- Provenance check: with the loop ON, every selected rule has a non-empty
  attribution record (fixed/broken/neutral) — i.e. the Phase 4 "every rule traced
  to cases" claim now holds. Rules with zero validation impact are flagged
  (already a Phase 4 requirement; now backed by real data).

---

## Naming / honesty fix (small but do it)

- Rename the *concept* so the codebase stops calling the static pass "conflict
  resolution". The static per-rule check is **static rule review**; the new
  iterative loop is **conflict resolution** (matching the paper). Keep the static
  reviewer — it is legitimate pre-eval hygiene the paper does not have — but do not
  let its name imply it is the paper's Phase 3. Update docstrings/README references
  accordingly. Do not break public API/artifact schema names without a
  `schema_version` bump (Phase 8 rule).

---

## Guardrails for the agent

- Read `AGENTS.md`, `pyproject.toml`, and the existing pipeline orchestrator before
  editing. Do the Task 0 report first.
- Preserve strict Pyright compliance. Use Pydantic for the new refinement and
  ablation schemas. No `Any` without justification.
- Do not add new dependencies.
- Keep changes PR-sized and sequenced: Task 1 lands and is green before Task 3
  depends on it.
- Do not change DBSCAN/HDBSCAN strategy behavior, provider routing, or the
  quality-gate schema beyond adding versioned fields.
- **Do not begin Phase 5 (task diversity / new datasets).** This spec ends at the
  point where Contract-NLI replication becomes meaningful; running it is the next
  step after this spec, not part of it.

## Definition of done

- `docs/dev/loop_validation_report.md` answers Task 0.
- Eval-to-rule mapping attributes failures to real rule IDs (Task 1, tested).
- Refinement teacher call exists and is offline-testable (Task 2).
- The loop runs, is seed-deterministic, never lowers the final metric, exercises
  real `utility_signals`, and is flag-toggleable (Task 3).
- `refinement_ablation.json` (loop on/off) is produced by one command (Task 4).
- The static reviewer is renamed away from "conflict resolution"; the loop owns
  that name.
- Required commands pass:
  - `uv run ruff check src/ tests/`
  - `uv run pyright`
  - `DATABASE_URL="sqlite+aiosqlite://" MLFLOW_TRACKING_URI="file:///tmp/mlflow-ci" uv run pytest -m "not external" --tb=short -q`
