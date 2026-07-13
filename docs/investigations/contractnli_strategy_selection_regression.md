# ContractNLI Strategy Selection Regression

Date: 2026-07-05
Job under investigation: `1521e48b-2cd6-4607-af4a-1cb061cbcef6` (300 validation cases, student `gemma-4-E4B-it-Q4_K_M.gguf` via llama.cpp)
Evidence source: structured events in `logs/rulekiln.log*` (this run had the new
`eval_complete` / `quality_gate_result` / `malformed_output_exception` logging live, so
every claim below is grounded in run data, not code inference).

## Observed issue

The evaluation report selected `embedding_knn_k1` (macro_f1 0.6144) even though
`baseline_scaffold` scored 0.7088, reported `selected_family: baseline` for an
embedding strategy, showed a net paired regression (41 fixed / 74 broken), 100%
malformed output for `hdbscan`, macro_f1 ≈ 0.02 for all few-shot strategies, and a
weak `dbscan` (accuracy 0.6667, macro_f1 0.2722).

## Root causes

### 1. Selection ranked gate-failing candidates instead of falling back to baseline

`select_strategy_generic` (src/rulekiln/pipeline/strategy_selection.py) excludes the
baseline scaffold from candidacy by design, filters candidates to those whose quality
gate passed — and then, **when zero candidates passed, silently ranked the failed pool
anyway** (`ranked_pool = passing_candidates if passing_candidates else candidate_evals`).

In this run every one of the 9 candidates failed its gate (all had
`metric_delta < 0` vs baseline_scaffold; log event `quality_gate_result`, all
`passed: false`). The failed pool was then ranked by macro_f1:

- top score: `embedding_knn_k5` 0.6190
- `embedding_knn_k1` 0.6144 is within the `_PRIMARY_METRIC_EPSILON = 0.005` tie window (Δ 0.0046)
- tie-breakers (golden failures 0/0, malformed 0/0, prompt tokens equal) all tied
- final tie-break is lexicographic → `embedding_knn_k1`

The log confirms: `strategy_selected: embedding_knn_k1, reason: "All tie-breaks equal;
defaulting to lexicographically stable strategy order."` The gates worked; the selector
ignored them.

### 2. `selected_family` lumped everything non-clustered into "baseline"

`_strategy_family` returned `"distilled"` only for dbscan/hdbscan and `"baseline"` for
everything else, so an embedding-KNN selection was reported as a baseline-family (and
implicitly prompt-level) result.

### 3. HDBSCAN 100% malformed = context overflow, not model quality

Extraction produced 520 micro-rules. HDBSCAN formed 26 clusters → 26 synthesized rules
→ compiled prompt of ~11,006 approx tokens. Every one of the 300 student calls failed
with `ModelHTTPError 400: request (9026 tokens) exceeds the available context size
(8192 tokens)` (300 `malformed_output_exception` events in the log). Each failed call
was scored `malformed=True` → 100% malformed. The token gate did catch it afterwards
(`prompt_tokens 11006 > max_prompt_tokens 8000`), but only after burning 300 doomed
calls, and the report never explained the failure.

### 4. Few-shot macro_f1 ≈ 0.02 = spurious-class dilution by out-of-vocabulary predictions

The student echoes the input for most few-shot cases (known failure documented in
`docs/dev/run3_investigation.md` / `few_shot_k5_investigation.md`). The parser accepted
those echoes as "valid" predictions (`malformed=False`), and
`compute_per_label_metrics` averages macro F1 over the **union** of gold and predicted
label strings — so every unique echoed string became its own zero-F1 class. With ~3
real classes and dozens of unique garbage "labels", macro F1 ≈ (sum of real-class
F1)/(3 + N_garbage) ≈ 0.02 while accuracy (same predictions, exact-match only) stayed
0.35–0.47 and malformed reported 0%. Accuracy and macro F1 *were* computed from the
same prediction column; the anomaly was the inflated label space plus a parser that
hid the contract violation.

### 5. DBSCAN collapsed 520 rules into one cluster

`clustering_diagnostics` reconstruction: DBSCAN (eps=0.3, min_samples=2, cosine) put
the mxbai rule embeddings into **1 cluster** → 1 synthesized rule (log:
`synthesis_done dbscan processed_clusters=1 inserted_rules=1`). A single generic rule
loses all label-specific reasoning; the strategy degenerates toward majority-class
prediction (accuracy 0.6667 vs macro_f1 0.2722). HDBSCAN found 26 clusters on the same
embeddings, i.e. the structure exists — the DBSCAN eps default is too coarse for this
embedding space. Not tuned here (would be blind tuning); surfaced via new diagnostics.

## Fixes

| # | File | Change |
|---|------|--------|
| 1 | `src/rulekiln/pipeline/strategy_selection.py` | When no candidate passes its gate, select the baseline scaffold and put every candidate's gate failures into the selection reason. Never rank a failed pool. |
| 2 | `src/rulekiln/pipeline/strategy_families.py` (new) | Families: `baseline`, `embedding_baseline`, `retrieval_baseline`, `distilled`. `strategy_utils._strategy_family` and the UI fallback delegate to it, so `selected_family` for `embedding_knn_k1` is now `embedding_baseline`. `best_by_family` also reports the two new families. |
| 3 | `src/rulekiln/pipeline/quality_gates.py`, `config/settings.py` | New Gate 6: `paired_broken > paired_fixed` is a violation (configurable `require_paired_net_improvement`, default **true**; per-task override supported). `QualityGateResult` now records `paired_fixed` / `paired_broken`. Combined with the existing `min_metric_delta` (≥ baseline) and `max_malformed_output_rate` gates this implements the suggested default promotion gate. |
| 4 | `src/rulekiln/workers/pipeline/distilled_eval_service.py` | Fail-fast: compiled distilled prompt is token-counted before evaluation; if over `max_prompt_tokens` the strategy is skipped, marked unavailable with the reason, and no model calls are made. Also logs `distilled_prompt_compiled` (rules count, token count, prompt preview). |
| 5 | `src/rulekiln/workers/pipeline/early_stages.py` | `clustering_diagnostics` log event per algorithm: cluster_count, clustered_rules, noise_count, noise_percentage, largest/median cluster size. |
| 6 | `src/rulekiln/pipeline/evaluator.py` | For classification/routing tasks with an enum label vocabulary, a parseable output whose label is not an allowed value is scored `malformed=True` (`error="prediction_not_in_allowed_labels"`). |
| 7 | `src/rulekiln/schemas/pipeline.py`, `workers/distillation_worker.py` | `StrategyComparison.selection_table`: one row per strategy with `strategy`, `family`, `eligible`, `gate_failures`, `selection_score`, `selection_rank`, `selected`, `reason`; strategies skipped before evaluation appear as ineligible with the skip reason. Lands automatically in `strategy_comparison.json`. |
| 8 | `src/rulekiln/ui/routes.py`, `templates/jobs/eval_report.html` | Eval report shows the selection reason and a Strategy Selection table (eligibility, score, rank, gate failures per strategy). Also removed a duplicated dead parsing block that clobbered the family fallback. |

### Metric-definition impact (fix 6) — read before comparing runs

Marking out-of-vocabulary predictions malformed changes reported numbers for
strategies that emit garbage:

- `malformed_output_rate` becomes honest (few-shot strategies will now show high
  malformed rates instead of 0%), which correctly makes them ineligible via the
  existing `max_malformed_output_rate` gate.
- macro F1 is no longer diluted by spurious classes; accuracy/macro F1 are computed
  over the remaining valid predictions only (unchanged existing behavior for
  malformed cases). Both still come from the same normalized prediction column.
- Well-behaved strategies (all predictions in vocabulary) are completely unaffected.
- Scores are **not comparable** with pre-fix runs for garbage-emitting strategies.

## Validation evidence

- Replaying the exact metrics/gates of job `1521e48b` through the fixed selector
  returns `baseline_scaffold` with reason "All 9 candidate strategies failed quality
  gates; selecting baseline scaffold. Gate failures — …".
- New tests (all passing):
  - `tests/unit/test_strategy_selection_regression.py` — higher macro_f1 wins;
    malformed strategy never selected; all-fail → baseline fallback (regression repro);
    paired gate fires on broken > fixed and is per-task disableable; family
    classification; selection table contains gate reasons for every rejected strategy,
    rank ordering, and skip reasons for strategies skipped before evaluation.
  - `tests/unit/test_evaluator_label_vocabulary.py` — vocabulary extraction from the
    output schema; echoed dict/prose marked malformed; valid-but-wrong labels not
    malformed; end-to-end eval shows malformed rate 0.4 and metrics over valid labels
    only, with no spurious classes.
- Full suite: 554 passed (`tests/unit`, `tests/integration`, `tests/ui`,
  `-m "not external"`). `ruff` and `pyright` clean on all touched files (one
  pre-existing S101 and pre-existing pyright errors in untouched files remain).

## Remaining risks / follow-ups

1. **Few-shot echo failure is still open.** The vocabulary fix reports it honestly; it
   does not make the local Gemma student answer few-shot prompts. The `Output:` answer
   cue work (see `docs/dev/run3_investigation.md`, Option B) is the open lead.
2. **DBSCAN eps for mxbai embeddings.** 1 cluster from 520 rules means the default
   eps=0.3 is too coarse for this embedding space. Use the new `clustering_diagnostics`
   event to pick eps empirically; deliberately not tuned blind here.
3. **Epsilon tie-break.** Within the 0.005 primary-metric window a marginally lower
   macro_f1 strategy can still win on tie-breakers; this is documented C005 behavior
   and now only applies among gate-passing candidates, with the tie-break named in the
   selection reason.
4. **Token counting is approximate.** `count_tokens_approx` (len/4) said 11,006 while
   llama.cpp counted 9,026 for prompt+input; both were over the limit here, but a
   prompt near the boundary could pass the approximate check and still overflow at the
   provider. The per-case malformed handling still catches that case.
5. **Resume semantics.** Case results persisted before this fix keep their old
   malformed flags; a resumed old job can mix old/new semantics. Fresh runs are clean.
