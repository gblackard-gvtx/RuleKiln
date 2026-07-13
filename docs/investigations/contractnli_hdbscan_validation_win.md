# ContractNLI HDBSCAN Validation "Win" — Integrity Review

Date: 2026-07-12
Job under investigation: `36bfc5ab-0ba1-4837-beaf-a7bdfb0878a1`
(300 ContractNLI validation cases, student label vocabulary `{entailment, neutral, contradiction}`)
Artifact root: `.rulekiln/runs/36bfc5ab-0ba1-4837-beaf-a7bdfb0878a1/`
Evidence source: the persisted run artifacts on disk (per-strategy `*_eval.json`,
`outputs/strategy_comparison.json`, `outputs/paired_comparison/*.jsonl`, `task.yaml`,
`metadata/manifest.json`). Every count and metric below was **recomputed independently
from the raw per-case records**, not read back from the summary fields.

This run is uniquely identified as the reported result: it is the only run on disk whose
paired signature is `fixed=29 / broken=18 / unchanged_correct=206 / unchanged_wrong=47`
with `candidate=hdbscan` selected.

---

## Executive summary

| Question | Answer |
|---|---|
| Is the score discrepancy (0.7142 vs 0.5350) explained? | **Yes.** 0.5350 is a **stale, pre-hardening aggregate** that counted one echoed-input baseline output as a spurious 4th class (÷4 instead of ÷3). 0.7142 is the correct 3-class macro-F1. Proven to the 4th decimal. |
| Are the paired counts correct? | **Yes.** 29/18/206/47 reproduce exactly when joined by `case_id` from raw records. |
| Did the job use the promoted HDBSCAN config? | **No.** The persisted job snapshot used `min_cluster_size=2, min_samples=null`, not the promoted `5 / 1`. |
| Is the malformed gate handling correct? | **Yes, but on the boundary.** hdbscan malformed = exactly 3/300 = 0.0100; the gate is strict `>`, so 0.0100 passes. One more malformed case (4/300) would fail. |
| Is the win statistically credible? | **No.** Paired-bootstrap 95% CI on Δmacro-F1 is **[−0.037, +0.077] (includes 0)**; McNemar exact p = **0.144**. |
| Does the gain hold across labels? | **No.** Gain is entirely entailment/neutral; **contradiction (the minority) degrades** (F1 0.618 → 0.556, precision 0.52 → 0.40). |
| Recommended status | **Experimental.** Not a promotion candidate as-run: params ≠ promoted config, win not significant, malformed on the gate boundary, and the displayed comparison mixed a stale baseline. |

---

## A. Score-field lineage — the 0.7142 vs 0.5350 discrepancy

### The two tables read two different sources

`GET /ui/jobs/{job}/eval` (`src/rulekiln/ui/routes.py::job_eval_report`) renders **two
independent tables from two independent sources**:

- **Evaluation table** ("Score / Accuracy / Macro F1"): `eval_runs = get_eval_runs_for_job(...)`
  — read live from the Postgres `eval_runs` table.
- **Strategy Selection table** ("Score" column = `row.selection_score`): read from the
  file `outputs/strategy_comparison.json` (`comparison_payload["selection_table"]`).

When the two sources disagree, the page shows two different numbers for the same strategy
with no explanation. That is exactly what happened.

### What `selection_score` actually is

`build_selection_table` (`strategy_selection.py:231`) sets
`selection_score = _primary_score(result, task_mode)`, which for a `classification` task is
`result.macro_f1`. So the Strategy Selection "Score" **is** macro-F1 — it is *not* an
adjusted or penalized score. The label "Score" is therefore not itself the bug; the bug is
that the macro-F1 it is showing is a **stale one**.

### Root cause: a spurious 4th class in the stale aggregate

The stale `baseline_scaffold` aggregate stored on disk contains a fourth "label" in its
confusion matrix and per-label metrics:

```
'Confidential Information may be disclosed in written or other tangible form ...
 Relationship: entailment'   (support=0, fp=1, f1=0.0)
```

This is case `contractnli_validation_000222`, where the student **echoed the input clause**
and appended `Relationship: entailment` instead of emitting a bare label. The
*pre-hardening* evaluator accepted that whole string as a valid predicted class. Averaging
four classes (three real + one zero-F1 garbage) yields the stale value:

```
(0.6182 + 0.7463 + 0.7754 + 0.0) / 4 = 0.53495   → displayed "0.5350"
```

The correct 3-class macro-F1, with case 222 excluded as malformed (what the current
evaluator does), is:

```
(0.6182 + 0.7491 + 0.7754) / 3 = 0.71420          → DB / eval-table "0.7142"
```

Accuracy shows the same fingerprint: stale = `224/300 = 0.7467`, correct = `224/299 = 0.7492`
(the malformed case dropped from the denominator).

### Proof that the stale aggregate is internally inconsistent

Within `baseline_scaffold_eval.json`, case 222's own `case_results` entry is already
`malformed=True, error="prediction_not_in_allowed_labels"` — yet the **aggregate confusion
matrix still counts it as a 4th class**. `_compute_metrics` skips malformed cases
(`evaluator.py:392 if res.malformed: continue`), so a single consistent run of the current
code could never produce this. The aggregates were computed by an older evaluator when case
222 was still a "valid" prediction, then the case-level flag was hardened without
recomputing the aggregates. **The file mixes v0 aggregates with v1 case flags.**

### Classification of the discrepancy

Primarily **(4) stale data from a previous run**, surfacing as **(2) an artifact-aggregate
inconsistency**. It is *not* a UI mapping bug (names line up), *not* an intentional adjusted
score, and *not* a selection-logic error. The current evaluator is already correct and is
covered by `tests/unit/test_evaluator_label_vocabulary.py::test_garbage_predictions_raise_malformed_rate_not_spurious_classes`.

### Consequence for the gate

`hdbscan_gate.metric_delta` is stored as **0.1988** = `0.7338 − 0.5350` — i.e. the gate
consumed the stale baseline. The **true** delta is **+0.0196**. The gate still passes
(`min_metric_delta=0.0`), so the *selection* was not changed, but any report reader sees a
delta inflated ~10×.

### Field-naming recommendation

The `StrategySelectionRow.selection_score` field name is fine, but the UI header "Score" and
the stored `metric_delta` should be disambiguated. Recommended vocabulary going forward:

- `primary_metric_value` — the metric used for ranking (here macro-F1)
- `raw_macro_f1` — always the 3-class macro-F1
- `selection_score` — reserved for any future *adjusted* score; must never be shown as plain "Score"
- render the column as **"Score (macro-F1)"** and warn when it diverges from the live eval row.

---

## B. HDBSCAN parameter provenance

Recovered from `.rulekiln/runs/36bfc5ab.../task.yaml` (the job's own persisted snapshot):

| Field | Promoted (expected) | **This job (persisted snapshot)** | Match |
|---|---|---|---|
| algorithm | hdbscan | hdbscan | ✓ |
| min_cluster_size | 5 | **2** | ✗ |
| min_samples | 1 | **null** | ✗ |
| cluster_selection_method | eom | eom | ✓ |
| cluster_selection_epsilon | 0.0 | 0.0 | ✓ |
| text_variant | rule_only | **not persisted** | ? |
| embedding model | mxbai-embed-large-v1.Q6_K | **not persisted per-run** | ? |
| embedding dimension | — | **not persisted** | ? |
| normalization / distance | cosine, normalized | **not persisted** | ? |
| cluster assignments | — | **no `clusters.json` in manifest** | ✗ |

Findings:

1. **The winning run did not use the promoted config.** It used `min_cluster_size=2,
   min_samples=null`. The source `examples/datasets/contractnli/task.yaml` in the working
   tree *now* carries the promoted `5 / 1`, but that edit post-dates this run.
2. **text_variant, embedding model/dimension, normalization, and cluster assignments are not
   persisted anywhere in the run.** `strategy_metadata` has no `hdbscan` entry at all. The
   only embedding evidence is a `local_mxbai_embed` provider in `settings_snapshot.json`.
3. Because cluster assignments are not persisted, Investigation E's request for "source
   cluster IDs / sizes / label distribution" per regressed case **cannot be satisfied from
   this run's artifacts**.

**Recommendation (not implemented here — requires wiring the clustering stage):** persist a
`clustering_provenance` block on the strategy comparison / job snapshot:

```jsonc
{
  "algorithm": "hdbscan",
  "text_variant": "rule_only",
  "min_cluster_size": 2, "min_samples": null,
  "cluster_selection_method": "eom", "cluster_selection_epsilon": 0.0,
  "embedding_model": "mxbai-embed-large-v1.Q6_K.gguf", "embedding_dim": 1024,
  "vectors_prenormalized": true, "distance_metric": "cosine",
  "cluster_assignment_hash": "…", "n_clusters": N, "noise_pct": …
}
```

Until this exists, "the full job used the promoted config" **cannot be verified** and, for
this run, is **false**.

---

## C. Paired-comparison integrity

Recomputed independently by joining `baseline_scaffold_eval.json` and `hdbscan_eval.json`
**on `case_id`** (dict join, not array position):

```
fixed=29  broken=18  unchanged_correct=206  unchanged_wrong=47  total=300
29 + 18 + 206 + 47 = 300 ✓
net fixes = +11   overall net fix rate = +3.67%   net-of-changed = +23.40%
```

- Same 300-case ID set on both sides; **no duplicate case IDs**; every case present in both.
- The paired comparison uses the **hardened** case-level `passed` (case 222 → malformed →
  `passed=False`), so it is consistent with the corrected view even though the stale
  *aggregate* macro-F1 is not. **The paired counts are trustworthy.**

**Latent join risk.** `compute_paired_comparison` (`statistics.py:221`) pairs by
`zip(..., strict=True)` over parallel arrays — i.e. **by position, not by case_id**. It is
safe *only* because the caller `_build_runtime_paired_comparison` builds both prediction
columns from a shared `eval_cases` order via a `case_id → result` dict lookup
(`classification_metrics.py:_aligned_classification_columns`). If that function is ever
called with independently ordered arrays, the join silently corrupts. A regression test
locking the by-id alignment is added (see §Tests).

---

## D. HDBSCAN malformed outputs (3 / 300 = 1.00%)

All three carry `error="prediction_not_in_allowed_labels"`; all three have
`expected_label=entailment`; all three are baseline-correct → **all three land in the
`broken` set** (malformed-output regressions).

| case_id | class | raw output (truncated) | recoverable? |
|---|---|---|---|
| `…_000008` | echoed clause | `"The Receiving Party may retain one (1) copy…"` | no clean trailing label → stays malformed |
| `…_000013` | echoed clause **ending in valid JSON** | `"…third party…{ "label": "entailment" }"` | **parser gap** — a valid `{"label":"entailment"}` is embedded but the leading echo defeated extraction |
| `…_000272` | label + suffix | `"entailment_verbal_included"` | correctly **not** recovered — `_` is a regex word char, no boundary (locked by an existing test) |

Classification: 000008 = echo without clean final label (true malformed); 000013 = parser
normalization gap (a valid label *was* present and arguably recoverable); 000272 = label
normalization gap deliberately left malformed to avoid laundering an ambiguous token.

**Do not relax the parser globally.** 000013 is the only unambiguous recovery candidate; a
narrow "trailing `{"label": X}` wins" rule would recover it, but that is a separate,
evidence-gated change, not part of this integrity review.

### Threshold semantics (verified)

- `malformed_output_rate = 3/300`, which in IEEE-754 is **exactly `0.01`** (`3/300 == 0.01`
  is `True` in Python).
- Gate (`quality_gates.py:119`): `if rate > max_malformed` with `max_malformed=0.01`.
  `0.01 > 0.01` is `False` → **passes**. The gate is effectively **inclusive (≤)**.
- Boundary: `0.009999 → pass`, `0.010000 → pass`, `0.010001 → fail`. Locked by a new
  boundary test.
- The UI renders `"%.2f" % (rate*100)` = `1.00%` and does **not** show the raw count. The
  report should show **both** `3 / 300` and the precise rate. (UI count display recommended;
  see §K.)

This result sits **exactly on the gate line**. It is fragile: a 4th malformed case flips the
gate to fail.

---

## E. Fixed (29) and broken (18) — what HDBSCAN changes

Reconstructed from `paired_comparison/{fixed,broken}.jsonl` (which carry
expected/baseline/candidate labels for all 300 cases).

**Fixed (29):** expected {entailment: 25, contradiction: 4}. Transitions:
- `neutral → entailment` ×22  ← dominant: baseline under-called entailment
- `neutral → contradiction` ×4
- `contradiction → entailment` ×2
- `<echoed garbage> → entailment` ×1 (case 222 — baseline's own malformed, now correct)

**Broken (18):** expected {neutral: 14, entailment: 3, contradiction: 1}. Transitions:
- `neutral → contradiction` ×8  ← HDBSCAN over-fires contradiction on neutrals
- `neutral → entailment` ×6
- `entailment → <malformed>` ×3 (cases 000008, 000013, 000272 — the three §D malformeds)
- `contradiction → neutral` ×1

**Reading:** HDBSCAN fixes the baseline's **entailment-recall** problem (25 of 29 fixes are
neutral/contradiction → entailment), but pays for it by **over-firing on genuinely neutral
cases** (14 of 18 breaks were neutral) and by **3 self-inflicted malformed regressions**.
Excluding the 3 malformed, the "reasoning" broken count is 15, net +14. The gain is
**concentrated in entailment**, not broad.

---

## F. Per-label metrics (consistent normalization, both strategies)

Recomputed for both strategies with **identical** rules (out-of-vocab prediction → malformed
→ excluded; 3 real labels only):

| label | support | baseline P / R / F1 | hdbscan P / R / F1 | ΔF1 |
|---|---|---|---|---|
| entailment | 149/147 | 0.847 / 0.671 / **0.749** | 0.830 / 0.830 / **0.830** | **+0.081** |
| neutral | 128 | 0.723 / 0.836 / **0.775** | 0.930 / 0.727 / **0.816** | **+0.040** |
| contradiction | 22 | 0.515 / 0.773 / **0.618** | 0.400 / 0.909 / **0.556** | **−0.063** |
| **macro-F1** | | **0.7142** | **0.7338** | **+0.0196** |
| accuracy | | 0.7492 (n=299) | 0.7912 (n=297) | +0.0421 |

- The macro-F1 gain is **entailment (+0.081) + neutral (+0.040)**, partly offset by
  **contradiction (−0.063)**.
- **Contradiction — the minority class — degrades**: HDBSCAN raises its recall (0.77 → 0.91)
  but collapses its precision (0.52 → 0.40) by over-predicting it. This is the opposite of
  what a promotion story wants for the minority label.

**Malformed-handling consistency caveat.** Both strategies **exclude** malformed cases from
accuracy/macro-F1 (they are not scored as wrong). Because HDBSCAN has 3 malformed vs
baseline's 1, HDBSCAN's accuracy denominator is 297 vs baseline's 299 — HDBSCAN gets a
2-case "free pass". The rule is applied identically to both, but it structurally favors the
noisier strategy. If malformed were scored as **wrong** instead, HDBSCAN's accuracy edge
would shrink. This is a policy choice worth stating explicitly in the report UI.

---

## G. Reproducibility & determinism

Deterministic given the same inputs:
- Metric computation, bootstrap CI (`bootstrap_seed=1729`), paired comparison, selection,
  gates — all pure functions of the case records.
- Clustering (DBSCAN/HDBSCAN) is deterministic for fixed embeddings + params.

Non-deterministic / unverifiable from artifacts:
- **Student decoding**: temperature/top_p/seed/max_tokens/stop are provider-runtime settings
  and are **not persisted per run** — the predictions themselves are the only record. Two
  reruns of the student model are **not guaranteed identical** unless the llama.cpp server is
  pinned to `temperature=0` (greedy) with a fixed seed. This is unconfirmed for this run.
- **Embeddings / cluster assignments**: not persisted → cannot confirm identical clusters on
  rerun.

Hashes present today: none of dataset-split / task-config / clustering-config / embeddings /
cluster-assignments / prompt are exposed in the evaluation report. **Recommendation:** add a
`hashes` block (dataset split ID, task-config hash, clustering-config hash, compiled-prompt
hash, prediction-set hash) to `strategy_comparison.json` and surface it in the report. Until
then, byte-for-byte reproducibility **cannot be asserted**.

---

## H. Repeat-run stability (procedure — not executed)

No new model calls were made (no authorized local student server in this session). To
establish stability, run the identical 300-case validation **3×** with dataset, HDBSCAN
params, prompt, model, decoding, parser and gates all fixed, and record per repeat:
HDBSCAN macro-F1, baseline macro-F1, Δ, HDBSCAN malformed count, fixed, broken, selected
strategy, prompt hash, cluster-assignment hash.

Commands (adjust job/config to your harness):

```bash
# Re-run the same validation job config 3× (student server must be pinned to greedy decoding)
for i in 1 2 3; do
  DATABASE_URL="$DATABASE_URL" MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
  uv run rulekiln-bench run \
    --task examples/datasets/contractnli/task.yaml \
    --split validation \
    --label "hdbscan-repeat-$i"
done
```

**Flag as unstable if** any repeat: HDBSCAN loses to baseline, malformed > 1%, broken >
fixed, prompt/cluster hash changes, or macro-F1 moves materially (> the bootstrap CI
half-width ≈ 0.055).

---

## I. Statistical confidence

Paired bootstrap over the 300 cases, resampling cases with replacement, `seed=1729`,
`B=10000`, recomputing macro-F1 for both arms in each resample:

```
observed Δmacro-F1 = +0.0196
bootstrap mean Δ   = +0.0196
95% CI             = [-0.0371, +0.0773]     ← includes 0
P(HDBSCAN > baseline) = 0.754
```

Paired correctness (McNemar), discordant pairs fixed=29 / broken=18:

```
exact binomial two-sided p = 0.1439
McNemar χ² (cont. corrected) = 2.13   (< 3.841 critical @ 0.05)
```

**Conclusion:** the improvement is **not statistically significant** at n=300. The point
estimate favors HDBSCAN, but the CI crosses zero and McNemar p ≈ 0.14. Do **not** claim a
significant win. (Bootstrap/McNemar implemented as a deterministic report utility; see
§Tests.)

---

## J. Selection & promotion semantics

Current selector correctly distinguishes *eligible winner* (passes gates, highest primary
metric among non-baseline candidates) from *baseline fallback*. HDBSCAN **is** the legitimate
selected strategy under current policy: it is eligible and has the top primary metric. The
baseline's depressed stale score did **not** change the selection (baseline is never a
candidate; it is the fallback).

What is **missing** is a distinction between *selected* and *promotable*. Recommended
explicit fields on the comparison artifact (policy change — document before enabling):

```
selected_strategy        = hdbscan          # already exists
promotion_candidate      = hdbscan
promotion_status         = "blocked"
promotion_blockers       = [
  "params_do_not_match_promoted_config (mcs=2/ms=null vs 5/1)",
  "delta_not_significant (bootstrap CI includes 0; McNemar p=0.144)",
  "malformed_on_gate_boundary (3/300 == 0.0100)",
  "minority_label_regressed (contradiction F1 -0.063)"
]
reproducibility_status   = "unverified (no persisted hashes / decoding config)"
confidence_status        = "not_significant"
```

Promotion should require reproducibility + a minimum significant delta, not just
"eligible + top metric".

---

## K. UI & artifact improvements

Implemented in this change (small, safe, test-backed):

1. **Strategy Selection "Score" column relabeled** to `Score (macro-F1)` and a **lineage
   warning** is shown when a selection-table score diverges from the live DB eval macro-F1
   for the same strategy beyond `1e-3` — this is the concrete fix for "two tables, two
   numbers, no explanation". It flags the stale `strategy_comparison.json` instead of
   silently rendering it.

Recommended follow-ups (documented, not implemented — larger / need a run to validate):

2. Show malformed as **count + rate** (`3 / 300 (1.00%)`), not rate alone.
3. Persist and display `clustering_provenance` (§B) and a `hashes` block (§G).
4. Add `promotion_status` / `promotion_blockers` (§J).
5. Score malformed cases consistently and state the policy (exclude vs wrong) in the report.

---

## Tests added

| Test | Locks |
|---|---|
| `tests/unit/test_quality_gate_malformed_boundary.py` | malformed gate is inclusive at 0.01; 0.009999/0.010000 pass, 0.010001 fails (§D) |
| `tests/unit/test_paired_comparison_join_by_id.py` | runtime paired comparison joins by `case_id`; reordering candidate case_results does not change counts; totals equal sample count (§C) |
| `tests/unit/test_selection_table_score_lineage.py` | `build_selection_table` score == eval macro-F1 for classification; UI lineage-divergence helper flags a stale selection score (§A, §K.1) |

Already-present coverage relied on (not duplicated):
`tests/unit/test_evaluator_label_vocabulary.py` — out-of-vocab / echoed predictions are
malformed and never create spurious classes (the invariant the stale artifact violated).

---

## Remaining risks

- The reported HDBSCAN "win" was computed against a **stale baseline** in the selection table
  and its gate `metric_delta`; the true edge is **+0.0196**, not the implied +0.20.
- That edge is **not statistically significant** and the **minority label regresses**.
- The run's **HDBSCAN parameters do not match the promoted config**, and text_variant /
  embedding / cluster assignments are **not persisted**, so the result is **not reproducible
  from artifacts**.
- Malformed rate is **exactly on the gate boundary**.

## Verdict

**Experimental.** Not a promotion candidate in its current form. Before treating HDBSCAN as
the best RuleKiln strategy: re-run with the *promoted* `min_cluster_size=5 / min_samples=1`
config, persist clustering + hash provenance, pin greedy decoding, and re-evaluate stability
and significance across ≥3 repeats. Only promote if the delta stays positive and significant
and the contradiction class no longer regresses.

---

## Follow-up: provenance & promotion implementation

The recommendations in §B, §G, §J, and §K.3–K.4 are now implemented. New runs capture and
persist full clustering provenance, reproducibility hashes, and an explicit promotion
decision; this run (a legacy run) is untouched and surfaces as `incomplete`/`unverified`.

**What was built**

- **Clustering provenance** (`schemas/provenance.py`, `pipeline/clustering_provenance.py`):
  a versioned `ClusteringProvenance` record captured at clustering time — requested vs.
  effective parameters, embedding model/dimension/normalization/metric, text variant, cluster
  assignments, cluster summary, and hashes. Persisted under
  `outputs/clustering/<algo>/`. See `docs/architecture/clustering_provenance.md`.
- **Canonical hashing** (`artifacts/canonical_hash.py`): one SHA-256 canonicalization path
  (sorted keys, order-sensitive lists, explicit float/null handling, recorded
  `canonicalization_version`). Job/clustering/strategy hashes are aggregated into
  `metadata/hashes.json` and embedded per-strategy in `strategy_comparison.json`
  (`pipeline/artifact_hashing.py`).
- **Provenance validation** (`pipeline/provenance_validation.py`,
  `metadata/provenance_validation.json`): structured findings for cross-job artifacts,
  cluster-assignment/prompt hash mismatches, case-set mismatches, and runtime-vs-approved
  config drift. Errors block promotion; they never change selection.
- **Promotion status & policy** (`pipeline/promotion.py`, `schemas/promotion.py`,
  `metadata/promotion_status.json`): explicit `selected` / `promotion_candidate` /
  `promoted` semantics with a versioned `PromotionPolicy` and deterministic blockers. See
  `docs/architecture/strategy_promotion.md`.
- **Approved-config registry** (`config/approved_strategy_configs/hdbscan.json`): the
  promoted `5/1 eom` config, compared by canonical hash — so a `2/null` run is detected as
  not-yet-approved rather than assumed correct.
- **UI**: the eval report now has a "Provenance & Promotion" section with run/strategy
  hashes, requested vs. effective params, and promotion blockers, with distinct
  selected/candidate/promoted labels. Legacy runs show "Provenance unavailable".
- **Manifest**: `metadata/manifest.json` gains rich `entries` (type, schema version,
  sha256, size, timestamp, strategy) verifiable against the files on disk.

**How this run would read under the new system.** Re-evaluated under the default
`PromotionPolicy`, the HDBSCAN result is `promotion_status: blocked` with blockers
`CONFIG_NOT_PROMOTED_CONFIG` (ran `2/null`, approved is `5/1`), `MINORITY_LABEL_REGRESSION`
(contradiction F1 −0.063), and `REPRODUCIBILITY_UNVERIFIED` — while remaining the *selected*
strategy for analysis. This is exactly the selected-≠-promoted distinction §J asked for.
Nothing was backfilled onto the historical artifacts.
