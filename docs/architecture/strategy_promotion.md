# Strategy selection versus promotion

The integrity review found that RuleKiln conflated two decisions. This document defines them
separately and documents the promotion states, blocker codes, policy configuration, and how
promotion differs from quality gates.

## Selection versus promotion

* **Selection** (quality gates) decides whether a strategy is **eligible to be selected** as
  a job's best result. It is governed by `pipeline/quality_gates.py` and produces the
  `selected_strategy` in `strategy_comparison.json`. Selection is for *exploratory analysis*:
  it identifies the best available result even when provenance is incomplete.
* **Promotion** (this system) decides whether an eligible, selected strategy is **approved for
  reuse/deployment** under a versioned `PromotionPolicy`. It is governed by
  `pipeline/promotion.py` and produces `metadata/promotion_status.json`.

**Selected does not mean production-ready.** A strategy can be the top-ranked eligible result
and still be blocked from promotion.

### Why provenance errors block promotion but not selection

Selection must still work for older or partially-instrumented runs, so provenance validation
never changes selection. Promotion, by contrast, is a stronger claim ("safe to reuse"), so a
provenance **error** (a data-integrity failure such as a cross-job artifact or a
prompt/assignment hash mismatch) blocks promotion. Provenance **warnings** (for example, a
runtime config that differs from the approved config) do not, on their own, block on the
provenance axis — but the approved-config check is a separate promotion blocker.

## Promotion states

`promotion_status` (`schemas/promotion.py`, `PromotionStatus`,
`schema_version = "rulekiln.promotion_status.v1"`):

| Status | Meaning |
|---|---|
| `not_evaluated` | No selected strategy, or policy disabled. |
| `experimental` | Selected but not put through promotion (e.g. legacy run). |
| `candidate` | Under review. |
| `blocked` | One or more promotion blockers present. |
| `approved` | All policy checks pass; **ready for a human to promote**. |
| `promoted` | Explicitly promoted (set upstream, never automatically). |
| `rejected` | Explicitly rejected. |

The engine never sets `promoted`. Reaching `approved` means "nothing blocks promotion";
`promoted_strategy` stays `null` until an explicit human/registry approval is recorded.

Supporting axes: `provenance_status` (`missing`/`incomplete`/`valid`/`invalid`),
`reproducibility_status` (`unknown`/`unverified`/`partially_verified`/`reproducible`/
`unstable`), `confidence_status` (`not_computed`/`insufficient`/`suggestive`/`significant`).

## Blocker codes

Blockers are emitted deterministically in this order (`promotion.BLOCKER_ORDER`):

| Code | Raised when |
|---|---|
| `SELECTED_STRATEGY_NOT_ELIGIBLE` | Selected strategy failed its quality gate. |
| `CONFIG_NOT_PROMOTED_CONFIG` | Runtime clustering config hash ≠ approved config hash. |
| `METRIC_DELTA_INSUFFICIENT` | Metric delta below `min_metric_delta`. |
| `PAIRED_FIXED_NOT_GT_BROKEN` | Paired fixed count ≤ broken count. |
| `MALFORMED_RATE_ABOVE_THRESHOLD` | Malformed rate above `max_malformed_rate`. |
| `MALFORMED_RATE_ON_BOUNDARY` | Malformed rate equals threshold (only if `block_on_threshold_boundary`). |
| `MINORITY_LABEL_REGRESSION` | A guarded label's F1 regressed beyond `max_f1_regression`. |
| `STATISTICAL_CONFIDENCE_INSUFFICIENT` | Confidence not `significant` (only if `require_confidence_check`). |
| `PROVENANCE_INVALID` | Provenance validation has errors, or provenance missing. |
| `REPRODUCIBILITY_UNVERIFIED` | Fewer than `required_repeat_count` repeats, or not `reproducible`. |

## Policy configuration

`PromotionPolicy` (`schema_version = "rulekiln.promotion_policy.v1"`) is versioned rather than
hardcoded. Defaults **preserve current product behavior**: no statistical-significance
requirement is enabled by default (`require_confidence_check = false`), and the malformed-rate
boundary is not blocked by default (`block_on_threshold_boundary = false`). Key inputs:

```yaml
promotion:
  enabled: true
  policy_version: v1
  require_selected_strategy_eligible: true
  require_positive_metric_delta: true
  min_metric_delta: 0.0
  require_paired_fixed_gt_broken: true
  require_reproducibility: true
  required_repeat_count: 3
  require_valid_provenance: true
  require_matching_approved_config: true
  require_confidence_check: false        # off by default; needs product review to enable
  confidence_level: 0.95
  max_mcnemar_p_value: 0.05
  max_malformed_rate: 0.01
  block_on_threshold_boundary: false
  minority_label_guard:
    enabled: true
    labels: [contradiction]
    max_f1_regression: 0.02
```

## Approved configuration identity

The approved/promoted clustering config for a strategy is defined by a checked-in registry at
`src/rulekiln/config/approved_strategy_configs/<strategy>.json` (loaded by
`pipeline/approved_configs.py`), **not** inferred from the current `task.yaml`. Promotion
compares a run's runtime config hash against the registry's `config_hash` (recomputed under
the current canonicalization version). A run that used HDBSCAN `2/null` instead of the
promoted `5/1` is therefore deterministically detected via `CONFIG_NOT_PROMOTED_CONFIG`.

## Example: the ContractNLI HDBSCAN run

Under the default policy, the investigated HDBSCAN run resolves to `blocked` with:

* `CONFIG_NOT_PROMOTED_CONFIG` — ran `min_cluster_size=2, min_samples=null`, approved is `5/1`.
* `MINORITY_LABEL_REGRESSION` — contradiction F1 regressed beyond tolerance.
* `REPRODUCIBILITY_UNVERIFIED` — no repeat runs / persisted decoding config.

with `provenance_status: incomplete`, `reproducibility_status: unverified`,
`confidence_status: insufficient`. It remains the **selected** strategy for analysis but is
**not** promoted — exactly the distinction the review asked for.

## How promotion differs from quality gates

| | Quality gates (selection) | Promotion policy |
|---|---|---|
| Question | Eligible to be selected? | Approved for reuse/deployment? |
| Scope | Per-strategy metric/malformed/golden checks | Whole selected result + provenance + reproducibility |
| Provenance errors | Ignored | Block |
| Config identity | Not checked | Compared to approved registry by hash |
| Reproducibility | Not checked | Required (repeat runs) |
| Output | `selected_strategy` | `promotion_status` + blockers |
