# Investigation: Baseline Fallback and Promotion Semantics

**Status:** Investigation complete (no production code changed)
**Type:** Investigation only — see [Non-goals](#non-goals-and-scope)

## 1. Executive summary

RuleKiln's promotion pipeline is broken by omission, not by faulty logic: `evaluate_promotion()`
(`src/rulekiln/pipeline/promotion.py:118-301`) never checks the *family* of the strategy it is
asked to evaluate. It treats `selected_strategy` — whatever strategy `strategy_selection.py`
picked as the best *eligible* result for analysis purposes, including a baseline used as a
fallback when no distilled strategy passes its quality gate — as unconditionally eligible to be
a *promotion candidate*. The selection side is correct and intentional: falling back to
`baseline_scaffold` when no distilled strategy is gate-eligible is documented, tested, and
exactly the exploratory-analysis behavior selection is supposed to provide. The defect is
entirely downstream, in promotion, which was designed and documented (see
`docs/architecture/strategy_promotion.md`) around the assumption that only distilled strategies
would ever reach it as a promotion candidate — but no code enforces that assumption. When a
baseline is selected, promotion evaluates blockers that were designed for distilled-vs-baseline
comparisons against a strategy that has no baseline to compare against, producing "blocked" with
blockers such as `METRIC_DELTA_INSUFFICIENT (metric_delta: null)` and
`PAIRED_FIXED_NOT_GT_BROKEN (fixed=0, broken=0)` that carry no real signal.

A second, independent, pre-existing gap was found during this investigation: the
`REPRODUCIBILITY_UNVERIFIED` blocker fires unconditionally for every job (baseline or distilled)
because `output_stage_services.py:750-752` hardcodes `reproducibility_status="unverified"`,
`confidence_status="not_computed"`, and `repeat_run_count=1` at the call site — reproducibility
is never actually measured for any strategy today. This is a separate stub gap, not specific to
baseline fallback, and is called out here only because it appears in the observed blocker list;
it should not be conflated with the primary fix.

## 2. Reproduction description

Smoke run observed:

- Selected evaluation strategy: `baseline_scaffold`
- Selected family: `baseline`
- Best distilled strategy (informational): `dbscan`
- Quality gate outcome: no distilled strategy passed its quality gate
- Promotion candidate: `baseline_scaffold` (identical to the selected strategy)
- Promotion status: `blocked`
- Blockers:
  - `SELECTED_STRATEGY_NOT_ELIGIBLE`
  - `METRIC_DELTA_INSUFFICIENT` (`metric_delta: null`)
  - `PAIRED_FIXED_NOT_GT_BROKEN` (`fixed=0, broken=0`)
  - `REPRODUCIBILITY_UNVERIFIED`

This is a straightforward consequence of the code paths described below, not an environment- or
data-specific artifact — it will reproduce for any job where the best distilled strategy fails
its quality gate and a baseline is selected as the fallback.

## 3. Current behavior

1. `distillation_worker.py` runs quality gates only for non-baseline strategies (`quality_gates.py`
   is invoked at `distillation_worker.py:514-523`; baseline strategies are never gated).
2. `strategy_selection.py` (`select_strategy_generic`, `build_strategy_comparison`) selects the
   best *eligible* strategy across all families. When no distilled strategy is gate-eligible, it
   correctly falls back to the best eligible baseline (`baseline_scaffold` in this run). This is
   intentional, documented, and covered by `tests/unit/test_strategy_selection_regression.py`.
3. `output_stage_services.py::_write_provenance_and_promotion` (595-789) reads
   `comparison.strategy_gates.get(selected_strategy)` (720) — `None` for a baseline strategy,
   since gates were never computed for it — and derives `metric_delta`, `paired_fixed`,
   `paired_broken` from that (mostly-empty) gate or from `comparison.paired_comparison`
   (732-742), yielding `None`/`0` values. It also hardcodes
   `reproducibility_status="unverified"`, `confidence_status="not_computed"`,
   `repeat_run_count=1` (750-752) for every call, regardless of strategy.
4. `evaluate_promotion()` (`promotion.py:118-301`) receives `selected_strategy="baseline_scaffold"`
   and sets `promotion_candidate=selected_strategy` unconditionally at line 148 — no check of
   `strategy_family(selected_strategy)` exists anywhere in the function. It then runs every
   configured blocker check against a baseline strategy that was never designed to be evaluated
   this way, producing the four observed blockers.

## 4. Expected semantic model

Confirmed (not merely assumed) as the correct model per the investigation: **only
`distilled`-family strategies should ever become promotion candidates.** `baseline`,
`embedding_baseline`, and `retrieval_baseline` strategies are comparators used to measure a
distilled strategy's improvement — they are never themselves subject to "approved for
reuse/deployment" as promotion is defined in `docs/architecture/strategy_promotion.md:13-15`.
When no eligible distilled strategy exists for a job, `promotion_candidate` should be `null`, and
the resulting status should communicate "there is nothing to promote" rather than "blocked" with
blockers computed against meaningless comparison values. This matches the "Selection versus
promotion" distinction the architecture doc already draws (`strategy_promotion.md:7-18`):
selection is exploratory and family-agnostic; promotion is a stronger, narrower claim that only
makes sense for the family selection was actually built to validate.

## 5. Code-path trace

| File:lines | Role | Family-aware? |
|---|---|---|
| `pipeline/strategy_families.py` (28 lines) | Sole source of truth for `strategy_family()`; `_DISTILLED_STRATEGIES = frozenset({"dbscan", "hdbscan"})` | Yes — but **never imported by `promotion.py`** |
| `pipeline/strategy_selection.py:68-147,188-254,257-321` | `select_strategy_generic`, `build_selection_table`, `_is_eligible`, `build_strategy_comparison` | Yes; correct, deliberate baseline-fallback, untouched by this investigation |
| `workers/pipeline/strategy_utils.py:50-78` | `_best_strategy_for_family` — gate-blind, informational "best distilled" | Yes, informational only |
| `pipeline/quality_gates.py:59-182` | `check_quality_gates()` — only invoked for non-baseline strategies (`distillation_worker.py:514-523`) | Effectively yes, by caller discipline |
| `pipeline/provenance_validation.py:29` | `_DISTILLED_STRATEGIES = ("dbscan", "hdbscan")` scopes clustering-hash checks | Yes |
| `pipeline/artifact_hashing.py:121-146` | `build_strategy_provenance_row` leaves clustering fields `None` for non-clustering strategies | Yes |
| `workers/pipeline/output_stage_services.py:595-789` | `_write_provenance_and_promotion` — call site into `evaluate_promotion`; hardcoded reproducibility stub at 750-752 | No family gate before calling promotion |
| `pipeline/promotion.py:118-301` | `evaluate_promotion()` — **bug location** | **No — never imports or checks `strategy_family`** |

## 6. Root-cause findings

1. **Primary root cause:** `evaluate_promotion()` has no strategy-family guard. Line 148
   (`promotion_candidate=selected_strategy`) and the unconditional blocker-evaluation block
   (166-298) run identically regardless of whether `selected_strategy` belongs to the `distilled`
   family or any baseline family. The function does not import `strategy_families.py` at all.
2. **Secondary, independent finding:** `output_stage_services.py:750-752` hardcodes
   reproducibility/confidence fields rather than computing them, so
   `REPRODUCIBILITY_UNVERIFIED` fires for every job today, distilled or baseline. This
   contributes to the observed blocker list but is not caused by, and should not be fixed as
   part of, the family-eligibility bug.

## 7. Strategy-family findings

`strategy_family()` in `pipeline/strategy_families.py` is the single, canonical, string-inferred
source of truth (not a DB column or enum member):

```python
DISTILLED_FAMILY = "distilled"
BASELINE_FAMILY = "baseline"
EMBEDDING_BASELINE_FAMILY = "embedding_baseline"
RETRIEVAL_BASELINE_FAMILY = "retrieval_baseline"
_DISTILLED_STRATEGIES: frozenset[str] = frozenset({"dbscan", "hdbscan"})

def strategy_family(strategy: str) -> str:
    if strategy in _DISTILLED_STRATEGIES:
        return DISTILLED_FAMILY
    if strategy.startswith("embedding_"):
        return EMBEDDING_BASELINE_FAMILY
    if strategy.startswith("retrieval_"):
        return RETRIEVAL_BASELINE_FAMILY
    return BASELINE_FAMILY
```

It is unambiguous, already consumed correctly by selection, provenance validation, and artifact
hashing (see the trace table above) — every part of the pipeline *except* `promotion.py` already
respects family boundaries. The gap is narrowly that `promotion.py` never imports it.

## 8. Blocker-by-blocker audit

| Blocker | Meaningful for a baseline promotion candidate? | Why |
|---|---|---|
| `SELECTED_STRATEGY_NOT_ELIGIBLE` | No | Baseline strategies are never run through `check_quality_gates()` (non-baseline-only per `distillation_worker.py:514-523`), so `selected_gate` is always `None` for a baseline — "not eligible" reads as a failure when it is structurally guaranteed to trip. |
| `METRIC_DELTA_INSUFFICIENT` (`metric_delta: null`) | No | There is no None-guard on this check (`promotion.py:196-208`), unlike the malformed-rate check which is properly guarded (`221-245`, `if malformed_rate is not None:`). The message f-string literally renders the text `None`. |
| `PAIRED_FIXED_NOT_GT_BROKEN` (`fixed=0, broken=0`) | No | This compares a distilled strategy's paired fix/break counts against its baseline; there is no such pairing when the "candidate" *is* the baseline. |
| `REPRODUCIBILITY_UNVERIFIED` | Legitimate concept, but not currently informative | Fires unconditionally for all strategies today due to the hardcoded stub in `output_stage_services.py:750-752` — a real, separate gap, not specific to baseline fallback. |

`CONFIG_NOT_PROMOTED_CONFIG`, `MALFORMED_RATE_*`, `MINORITY_LABEL_REGRESSION`,
`STATISTICAL_CONFIDENCE_INSUFFICIENT`, `PROVENANCE_INVALID`, and `STALE_ARTIFACT_DETECTED` did not
fire in the observed run (correctly guarded/None-safe or not applicable), and are not implicated
in this defect.

## 9. Provenance requirement matrix

| Axis | Meaningful for `distilled` candidate | Meaningful for `baseline`/`embedding_baseline`/`retrieval_baseline` |
|---|---|---|
| `provenance_status` (clustering hash/assignment checks) | Yes | No — `provenance_validation.py:29` already scopes `_DISTILLED_STRATEGIES` correctly |
| `stale_artifact_status` | Yes | Ambiguous; may still be worth checking generically, but not currently gated by family |
| Config-hash comparison (`CONFIG_NOT_PROMOTED_CONFIG`) | Yes | No — `runtime_config_hash`/`approved_config_hash` are `None` for non-clustering strategies (`artifact_hashing.py:121-146`), and the check is already skipped when either hash is `None` (`promotion.py:179-184`) |
| Reproducibility | Yes (once measured) | Arguably not applicable — a baseline is not "reproduced," it is regenerated deterministically or not at all |

The pattern of scoping clustering-specific checks to the `distilled` family already exists
consistently in provenance validation and artifact hashing — it simply was never extended to
promotion-candidate *eligibility itself*, which is the missing piece.

## 10. UI/report findings

- `ui/routes.py::_build_provenance_promotion_view` (283-391) reads `promotion_status.json` as an
  untyped dict, not a typed `PromotionStatus` model — it passes through whatever `promotion_status`
  string value is present.
- `templates/jobs/eval_report.html` (113-204), the Provenance & Promotion badge logic
  (`if`/`elif` around line ~135), has a safe catch-all `else` branch — a new `promotion_status`
  value would render without crashing, just without a dedicated label/color, until the template is
  updated.
- An adjacent, out-of-scope oddity: `ui/routes.py` near line 1435 (`/jobs/{job_id}/rules`) has a
  hardcoded `"hdbscan"` fallback unrelated to this defect — noted but not investigated further,
  per the spec's scope.

## 11. Artifact-regeneration findings

`promotion_status.json` is produced by `evaluate_promotion()`, a pure function of its inputs
(selected strategy, gate result, metric delta, paired counts, malformed rate, config hashes,
provenance/stale-artifact reports, reproducibility/confidence stub values). All of these inputs
are already persisted job artifacts. **`promotion_status.json` can be regenerated offline from
existing artifacts, with no teacher/student model calls and no re-clustering required** — a fix
to `evaluate_promotion()` (or its call site) can be validated by recomputing promotion status for
historical jobs already on disk, without touching the ContractNLI test split or rerunning any
model.

## 12. Compatibility analysis

`PromotionStatusValue` (`schemas/promotion.py:23-31`) already declares 7 states:
`not_evaluated | experimental | candidate | blocked | approved | promoted | rejected`. Of these,
only `not_evaluated`, `approved`, and `blocked` are ever assigned by code today (confirmed by
reading `promotion.py` in full — the only writer of `promotion_status`). `experimental`,
`candidate`, and `rejected` are documented in `strategy_promotion.md:34-42` but dead in
implementation.

Two structural remediation options, presented without a decision (out of scope to decide
unilaterally):

- **Option A (SELECTED — human decision recorded 2026-07-15)** — Add a new
  `PromotionStatusValue` literal (e.g. `not_applicable`) and a dedicated blocker/code such as
  `NO_ELIGIBLE_DISTILLED_STRATEGY`, set when `strategy_family(selected_strategy) != "distilled"`.
  `promotion_candidate` is set to `None` in this branch. This is additive to the schema (new
  Literal member) and does not remove any existing value.
- **Option B** — Reuse the existing `blocked` status, but set `promotion_candidate=None` and
  short-circuit to a single dedicated blocker (still needs a new blocker code, e.g. the same
  `NO_ELIGIBLE_DISTILLED_STRATEGY`), avoiding any schema change to `PromotionStatusValue`.
- **Reuse `experimental`** — The architecture doc already defines `experimental` as "Selected but
  not put through promotion (e.g. legacy run)" (`strategy_promotion.md:37`). A baseline-fallback
  selection arguably fits this description structurally ("not put through promotion" because it
  isn't promotion-eligible), though the doc's original intent appears to be about *legacy runs
  lacking provenance* rather than *non-distilled families*. Worth flagging as a third option for
  human judgment rather than resolving here.

## 13. Recommended patch scope

(Documented for planning purposes only — no code changes were made in this task, per the
"Production-code changes: Prohibited" constraint.)

A minimal fix would add a family check near the top of `evaluate_promotion()` (before line 162's
`if selected_strategy is None` check, or immediately after), short-circuiting to the new/reused
status *before* any blocker in the 166-298 block is evaluated, whenever
`strategy_family(selected_strategy) != DISTILLED_FAMILY`. This requires:

- Importing `strategy_families` into `promotion.py` (currently not imported).
- No change to `strategy_selection.py`, `quality_gates.py`, or any distilled-strategy code path —
  the fix is purely about gating what happens *after* a non-distilled strategy reaches the
  promotion function.

The `REPRODUCIBILITY_UNVERIFIED` stub gap (`output_stage_services.py:750-752`) is explicitly
**out of scope** for this patch and should be tracked separately.

**Additional recommendation (added 2026-07-16, human-suggested):** persist the resolved family
alongside the existing `selected_strategy` field on `PromotionStatus`, e.g.:

```json
{
  "selected_strategy": "baseline_scaffold",
  "selected_strategy_family": "baseline"
}
```

This is independent of, and complementary to, the Option A status/blocker gating: the gating
decides *behavior* (whether the strategy is treated as a candidate at all), while
`selected_strategy_family` records the *fact* the decision was based on, directly on the
artifact. Rationale:

- Makes `promotion_status.json` self-describing — a reader (UI, API consumer, or a human
  debugging a job) can see *why* `promotion_candidate` is `null`/status is `not_applicable`
  without re-deriving it from `strategy_family(selected_strategy)` against a family map that
  could change over time.
- Decouples historical artifacts from the current `strategy_families.py` definition. Today,
  `strategy_family()` is purely a string-inference function computed fresh on every read — if the
  distilled-strategy set ever changes (e.g. a new clustering algorithm is added to
  `_DISTILLED_STRATEGIES`), old artifacts would silently be reinterpreted under the new
  definition unless the family is captured at evaluation time.
- Cheap and additive: a new optional string field on an already-versioned Pydantic model
  (`schema_version = "rulekiln.promotion_status.v1"`); no existing consumer reads an unknown
  field, so this does not require a schema-version bump under the project's existing
  conventions.

## 14. File-by-file change forecast

| File | Expected change (not made in this task) |
|---|---|
| `src/rulekiln/pipeline/promotion.py` | Import `strategy_families`; add a family-eligibility short-circuit before blocker evaluation; set `selected_strategy_family` on the returned `PromotionStatus` |
| `src/rulekiln/schemas/promotion.py` | New `PromotionStatusValue` literal member (Option A, selected); possibly a new blocker-code constant in `promotion.py`'s `BLOCKER_ORDER`; new `selected_strategy_family: str \| None = None` field on `PromotionStatus` |
| `docs/architecture/strategy_promotion.md` | Update the promotion-states table to reflect the new/clarified state; add a baseline-fallback example alongside the existing HDBSCAN example (§"Example: the ContractNLI HDBSCAN run") |
| `src/rulekiln/workers/pipeline/output_stage_services.py` | Possibly pass `strategy_family(selected_strategy)` explicitly into `evaluate_promotion()` if the call site is the preferred place to resolve family rather than inside `promotion.py` itself (implementation detail, either location works) |
| `src/rulekiln/templates/jobs/eval_report.html` | Optional: dedicated badge/label for the new status, though the existing catch-all `else` means this is not strictly required |
| `tests/unit/test_promotion_policy.py` | (Not edited — forecast only) Would need a new case asserting baseline-family selection does not produce candidate-only blockers |
| `tests/unit/test_output_stage_integrity_failures.py` | (Not edited — forecast only) Existing `baseline_scaffold` fixture should be strengthened |

## 15. Test forecast

Not written, per the "do not modify tests" non-goal — listed for planning purposes:

- `test_promotion_policy.py`: a new case selecting a baseline-family strategy and asserting
  `promotion_candidate is None` (or whatever the chosen semantics dictate) and that none of
  `METRIC_DELTA_INSUFFICIENT` / `PAIRED_FIXED_NOT_GT_BROKEN` / `SELECTED_STRATEGY_NOT_ELIGIBLE`
  appear in the blocker list for that case.
- `test_output_stage_integrity_failures.py`: its existing `baseline_scaffold` fixture currently
  only asserts `promotion_status is not None` — a coverage gap that masks this exact bug today.
  It should assert the specific expected status/blockers once the fix lands.

## 16. Risks

- Adding a new `PromotionStatusValue` literal member is additive and should not break existing
  consumers, given the UI's safe catch-all `else` branch (`eval_report.html`) and the fact that
  `_build_provenance_promotion_view` reads the status as an untyped string/dict rather than a
  strict enum.
- The main risk is process, not code: bikeshedding among Option A / Option B / reusing
  `experimental` could stall the fix. A secondary risk is that a future implementer bundles the
  unrelated `REPRODUCIBILITY_UNVERIFIED` stub fix into the same change, which this investigation
  recommends against — they are independent defects with independent fixes.

## 17. Open human decisions

1. ~~Option A (new `not_applicable`-style literal) vs. Option B (reuse `blocked` + null
   candidate) vs. reusing the already-documented-but-dead `experimental` state.~~ **Resolved
   2026-07-15: Option A selected by human sign-off.**
2. The exact name of any new blocker code (this doc suggests `NO_ELIGIBLE_DISTILLED_STRATEGY` as
   a placeholder — carried forward as the working name pending implementation; not separately
   contested).
3. Whether the `REPRODUCIBILITY_UNVERIFIED` stub gap (`output_stage_services.py:750-752`) should
   be tracked as its own follow-up investigation/ticket. **Still open** — this investigation
   continues to recommend it stay out of scope for the Option A implementation.
4. Whether `DistillationJob.selected_strategy` (a DB column declared in `db/models.py` but never
   written anywhere) is worth a separate follow-up — noted here only because it was encountered
   while tracing selected-strategy provenance, not because it contributes to this defect. **Still
   open.**

## 18. Final recommendation

Option A (add a dedicated `not_applicable`-style status plus a dedicated
`NO_ELIGIBLE_DISTILLED_STRATEGY`-style blocker, gated by `strategy_family()`) is the cleanest
long-term signal: it makes "there was nothing eligible to promote" structurally distinguishable
from "something was evaluated and failed," which the UI, API consumers, and future promotion
policies can all rely on unambiguously. **Human sign-off received 2026-07-15 (Option A).**

**Additional recommendation (2026-07-16):** persist `selected_strategy_family` on
`PromotionStatus` alongside the Option A gating logic (see §13), so the artifact records the
family fact directly rather than requiring downstream readers to re-derive it from
`strategy_family()`. This is additive and low-risk to bundle into the same implementation pass
as Option A, since both touch `promotion.py` and `schemas/promotion.py` in the same
`evaluate_promotion()` call.

No code was changed as part of this investigation; implementation is tracked as a separate,
follow-up task.

## Non-goals and scope

Per the governing spec, this investigation did not: modify production code or tests; rerun
teacher or student models; access the ContractNLI test split; change strict metric semantics;
retune clustering; build `.rkprompt`; redesign the complete promotion lifecycle; weaken gates for
actual distilled candidates; or rewrite historical artifacts. All findings above were verified by
direct reading of `promotion.py`, `schemas/promotion.py`, `strategy_families.py`,
`strategy_selection.py`, `quality_gates.py`, `provenance_validation.py`, `artifact_hashing.py`,
`output_stage_services.py`, `docs/architecture/strategy_promotion.md`, and the relevant UI/test
files, with no code paths modified.
