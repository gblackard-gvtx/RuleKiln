# Spec: Classroom-Aware Benchmark Phases 1–4

**Status:** Active. Supersedes the phase-level benchmark guidance in
`SPEC_tiered_teachers_and_classroom_students.md` (Tasks 5–7).
**Depends on:** `SPEC_tiered_teachers_and_classroom_students.md` (Tasks 1–4 complete —
`TeacherConfig`, `ClassroomConfig`, `evaluate_classroom`, `BenchmarkManifest v2`).
**Scope:** Benchmark correctness, statistical credibility, and rule quality reporting.
No new ML concepts.

---

## Primary benchmark unit

The primary benchmark unit is **strategy × student**, not strategy alone.

Every metric, confidence interval, confusion matrix, paired comparison, and lift figure
must be producible at the strategy × student level. Aggregate (classroom-level) numbers
are permitted as summaries, but they must not replace per-student detail.

The constraint across all four phases:

> Non-anchor students do NOT participate in refinement loop iterations. They run
> final evaluation across all strategies after the anchor-driven loop has converged.

---

## Phase 1 — Reproducibility

**Goal:** Every benchmark run is fully reproducible from its manifest alone.

### 1.1 Benchmark manifest must include classroom provenance

`BenchmarkManifest` already carries `teacher_config`, `classroom_config`,
`conflict_resolution_anchor_id`, `extraction_cache_hits`, and
`extraction_cache_misses` (added in v2). The following fields are still missing
and must be added:

```python
class BenchmarkManifest(BaseModel):
    # … existing fields …

    # Provider profile names used per role (not full credentials — just the name)
    teacher_provider_profile: str | None = None       # e.g. "anthropic-default"
    student_provider_profiles: list[str] = Field(default_factory=list)  # one per student in order
    embedding_provider_profile: str | None = None

    # Concurrency-relevant config captured at run time
    max_concurrent_students: int | None = None
    max_concurrent_cases: int | None = None
```

The schema version must be bumped to `rulekiln.benchmark_manifest.v3` when these
fields are added.

**Acceptance:**
- A re-run from the stored manifest produces byte-identical prompt hashes.
- Provider profile names appear in the manifest; no API keys or secrets do.
- `extraction_cache_hits` / `extraction_cache_misses` are accurate.

### 1.2 strategy_comparison.json — student_results keyed by student_id

`BenchmarkStrategyComparison.student_results` already has type
`dict[str, StudentEvalSummary]`. It must be populated for every strategy that
runs a student model. The key is `student_id`, not student index.

The JSON written to disk must be directly addressable:

```json
{
  "schema_version": "rulekiln.strategy_comparison.v2",
  "strategy": "rulekiln_dbscan",
  "student_results": {
    "qwen_7b":  { "student_id": "qwen_7b", "macro_f1": 0.71, ... },
    "qwen_14b": { "student_id": "qwen_14b", "macro_f1": 0.79, ... }
  }
}
```

Embedding-only strategies (`embedding_centroid`, `embedding_knn_*`) produce no
student model calls and must not appear in `student_results`. They are reported
separately as non-LLM baselines (see Phase 2).

**Acceptance:**
- Deserializing `strategy_comparison.json` yields a `BenchmarkStrategyComparison`
  with `student_results` populated for all prompt-based strategies.
- Each key in `student_results` matches a `student_id` from `classroom_config.students`.

### 1.3 summary.md — cross-strategy × student matrix

The current `_render_student_matrix_section` renders a single-strategy student
column. The summary must instead render a full cross-strategy matrix.

Required table layout:

```markdown
## Strategy × Student Results (macro_f1)

| Strategy               | qwen_7b | qwen_14b | qwen_32b | haiku  |
|------------------------|--------:|---------:|---------:|-------:|
| baseline_scaffold      |  0.41   |   0.52   |   0.61   |  0.67  |
| baseline_few_shot_k5   |  0.55   |   0.63   |   0.70   |  0.74  |
| rulekiln_dbscan        |  0.71   |   0.79   |   0.84   |  0.88  |
| rulekiln_hdbscan       |  0.72   |   0.80   |   0.85   |  0.89  |
```

Column headers use `display_name` from `StudentConfig`, not the raw `id`.

A second matrix for accuracy is also required; confidence interval notation
(`[lo, hi]`) is recommended when CI data is available.

**Acceptance:**
- Summary renders without error when `student_results` is empty (single-student
  backward-compat path produces a single-column matrix).
- Column ordering follows `classroom_config.students` list order, not sort order.

---

## Phase 2 — Baselines

**Goal:** Every prompt-based baseline produces a `student_results` map across all
configured students.

Baseline evaluation is **not** anchor-only. All students run all baseline strategies.
This is required because the classroom lift claim ("RuleKiln improves every student
relative to its own baseline") must be grounded in per-student baseline numbers.

### 2.1 Strategies that must produce student_results

| Strategy | Category | Produces student_results |
|---|---|---|
| `baseline_scaffold` | Prompt baseline | Yes — all students |
| `baseline_few_shot_k3` | Prompt baseline | Yes — all students |
| `baseline_few_shot_k5` | Prompt baseline | Yes — all students |
| `retrieval_few_shot_k5` | Prompt baseline | Yes — all students |
| `rulekiln_dbscan` | Distilled | Yes — all students |
| `rulekiln_hdbscan` | Distilled | Yes — all students |
| `embedding_centroid` | Non-LLM baseline | No student_results |
| `embedding_knn_k1` | Non-LLM baseline | No student_results |
| `embedding_knn_k3` | Non-LLM baseline | No student_results |
| `embedding_knn_k5` | Non-LLM baseline | No student_results |

### 2.2 Non-LLM baseline reporting

Embedding-only strategies are reported in a separate section of `summary.md` and
`strategy_comparison.json` because they run no student model. They are not excluded
from the benchmark; they are classified as a distinct category without
`student_results`.

```markdown
## Non-LLM Baselines (embedding-only)

| Strategy          | macro_f1 | accuracy |
|-------------------|--------:|---------:|
| embedding_knn_k5  |  0.38   |  0.40    |
| embedding_centroid|  0.33   |  0.35    |
```

### 2.3 Baseline runs are not anchor-only

`evaluate_classroom` for baseline strategies must be called with `anchor_only=False`.
The anchor restriction applies **only** within loop iterations of conflict resolution.

**Acceptance:**
- `baseline_scaffold` `student_results` contains one entry per configured student.
- `retrieval_few_shot_k5` runs per-case retrieval × per-student inference; the
  total call count is `n_validation_cases × n_students`.
- Embedding-only strategies appear in a dedicated section, not in the student matrix.

---

## Phase 3 — Statistical Credibility

**Goal:** Every claim about lift has a per-student confidence interval and a valid
paired comparison grounding it.

### 3.1 Confidence intervals per strategy × student

`StudentEvalSummary` already carries `macro_f1_ci_95` and `accuracy_ci_95`. These
must be populated for every student × strategy combination. The bootstrap seed must
be identical across students within a run (to make CI widths comparable); it must
be stored in `BenchmarkManifest.seed`.

```python
class StudentEvalSummary(BaseModel):
    student_id: str
    macro_f1: float | None = None
    macro_f1_ci_95: tuple[float, float] | None = None
    accuracy: float | None = None
    accuracy_ci_95: tuple[float, float] | None = None
    malformed_rate: float = 0.0
    cost_usd: float | None = None
    latency_p95_ms: float | None = None
```

No aggregate-only CI is acceptable. If a student's CI cannot be computed (sample
too small), the field is `None` and a warning is logged; the run is not aborted.

### 3.2 Paired comparisons per student

Paired comparison compares each student's distilled result against that **same
student's** baseline result — not against a different student's or an aggregate.

Each strategy × student pair (rulekiln_strategy, student_id) must produce:
- `fixed_count` — baseline wrong, distilled right
- `broken_count` — baseline right, distilled wrong
- `net_fix_rate` — (fixed − broken) / n_cases
- `overall_net_fix_rate` — (fixed − broken) / n_total (including unchanged)

These go into `BenchmarkStrategyComparison.student_results[student_id]` via a
`PairedComparisonSummary` sub-field (or equivalent extension of `StudentEvalSummary`).

### 3.3 Per-label metrics and confusion matrices per strategy × student

The artifact writer must emit per-student artifacts under a namespaced path:

```
outputs/
  per_student/
    {student_id}/
      {strategy}_eval.json
      {strategy}_per_label_metrics.csv
      {strategy}_confusion_matrix.csv
```

This is additive — the existing top-level `{strategy}_eval.json` (for the anchor
student or single-student path) is preserved for backward compatibility.

### 3.4 summary.md — per-student lift and classroom aggregate lift

The strategy × student matrix from Phase 1 must be accompanied by a lift table
showing the delta from each student's own baseline:

```markdown
## Lift vs. Baseline (macro_f1 delta from baseline_scaffold)

| Strategy             | qwen_7b | qwen_14b | qwen_32b | haiku  |
|----------------------|--------:|---------:|---------:|-------:|
| baseline_few_shot_k5 |  +0.14  |  +0.11   |  +0.09   |  +0.07 |
| rulekiln_dbscan      |  +0.30  |  +0.27   |  +0.23   |  +0.21 |
| rulekiln_hdbscan     |  +0.31  |  +0.28   |  +0.24   |  +0.22 |

Classroom aggregate lift (mean across students):
  rulekiln_hdbscan vs. baseline_scaffold: +0.2625
```

The aggregate is the unweighted mean across students, clearly labeled as such.

**Acceptance:**
- Lift deltas are computed against the same student's baseline strategy, not a
  global aggregate baseline.
- The classroom aggregate is labeled "mean across students, unweighted."
- Per-student artifact paths are emitted for every prompt-based strategy.

---

## Phase 4 — Rule Quality

**Goal:** Rule provenance, ablation, and pruning mode selection are grounded in
student-level utility signals.

### 4.1 Conflict-resolution refinement uses only classroom.anchor_student

The closed-loop refinement (SPEC_close_conflict_resolution_loop) uses only the
anchor student's failure cases. Non-anchor students do not participate in any
refinement iteration. This remains unchanged from the existing implementation.

**What changes:** The `refining_rules` stage log must explicitly record the
anchor student ID used in each iteration so the manifest is auditable.

### 4.2 Rule utility reported per student

After convergence, the final rule set's utility must be attributed per student:

```python
class RuleStudentUtility(BaseModel):
    rule_id: str
    student_id: str
    fixed_count: int       # cases this rule helped this student
    broken_count: int      # cases this rule hurt this student
    net_utility: int       # fixed_count - broken_count
    utility_per_token: float  # net_utility / rule_token_count

class RuleProvenanceReport(BaseModel):
    rule_id: str
    topic: str
    support_count: int
    source_case_ids: list[str]
    student_utility: list[RuleStudentUtility]  # one per student
    anchor_net_utility: int | None   # shortcut for anchor student's net_utility
    mean_classroom_net_utility: float | None  # mean across all students
    worst_student_net_utility: int | None     # min across students
```

The `rulekiln-benchmark refinement-ablation` CLI must emit these fields when
classroom config is present.

### 4.3 Prompt pruning modes extended for classroom

Current modes: `support_count`, `utility`, `utility_per_token`.

New classroom-aware modes to add:

| Mode | Description |
|---|---|
| `anchor_utility` | Rank by anchor student's `net_utility` (equivalent to `utility` when single-student) |
| `mean_classroom_utility` | Rank by unweighted mean `net_utility` across all students |
| `worst_student_utility` | Rank by minimum `net_utility` across students (maximizes worst-case coverage) |

The `utility_per_token` mode remains unchanged and uses the anchor student's utility
(consistent with its current behavior as anchor-only).

`PruningModeComparison` (already in `schemas.pipeline`) must be extended to include
the three new modes:

```python
PruningMode = Literal[
    "support_count",
    "utility",
    "utility_per_token",
    "anchor_utility",
    "mean_classroom_utility",
    "worst_student_utility",
]
```

**Acceptance:**
- `worst_student_utility` mode selects rules that help all students, not just the
  anchor. Unit test: a rule with high anchor utility but negative utility for one
  other student ranks lower under `worst_student_utility` than under `anchor_utility`.
- When classroom has one student, `mean_classroom_utility` == `anchor_utility` ==
  `utility` (numerically identical).
- `PruningModeComparison` rows include all active modes in the ablation report.

### 4.4 Non-anchor students run final eval across all strategies after convergence

After the conflict-resolution loop converges on the anchor student, the full
classroom evaluation runs once across all strategies × all students. This is the
single pass that populates the strategy × student matrix.

The pipeline stage order for this is:

```
refining_rules (anchor only, loop until convergence)
  → evaluating_classroom_final (all students × all strategies)
  → checking_quality_gates
  → logging_artifacts
```

`evaluating_classroom_final` is a new stage name that clarifies its scope. Existing
stage names for per-strategy evaluation are unaffected.

---

## Implementation priority

| Phase | Status | Priority |
|---|---|---|
| Phase 1.1 — Manifest provider/concurrency fields | Not implemented | High |
| Phase 1.2 — student_results population | Partially implemented (schema exists) | High |
| Phase 1.3 — Cross-strategy × student matrix in summary.md | Not implemented | High |
| Phase 2.1 — All baselines run for all students | Not implemented | High |
| Phase 2.2 — Non-LLM baseline section | Not implemented | Medium |
| Phase 3.1 — CI per strategy × student | Not implemented | High |
| Phase 3.2 — Paired comparisons per student | Not implemented | High |
| Phase 3.3 — Per-student artifact paths | Not implemented | Medium |
| Phase 3.4 — Per-student lift in summary.md | Not implemented | High |
| Phase 4.1 — Anchor ID in refinement log | Trivial | Low |
| Phase 4.2 — Rule utility per student | Not implemented | Medium |
| Phase 4.3 — New pruning modes | Not implemented | Medium |
| Phase 4.4 — `evaluating_classroom_final` stage | Not implemented | High |

Phases 1 and 2 are prerequisites for Phases 3 and 4.
