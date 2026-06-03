# Spec: Tiered Teacher Routing, Extraction Caching, and Multi-Student Classroom Evaluation

**Status:** Pre-Phase 5. Implement after SPEC_close_conflict_resolution_loop is complete.
**Depends on:** The loop-closure spec must land first; this spec assumes per-iteration
re-pruning and `utility_signals` are working.
**Scope:** Cost architecture, benchmark richness, and a stronger differentiation story
from the paper. No new ML concepts — only a richer configuration and evaluation surface.

---

## Why this matters

The paper tested one teacher model pair and two student models. RuleKiln, as a
production implementation, should do better on both axes:

**Teacher cost is phase-asymmetric.** Extraction is O(n_training_cases) calls — the
budget killer. Synthesis and conflict resolution are O(n_clusters) and
O(iterations × rules) respectively — an order of magnitude fewer calls. The reasoning
demand is also asymmetric: extraction is mechanical ("given this input and label,
write a rule"), synthesis is moderate, conflict resolution is the hardest and most
important. Paying frontier rates uniformly across all three is wasteful and
unnecessary. The paper itself uses a split: Gemini Flash with thinking for extraction,
Gemini Pro for synthesis and conflict resolution. We should make this explicit,
configurable, and empirically validatable.

**Multi-student evaluation is the paper's core thesis, stated but not fully explored.**
The paper's claim is that PLD enables compact models to match frontier performance.
Testing one student proves it for one model. Testing a range of students — your local
7B, 14B, 32B, and a mid-tier API model — proves it across a capability spectrum and
answers the question the paper leaves open: at what model size do the distilled rules
stop helping? That is a publishable result. It also means your local compute does
real scientific work rather than just being a student proxy.

**The classroom metaphor is precise.** One teacher (or teaching team) produces the
curriculum (synthesized rules). Multiple students take the exam (inference on the
test set). The curriculum is fixed; only the students vary at evaluation time. This
means multi-student evaluation costs only additional student inference — zero extra
teacher API calls.

---

## Design decisions (resolve these before coding)

### Decision 1: Per-phase teacher config shape

Each of the three teacher-intensive stages gets its own model config. The global
`teacher_model` field becomes a fallback default; per-phase configs override it.

```yaml
teacher_config:
  default:                          # fallback if a phase config is absent
    provider: anthropic
    model: claude-sonnet-4-6

  instruction_extraction:           # O(n_cases) calls — use cheapest capable model
    provider: local
    model: qwen2.5-32b-instruct     # or whatever is running locally
    # Can also be: provider: openai, model: gpt-4o-mini — configurable, not hardcoded

  cluster_consolidation:            # O(n_clusters) calls — moderate synthesis
    provider: anthropic
    model: claude-haiku-4-5

  conflict_resolution:              # O(iterations × rules) — highest reasoning demand
    provider: anthropic
    model: claude-opus-4-6
```

**Provider-neutrality requirement:** All three phase configs must accept any
configured provider (local/anthropic/openai/gemini). This is not hardcoded to
any specific model — it is configuration. The default values above are suggestions,
not requirements.

**Backward compatibility requirement:** If `teacher_config` is a flat
`{provider, model}` (the current shape), that is treated as the `default` and all
phases inherit it. No existing job configurations break.

### Decision 2: Student config as a list ("the classroom")

The current `student_model` field (singular) becomes `students` (list). Each student
gets an ID, a display name, and a model config.

```yaml
students:
  - id: qwen_7b
    display_name: "Qwen2.5 7B"
    provider: local
    model: qwen2.5-7b-instruct

  - id: qwen_14b
    display_name: "Qwen2.5 14B"
    provider: local
    model: qwen2.5-14b-instruct

  - id: qwen_32b
    display_name: "Qwen2.5 32B"
    provider: local
    model: qwen2.5-32b-instruct

  - id: haiku
    display_name: "Claude Haiku 4.5"
    provider: anthropic
    model: claude-haiku-4-5
```

**Backward compatibility:** If `student_model` is a flat config (current shape), it is
wrapped as a single-element `students` list with `id: default`. No existing jobs break.

**Ordering guarantee:** Students are evaluated in list order. Results are always
indexed by `student_id`. Order is stable across runs given the same config.

### Decision 3: Which student(s) drive the conflict resolution loop?

This is the most important non-obvious design decision in this spec.

The conflict resolution loop (from the previous spec) feeds student failure cases back
to the teacher to revise rules. With multiple students, the question is: whose
failures?

**Option A — Weakest student only.** If the rules work for the weakest student in the
classroom, they work for all. This minimizes loop cost (one eval pass per iteration)
and produces rules calibrated for the hardest use case.

**Option B — Aggregate failures across all students.** Use the union of failures from
all students. More failures → more signal → potentially better rules. But cost scales
with classroom size and may produce rules that are over-fitted to the class average.

**Option C — Configurable anchor student.** Designate one student as the
`conflict_resolution_anchor` in config. Loop uses only that student. Defaults to
the first student in the list (usually the weakest).

**Recommendation: Option C.** It is explicit, configurable, and cheap. Advanced
users can set the anchor to their production target model. Default to first student.
Document the rationale in config comments.

```yaml
conflict_resolution_anchor: qwen_7b   # which student drives failure signal
                                       # defaults to first student if omitted
```

### Decision 4: Caching architecture for extraction

Extraction results are deterministic given the same inputs. Cache keyed on:

```
cache_key = sha256(
    input_text +
    gold_label +
    teacher_extraction_model_id +   # provider + model name, normalized
    extraction_prompt_version        # from prompt_hashes in the manifest
)
```

**Cache location:** `.rulekiln/extraction_cache/{dataset_name}/{cache_key[:2]}/{cache_key}.json`
(sharded by first 2 chars to avoid filesystem limits on large datasets).

**Cache scope:** Shared across benchmark runs for the same dataset. Re-running with
a different student model, different synthesis model, or different CR model does NOT
invalidate extraction cache — extraction outputs are teacher-only artifacts.

**Cache invalidation:** Changing the extraction model or the extraction prompt version
produces new cache keys. Old entries are not deleted automatically (they are cheap
storage and allow rollback comparisons).

**Cache content:** Full `MicroRuleSchema` output per case, plus the raw
`reasoning_trace`, the `cache_key`, and a `cached_at` timestamp.

**Cache hit reporting:** Benchmark manifest must include `extraction_cache_hits` and
`extraction_cache_misses` so cost attribution is honest. A run that was 100% cached
should clearly say so.

### Decision 5: Benchmark output shape — the strategy × student matrix

With multiple students, the benchmark output is no longer a flat table. It becomes
a matrix: strategy (scaffold_only, few_shot, rulekiln_dbscan, etc.) × student
(qwen_7b, qwen_14b, qwen_32b, haiku).

**strategy_comparison.json** gains a `student_results` map:

```json
{
  "strategy": "rulekiln_dbscan",
  "student_results": {
    "qwen_7b":  { "macro_f1": 0.71, "accuracy": 0.73, "macro_f1_ci_95": [0.68, 0.74] },
    "qwen_14b": { "macro_f1": 0.79, "accuracy": 0.81, "macro_f1_ci_95": [0.76, 0.82] },
    "qwen_32b": { "macro_f1": 0.84, "accuracy": 0.85, "macro_f1_ci_95": [0.81, 0.87] },
    "haiku":    { "macro_f1": 0.88, "accuracy": 0.89, "macro_f1_ci_95": [0.86, 0.91] }
  }
}
```

**summary.md** must include a matrix table:

```markdown
| Strategy              | Qwen 7B | Qwen 14B | Qwen 32B | Haiku  |
|-----------------------|--------:|---------:|---------:|-------:|
| Zero-shot baseline    |   0.41  |   0.52   |   0.61   |  0.67  |
| Few-shot k5           |   0.55  |   0.63   |   0.70   |  0.74  |
| RuleKiln (DBSCAN)     |   0.71  |   0.79   |   0.84   |  0.88  |
| RuleKiln (HDBSCAN)    |   0.72  |   0.80   |   0.85   |  0.89  |
```

This table directly demonstrates the paper's thesis — "PLD lifts compact models
toward frontier performance" — across a capability spectrum, not just for one model.

---

## Tasks

### Task 1 — Per-phase teacher config schema

**New Pydantic model:** `PhaseTeacherConfig`

```python
class PhaseTeacherConfig(BaseModel):
    schema_version: str = "rulekiln.phase_teacher_config.v1"
    provider: str
    model: str
    extra_params: dict[str, object] = {}   # thinking_budget, temperature, etc.

class TeacherConfig(BaseModel):
    schema_version: str = "rulekiln.teacher_config.v1"
    default: PhaseTeacherConfig
    instruction_extraction: PhaseTeacherConfig | None = None
    cluster_consolidation: PhaseTeacherConfig | None = None
    conflict_resolution: PhaseTeacherConfig | None = None

    def for_phase(self, phase: Literal[
        "instruction_extraction",
        "cluster_consolidation",
        "conflict_resolution"
    ]) -> PhaseTeacherConfig:
        """Return phase-specific config, falling back to default."""
        return getattr(self, phase) or self.default
```

**Migration:** A flat `{provider, model}` config is deserialized as `TeacherConfig`
with `default` set and all phase overrides as `None`. This is a non-breaking change.

**Acceptance:**
- Unit test: `TeacherConfig.for_phase("instruction_extraction")` returns the
  phase-specific config when set, and the default when not set.
- Existing flat-config test fixtures still deserialize without error.
- Pyright passes with no `Any`.

---

### Task 2 — Multi-student config schema

**New Pydantic model:** `StudentConfig` and updated `ClassroomConfig`

```python
class StudentConfig(BaseModel):
    id: str                           # stable identifier, used in output keys
    display_name: str
    provider: str
    model: str
    extra_params: dict[str, object] = {}

class ClassroomConfig(BaseModel):
    schema_version: str = "rulekiln.classroom_config.v1"
    students: list[StudentConfig]
    conflict_resolution_anchor: str | None = None  # student id; defaults to students[0]

    @property
    def anchor_student(self) -> StudentConfig:
        if self.conflict_resolution_anchor:
            match = next(
                (s for s in self.students if s.id == self.conflict_resolution_anchor),
                None
            )
            if match is None:
                raise ValueError(
                    f"conflict_resolution_anchor '{self.conflict_resolution_anchor}' "
                    f"not found in students list"
                )
            return match
        return self.students[0]
```

**Migration:** A flat `student_model: {provider, model}` config is wrapped as
`ClassroomConfig(students=[StudentConfig(id="default", ...)])`.

**Acceptance:**
- `anchor_student` returns the configured anchor, or `students[0]` when not set.
- Invalid anchor ID raises `ValueError` at config load time, not at eval time.
- Single-student config (backward compat) still produces single-student output shape
  (no regression on existing benchmark artifacts).

---

### Task 3 — Extraction cache

**New module:** `src/rulekiln/pipeline/extraction_cache.py`

**Interface:**

```python
class ExtractionCacheKey(BaseModel):
    input_hash: str        # sha256 of normalized input text
    gold_label: str
    model_id: str          # "{provider}/{model_name}" normalized to lowercase
    prompt_version: str    # from benchmark manifest prompt_hashes

    def cache_key(self) -> str:
        """Returns hex digest used for filesystem path."""
        ...

class ExtractionCacheEntry(BaseModel):
    schema_version: str = "rulekiln.extraction_cache.v1"
    cache_key: str
    cached_at: datetime
    model_id: str
    prompt_version: str
    micro_rule: MicroRuleSchema
    reasoning_trace: str | None

class ExtractionCache:
    def __init__(self, cache_root: Path) -> None: ...
    def get(self, key: ExtractionCacheKey) -> ExtractionCacheEntry | None: ...
    def put(self, key: ExtractionCacheKey, entry: ExtractionCacheEntry) -> None: ...
    def stats(self) -> dict[str, int]: ...  # hits, misses, total_entries
```

**Cache location:** `{cache_root}/{dataset_name}/{key[:2]}/{key}.json`

**Atomic writes:** Use write-to-temp-then-rename to prevent corrupt partial entries
if a worker crashes mid-write.

**Acceptance:**
- Unit test: put an entry, get it back, assert byte-identical `micro_rule`.
- Cache miss returns `None`, not an exception.
- Cache stats are accurate across put/get sequences.
- Changing any field of the cache key produces a different `cache_key()` (property test).
- Benchmark manifest `extraction_cache_hits` and `extraction_cache_misses` are
  populated from `ExtractionCache.stats()`.

---

### Task 4 — Multi-student evaluation runner

**New module (or extend existing):** `src/rulekiln/pipeline/classroom_evaluator.py`

**Behavior:**
- Accept a compiled prompt and a `ClassroomConfig`.
- Run inference for each student against the test split.
- Return `dict[student_id, EvalResult]`.
- Students can run concurrently (up to configured parallelism limit); results are
  collected and keyed by `student_id`.

**Interface:**

```python
async def evaluate_classroom(
    compiled_prompt: str,
    cases: list[TaskCase],
    classroom: ClassroomConfig,
    *,
    max_concurrent_students: int = 4,
    seed: int,
) -> dict[str, EvalResult]:
    ...
```

**Acceptance:**
- Single-student classroom produces `{"default": EvalResult}` (backward compat).
- Results are keyed by `student_id`, not by index.
- Concurrent execution does not produce data races on shared state.
- Each student's eval is independently resumable (DBOS per-student stage).
- Test with fake provider: 3 students, assert 3 result keys.

---

### Task 5 — Updated benchmark output schemas

**`BenchmarkManifest` additions (extend from loop-closure spec):**

```python
teacher_config: TeacherConfig            # replaces flat teacher_model field
classroom_config: ClassroomConfig        # replaces flat student_model field
extraction_cache_hits: int
extraction_cache_misses: int
conflict_resolution_anchor_id: str
```

**`StrategyResult` additions:**

```python
student_results: dict[str, StudentEvalSummary]  # keyed by student_id

class StudentEvalSummary(BaseModel):
    schema_version: str = "rulekiln.student_eval_summary.v1"
    student_id: str
    macro_f1: float
    macro_f1_ci_95: tuple[float, float] | None
    accuracy: float
    accuracy_ci_95: tuple[float, float] | None
    malformed_rate: float
    cost_usd: float | None
    latency_p95_ms: float | None
```

**`summary.md`** must include the strategy × student matrix table described in
Decision 5. The report must explicitly label columns with `display_name`, not `id`.

**Acceptance:**
- `schema_version` is present on all new schemas.
- `summary.md` matrix table is generated and parseable as Markdown.
- Single-student run produces a valid single-column matrix (not a broken table).

---

### Task 6 — Wire per-phase teacher routing into the pipeline stages

**Stages to update:**

- `extracting_instructions`: use `teacher_config.for_phase("instruction_extraction")`.
  Check extraction cache before calling teacher. Record cache hit/miss.
- `synthesizing_clusters`: use `teacher_config.for_phase("cluster_consolidation")`.
- `refining_rules` (from loop-closure spec): use
  `teacher_config.for_phase("conflict_resolution")`.
  Use `classroom.anchor_student` to run the student eval pass inside the loop.
- `evaluating_distilled` → `evaluating_classroom`: replace with the classroom
  evaluator (Task 4). All students run; anchor student's results feed conflict
  resolution.

**Acceptance:**
- Integration test: pipeline runs with three students and a two-phase teacher split.
  Assert that extraction used the extraction model, synthesis used the synthesis
  model, and conflict resolution used the CR model (log assertions or mock call tracking).
- Anchor student results feed loop-closure spec's failure analysis. Non-anchor
  students run after convergence (or in parallel at final iteration — configurable).
- All existing single-teacher, single-student tests still pass.

---

### Task 7 — Cost attribution by phase and student

**Extend `CostSummary` schema:**

```python
class PhaseCostBreakdown(BaseModel):
    instruction_extraction: float = 0.0   # USD, teacher only
    cluster_consolidation: float = 0.0
    conflict_resolution: float = 0.0
    student_evaluation: dict[str, float] = {}  # keyed by student_id

class CostSummary(BaseModel):
    schema_version: str = "rulekiln.cost_summary.v2"   # bump from v1
    total_teacher_usd: float
    total_student_usd: float
    total_usd: float
    by_phase: PhaseCostBreakdown
    extraction_cache_savings_usd: float | None  # estimated based on model pricing
```

**`extraction_cache_savings_usd`:** estimated cost of cache hits had they been real
calls. This is an estimate (requires knowing the model's per-token price) but is
useful for communicating the value of caching to users. Mark as `| None` when
pricing data is unavailable.

**Acceptance:**
- Phase totals sum to `total_teacher_usd`.
- Student totals sum to `total_student_usd`.
- Schema version bumped; migration note added.

---

## What this enables that the paper doesn't have

The paper tests one teacher configuration on two simple datasets with two student
models. This architecture enables:

**Teacher quality experiment:** Configure extraction at local-32B, synthesis at
mid-tier, CR at frontier. Then run the same benchmark with extraction at frontier.
Compare rule quality (via CR loop convergence and final student F1). This directly
characterizes the minimum teacher quality per phase — an open question the paper
leaves entirely unaddressed. That is a citable result.

**Student capability spectrum:** Run the same distilled rules against 7B, 14B, 32B,
and a mid-tier API model. Plot F1 against parameter count. The inflection point —
where the rules stop helping — tells you the minimum capable student. The paper
shows PLD helps Gemma 4B; you can show it helps or doesn't help models below that.

**Cost transparency:** The phase-level cost breakdown shows users exactly what they're
paying for and where caching saves them money. This is a practical differentiator from
the paper, which only reports aggregate cost.

---

## Guardrails for the agent

- Read the loop-closure spec and confirm it has landed before starting Task 6.
- Do not break `BenchmarkManifest` schema compatibility without a version bump.
- `ClassroomConfig` must validate at load time (bad anchor ID = immediate error, not
  silent wrong behavior at eval time).
- Extraction cache must be safe for concurrent workers (atomic writes).
- Per-phase teacher routing must not require changes to existing prompt construction
  logic — it changes only which model config is passed to `ChatModelClient`.
- Do not add new third-party dependencies for the cache (stdlib `hashlib` + `json`
  + `pathlib` are sufficient).
- Non-anchor students must not add cost to the conflict resolution loop iterations.
  They run only once at the final iteration (or in parallel with the final eval pass).

## Definition of done

- `TeacherConfig.for_phase()` routes correctly and falls back to default (Task 1, tested).
- `ClassroomConfig.anchor_student` resolves correctly, validates at load (Task 2, tested).
- Extraction cache produces hits on re-run, misses on first run, and is concurrent-safe
  (Task 3, tested).
- Classroom evaluator returns per-student results keyed by `student_id` (Task 4, tested).
- `BenchmarkManifest` and `StrategyResult` carry classroom and phase-teacher fields
  with `schema_version` (Task 5).
- Pipeline stages use phase-specific teacher models (Task 6, integration-tested).
- `CostSummary` breaks down cost by phase and student (Task 7, tested).
- `summary.md` renders a strategy × student matrix table.
- Required commands pass:
  - `uv run ruff check src/ tests/`
  - `uv run pyright`
  - `DATABASE_URL="sqlite+aiosqlite://" MLFLOW_TRACKING_URI="file:///tmp/mlflow-ci" uv run pytest -m "not external" --tb=short -q`
