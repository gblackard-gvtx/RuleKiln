# Spec: Refactor Distillation Worker into Class-Based Pipeline Services

## Status

Proposed.

## Motivation

`src/rulekiln/workers/distillation_worker.py` has grown into a large orchestration module that mixes stage control, provider resolution, evaluation, baseline generation, classroom evaluation, refinement, quality gates, artifact writing, MLflow logging, database persistence, and helper conversions in one file.

The worker currently contains:

- stage definitions
- public entry points
- a long `_run(...)` orchestration function
- inline implementation for most pipeline stages
- stage-marker idempotency checks
- provider/client resolution
- baseline strategy evaluation
- distilled strategy evaluation
- classroom evaluation
- empirical refinement loop
- ablation and pruning optimization
- quality gates
- artifact export
- MLflow logging
- DB conversion helpers

This makes the file difficult to edit safely and increases the chance that unrelated features regress each other.

The goal of this refactor is to introduce class-based services around existing behavior while preserving current stage semantics, resume behavior, artifacts, and public entry points.

---

## Goals

1. Reduce `distillation_worker.py` to a thin orchestration entry point.
2. Move stage implementations into focused service classes.
3. Preserve existing pipeline behavior.
4. Preserve existing stage-marker idempotency and resume semantics.
5. Preserve `run_distillation_pipeline(...)` and `run_pipeline_phase(...)` public APIs.
6. Improve testability by allowing each stage/service to be unit-tested independently.
7. Make the empirical conflict-resolution loop, classroom evaluation, baseline evaluation, and artifact export easier to modify without editing the whole worker.
8. Avoid broad schema or artifact changes in this refactor unless required to preserve existing behavior.

---

## Non-Goals

This refactor must not:

- redesign the pipeline stages
- change the PLD algorithm
- change benchmark semantics
- change artifact schemas
- change database schemas
- change provider behavior
- change retry/queue semantics
- change classroom evaluation semantics
- add new third-party dependencies
- replace the stage-marker system
- make the worker synchronous

Any functional changes should be handled in separate PRs after the service split is complete.

---

## Architectural Principle

Use a staged strangler refactor.

The first PRs should move code without changing behavior. The worker should continue to call the same logic, but the logic should live behind class-based services.

Target direction:

```text
run_distillation_pipeline()
  -> DistillationPipelineRunner.run()
     -> PipelineContextFactory.create()
     -> StageOrchestrator.run()
        -> ValidationStageService.run()
        -> ExtractionStageService.run()
        -> SynthesisStageService.run()
        -> BaselineEvaluationService.run()
        -> DistilledEvaluationService.run()
        -> StrategySelectionService.run()
        -> RefinementLoopService.run()
        -> RuleAblationService.run()
        -> PruningOptimizationService.run()
        -> ClassroomEvaluationService.run()
        -> QualityGateService.run()
        -> ArtifactExportService.run()
```

---

## Proposed Module Layout

Create a new package:

```text
src/rulekiln/workers/pipeline/
  __init__.py
  context.py
  runner.py
  stages.py
  stage_guard.py
  provider_bundle.py
  db_mappers.py
  evaluation_persistence.py
  services/
    __init__.py
    validation.py
    extraction.py
    rule_embedding.py
    clustering.py
    synthesis.py
    static_conflict_review.py
    pruning.py
    prompt_compilation.py
    baseline_evaluation.py
    distilled_evaluation.py
    strategy_selection.py
    failure_analysis.py
    refinement.py
    ablation.py
    pruning_optimization.py
    classroom_evaluation.py
    quality_gates.py
    mlflow_logging.py
    artifact_export.py
```

Keep `src/rulekiln/workers/distillation_worker.py` as the public compatibility entry point.

---

## Public Entry Points

Keep these public functions unchanged:

```python
async def run_distillation_pipeline(
    job_id: str,
    payload: DistillationRequest,
) -> None:
    ...

async def run_pipeline_phase(
    session: AsyncSession,
    job_id: str,
    payload: DistillationRequest,
    *,
    phase: PipelinePhase,
) -> None:
    ...
```

They should delegate to:

```python
runner = DistillationPipelineRunner(session=session, job_id=job_id, payload=payload)
await runner.run(phase=phase)
```

or equivalent.

---

## Core Data Structures

### PipelineContext

Add a `PipelineContext` dataclass or Pydantic model that holds derived state currently spread through `_run(...)`.

Suggested shape:

```python
@dataclass(slots=True)
class PipelineContext:
    session: AsyncSession
    job_id: str
    payload: DistillationRequest
    settings: AppSettings

    task: RuleKilnTask
    cases: list[RuleKilnCase]
    case_by_id: dict[str, RuleKilnCase]

    split_policy: SplitPolicy
    extraction_cases: list[RuleKilnCase]
    eval_cases: list[RuleKilnCase]
    eval_split: str
    golden_cases: list[RuleKilnCase]

    primary_metric: str
    dataset_id: str

    collector: ModelCallCollector

    provider_bundle: ProviderBundle

    evaluation_bootstrap_enabled: bool
    evaluation_bootstrap_iterations: int
    evaluation_bootstrap_seed_offset: int
```

### ProviderBundle

Move provider/client resolution out of `_run(...)`.

Suggested shape:

```python
@dataclass(slots=True)
class ProviderBundle:
    teacher_config: ProviderConfig
    student_config: ProviderConfig
    embedding_config: ProviderConfig
    judge_config: ProviderConfig

    extraction_teacher_config: ProviderConfig
    synthesis_teacher_config: ProviderConfig
    refinement_teacher_config: ProviderConfig

    teacher_chat: ChatModelClient
    student_chat: ChatModelClient
    judge_chat: ChatModelClient
    extraction_teacher_chat: ChatModelClient
    synthesis_teacher_chat: ChatModelClient
    refinement_teacher_chat: ChatModelClient
    embedding_client: EmbeddingClient

    eval_student_id: str
```

### PipelineArtifactsState

Optional but useful for aggregation and export stages:

```python
@dataclass(slots=True)
class PipelineArtifactsState:
    baseline_prompt_text: str | None = None
    baseline_prompt_source: str | None = None
    baseline_prompt_tokens: int = 0

    eval_map: dict[str, EvalResult] = field(default_factory=dict)
    gate_map: dict[str, QualityGateResult] = field(default_factory=dict)
    compiled_prompts: dict[str, str] = field(default_factory=dict)
    strategy_prompt_tokens: dict[str, int] = field(default_factory=dict)
    strategy_metadata: dict[str, dict[str, str | int | float | bool]] = field(default_factory=dict)

    comparison: StrategyComparison | None = None
    selected_strategy: str | None = None

    paired_comparison_artifacts: PairedComparisonArtifacts | None = None
    ablation_artifact: RuleAblationArtifact | None = None
    classroom_student_evals: dict[str, dict[str, EvalResult]] = field(default_factory=dict)
```

This can be introduced later. Early PRs can return explicit result objects from each service instead.

---

## Stage Guard

Centralize stage idempotency.

Current code repeatedly uses:

```python
if not await is_stage_complete(...):
    await _set_stage(...)
    ...
    await mark_stage_complete(...)
```

Add:

```python
class StageGuard:
    def __init__(self, session: AsyncSession, job_id: str) -> None: ...

    async def is_complete(
        self,
        stage: PipelineStage | str,
        *,
        strategy: str | None = None,
        artifact_type: str | None = None,
    ) -> bool: ...

    async def start(self, stage: PipelineStage) -> None: ...

    async def complete(
        self,
        stage: PipelineStage | str,
        *,
        strategy: str | None = None,
        artifact_type: str | None = None,
    ) -> None: ...

    async def run_once(
        self,
        stage: PipelineStage,
        *,
        strategy: str | None = None,
        artifact_type: str | None = None,
        fn: Callable[[], Awaitable[T]],
    ) -> T | None:
        ...
```

Do not over-abstract immediately. A small `StageGuard` with `is_complete`, `start`, and `complete` is enough for the first PR.

---

## Service Interfaces

Each service should be small, explicit, and testable.

### ValidationStageService

Responsibilities:

- insert normalized cases into DB
- preserve `VALIDATING_PROJECT` stage marker

```python
class ValidationStageService:
    async def run(self, ctx: PipelineContext) -> None:
        ...
```

### ExtractionStageService

Responsibilities:

- choose batch vs sequential extraction
- submit batch extraction
- collect batch extraction
- sequential extraction
- persist micro-rules
- mark per-case extraction artifacts
- preserve `EXTRACTING_RULES`, `EXTRACTING_RULES_BATCH_SUBMITTED`, and `EXTRACTING_RULES_BATCH_COLLECTED`

```python
class ExtractionStageService:
    async def run(self, ctx: PipelineContext, *, phase: PipelinePhase) -> None:
        ...
```

This service can internally delegate to:

```python
BatchExtractionService
SequentialExtractionService
```

but do not split further until tests are stable.

### RuleEmbeddingStageService

Responsibilities:

- load micro-rules
- build rule texts
- call embedding provider
- return `RuleEmbeddingResult`

```python
@dataclass(slots=True)
class RuleEmbeddingResult:
    db_micro_rules: list[MicroRule]
    rule_ids: list[str]
    rule_texts: list[str]
    embeddings: list[list[float]]
```

### ClusteringStageService

Responsibilities:

- run DBSCAN and HDBSCAN
- persist clusters
- return cluster results

```python
@dataclass(slots=True)
class ClusteringResult:
    dbscan_clusters: list[RuleClusterSchema]
    hdbscan_clusters: list[RuleClusterSchema]
```

### SynthesisStageService

Responsibilities:

- synthesize DBSCAN/HDBSCAN clusters
- persist synthesized rules
- preserve per-cluster idempotency markers

### StaticConflictReviewService

Responsibilities:

- run static rule conflict review
- persist conflict metadata
- preserve static review stage markers

Note: this service should not be confused with empirical conflict resolution.

### PruningStageService

Responsibilities:

- prune synthesized rules
- persist pruning metadata

### PromptCompilationService

Responsibilities:

- compile selected rules into prompts
- insert `PromptVersion`
- return compiled prompts and token counts

### BaselineEvaluationService

Responsibilities:

- compile scaffold baseline
- evaluate legacy baseline
- evaluate scaffold baseline
- evaluate few-shot baselines
- evaluate embedding centroid baseline
- evaluate embedding kNN baselines
- evaluate retrieval few-shot baseline
- persist case-level results and eval runs

This service is currently one of the largest inline sections. It should be moved early because it changes often.

Suggested return:

```python
@dataclass(slots=True)
class BaselineEvaluationResult:
    baseline_prompt_text: str
    baseline_prompt_source: str
    baseline_prompt_tokens: int
    baseline_eval: EvalResult | None
    baseline_scaffold_eval: EvalResult | None
    strategy_evals: dict[str, EvalResult]
    strategy_prompts: dict[str, str]
    strategy_prompt_tokens: dict[str, int]
    strategy_metadata: dict[str, dict[str, str | int | float | bool]]
```

### DistilledEvaluationService

Responsibilities:

- evaluate DBSCAN/HDBSCAN prompts
- persist case-level results
- persist eval runs
- support phase-specific execution: `evaluate_dbscan`, `evaluate_hdbscan`

### EvaluationPersistenceService

Responsibilities:

Move reusable functions here:

- `_load_eval_result_from_db`
- `_insert_eval_run_if_missing`
- `_persist_case_results`
- `_evaluate_prompt_strategy`

Suggested class:

```python
class EvaluationPersistenceService:
    async def evaluate_prompt_strategy(...) -> EvalResult: ...
    async def load_eval_result(...) -> EvalResult | None: ...
    async def persist_case_results(...) -> None: ...
```

### StrategySelectionService

Responsibilities:

- load/refresh eval results
- compute classification statistics
- build strategy comparison
- select strategy
- mark selected prompt version
- compute paired comparison artifacts
- compute regressed labels

### FailureAnalysisService

Responsibilities:

- run `analyze_failures`
- preserve `ANALYZING_FAILURES`

### RefinementLoopService

Responsibilities:

- enforce unsafe split guards
- run empirical closed-loop conflict resolution
- use golden cases as gate input
- emit refinement artifacts/provenance
- return best rules/eval if improved

Move `_run_refinement_loop(...)` into this service.

Suggested interface:

```python
class RefinementLoopService:
    async def run(
        self,
        ctx: PipelineContext,
        *,
        selected_strategy: str,
        selected_eval: EvalResult,
        failure_analysis_result: FailureAnalysisResult,
    ) -> RefinementLoopResult:
        ...
```

Suggested result:

```python
@dataclass(slots=True)
class RefinementLoopResult:
    refined_rules: list[SynthesizedRuleSchema] | None
    refined_eval: EvalResult | None
    stop_reason: str | None
    iterations_run: int
```

### RuleAblationService

Responsibilities:

- move `_run_rule_ablation(...)`
- preserve `ABLATING_RULES`

### PruningOptimizationService

Responsibilities:

- move `_run_two_pass_optimizer(...)`
- preserve `OPTIMIZING_PRUNING`

### ClassroomEvaluationService

Responsibilities:

- run final classroom evaluation
- evaluate every LLM prompt strategy across all configured students
- attach `all_student_results`
- persist/write per-student artifacts if needed
- preserve `EVALUATING_CLASSROOM`

### QualityGateService

Responsibilities:

- run gates for distilled and baseline variant strategies
- evaluate golden cases separately
- preserve `CHECKING_QUALITY_GATES`

### MlflowLoggingService

Responsibilities:

- create MLflow run
- log params
- log metrics
- set MLflow run ID
- preserve `LOGGING_ARTIFACTS`

### ArtifactExportService

Responsibilities:

- write all task/case/prompt/eval/report/artifact outputs
- write manifest
- write token/cost summary
- upload/log artifacts to MLflow
- preserve `EXPORTING_ARTIFACTS`

---

## Helper Extraction

Move helper functions out of the worker into focused modules.

### `db_mappers.py`

Move:

- `_to_db_case`
- `_db_case_id`
- `_payload_case_id_from_db_case_id`
- `_to_db_cluster`
- `_synth_to_db`
- `_db_synth_to_schema`
- `_eval_to_db`
- `_eval_case_record_to_schema`
- `_build_eval_case_upsert_payload`
- `_actual_output_to_raw_text`
- `_is_invalid_label`

### `strategy_utils.py` or `services/strategy_selection.py`

Move:

- `_strategy_family`
- `_best_strategy_for_family`
- `_primary_score_for_metric`
- `_delta_vs_baseline`
- `_resolve_primary_metric`

### `classification_metrics.py` or existing statistics module

Move:

- `_prediction_label_from_case_result`
- `_case_input_text`
- `_aligned_classification_columns`
- `_refresh_eval_classification_statistics`
- `_build_runtime_paired_comparison`

### `paths.py`

Move:

- `_artifact_root`
- marker helpers:
  - `_extracting_case_marker`
  - `_synthesis_cluster_marker`
  - `_conflict_review_rule_marker`

### `seeding.py`

Move:

- `_evaluation_bootstrap_seed`

---

## Orchestrator Design

Create `DistillationPipelineRunner`.

Suggested interface:

```python
class DistillationPipelineRunner:
    def __init__(
        self,
        *,
        session: AsyncSession,
        job_id: str,
        payload: DistillationRequest,
    ) -> None:
        ...

    async def run(self, *, phase: PipelinePhase = "full") -> None:
        ...
```

The runner should:

1. create context
2. compute phase flags
3. call services in current order
4. preserve early returns for phase-specific execution
5. persist model call collector records before returning from phase-specific paths
6. catch nothing; error handling stays in `run_distillation_pipeline(...)`

The public entry point keeps the current exception behavior.

---

## Phase Flag Handling

Current phase flags:

```python
run_validate = phase in {"full", "validate_project", "compile_prompts"}
run_compile_phase = phase in {
    "full",
    "compile_prompts",
    "extraction_batch_submit",
    "extraction_batch_collect",
}
run_post_extraction = phase in {
    "full",
    "compile_prompts",
    "extraction_batch_collect",
}
run_baseline_eval = phase in {"full", "evaluate_baseline"}
run_aggregate = phase in {"full", "aggregate_evaluation_report"}
```

Move these to:

```python
@dataclass(frozen=True, slots=True)
class PhasePlan:
    run_validate: bool
    run_compile_phase: bool
    run_post_extraction: bool
    run_baseline_eval: bool
    run_aggregate: bool

    @classmethod
    def from_phase(cls, phase: PipelinePhase) -> PhasePlan:
        ...
```

Acceptance: exact same behavior as current phase handling.

---

## Migration Plan

### PR 1: Introduce Package, Context, ProviderBundle, StageGuard

- Add `workers/pipeline/` package.
- Add `PipelineContext`.
- Add `ProviderBundle`.
- Add `PipelineContextFactory`.
- Add `StageGuard`.
- Keep `_run(...)` mostly intact.
- Replace inline provider resolution with `PipelineContextFactory` or `ProviderBundleFactory`.

Acceptance:

- No behavior change.
- Existing tests pass.
- `distillation_worker.py` smaller but still owns most stages.

### PR 2: Move DB Mappers and Evaluation Persistence

- Move helper conversion functions to `db_mappers.py`.
- Move eval persistence helpers to `evaluation_persistence.py`.
- Update imports.

Acceptance:

- No behavior change.
- Unit tests added for mapper functions if not already present.

### PR 3: Extract Validation, Extraction, Embedding, Clustering, Synthesis

- Move early compile-phase stages into services:
  - validation
  - extraction
  - embedding
  - clustering
  - synthesis
- Preserve batch extraction behavior and stage markers.

Acceptance:

- `compile_prompts`, `extraction_batch_submit`, and `extraction_batch_collect` phases still work.
- Existing extraction tests pass.

### PR 4: Extract Static Conflict Review, Pruning, Prompt Compilation

- Move static conflict review service.
- Move pruning service.
- Move prompt compilation service.

Acceptance:

- Prompt compilation artifacts and prompt versions are unchanged.
- Stage markers unchanged.

### PR 5: Extract Baseline and Distilled Evaluation

- Move baseline evaluation into `BaselineEvaluationService`.
- Move distilled evaluation into `DistilledEvaluationService`.
- Move shared evaluation helpers into `EvaluationPersistenceService`.

Acceptance:

- Baseline progress and results are unchanged.
- Retrieval few-shot resume behavior is unchanged.
- Case-level persistence is unchanged.

### PR 6: Extract Aggregation Services

Move:

- strategy selection
- quality gates
- failure analysis
- refinement loop
- ablation
- pruning optimization

Acceptance:

- Strategy selection unchanged.
- Quality gate behavior unchanged.
- Refinement artifacts unchanged.
- Ablation artifacts unchanged.
- Pruning comparison unchanged.

### PR 7: Extract Classroom, MLflow, Artifact Export

Move:

- final classroom evaluation
- MLflow logging
- artifact export

Acceptance:

- Manifest contents unchanged.
- Classroom artifacts unchanged.
- MLflow metrics/params unchanged.
- Final job completion unchanged.

### PR 8: Delete Old Helpers and Tighten Worker

- Remove dead functions from `distillation_worker.py`.
- Leave only:
  - constants if still needed
  - `PipelineStage`
  - `PipelinePhase`
  - public entry points
  - error handling wrapper
  - delegation to `DistillationPipelineRunner`

Acceptance:

- `distillation_worker.py` is under roughly 250 lines.
- No `# noqa: C901` needed on worker entry point.
- Full test suite passes.

---

## Behavior Preservation Requirements

For every PR:

- Do not change stage names.
- Do not change strategy names.
- Do not change artifact paths.
- Do not change DB schemas.
- Do not change `PipelinePhase` values.
- Do not change public worker function signatures.
- Do not change idempotency behavior.
- Do not change provider resolution semantics.
- Do not change tracking context semantics.
- Do not change classroom semantics.
- Do not change quality gate semantics.
- Do not change refinement loop semantics unless explicitly covered by another spec.

---

## Testing Strategy

### Unit Tests

Add tests for:

- `PhasePlan.from_phase`
- `PipelineContextFactory`
- `ProviderBundleFactory`
- `StageGuard`
- DB mappers
- EvaluationPersistenceService
- each service with mocked dependencies where practical

### Integration Tests

Existing integration tests should continue to validate:

- full pipeline run
- compile-prompts phase
- batch extraction submit/collect
- baseline evaluation
- DBSCAN evaluation
- HDBSCAN evaluation
- aggregate report
- refinement loop
- classroom final evaluation
- artifact export

### Regression Tests

Add a golden behavior snapshot test if feasible:

1. Run a small fake-provider job through the old path before refactor.
2. Capture key artifacts and stage markers.
3. Run through refactored path.
4. Assert same stage completions and equivalent output schema.

If direct old/new comparison is not feasible, assert stable output fields and counts.

---

## Acceptance Criteria

This refactor is complete when:

- `distillation_worker.py` is a thin entry point.
- Each major pipeline stage is implemented in a service class.
- All public entry points are unchanged.
- All stage names and markers are unchanged.
- Existing phase-specific runs still work.
- Resume/idempotency behavior is preserved.
- Artifacts remain in the same paths with the same schema versions.
- Full test suite passes.
- Type checking passes.
- Ruff passes.
- No major service has excessive cyclomatic complexity.
- New tests cover the extracted services.

Required commands:

```bash
uv run ruff format src/ tests/
uv run ruff check src/ tests/ --fix
uv run pyright
DATABASE_URL="sqlite+aiosqlite://" MLFLOW_TRACKING_URI="file:///tmp/mlflow-ci" uv run pytest -m "not external" --tb=short -q
```

---

## Suggested Claude Code Instruction

Use this implementation instruction:

```text
Refactor the distillation worker into class-based pipeline services without changing behavior.

Start by adding a `src/rulekiln/workers/pipeline/` package with PipelineContext,
ProviderBundle, StageGuard, and DistillationPipelineRunner. Keep the existing public
functions in `distillation_worker.py`.

Proceed in small PR-sized steps. Do not rewrite the algorithm. Preserve all stage markers,
strategy names, artifact paths, database schemas, and public function signatures.

After each extraction, run ruff, pyright, and the non-external pytest suite.
```

---

## Risks and Mitigations

### Risk: Resume semantics change

Mitigation:

- centralize stage marker logic in `StageGuard`
- add tests for resume after partial completion
- avoid changing marker strings

### Risk: Artifact paths change

Mitigation:

- do not move writer functions in this refactor
- keep export service as a thin caller around existing writer functions

### Risk: Provider tracking changes

Mitigation:

- preserve `set_tracking_context` and `update_tracking_context` call locations initially
- only later centralize tracking once behavior is stable

### Risk: Giant PR

Mitigation:

- follow PR sequence above
- each PR should move one coherent group of code
- avoid broad formatting-only changes mixed with logic moves

### Risk: Type complexity increases

Mitigation:

- use dataclasses with explicit fields
- avoid `Any`
- add small result types for service outputs
- prefer explicit return objects over large mutable dictionaries

---

## Final Target

After the refactor, the worker should read like an orchestration plan, not an implementation dump:

```python
async def run(self, *, phase: PipelinePhase = "full") -> None:
    ctx = await self.context_factory.create()

    await self.validation.run(ctx)

    if phase == "validate_project":
        return

    compile_result = await self.compile_pipeline.run(ctx, phase=phase)

    if phase in COMPILE_ONLY_PHASES:
        await self.model_calls.persist(ctx)
        return

    baseline_result = await self.baselines.run(ctx)
    distilled_result = await self.distilled.run(ctx, baseline_result)
    aggregate_result = await self.aggregation.run(ctx, baseline_result, distilled_result)

    await self.artifacts.export(ctx, aggregate_result)
    await self.jobs.complete(ctx)
```

The implementation details should live in services that can be tested and edited independently.
