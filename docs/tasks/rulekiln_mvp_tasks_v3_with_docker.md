---

description: "MVP implementation task list for RuleKiln"

---

# Tasks: RuleKiln MVP

> Status: Canonical MVP task list.

**Input**: Design documents from docs/specs/rulekiln_mvp_plan_spec_v2.md and docs/tasks/task_template.md

**Prerequisites**: RuleKiln MVP plan spec (required), task template (required)

**Tests**: Included for API contracts, deterministic compilation, pipeline orchestration, and artifact/MLflow verification.

**Organization**: Tasks are grouped by user story so each story can be implemented and validated independently.

## Format: [ID] [P?] [Story] Description

- [P]: Can run in parallel (different files, no dependencies)
- [Story]: User story label (US1, US2, US3)
- Every task includes exact file paths

## Path Conventions

- Single project layout at repository root
- Runtime code under src/rulekiln/
- Tests under tests/
- Migrations under migrations/

---

## MVP Clarification Decisions

These decisions resolve known ambiguities in the initial MVP task list and should be treated as implementation constraints.

### C001 Canonical Request Contract

The MVP request contract is:

```json
{
  "task": {},
  "cases": [],
  "teacher": {
    "provider_profile": "bedrock-primary",
    "model": "..."
  },
  "student": {
    "provider_profile": "bedrock-primary",
    "model": "..."
  },
  "embedding": {
    "provider_profile": "openai-compatible",
    "model": "..."
  },
  "judge": null,
  "baseline_prompt": null,
  "metric": null
}
```

Rules:

- `task`, `cases`, `teacher`, `student`, and `embedding` are required for MVP.
- `judge`, `baseline_prompt`, and `metric` are optional.
- Strict mode means the API accepts only the canonical envelope above.
- Strict mode rejects legacy top-level fields such as `task_name`, `task_description`, `labels`, and `examples`.
- Provider defaults may exist in `task.provider_model_defaults`, but MVP job submission still requires explicit `teacher`, `student`, and `embedding` routes to avoid hidden routing behavior.

### C002 MVP Provider Scope

The MVP does not require all provider adapters to be fully implemented.

Required MVP providers:

- `fake` provider for offline tests and local deterministic development
- `openai` chat and embedding provider
- `openai_compatible` chat and embedding provider
- `bedrock` chat provider
- `bedrock` embedding provider if available in the target AWS account; otherwise the adapter may return a clear `ProviderNotConfiguredError`

Deferred providers with stubs only:

- `anthropic`
- `vertex_gemini`
- `azure_openai`
- `custom`

A stub provider must fail fast with a typed `ProviderNotImplementedError`. It must not silently fall back to another provider.

### C003 Worker Stage Enum, Transitions, and Resume Semantics

The MVP stage enum is:

```text
created
validating_project
extracting_rules
embedding_rules
clustering_rules
synthesizing_rules
compiling_prompts
evaluating_baseline
evaluating_distilled
selecting_strategy
analyzing_failures
checking_quality_gates
logging_artifacts
exporting_artifacts
completed
failed
```

Allowed transition model:

```text
created -> validating_project
validating_project -> extracting_rules
extracting_rules -> embedding_rules
embedding_rules -> clustering_rules
clustering_rules -> synthesizing_rules
synthesizing_rules -> compiling_prompts
compiling_prompts -> evaluating_baseline
evaluating_baseline -> evaluating_distilled
evaluating_distilled -> selecting_strategy
selecting_strategy -> analyzing_failures
analyzing_failures -> checking_quality_gates
checking_quality_gates -> logging_artifacts
logging_artifacts -> exporting_artifacts
exporting_artifacts -> completed
any non-terminal stage -> failed
failed -> same failed stage on explicit retry
```

Resume semantics:

- Every stage writes a durable stage-completion marker.
- A resumed job starts at the first incomplete stage.
- Stage writes must be idempotent using `(job_id, stage, strategy, artifact_type)` or equivalent uniqueness.
- Large outputs should be written as artifacts and referenced from database rows.
- LLM calls should use cache keys based on case hash, prompt version, model route, and output schema version.
- `completed` and `failed` are terminal unless an explicit retry endpoint or admin command is added later.

### C004 Test Strategy and External API Policy

CI must run fully offline by default.

Required test modes:

- unit tests: fake providers only
- contract tests: fake providers only
- integration tests: fake providers plus temporary/local database
- artifact and MLflow tests: local file-backed MLflow tracking URI
- external provider smoke tests: opt-in only, excluded from default CI

No default CI test may require:

- OpenAI API access
- AWS credentials
- Bedrock model access
- internet access
- paid model calls

External smoke tests should be marked separately, for example:

```text
pytest -m external
```

### C005 Evaluation Contract by Task Mode

Primary metric selection rules:

- `classification`: default primary metric is `macro_f1`
- `routing`: default primary metric is `macro_f1`
- `tool_use`: default primary metric is `weighted_case_score`
- `extraction`: default primary metric is `weighted_case_score`
- `summarization`: default primary metric is `weighted_case_score`
- `rubric_review`: default primary metric is `weighted_case_score`
- `freeform_generation`: default primary metric is `weighted_case_score`
- `agent_behavior`: default primary metric is `weighted_case_score`

Malformed output handling:

- If the output is not parseable according to the task output schema, the case receives score `0.0`.
- Malformed outputs increment `malformed_output_rate`.
- Malformed outputs fail all path-based assertions for that case.

Weighted case score:

```text
case_score = weighted average of assertion scores and rubric scores
weighted_case_score = weighted average of case_score across cases using case.weight
```

Tie-break logic for strategy selection:

1. Prefer the strategy that passes all quality gates.
2. Prefer higher primary metric.
3. If tied within `0.005`, prefer lower golden failures.
4. If still tied, prefer lower malformed output rate.
5. If still tied, prefer lower prompt token count.
6. If still tied, prefer HDBSCAN because it is the intended production default.
7. If HDBSCAN fails gates and DBSCAN passes, select DBSCAN.

### C006 Default Quality Gate Thresholds

Default thresholds live in application settings and may be overridden by `task.quality_gates`.

Precedence:

```text
task.quality_gates > AppSettings.default_quality_gates > hardcoded safe defaults
```

MVP safe defaults:

```yaml
min_metric_delta: 0.0
max_regression_rate: 0.10
max_golden_failures: 0
max_malformed_output_rate: 0.01
max_prompt_tokens: 8000
require_human_approval: true
```

For initial development, `min_metric_delta` defaults to `0.0` so that the pipeline can select a non-regressing candidate without requiring immediate measurable lift on small fake datasets. Production task configs may set `min_metric_delta: 0.03`.

### C007 Artifact Export Location and Naming

Artifacts must be scoped by job ID to avoid collisions.

Default artifact root:

```text
.rulekiln/runs/{job_id}/
```

Required layout:

```text
.rulekiln/runs/{job_id}/
  task.yaml
  cases.normalized.jsonl
  outputs/
    distilled_prompt_dbscan.md
    distilled_prompt_hdbscan.md
    selected_distilled_prompt.md
    rules_dbscan.jsonl
    rules_hdbscan.jsonl
    eval_report.json
    strategy_comparison.json
    failures_fixed.jsonl
    failures_broken.jsonl
    failures_unchanged.jsonl
  exports/
    promptfoo.yaml
    mlflow_run_id.txt
  metadata/
    settings_snapshot.json
    manifest.json
```

All artifact-writing code must accept an explicit `artifact_root` from settings.

### C008 MLflow Version and Prompt Registry Policy

MVP should pin MLflow to a version that supports the APIs used by the implementation.

Default dependency policy:

```text
mlflow>=3.5,<4
```

Prompt registry usage is optional in MVP unless confirmed working in the selected MLflow version and deployment target.

Required MVP MLflow behavior:

- create one MLflow run per distillation job
- log params
- log metrics
- log artifacts
- save the MLflow run ID back to the job record

Optional MVP behavior:

- register the selected prompt in MLflow Prompt Registry
- save `mlflow_prompt_uri` when registration succeeds

If prompt registration is unavailable, RuleKiln must still complete the job and log `selected_distilled_prompt.md` as an artifact.

### C009 Vector Storage and pgvector Decision

pgvector is optional for MVP.

MVP behavior:

- embeddings may be computed and kept in memory during the job
- embeddings may be persisted as JSON artifacts
- database schema may include nullable embedding/vector columns only if pgvector is enabled
- migrations must not require pgvector unless `ENABLE_PGVECTOR=true`

Implementation rule:

- Feature-flag pgvector support.
- Default local/dev setup should work without pgvector.
- Do not block MVP on vector database setup.

### C010 Parallelization Clarification

The `[P]` marker means the task can be implemented in parallel at the file/module level once its phase prerequisites are met. It does not mean the runtime workflow stages execute in parallel.

Runtime dependency order remains sequential for the core job:

```text
US1 endpoint and worker skeleton
  -> US2 pipeline orchestration
  -> US3 artifact and MLflow surfacing
```

Within US2, modules can be developed in parallel using fake providers and fixtures:

- schemas
- provider adapters
- clustering
- prompt compiler
- evaluator
- quality gates
- failure analysis

The full worker orchestration depends on those modules being available.



### C011 Docker Local Development Policy

RuleKiln should support Docker for local infrastructure and onboarding, but Docker is not the only supported development path.

Supported local workflows:

```text
1. Native Python workflow:
   uvicorn + local/remote Postgres + local file-backed MLflow

2. Docker Compose workflow:
   postgres + mlflow + rulekiln-api
```

Docker is developer infrastructure, not the production architecture for MVP.

Required Docker assets:

```text
Dockerfile
docker-compose.yml
.dockerignore
scripts/dev-up.sh
scripts/dev-down.sh
```

Default Docker Compose services:

```text
postgres
mlflow
api
```

Optional future services:

```text
pgvector postgres image override
minio for artifact storage
worker service if DBOS/Celery/Temporal is introduced
```

Rules:

- Local development must still work without Docker.
- Docker Compose must use the same `.env` contract as native development.
- The default Compose stack should not require external provider credentials because fake providers must work offline.
- pgvector must remain optional and feature-flagged.
- The app container should mount the project directory for development reloads.
- Runtime artifacts should be written to a job-scoped `.rulekiln/runs/{job_id}/` path.

Recommended Compose defaults:

```text
DATABASE_URL=postgresql+asyncpg://rulekiln:rulekiln@postgres:5432/rulekiln
MLFLOW_TRACKING_URI=http://mlflow:5000
ARTIFACT_ROOT=.rulekiln/runs
ENABLE_PGVECTOR=false
```

Host-machine `.env.example` may also include localhost equivalents:

```text
DATABASE_URL=postgresql+asyncpg://rulekiln:rulekiln@localhost:5432/rulekiln
MLFLOW_TRACKING_URI=http://localhost:5000
ARTIFACT_ROOT=.rulekiln/runs
ENABLE_PGVECTOR=false
```


---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Initialize project layout and baseline tooling for the MVP.

- [ ] T001 Create package structure in src/rulekiln/ with subpackages api/, agents/, artifacts/, db/, pipeline/, providers/, integrations/, observability/, schemas/, workers/
- [ ] T002 Update pyproject.toml with MVP dependencies (fastapi, uvicorn, pydantic, pydantic-settings, pydantic-ai, sqlalchemy, asyncpg, alembic, mlflow>=3.5,<4, scikit-learn, hdbscan, structlog, httpx, pytest)
- [ ] T003 [P] Add lint and type-check configuration in pyproject.toml for ruff and pyright
- [ ] T004 [P] Create environment template in .env.example for DATABASE_URL, MLFLOW_TRACKING_URI, ARTIFACT_ROOT, ENABLE_PGVECTOR, provider API key env vars, default quality gates, and provider profile examples
- [ ] T005 Create application entrypoint in src/rulekiln/api/app.py and update main.py to launch the FastAPI app
- [ ] T005A [P] Add Dockerfile for the RuleKiln FastAPI app at Dockerfile
- [ ] T005B [P] Add docker-compose.yml for postgres, mlflow, and rulekiln-api local development
- [ ] T005C [P] Add .dockerignore for Python caches, virtualenvs, test artifacts, local run artifacts, and secret files
- [ ] T005D [P] Add scripts/dev-up.sh and scripts/dev-down.sh for local Docker Compose workflow
- [ ] T005E [P] Document Docker and non-Docker local development commands in docs/dev/docker.md

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build core platform pieces required by all user stories.

**CRITICAL**: No user story work starts until this phase is complete.

- [ ] T006 Implement centralized settings, default quality gates, artifact root, pgvector flag, and provider profile models in src/rulekiln/config/settings.py
- [ ] T007 [P] Implement provider contracts (ProviderConfig, ChatModelClient, EmbeddingClient) in src/rulekiln/providers/contracts.py
- [ ] T008 [P] Implement provider profile normalization and route resolution in src/rulekiln/providers/resolver.py
- [ ] T009 Implement structured logging setup with structlog in src/rulekiln/observability/logging.py
- [ ] T010 [P] Implement database engine/session wiring in src/rulekiln/db/session.py
- [ ] T011 Define SQLAlchemy models for distillation_jobs, cases, micro_rules, rule_clusters, synthesized_rules, prompt_versions, eval_runs, and stage completion markers in src/rulekiln/db/models.py
- [ ] T012 Create initial migration for MVP tables in migrations/versions/0001_mvp_schema.py; pgvector columns/extensions must be optional behind ENABLE_PGVECTOR
- [ ] T013 [P] Implement repository interfaces and persistence helpers in src/rulekiln/db/repositories/jobs.py and src/rulekiln/db/repositories/artifacts.py
- [ ] T014 Implement FastAPI lifespan startup validation (settings load and provider profile validation) in src/rulekiln/api/lifespan.py
- [ ] T015 Implement API error schema and global exception handlers in src/rulekiln/api/errors.py

**Checkpoint**: Foundation is ready; user stories can proceed.

---

## Phase 3: User Story 1 - Submit and Track Distillation Jobs (Priority: P1) MVP

**Goal**: Accept strict task and case payloads, queue jobs, and expose job status progress.

**Independent Test**: A valid POST /distillation-jobs request returns queued job_id; GET /distillation-jobs/{id} returns status and stage updates.

### Tests for User Story 1

- [ ] T016 [P] [US1] Add contract tests for POST /distillation-jobs request and response shape in tests/contract/test_create_distillation_job_contract.py
- [ ] T017 [P] [US1] Add contract tests for GET /distillation-jobs/{job_id} response shape in tests/contract/test_get_distillation_job_contract.py
- [ ] T018 [P] [US1] Add integration test for job creation plus initial state persistence in tests/integration/test_job_creation_persistence.py

### Implementation for User Story 1

- [ ] T019 [P] [US1] Implement task and case schemas (RuleKilnTask, RuleKilnCase, evaluation models) in src/rulekiln/schemas/task_case.py
- [ ] T020 [P] [US1] Implement request and job status schemas (DistillationRequest, JobStatusResponse) in src/rulekiln/schemas/job.py
- [ ] T021 [US1] Implement strict payload validation guard (reject legacy top-level fields) in src/rulekiln/api/validators/request_shape.py
- [ ] T022 [US1] Implement POST /distillation-jobs endpoint in src/rulekiln/api/routes/distillation_jobs.py
- [ ] T023 [US1] Implement GET /distillation-jobs/{job_id} endpoint in src/rulekiln/api/routes/distillation_jobs.py
- [ ] T024 [US1] Implement background worker entrypoint skeleton run_distillation_job(job_id) with canonical stage enum, allowed transitions, completion markers, and stage updates in src/rulekiln/workers/distillation_worker.py
- [ ] T025 [US1] Wire distillation job routes and lifespan into app factory in src/rulekiln/api/app.py

**Checkpoint**: Job submission and status tracking work independently.

---

## Phase 4: User Story 2 - Run Distillation Pipeline and Select Winning Strategy (Priority: P1)

**Goal**: Execute extraction, clustering (DBSCAN and HDBSCAN), synthesis, deterministic prompt compilation, evaluation, quality gates, and strategy selection.

**Independent Test**: A queued job runs end-to-end and produces baseline, DBSCAN, and HDBSCAN eval outputs with a selected strategy.

### Tests for User Story 2

- [ ] T026 [P] [US2] Add unit tests for deterministic prompt compiler and stable hash output in tests/unit/test_prompt_compiler_determinism.py
- [ ] T027 [P] [US2] Add unit tests for provider route resolution and role capability checks in tests/unit/test_provider_resolution.py
- [ ] T028 [P] [US2] Add integration test for orchestration flow and selected strategy in tests/integration/test_distillation_pipeline_selection.py
- [ ] T029 [P] [US2] Add integration test for golden-case regression gate behavior in tests/integration/test_golden_case_quality_gate.py
- [ ] T029A [P] [US2] Add unit tests for task-mode scoring, malformed-output penalties, weighted case score, and strategy tie-breaks in tests/unit/test_evaluation_contract.py
- [ ] T029B [P] [US2] Add unit tests for worker stage enum, allowed transitions, and resume semantics in tests/unit/test_worker_stage_model.py
- [ ] T029C [P] [US2] Add offline end-to-end pipeline test using fake providers in tests/integration/test_offline_e2e_fake_providers.py

### Implementation for User Story 2

- [ ] T030 [P] [US2] Implement pipeline domain schemas (MicroRule, RuleCluster, SynthesizedRule, EvalResult, QualityGateResult) in src/rulekiln/schemas/pipeline.py
- [ ] T031 [US2] Implement Pydantic AI rule extraction agent wrapper in src/rulekiln/agents/rule_extraction.py
- [ ] T032 [US2] Implement Pydantic AI rule synthesis agent wrapper in src/rulekiln/agents/rule_synthesis.py
- [ ] T033 [US2] Implement MVP chat provider adapters (fake, openai, openai_compatible, bedrock) behind ChatModelClient in src/rulekiln/providers/chat/; add typed not-implemented stubs for anthropic, vertex_gemini, azure_openai, and custom
- [ ] T034 [US2] Implement MVP embedding provider adapters (fake, openai, openai_compatible, optional bedrock) behind EmbeddingClient in src/rulekiln/providers/embedding/; add typed not-implemented stubs for deferred providers
- [ ] T035 [US2] Implement clustering service for DBSCAN and HDBSCAN in src/rulekiln/pipeline/clustering.py
- [ ] T036 [US2] Implement deterministic prompt compiler in src/rulekiln/pipeline/prompt_compiler.py
- [ ] T037 [US2] Implement evaluator for baseline and distilled prompts, including task-mode primary metric rules, malformed-output penalties, weighted case score, and assertion/rubric scoring in src/rulekiln/pipeline/evaluator.py
- [ ] T038 [US2] Implement strategy selection, tie-break logic, and regression analysis in src/rulekiln/pipeline/strategy_selection.py
- [ ] T039 [US2] Implement quality gates (metric delta, malformed output, golden failures, regression rate, token budget) with task-over-settings threshold precedence in src/rulekiln/pipeline/quality_gates.py
- [ ] T040 [US2] Implement failure analysis outputs (fixed, broken, unchanged) in src/rulekiln/pipeline/failure_analysis.py
- [ ] T041 [US2] Implement full worker stage orchestration, allowed transitions, resume-from-first-incomplete-stage behavior, and resumable/idempotent stage writes in src/rulekiln/workers/distillation_worker.py
- [ ] T042 [US2] Persist pipeline artifacts and run outputs through repositories in src/rulekiln/db/repositories/artifacts.py

**Checkpoint**: Core distillation engine works and selects a winning strategy.

---

## Phase 5: User Story 3 - Surface Artifacts, Reports, and MLflow Audit Trail (Priority: P2)

**Goal**: Expose prompt/rule/eval outputs, export canonical artifacts, and log full run details to MLflow.

**Independent Test**: Completed jobs expose prompt, rules, and eval report endpoints; artifacts are exported; MLflow run includes params, metrics, and artifacts.

### Tests for User Story 3

- [ ] T043 [P] [US3] Add contract tests for GET /distillation-jobs/{job_id}/prompt in tests/contract/test_get_prompt_contract.py
- [ ] T044 [P] [US3] Add contract tests for GET /distillation-jobs/{job_id}/rules and /eval-report in tests/contract/test_get_rules_eval_contract.py
- [ ] T045 [P] [US3] Add integration test for local file-backed MLflow run logging and artifact upload in tests/integration/test_mlflow_artifact_logging.py

### Implementation for User Story 3

- [ ] T046 [P] [US3] Implement per-job artifact packager for canonical outputs under .rulekiln/runs/{job_id}/ in src/rulekiln/artifacts/writer.py
- [ ] T047 [P] [US3] Implement settings snapshot export with secret redaction in src/rulekiln/artifacts/settings_snapshot.py
- [ ] T048 [US3] Implement MLflow integration for params, metrics, artifacts, run ID persistence, and optional prompt URI registration fallback in src/rulekiln/integrations/mlflow_tracker.py
- [ ] T049 [US3] Implement GET /distillation-jobs/{job_id}/prompt endpoint in src/rulekiln/api/routes/distillation_outputs.py
- [ ] T050 [US3] Implement GET /distillation-jobs/{job_id}/rules endpoint in src/rulekiln/api/routes/distillation_outputs.py
- [ ] T051 [US3] Implement GET /distillation-jobs/{job_id}/eval-report endpoint in src/rulekiln/api/routes/distillation_outputs.py
- [ ] T052 [US3] Implement strategy comparison payload and selected prompt retrieval in src/rulekiln/db/repositories/artifacts.py
- [ ] T053 [US3] Export selected_distilled_prompt.md, rules.jsonl, eval_report.json, cases.normalized.jsonl, promptfoo.yaml, mlflow_run_id.txt to .rulekiln/runs/{job_id}/outputs/ and .rulekiln/runs/{job_id}/exports/
- [ ] T054 [US3] Wire output routes into app and ensure OpenAPI docs include endpoint schemas in src/rulekiln/api/app.py

**Checkpoint**: Auditable outputs and MLflow tracking are available to users.

---

## Phase 6: Polish and Cross-Cutting Concerns

**Purpose**: Final hardening across all stories.

- [ ] T055 [P] Update README.md with native setup, Docker Compose setup, environment, provider profile examples, and run instructions
- [ ] T056 [P] Add observability events for model calls, stage timings, retries, token usage, and cost in src/rulekiln/observability/events.py
- [ ] T057 Add security masking helpers for secrets and URLs in logs in src/rulekiln/observability/security.py
- [ ] T058 Add CI checks for lint, type check, and offline tests in .github/workflows/ci.yml; external provider smoke tests must be opt-in and excluded from default CI
- [ ] T058A [P] Add CI validation that Dockerfile builds successfully without provider credentials
- [ ] T058B [P] Add docker-compose config validation in CI without starting paid/external provider services
- [ ] T059 Run end-to-end MVP validation against acceptance criteria and document results in docs/mvp_validation.md

---

## Dependencies and Execution Order

### Phase Dependencies

- Setup (Phase 1): No dependencies
- Docker tasks T005A-T005E may run in parallel with app setup but depend on the dependency list and environment contract being stable
- Foundational (Phase 2): Depends on Phase 1 and blocks user stories
- User Stories (Phases 3, 4, 5): Depend on Phase 2
- Polish (Phase 6): Depends on completion of required user stories

### User Story Dependencies

- US1 (P1): Starts after Foundational phase
- US2 (P1): Starts after Foundational phase; depends on US1 endpoint and worker wiring
- US3 (P2): Starts after US2 produces persisted outputs and run metadata

### Within Each User Story

- Write tests first, confirm failures
- Implement schemas and models before services
- Implement services before API routes
- Finish persistence wiring before endpoint response contracts

### Parallel Opportunities

- Phase 1 and 2 tasks marked [P] can run concurrently
- US1 schema and contract tests can run in parallel
- US2 adapters, clustering, compiler, evaluator modules can be split across developers
- US3 endpoint contracts and artifact writer tasks can run in parallel
- `[P]` indicates file/module-level implementation parallelism, not runtime workflow parallelism.
- Runtime stage execution remains ordered according to the canonical worker stage enum.


---

## Parallel Example: User Story 2

Execute in parallel after foundational phase:

- T026 test_prompt_compiler_determinism.py
- T027 test_provider_resolution.py
- T030 src/rulekiln/schemas/pipeline.py
- T033 src/rulekiln/providers/chat/
- T034 src/rulekiln/providers/embedding/

Then sequence dependent tasks:

- T035 clustering.py -> T036 prompt_compiler.py -> T037 evaluator.py -> T038 strategy_selection.py -> T041 distillation_worker.py

---

## Implementation Strategy

### MVP First Path

1. Complete Phase 1 and Phase 2
2. Deliver US1 (Phase 3) and validate job lifecycle
3. Deliver US2 (Phase 4) and validate strategy selection plus quality gates
4. Deliver US3 (Phase 5) and validate artifact and MLflow audit trail
5. Complete Phase 6 hardening and acceptance check

### Incremental Delivery

1. Ship API job submission and status tracking (US1)
2. Ship core distillation pipeline (US2)
3. Ship reporting and audit exports (US3)
4. Finalize reliability and operations polish

---

## Notes

- [P] tasks should target separate files to avoid merge conflicts
- Keep stage writes idempotent to support safe retries
- Do not store secrets in logs, artifacts, or MLflow params
- Ensure prompt compilation remains deterministic and pure (no LLM calls)
- Docker Compose is for local development and onboarding; native Python execution remains supported