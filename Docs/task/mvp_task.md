---

description: "MVP implementation task list for RuleKiln"

---

# Tasks: RuleKiln MVP

**Input**: Design documents from Docs/plans/rulekiln_mvp_plan_spec_v2.md and Docs/task/task_template.md

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

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Initialize project layout and baseline tooling for the MVP.

- [ ] T001 Create package structure in src/rulekiln/ with subpackages api/, agents/, artifacts/, db/, pipeline/, providers/, integrations/, observability/, schemas/, workers/
- [ ] T002 Update pyproject.toml with MVP dependencies (fastapi, uvicorn, pydantic, pydantic-settings, pydantic-ai, sqlalchemy, asyncpg, alembic, mlflow, scikit-learn, hdbscan, structlog, httpx, pytest)
- [ ] T003 [P] Add lint and type-check configuration in pyproject.toml for ruff and pyright
- [ ] T004 [P] Create environment template in .env.example for DATABASE_URL, MLFLOW_TRACKING_URI, provider API key env vars, and provider profile examples
- [ ] T005 Create application entrypoint in src/rulekiln/api/app.py and update main.py to launch the FastAPI app

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build core platform pieces required by all user stories.

**CRITICAL**: No user story work starts until this phase is complete.

- [ ] T006 Implement centralized settings and provider profile models in src/rulekiln/config/settings.py
- [ ] T007 [P] Implement provider contracts (ProviderConfig, ChatModelClient, EmbeddingClient) in src/rulekiln/providers/contracts.py
- [ ] T008 [P] Implement provider profile normalization and route resolution in src/rulekiln/providers/resolver.py
- [ ] T009 Implement structured logging setup with structlog in src/rulekiln/observability/logging.py
- [ ] T010 [P] Implement database engine/session wiring in src/rulekiln/db/session.py
- [ ] T011 Define SQLAlchemy models for distillation_jobs, cases, micro_rules, rule_clusters, synthesized_rules, prompt_versions, eval_runs in src/rulekiln/db/models.py
- [ ] T012 Create initial migration for MVP tables in migrations/versions/0001_mvp_schema.py
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
- [ ] T024 [US1] Implement background worker entrypoint skeleton run_distillation_job(job_id) with stage updates in src/rulekiln/workers/distillation_worker.py
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

### Implementation for User Story 2

- [ ] T030 [P] [US2] Implement pipeline domain schemas (MicroRule, RuleCluster, SynthesizedRule, EvalResult, QualityGateResult) in src/rulekiln/schemas/pipeline.py
- [ ] T031 [US2] Implement Pydantic AI rule extraction agent wrapper in src/rulekiln/agents/rule_extraction.py
- [ ] T032 [US2] Implement Pydantic AI rule synthesis agent wrapper in src/rulekiln/agents/rule_synthesis.py
- [ ] T033 [US2] Implement chat provider adapters (bedrock, openai, anthropic, vertex_gemini, azure_openai, openai_compatible, custom) behind ChatModelClient in src/rulekiln/providers/chat/
- [ ] T034 [US2] Implement embedding provider adapters behind EmbeddingClient in src/rulekiln/providers/embedding/
- [ ] T035 [US2] Implement clustering service for DBSCAN and HDBSCAN in src/rulekiln/pipeline/clustering.py
- [ ] T036 [US2] Implement deterministic prompt compiler in src/rulekiln/pipeline/prompt_compiler.py
- [ ] T037 [US2] Implement evaluator for baseline and distilled prompts in src/rulekiln/pipeline/evaluator.py
- [ ] T038 [US2] Implement strategy selection and regression analysis in src/rulekiln/pipeline/strategy_selection.py
- [ ] T039 [US2] Implement quality gates (metric delta, malformed output, golden failures, token budget) in src/rulekiln/pipeline/quality_gates.py
- [ ] T040 [US2] Implement failure analysis outputs (fixed, broken, unchanged) in src/rulekiln/pipeline/failure_analysis.py
- [ ] T041 [US2] Implement full worker stage orchestration and resumable/idempotent stage writes in src/rulekiln/workers/distillation_worker.py
- [ ] T042 [US2] Persist pipeline artifacts and run outputs through repositories in src/rulekiln/db/repositories/artifacts.py

**Checkpoint**: Core distillation engine works and selects a winning strategy.

---

## Phase 5: User Story 3 - Surface Artifacts, Reports, and MLflow Audit Trail (Priority: P2)

**Goal**: Expose prompt/rule/eval outputs, export canonical artifacts, and log full run details to MLflow.

**Independent Test**: Completed jobs expose prompt, rules, and eval report endpoints; artifacts are exported; MLflow run includes params, metrics, and artifacts.

### Tests for User Story 3

- [ ] T043 [P] [US3] Add contract tests for GET /distillation-jobs/{job_id}/prompt in tests/contract/test_get_prompt_contract.py
- [ ] T044 [P] [US3] Add contract tests for GET /distillation-jobs/{job_id}/rules and /eval-report in tests/contract/test_get_rules_eval_contract.py
- [ ] T045 [P] [US3] Add integration test for MLflow run logging and artifact upload in tests/integration/test_mlflow_artifact_logging.py

### Implementation for User Story 3

- [ ] T046 [P] [US3] Implement artifact packager for canonical outputs in src/rulekiln/artifacts/writer.py
- [ ] T047 [P] [US3] Implement settings snapshot export with secret redaction in src/rulekiln/artifacts/settings_snapshot.py
- [ ] T048 [US3] Implement MLflow integration for params, metrics, artifacts, and prompt URI in src/rulekiln/integrations/mlflow_tracker.py
- [ ] T049 [US3] Implement GET /distillation-jobs/{job_id}/prompt endpoint in src/rulekiln/api/routes/distillation_outputs.py
- [ ] T050 [US3] Implement GET /distillation-jobs/{job_id}/rules endpoint in src/rulekiln/api/routes/distillation_outputs.py
- [ ] T051 [US3] Implement GET /distillation-jobs/{job_id}/eval-report endpoint in src/rulekiln/api/routes/distillation_outputs.py
- [ ] T052 [US3] Implement strategy comparison payload and selected prompt retrieval in src/rulekiln/db/repositories/artifacts.py
- [ ] T053 [US3] Export selected_distilled_prompt.md, rules.jsonl, eval_report.json, cases.normalized.jsonl, promptfoo.yaml, mlflow_run_id.txt to rulekiln_project/outputs/ and rulekiln_project/exports/
- [ ] T054 [US3] Wire output routes into app and ensure OpenAPI docs include endpoint schemas in src/rulekiln/api/app.py

**Checkpoint**: Auditable outputs and MLflow tracking are available to users.

---

## Phase 6: Polish and Cross-Cutting Concerns

**Purpose**: Final hardening across all stories.

- [ ] T055 [P] Update README.md with setup, environment, provider profile examples, and run instructions
- [ ] T056 [P] Add observability events for model calls, stage timings, retries, token usage, and cost in src/rulekiln/observability/events.py
- [ ] T057 Add security masking helpers for secrets and URLs in logs in src/rulekiln/observability/security.py
- [ ] T058 Add CI checks for lint, type check, and tests in .github/workflows/ci.yml
- [ ] T059 Run end-to-end MVP validation against acceptance criteria and document results in Docs/mvp_validation.md

---

## Dependencies and Execution Order

### Phase Dependencies

- Setup (Phase 1): No dependencies
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