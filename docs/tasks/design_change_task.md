---
description: "Task list for RuleKiln Backend Design Adjustment"
---

# Tasks: RuleKiln Backend Design Adjustment

> **Status:** Historical/archival implementation task list. It tracks legacy queue/background migration work and is not the current DBOS-only runtime plan.

**Input**: `docs/specs/rulekiln_backend_design_adjustment_spec.md`

**Prerequisites**: Existing MVP backend is complete and all current tests pass.

**Tests**: Included per phase as specified in the design spec (Section 18).

---

## Format: `[ID] [P?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)

---

## Phase 1: Durable Execution Backend (Postgres Job Queue)

**Goal**: Replace FastAPI `BackgroundTasks` with a Postgres-backed job queue so pod restarts and worker crashes do not orphan running jobs.

**Independent Test**: Submit a job, kill the API process, restart it — job should remain `pending` and be picked up by the worker.

### Tests for Phase 1

- [ ] A008a [P] Add `test_claim_next_job_uses_skip_locked` in `tests/unit/test_queue_worker.py`
- [ ] A008b [P] Add `test_two_workers_do_not_claim_same_job` in `tests/unit/test_queue_worker.py`
- [ ] A008c [P] Add `test_expired_lease_returns_job_to_pending` in `tests/unit/test_queue_worker.py`
- [ ] A008d [P] Add `test_max_attempts_marks_job_failed` in `tests/unit/test_queue_worker.py`
- [ ] A008e [P] Add `test_completed_job_is_not_reclaimed` in `tests/unit/test_queue_worker.py`
- [ ] A008f [P] Add `test_stage_resume_starts_at_first_incomplete_stage` in `tests/unit/test_queue_worker.py`

### Implementation for Phase 1

- [ ] A001 Add queue columns (`queue_status`, `locked_by`, `locked_at`, `lease_expires_at`, `attempt_count`, `max_attempts`, `next_run_at`) and result columns (`selected_strategy`, `selected_prompt_version_id`, `primary_metric`, `baseline_score`, `selected_score`, `metric_delta`, `quality_gates_passed`) to `migrations/versions/0003_add_queue_columns.py`
- [ ] A002 Add optional `job_stage_markers` table with `(job_id, stage, strategy, artifact_type)` unique constraint to `migrations/versions/0003_add_queue_columns.py` alongside A001
- [ ] A003 Implement `JobQueueRepository.claim_next_job()` using `FOR UPDATE SKIP LOCKED` and `JobQueueRepository.renew_lease()` in `src/rulekiln/db/repositories/jobs.py`
- [ ] A004 Implement expired lease recovery (`recover_expired_leases()`) and max-attempts failure marking in `src/rulekiln/db/repositories/jobs.py`
- [ ] A005 Implement `queue_worker.py` with `worker_loop()`, lease renewal task, and `rulekiln-worker` CLI entrypoint in `src/rulekiln/workers/queue_worker.py`
- [ ] A006 Update `POST /distillation-jobs` to create a `queue_status='pending'` record without spawning `BackgroundTasks` when `EXECUTION_BACKEND=postgres_queue`; keep old path behind feature flag in `src/rulekiln/api/routes/`
- [ ] A007 Add `worker` service to `docker-compose.yml` with `command: rulekiln-worker`; keep `api` service with `uvicorn` command separate
- [ ] A008 Add `execution_backend`, `worker_poll_interval_seconds`, and `worker_lease_seconds` to `src/rulekiln/config/settings.py` and `.env.example`

**Checkpoint**: Worker picks up queued jobs; API returns 202 without blocking; two concurrent workers never claim the same job.

---

## Phase 2: Provider Rate Limiting

**Goal**: Prevent provider quota failures by enforcing `max_concurrency` and RPM limits on every provider call path.

**Independent Test**: Configure `max_concurrency=1` and `rate_limit_rpm=6`; submit a job with many cases; verify calls are serialized and wait times are logged.

### Tests for Phase 2

- [ ] B008a [P] Add `test_provider_max_concurrency_is_enforced` in `tests/unit/test_rate_limiter.py`
- [ ] B008b [P] Add `test_provider_rpm_limit_waits` in `tests/unit/test_rate_limiter.py`
- [ ] B008c [P] Add `test_route_override_takes_precedence_over_profile` in `tests/unit/test_rate_limiter.py`
- [ ] B008d [P] Add `test_profile_limit_takes_precedence_over_app_default` in `tests/unit/test_rate_limiter.py`

### Implementation for Phase 2

- [ ] B001 Add `rate_limit_rpm: int | None`, `rate_limit_tpm: int | None`, and `max_concurrency: int` fields to `ProviderProfile` in `src/rulekiln/providers/contracts.py`
- [ ] B002 Add optional `rate_limit_rpm`, `rate_limit_tpm`, and `max_concurrency` override fields to `ModelRoute` in `src/rulekiln/providers/contracts.py`
- [ ] B003 Implement effective limit resolution (route override → profile → app default) in `src/rulekiln/providers/resolver.py`
- [ ] B004 Implement `ProviderRateLimiter` using `asyncio.Semaphore` for concurrency and a sliding-window counter for RPM in `src/rulekiln/providers/rate_limiter.py`
- [ ] B005 Wrap chat provider `complete()` calls with `ProviderRateLimiter.acquire()` in `src/rulekiln/providers/chat/`
- [ ] B006 Wrap embedding provider `embed()` calls with `ProviderRateLimiter.acquire()` in `src/rulekiln/providers/embedding/`
- [ ] B007 Log `rate_limit_wait_seconds` and effective limits via structlog in provider call sites; add `DEFAULT_PROVIDER_MAX_CONCURRENCY`, `DEFAULT_PROVIDER_RATE_LIMIT_RPM`, and `DEFAULT_PROVIDER_RATE_LIMIT_TPM` to `src/rulekiln/config/settings.py` and `.env.example`

**Checkpoint**: Provider calls respect concurrency and RPM caps; wait time appears in structured logs; route overrides take precedence over profile defaults.

---

## Phase 3: Conflict Detection and Resolution

**Goal**: Detect contradictions in synthesized rules and exclude unresolved conflicts from prompt compilation.

**Independent Test**: Inject a conflicting synthesized rule; verify it is marked `has_conflicts=True`, excluded from prompt compilation, and written to `rule_conflicts_*.jsonl`.

### Tests for Phase 3

- [ ] C007a [P] Add `test_conflicting_rule_is_marked_has_conflicts` in `tests/unit/test_conflict_review.py`
- [ ] C007b [P] Add `test_discarded_conflict_rule_is_not_compiled` in `tests/unit/test_conflict_review.py`
- [ ] C007c [P] Add `test_split_conflict_rule_produces_multiple_rules` in `tests/unit/test_conflict_review.py`
- [ ] C007d [P] Add `test_conflict_artifacts_are_written` in `tests/unit/test_conflict_review.py`

### Implementation for Phase 3

- [ ] C001 Add `has_conflicts`, `conflict_summary`, and `conflicting_micro_rule_ids` fields to `SynthesizedRule` in `src/rulekiln/schemas/pipeline.py`
- [ ] C002 Add `RuleConflictReview` Pydantic model with `resolution: Literal["keep", "modify", "split", "discard"]` and `resolved_rules` in `src/rulekiln/schemas/pipeline.py`
- [ ] C003 Implement `RuleConflictReviewAgent` using PydanticAI in `src/rulekiln/agents/rule_conflict_review.py`
- [ ] C004 Add `reviewing_rule_conflicts` stage to the pipeline in `src/rulekiln/pipeline/` and register it in the worker stage sequence in `src/rulekiln/workers/queue_worker.py`
- [ ] C005 Update `src/rulekiln/pipeline/prompt_compiler.py` to filter out rules where `has_conflicts=True` and `resolution` is not `"keep"` or `"modify"`
- [ ] C006 Export `rule_conflicts_dbscan.jsonl`, `rule_conflicts_hdbscan.jsonl`, and `rules_discarded_conflicts.jsonl` via `src/rulekiln/artifacts/writer.py`

**Checkpoint**: Conflicting rules are flagged, excluded from the compiled prompt, and written to conflict artifacts.

---

## Phase 4: Rule Pruning and Prompt Bloat Control

**Goal**: Enforce `max_rules`, `min_rule_support_count`, and `max_prompt_tokens` budget before prompt compilation.

**Independent Test**: Create more rules than `max_rules` allows; verify lower-support rules are pruned and the pruning report lists reasons.

### Tests for Phase 4

- [ ] D007a [P] Add `test_rules_below_min_support_are_pruned` in `tests/unit/test_rule_pruning.py`
- [ ] D007b [P] Add `test_golden_backed_rule_is_preserved` in `tests/unit/test_rule_pruning.py`
- [ ] D007c [P] Add `test_max_rules_is_enforced` in `tests/unit/test_rule_pruning.py`
- [ ] D007d [P] Add `test_prompt_token_budget_is_enforced` in `tests/unit/test_rule_pruning.py`
- [ ] D007e [P] Add `test_pruning_report_contains_reasons` in `tests/unit/test_rule_pruning.py`

### Implementation for Phase 4

- [ ] D001 Add `max_rules: int = 40`, `max_prompt_tokens: int = 8000`, `min_rule_support_count: int = 2`, and `preserve_golden_rules: bool = True` to `RuleKilnTask` in `src/rulekiln/schemas/task_case.py`; add corresponding `DEFAULT_MAX_RULES`, `DEFAULT_MIN_RULE_SUPPORT_COUNT`, and `DEFAULT_MAX_PROMPT_TOKENS` to `src/rulekiln/config/settings.py`
- [ ] D002 Add `support_count`, `support_ratio`, `golden_case_backed`, and `estimated_token_count` fields to `SynthesizedRule` in `src/rulekiln/schemas/pipeline.py`
- [ ] D003 Implement `RulePruningService` with pruning order (conflict removal → golden preservation → min-support filter → priority/support sort → `max_rules` cap → token budget cap) and explicit `PruningReason` literals in `src/rulekiln/pipeline/rule_pruning.py`
- [ ] D004 Add `pruning_rules` stage to the pipeline sequence after `reviewing_rule_conflicts` in `src/rulekiln/workers/queue_worker.py`
- [ ] D005 Update `src/rulekiln/pipeline/prompt_compiler.py` to receive only the selected rules list from pruning output
- [ ] D006 Export `rules_selected_dbscan.jsonl`, `rules_selected_hdbscan.jsonl`, `rules_pruned_dbscan.jsonl`, `rules_pruned_hdbscan.jsonl`, and `rule_pruning_report.json` via `src/rulekiln/artifacts/writer.py`

**Checkpoint**: Compiled prompts respect rule count and token budget; pruned rules appear in artifacts with explicit reasons.

---

## Phase 5: Eval-to-Rule Failure Mapping

**Goal**: Map evaluation failures to the rules that were matched, violated, or missed — making the rule layer auditable.

**Independent Test**: Run offline end-to-end with fake providers; verify `failures_broken.jsonl` includes `matched_rule_ids`, `violated_rule_ids`, and `failed_assertion_paths`.

### Tests for Phase 5

- [ ] E007a [P] Add `test_failed_assertion_path_maps_to_rule_output_path` in `tests/unit/test_eval_rule_mapping.py`
- [ ] E007b [P] Add `test_failure_artifact_contains_violated_rule_ids` in `tests/unit/test_eval_rule_mapping.py`
- [ ] E007c [P] Add `test_violated_rule_summary_counts_failures` in `tests/unit/test_eval_rule_mapping.py`
- [ ] E007d [P] Add `test_failures_page_view_model_includes_rule_mapping` in `tests/ui/test_results.py`

### Implementation for Phase 5

- [ ] E001 Add `CaseEvaluationFailure` Pydantic model with `matched_rule_ids`, `violated_rule_ids`, `failed_assertion_paths`, `failed_assertion_types`, and `explanation` fields in `src/rulekiln/schemas/pipeline.py`
- [ ] E002 Update `EvalResult` in `src/rulekiln/schemas/pipeline.py` to include `violated_rule_counts: dict[str, int]`, `failed_assertion_path_counts: dict[str, int]`, and `failures: list[CaseEvaluationFailure]`
- [ ] E003 Update `src/rulekiln/pipeline/failure_analysis.py` to map `failed_assertion.path` → `rule.output_path` → `violated_rule_id`; fall back to `matched_rule_ids` from model output when no path match exists
- [ ] E004 Update `src/rulekiln/pipeline/evaluator.py` to populate `violated_rule_counts` and `failed_assertion_path_counts` aggregates on `EvalResult`
- [ ] E005 Export `violated_rule_summary.json` (keyed by rule ID with `violated_count`, `broken_count`, `unchanged_wrong_count`) via `src/rulekiln/artifacts/writer.py`
- [ ] E006 Update `src/rulekiln/ui/view_models.py` to include `matched_rule_ids`, `violated_rule_ids`, and `failed_assertion_paths` columns in failure view models; update rules view model to include `violated_count` and `pruned_status`

**Checkpoint**: Failure artifacts include rule mapping fields; violated rule summary JSON is written; failures UI page shows rule columns.

---

## Phase 6: MLflow Logging, UI Polish, and Documentation

**Goal**: Surface new pipeline metadata in MLflow, update UI stage labels and results pages, and document the worker service.

### Implementation for Phase 6

- [ ] F001 Log `execution_backend`, `worker_id`, `max_rules`, `min_rule_support_count`, `provider_rate_limit_rpm`, and `provider_max_concurrency` as MLflow params in `src/rulekiln/integrations/mlflow_tracker.py`
- [ ] F002 Log `num_rules_discarded_conflicts`, `num_rules_pruned`, `num_rules_selected`, `num_violated_rules`, and `rate_limit_wait_seconds_total` as MLflow metrics; log `rule_conflicts_*.jsonl`, `rules_selected_*.jsonl`, `rules_pruned_*.jsonl`, `rule_pruning_report.json`, and `violated_rule_summary.json` as MLflow artifacts
- [ ] F003 [P] Add `reviewing_rule_conflicts` and `pruning_rules` to the stage label map in UI templates under `src/rulekiln/templates/`
- [ ] F004 [P] Update results page template/view model to show `rules_selected`, `rules_pruned`, `rules_discarded_conflicts`, and top violated rules in `src/rulekiln/ui/` and `src/rulekiln/templates/jobs/`
- [ ] F005 [P] Update rules page template/view model to show `has_conflicts`, `conflict_summary`, `support_count`, `support_ratio`, and `violated_count` in `src/rulekiln/ui/` and `src/rulekiln/templates/`
- [ ] F006 [P] Update failures page template to add `matched_rules`, `violated_rules`, and `failed_assertion_paths` columns in `src/rulekiln/templates/`
- [ ] F007 [P] Update `README.md` with backend architecture notes covering the Postgres queue and `rulekiln-worker` process
- [ ] F008 [P] Update `docs/dev/docker.md` to document the `api` + `worker` dual-service Docker Compose setup

**Checkpoint**: MLflow run contains new metrics and artifacts; UI shows new stages, rule counts, and failure rule columns; docs reflect the worker service.

---

## Dependencies & Execution Order

| Phase | Depends On | Can Parallelize Within |
|---|---|---|
| Phase 1 (Queue) | None — start immediately | A001+A002 together; A003+A004 together after migration |
| Phase 2 (Rate Limiting) | Phase 1 complete | B001+B002 together; B005+B006 after B004 |
| Phase 3 (Conflict Review) | Phase 1 complete | C001+C002 together; C003 after C002 |
| Phase 4 (Rule Pruning) | Phase 3 complete | D001+D002 together; D003 after D001+D002 |
| Phase 5 (Eval Mapping) | Phase 4 complete | E001+E002 together; E003+E004 after E002 |
| Phase 6 (MLflow/UI/Docs) | Phase 5 complete | F003+F004+F005+F006+F007+F008 all parallel |

Phases 2, 3, and 4 can begin in parallel once Phase 1 is stable.

---

## Rollback Plan

Set `EXECUTION_BACKEND=background_tasks` in `.env` to revert to the original FastAPI `BackgroundTasks` path until the queue worker is proven stable.
