## Plan: Expand MLflow Observability Coverage

Expand the existing late-stage MLflow logging so each distillation job records complete task/model/dataset/strategy context as tags, logs strategy metrics (baseline, DBSCAN, HDBSCAN, selected) including deltas and quality-gate metrics, exports the full artifact set (prompts, rules, eval report, failures, manifest), and keeps token/cost metrics staged behind usage-summary availability. Prompt versioning follows a hybrid approach: selected prompt artifacts are always exported, and Prompt Registry registration is attempted opportunistically with graceful fallback.

**Steps**
1. Phase 1: Define the observability contract and key taxonomy (blocking foundation).
2. In [src/rulekiln/integrations/mlflow_tracker.py](src/rulekiln/integrations/mlflow_tracker.py), add canonical helper builders for:
3. task/model/dataset/strategy tags.
4. per-strategy core metrics for baseline, dbscan, hdbscan, and selected.
5. selected-vs-baseline delta metrics.
6. quality-gate metrics (metric_delta, regression_rate, golden_failures, malformed_output_rate, prompt_tokens, pass/fail).
7. Keep existing token/cost metric names unchanged and layered on top when usage summaries are present (depends on current build_token_cost_metrics).
8. Phase 2: Close persistence plumbing gaps (depends on Phase 1).
9. In [src/rulekiln/workers/distillation_worker.py](src/rulekiln/workers/distillation_worker.py), after create_run, persist run_id to both DB and artifact storage using existing repository and writer helpers.
10. Use [src/rulekiln/db/repositories/jobs.py](src/rulekiln/db/repositories/jobs.py) set_mlflow_run_id from LOGGING_ARTIFACTS so downstream stages do not rely on implicit files.
11. Add a repository helper to persist PromptVersion.mlflow_prompt_uri for the selected strategy (new helper in [src/rulekiln/db/repositories/jobs.py](src/rulekiln/db/repositories/jobs.py)).
12. Phase 3: Wire complete metric logging in worker orchestration (depends on Phases 1-2).
13. In [src/rulekiln/workers/distillation_worker.py](src/rulekiln/workers/distillation_worker.py), retain computed baseline eval, strategy evals, quality-gate results, and strategy comparison outputs for use in LOGGING_ARTIFACTS and EXPORTING_ARTIFACTS.
14. During LOGGING_ARTIFACTS, log:
15. tags: task/model/dataset/strategy context and selection reason.
16. params: existing run params/provider params plus clustering and task metadata available in payload/task.
17. metrics: baseline/dbscan/hdbscan core metrics, selected strategy metrics, delta metrics, and per-strategy quality-gate metrics.
18. Preserve idempotent behavior by keeping stage-marker semantics unchanged (only log once per stage marker).
19. Phase 4: Export full artifact set and log artifacts to MLflow (depends on Phase 3).
20. In [src/rulekiln/workers/distillation_worker.py](src/rulekiln/workers/distillation_worker.py), expand EXPORTING_ARTIFACTS to call existing writers in [src/rulekiln/artifacts/writer.py](src/rulekiln/artifacts/writer.py) for:
21. prompts: baseline, distilled_prompt_dbscan, distilled_prompt_hdbscan, selected_distilled_prompt.
22. rules: rules_dbscan.jsonl, rules_hdbscan.jsonl.
23. reports: eval_report.json, strategy_comparison.json.
24. failures: failures_fixed.jsonl, failures_broken.jsonl, failures_unchanged_failing.jsonl (or chosen canonical naming aligned to existing writer categories).
25. metadata: manifest.json plus token_cost_summary.json.
26. Include task.yaml and cases.normalized.jsonl so run artifacts are self-contained.
27. Build manifest from returned writer paths, then call log_artifacts_dir for the full job artifact root.
28. Phase 5: Prompt versioning hybrid workflow (depends on Phase 3; parallelizable with Phase 4 after run creation).
29. Always export selected_distilled_prompt artifact and selected PromptVersion metadata.
30. Attempt Prompt Registry registration via log_prompt_to_registry in [src/rulekiln/integrations/mlflow_tracker.py](src/rulekiln/integrations/mlflow_tracker.py).
31. On success, store mlflow_prompt_uri on selected PromptVersion via repository helper.
32. On unavailable/failure, continue job completion and emit structured warning only (no stage failure).
33. Phase 6: Staged token/cost rollout (depends on existing usage summary behavior; parallel with Phase 4).
34. Keep log_token_cost_metrics in EXPORTING_ARTIFACTS gated by run_id plus non-empty usage summary.
35. Prefer DB-backed summary fallback already implemented for resumed phases; do not fail the pipeline when cost logging fails.
36. Ensure staged behavior is explicit in docs/tests: token/cost metrics are emitted when usage tracking has produced aggregates.
37. Phase 7: Verification and regression safety net (depends on Phases 1-6).
38. Extend unit coverage in [tests/unit/test_mlflow_token_metrics.py](tests/unit/test_mlflow_token_metrics.py) for new metric builder helpers and naming.
39. Add unit coverage for artifact completeness and manifest generation in [tests/unit/test_token_cost_artifact.py](tests/unit/test_token_cost_artifact.py) or a new artifact-writer unit test file.
40. Add integration test for MLflow run observability (new file under tests/integration/) to assert tags, metric keys, and artifact presence for a completed offline fake-provider run.
41. Keep existing contract and UI artifact tests green by validating no regression in [tests/contract/test_output_routes_contract.py](tests/contract/test_output_routes_contract.py) and [tests/ui/test_artifacts.py](tests/ui/test_artifacts.py).
42. Phase 8: Documentation sync (depends on implementation complete).
43. Update the MLflow/task docs to reflect final metric/tag taxonomy and hybrid Prompt Registry policy in [Docs/plans/rulekiln_mvp_plan_spec_v2.md](Docs/plans/rulekiln_mvp_plan_spec_v2.md) and/or [Docs/task/rulekiln_mvp_tasks_v3_with_docker.md](Docs/task/rulekiln_mvp_tasks_v3_with_docker.md).

**Relevant files**
- [src/rulekiln/workers/distillation_worker.py](src/rulekiln/workers/distillation_worker.py) - primary orchestration changes for tags/metrics/artifacts sequencing and persistence.
- [src/rulekiln/integrations/mlflow_tracker.py](src/rulekiln/integrations/mlflow_tracker.py) - helper builders and MLflow logging primitives.
- [src/rulekiln/artifacts/writer.py](src/rulekiln/artifacts/writer.py) - existing writers reused for required artifact set and manifest.
- [src/rulekiln/db/repositories/jobs.py](src/rulekiln/db/repositories/jobs.py) - run_id persistence and new prompt-uri persistence helper.
- [src/rulekiln/db/models.py](src/rulekiln/db/models.py) - PromptVersion and DistillationJob fields already present; no schema expansion expected for this scope.
- [src/rulekiln/pipeline/strategy_selection.py](src/rulekiln/pipeline/strategy_selection.py) - source of selection reason and selected strategy metadata.
- [src/rulekiln/pipeline/quality_gates.py](src/rulekiln/pipeline/quality_gates.py) - source of gate metrics to log.
- [src/rulekiln/pipeline/failure_analysis.py](src/rulekiln/pipeline/failure_analysis.py) - source of failure category artifacts.
- [tests/unit/test_mlflow_token_metrics.py](tests/unit/test_mlflow_token_metrics.py) - extend unit metric coverage.
- [tests/unit/test_token_cost_artifact.py](tests/unit/test_token_cost_artifact.py) - extend artifact writer coverage.
- [tests/contract/test_output_routes_contract.py](tests/contract/test_output_routes_contract.py) - confirm output API shape remains stable.
- [tests/ui/test_artifacts.py](tests/ui/test_artifacts.py) - confirm artifact listing/download remains stable.

**Verification**
1. Run focused unit tests for MLflow helpers and artifact writers, including new key naming assertions.
2. Run/author integration test with file-backed MLflow to assert:
3. one run per job.
4. required tags for task/model/dataset/strategy.
5. metrics for baseline/dbscan/hdbscan/selected plus deltas and gate metrics.
6. artifact upload contains prompts, rules, eval reports, failures, and manifest.
7. token/cost metrics are emitted when usage summary exists and gracefully skipped otherwise.
8. Prompt Registry behavior is non-blocking: URI saved on success, warning-only fallback on unsupported MLflow.
9. Re-run existing output and UI artifact tests to detect regressions.

**Decisions**
- Prompt versioning approach: Hybrid (artifacts required; Prompt Registry opportunistic).
- Metrics depth: Core per-strategy metrics plus selected-vs-baseline deltas plus quality-gate metrics.
- Token/cost rollout: Staged; do not hard-fail jobs when usage summary or registry capabilities are unavailable.
- In scope: logging taxonomy, worker sequencing, artifact completeness, run/prompt URI persistence, tests.
- Out of scope: new database schema/migration work, new UI surfaces, provider-side usage instrumentation redesign.

**Further Considerations**
1. Normalize failure artifact naming to one canonical convention before implementation (existing analysis uses unchanged_passing/unchanged_failing while historical docs mention unchanged); choose one and keep UI pattern list aligned.
2. Keep MLflow key cardinality bounded (avoid per-case tags/metrics) to prevent run bloat and preserve query performance.
3. Preserve backward compatibility for any existing dashboards by retaining current token/cost metric names and only adding prefixed strategy metrics.