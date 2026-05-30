# RuleKiln MVP Validation

**Date**: 2026-05-21
**Version**: 0.1.0

---

## Acceptance criteria results

### US1 — Submit and Track Distillation Jobs

| Criterion | Result | Evidence |
|-----------|--------|----------|
| `POST /v1/jobs/` returns `202 Accepted` with a `job_id` | PASS | `test_jobs_api.py::test_create_job_returns_202` |
| `GET /v1/jobs/{id}` returns current `status` and `stage` | PASS | `test_jobs_api.py::test_get_job_returns_status` |
| Unknown job returns `404` | PASS | `test_jobs_api.py::test_get_unknown_job_returns_404` |
| Legacy top-level fields (e.g. `task_name` at root) are rejected `422` | PASS | `test_request_validator.py` |
| Job is persisted to database on creation | PASS | `test_jobs_api.py` (SQLite in-memory) |

### US2 — Run Distillation Pipeline and Select Winning Strategy

| Criterion | Result | Evidence |
|-----------|--------|----------|
| Full pipeline runs end-to-end with fake providers | PASS | `test_offline_e2e_fake_providers.py::test_pipeline_runs_to_completion` |
| Pipeline with baseline runs end-to-end | PASS | `test_offline_e2e_fake_providers.py::test_pipeline_with_baseline_runs` |
| All 16 pipeline stages present in `PipelineStage` enum | PASS | `test_worker_stage_model.py` |
| Prompt compiler output is deterministic (stable hash) | PASS | `test_prompt_compiler_determinism.py` (7 tests) |
| DBSCAN and HDBSCAN strategies compiled to distinct prompts | PASS | `test_prompt_compiler_determinism.py::test_strategy_produces_different_hash` |
| Evaluation contract: accuracy range, malformed flag, weighted case score | PASS | `test_evaluation_contract.py` |
| Quality-gate check runs and produces `QualityGateResult` | PASS | `test_offline_e2e_fake_providers.py` |
| Strategy is selected and persisted at end of pipeline | PASS | `test_offline_e2e_fake_providers.py` |

### US3 — Surface Artifacts, Reports, and MLflow Audit Trail

| Criterion | Result | Evidence |
|-----------|--------|----------|
| `GET /v1/jobs/{id}/prompt` returns selected system prompt | PASS | `test_output_routes_contract.py::test_get_prompt_returns_correct_shape` |
| `GET /v1/jobs/{id}/rules` returns rule list | PASS | `test_output_routes_contract.py::test_get_rules_returns_list` |
| `GET /v1/jobs/{id}/eval-report` returns eval run list | PASS | `test_output_routes_contract.py::test_get_eval_report_returns_runs` |
| Endpoints return `404` for unknown or incomplete jobs | PASS | `test_output_routes_contract.py` (3 × 404 tests) |
| Artifact writer produces job-scoped directory layout | PASS | `artifacts/writer.py` (verified in e2e test) |
| Settings snapshot redacts secrets before writing | PASS | `test_security_masking.py` |
| Security masking covers URLs with embedded credentials | PASS | `test_security_masking.py::test_mask_url_*` |

---

## Test suite summary

```
36 passed, 0 failed
Offline only (sqlite+aiosqlite in-memory + file-backed MLflow)
Runtime: ~2.5 s
```

Test breakdown:

| Suite | Tests | Notes |
|-------|-------|-------|
| `tests/unit/` | 20 | Deterministic, no I/O |
| `tests/contract/` | 9 | Schema shape contracts via in-memory DB |
| `tests/integration/` | 7 | Full pipeline + HTTP layer |

---

## Known limitations and deferred items

| Item | Decision |
|------|----------|
| T027 `test_provider_resolution.py` | Deferred — covered implicitly by e2e tests; separate unit test not yet written |
| T028 `test_distillation_pipeline_selection.py` | Deferred — covered by `test_offline_e2e_fake_providers.py` |
| T029 `test_golden_case_quality_gate.py` | Deferred — quality gate logic exercised in e2e test |
| T045 `test_mlflow_artifact_logging.py` | Deferred — MLflow logging runs in e2e test; isolated unit test not yet written |
| pgvector embedding persistence | `ENABLE_PGVECTOR=false` by default; embeddings kept in memory |
| `require_human_approval` gate | Gate is checked and logged but no human-approval workflow is implemented in MVP |
| External provider smoke tests (`-m external`) | Excluded from CI; opt-in only |
| Anthropic / Vertex / Azure stubs | Raise `ProviderNotImplementedError` fast; full adapters deferred |

---

## CI baseline

Pipeline: `.github/workflows/ci.yml`

| Job | Status |
|-----|--------|
| Lint & type check (Ruff + Pyright) | Configured |
| Offline tests (36 tests, no external deps) | Configured |
| Docker image build validation | Configured (T058A) |
| Docker Compose config validation | Configured (T058B) |
