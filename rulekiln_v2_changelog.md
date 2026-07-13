# RuleKiln v2 Changelog

_Working-tree changes on `allow-batching-via-ui`, reviewed 2026-07-08. Uncommitted — not yet split into PRs._

## Highlights

- **Batch API robustness**: strict JSON-schema enforcement and consistent error extraction for OpenAI Batch/Responses API calls, plus a hard-fail (instead of silent zero-rule proceed) when batch extraction fails.
- **Strategy selection bug fix**: the pipeline could previously select a strategy that *regressed* against baseline when no candidate passed its quality gate. It now falls back to baseline with an explicit reason.
- **Metric-definition fix**: out-of-vocabulary predictions are now scored as malformed instead of silently counted as valid, which changes reported macro-F1/malformed-rate numbers for affected strategies (see Breaking Changes).
- **Few-shot echo-failure fix**: student prompts now render from the task's `input_template` instead of raw JSON, with a new answer-cue/transition separator, fixing cases where the model echoed the test input as "another example."
- **New clustering tooling**: schema-driven DBSCAN/HDBSCAN params, an offline clustering-sweep CLI, and cluster-health diagnostics to empirically tune eps/min_samples instead of guessing.
- **Clustering provenance & promotion**: every clustered evaluation now persists full, versioned clustering provenance, canonical reproducibility hashes, structured provenance validation, and an explicit promotion decision (`selected` vs. `promotion_candidate` vs. `promoted`) with deterministic blockers — closing the reproducibility gaps the ContractNLI HDBSCAN integrity review found.
- Four investigation docs (`docs/investigations/`, `docs/dev/`) document the root causes behind the above fixes.

## Batch Processing

- `providers/chat/openai_chat.py`: batch requests now build strict-mode JSON schemas (`_to_strict_json_schema`/`_strictify_schema_node`), forcing `additionalProperties: false` and `required` on every object node recursively, without mutating the input schema. This is required for the OpenAI Batch/Responses API to actually enforce output shape.
- Same file: new `_format_error_dict`/`_extract_batch_item_error` normalize the various provider batch-item error shapes (top-level dict/string error, nested response-body error, non-200 without body, truncated "incident" rows) into one consistent message, with a "dominant failure reason" picker that preserves retryable-vs-terminal distinction across a batch.
- `workers/pipeline/early_stages.py`: new `ExtractionYieldedNoRulesError`. If batch extraction returns 0 rules *because items failed* (vs. a legitimate empty extraction), the stage now hard-fails with the dominant error reason instead of silently continuing with zero rules.
- `schemas/classroom.py`: `PhaseTeacherConfig.batch_min_items` default lowered from **10 to 5** — batch mode now triggers more readily. Check any config relying on the old implicit threshold.

## Strategy Selection & Clustering

- **Regression bug fix** — `pipeline/strategy_selection.py`'s `select_strategy_generic` previously ranked the failed candidate pool when zero candidates passed their quality gate, which could select a strategy worse than baseline. It now falls back to `baseline_scaffold` with an explicit reason listing every candidate's gate failures. Root-caused in `docs/investigations/contractnli_strategy_selection_regression.md` (job `1521e48b`: `embedding_knn_k1` at 0.61 macro_f1 was selected over baseline's 0.71 despite a net paired regression of 41 fixed/74 broken).
- New `pipeline/strategy_families.py` classifies strategies into `baseline`, `embedding_baseline`, `retrieval_baseline`, `distilled` (previously only `baseline`/`distilled` existed, so embedding-KNN selections were misreported as `baseline`-family).
- New Gate 6, `require_paired_net_improvement` (`pipeline/quality_gates.py`, default **true**, per-task overridable via `config/settings.py`), fails a strategy if `paired_broken > paired_fixed` vs. baseline. `QualityGateResult` gains `paired_fixed`/`paired_broken`.
- New selection transparency: `StrategyComparison.selection_table` (one row per strategy: family, eligibility, gate failures, score, rank, selected flag, reason) lands in `strategy_comparison.json` and now renders as a "Strategy Selection" table + "Selection reason" in `templates/jobs/eval_report.html`, wired from `ui/routes.py`.
- Fail-fast token budget: `distilled_eval_service.py` now token-counts the compiled prompt *before* evaluation and skips strategies over `max_prompt_tokens` instead of burning all per-case calls first (root cause of an HDBSCAN 100%-malformed run: ~11,006 tokens vs. an 8,192-token model context / 8,000 max_prompt_tokens budget).
- **New clustering capability**:
  - `pipeline/clustering_diagnostics.py` (new, pure functions): k-distance percentile reports, eps-candidate derivation, Shannon entropy of label/hypothesis distributions per cluster, duplicate-rule rate, embedding-norm stats, and `evaluate_eligibility` producing `EligibilityFlags`. Backing schemas added to `schemas/pipeline.py`.
  - `benchmarks/clustering_sweep.py` (new) + `benchmarks/cli.py clustering-sweep` subcommand: offline sweep over embedding-text variants (rule-only / label+rule / hypothesis-degraded) and clustering params, without running the LLM pipeline.
  - `schemas/task_case.py` adds `DbscanParams`, `HdbscanParams`, `ClusteringParams`, `RuleKilnTask.clustering`; `ClusteringStageService` reads these instead of hardcoded eps/min_samples. Additive/backward compatible — an omitted block yields prior defaults.
  - `examples/datasets/contractnli/task.yaml` now sets `clustering: hdbscan, min_cluster_size=5, min_samples=1, method=eom, epsilon=0.0`.
  - Still open: DBSCAN's default eps=0.3 is not yet empirically tuned for mxbai embeddings — the sweep tooling exists to do this, but no tuned default is committed yet.

## Error Handling & Output Quality

- `pipeline/evaluator.py`: for classification/routing tasks with an enum output schema, a syntactically valid but out-of-vocabulary prediction is now scored `malformed=True` (`error="prediction_not_in_allowed_labels"`) instead of silently passing as a valid (wrong) label. New `_try_recover_label` salvages a single unambiguous valid label wrapped in extra prose/punctuation/casing before giving up.
- `pipeline/prompt_compiler.py`: fixes compound-label leakage where `_render_rule` rendered the internal dict key for an outcome instead of the canonical `oc.outcome` field, and where free-text rule prose could leak the same mnemonic. New `_canonical_outcome_label`, `_contains_forbidden_instruction`, `_normalize_compound_label_mentions`, `_sanitize_rule` fix this and strip instruction-injection patterns from synthesized rules. `compile_prompt` now appends a "# Final Answer Format (Required)" reinforcement section. Root-caused in `docs/investigations/contractnli_hdbscan_malformed_output.md` (18 malformed cases, 83.3% compound-leak).
- `agents/rule_synthesis.py`: system prompt strengthened against inventing label variants or embedding output-format instructions inside synthesized rules; explicitly enumerates the allowed label vocabulary when the task has an output schema.
- `artifacts/writer.py`: new `write_strategy_debug_artifacts` writes 8 debug artifact types per strategy under `outputs/debug/` (raw/parsed/malformed outputs, baseline-passed/candidate-failed regressions, truncated+hashed prompt preview, confusion matrix, classification report), wired into `output_stage_services.py` for every strategy.
- `integrations/mlflow_tracker.py`: `log_artifacts_dir` no longer silently swallows exceptions from `mlflow.log_artifacts` — previously-invisible failures now surface.

## Evaluator & Prompting

- `_call_student` now renders the user message from the task's `input_template` (`_render_input_from_template`) instead of `json.dumps(case.input)` — fixes a few-shot "echo failure" where, at k≥4 examples, the model pattern-matched the raw-JSON test input as "the next example" and echoed/continued it instead of answering. Root-caused in `docs/dev/few_shot_k5_investigation.md` and `docs/dev/run3_investigation.md`. Applies to all strategies including baseline — tasks without a working `input_template` need a fallback (flagged as a compatibility concern in the investigation doc).
- New `FEW_SHOT_ANSWER_CUE = "Output:"` and `render_few_shot_transition` add an answer cue plus a "this is the actual input, not another example" separator between few-shot examples and the real input.
- `evaluate_prompt` gains `user_message_prefix` and `check_cancelled: CancellationCheckFn | None`; new `JobCancelledError` propagates through per-case eval loops (`baseline_eval_service`, `distilled_eval_service`, `refinement_ablation`), so long-running per-case loops can now be cancelled mid-run rather than only at DBOS step boundaries. `db/repositories/jobs.py` adds `is_job_cancelled(session, job_id)`.
- `_call_student` return signature changed from a 2-tuple to a 3-tuple (adds `raw_content`); `CaseEvalResult` gains `raw_content`/`recovered` fields.

## Infrastructure & Config

- `Dockerfile`: rewritten for uv build caching — prebuilt uv binary layer, `--mount=type=cache`, dependency install split from source copy.
- `docker-compose.yml`: adds `./logs:/app/logs` volume and `LOG_FILE` env var to worker/web services.
- `observability/logging.py`: `configure_logging` now attaches an optional rotating file handler (10MB × 5 backups) when `LOG_FILE` is set; `dbos_worker.py main()` now calls `configure_logging()`.
- `workers/dbos_workflow.py`: `_run_stage_sequence` switched to `DBOS.sleep_async` from a wrapped sync `DBOS.sleep`, aligning with the async DBOS API used elsewhere and supporting proper async cancellation propagation.
- `.gitignore`: adds `logs/rulekiln.log*` patterns.
- Deleted `.serena/.gitignore`, `.serena/project.yml` — Serena MCP/IDE tooling config removed, no functional impact.
- `examples/datasets/contractnli/generate_cases.py`: adds `TEST_LIMIT`/`--train-limit`/`--validation-limit`/`--test-limit` args and writes a `cases.test.jsonl` split; adds `ensure_ascii=False`/UTF-8 encoding for non-ASCII contract text.
- `examples/datasets/contractnli/task.yaml`: adds `max_prompt_tokens: 15500`; quotes two colon-containing YAML strings in `task_scope`/`non_scope`, fixing a real parse bug where an unquoted string was silently parsed as a nested mapping and corrupted the rules section of every baseline prompt. Regression-guarded by new `test_task_yaml_scope_parsing.py`, parametrized over every `examples/**/task.yaml`.

## Clustering Provenance, Reproducibility Hashes & Promotion

Implements durable, end-to-end provenance and explicit promotion metadata for clustered
strategy evaluations, in response to `docs/investigations/contractnli_hdbscan_validation_win.md`.
Full file-by-file summary: `docs/dev/clustering_provenance_promotion_changes.md`; architecture
references: `docs/architecture/clustering_provenance.md`, `docs/architecture/strategy_promotion.md`.

- **Clustering provenance** (`schemas/provenance.py`, `pipeline/clustering_provenance.py`): a versioned `ClusteringProvenance` record captured at clustering time in `early_stages.py` — requested vs. effective parameters (HDBSCAN `min_samples=null` stays visibly `null`, effective shows the resolved value), embedding model/dimension/normalization/metric, text variant, deterministic cluster assignments, cluster summary, and hashes. Persisted under `outputs/clustering/<algo>/`. Raw embedding vectors are never written to JSON — only an `embedding_matrix_hash`.
- **Canonical hashing** (`artifacts/canonical_hash.py`): one SHA-256 path (sorted keys, order-sensitive lists, explicit float/`null` handling, recorded `canonicalization_version=v1`). `pipeline/artifact_hashing.py` aggregates job-level (dataset split, ordered case IDs, task/provider/parser/quality-gate config) + clustering + strategy hashes into `metadata/hashes.json` and embeds per-strategy `StrategyProvenanceRow`s in `strategy_comparison.json`.
- **Provenance validation** (`pipeline/provenance_validation.py` → `metadata/provenance_validation.json`): structured findings for cross-job artifacts, cluster-assignment/prompt hash mismatches, case-set mismatches (errors), and runtime-vs-approved config drift (warning). Errors block **promotion**; they never change **selection**.
- **Promotion status & policy** (`pipeline/promotion.py`, `schemas/promotion.py` → `metadata/promotion_status.json`): explicit `selected_strategy` / `promotion_candidate` / `promoted_strategy` semantics with a versioned `PromotionPolicy` (defaults preserve current behavior — no significance requirement, malformed boundary not blocked) and deterministic ordered blockers. Never auto-promotes: `approved` means "ready for a human", `promoted_strategy` stays `null`.
- **Approved-config registry** (`config/approved_strategy_configs/hdbscan.json`, `pipeline/approved_configs.py`): the promoted `5/1 eom` config, compared to a run's runtime config by canonical hash — so a `2/null` run is detected as `CONFIG_NOT_PROMOTED_CONFIG` rather than assumed correct. Not inferred from the current `task.yaml`.
- **Manifest enrichment** (`artifacts/manifest.py`, `write_manifest`): `metadata/manifest.json` keeps its legacy `artifacts` path list **and** gains rich `entries` (type, schema version, sha256, size, timestamp, strategy) verifiable against files on disk via `verify_manifest`. `output_stage_services.py` now also discovers clustering artifacts written in the earlier stage so they appear in the manifest.
- **UI** (`ui/routes.py`, `templates/jobs/eval_report.html`): new "Provenance & Promotion" section — promotion status/blockers, run-provenance hashes, requested-vs-effective params, embedding + cluster stats, per-strategy hashes, with distinct selected/candidate/promoted labels. Legacy runs show "Provenance unavailable for this historical run" and are never backfilled from the current repo config.

## Tests & Docs

- 20 new test files covering: batch strict-schema (5 tests) and error extraction (13 tests), label-vocabulary enforcement (17 tests), malformed-output parsing (17 tests), few-shot separator (7 tests), clustering params/diagnostics/sweep, strategy-selection regression, strategy debug artifacts, task.yaml scope parsing (all example tasks), shared provider config invariants, distilled-eval-with-no-rules integration (3 async tests).
- Provenance/promotion coverage: `test_promotion_policy.py` (11), `test_provenance_validation.py` (6), `test_artifact_hashing.py` (8), `test_run_manifest.py` (3), `test_canonical_hash.py`, `tests/ui/test_provenance_promotion_report.py` (4), plus a full clustered integration test in `test_offline_e2e_fake_providers.py` (artifacts written, manifest verifies, hash references match, promotion status round-trips).
- New investigation docs serve as design rationale: `docs/investigations/contractnli_hdbscan_malformed_output.md`, `docs/investigations/contractnli_strategy_selection_regression.md`, `docs/investigations/contractnli_hdbscan_validation_win.md` (with implementation follow-up), `docs/dev/run3_investigation.md`, `docs/dev/few_shot_k5_investigation.md`.
- New architecture + change docs: `docs/architecture/clustering_provenance.md`, `docs/architecture/strategy_promotion.md`, `docs/dev/clustering_provenance_promotion_changes.md`.

## Breaking Changes / Migration Notes

1. **Metric semantics changed**: label-vocabulary enforcement in `evaluator.py` makes `malformed_output_rate`/`macro_f1` non-comparable to pre-fix runs for any strategy that previously emitted out-of-vocabulary predictions (few-shot strategies especially). Persisted results from resumed jobs retain the old semantics.
2. **`batch_min_items` default changed 10 → 5** in `PhaseTeacherConfig`.
3. **`_call_student` return arity changed** (2-tuple → 3-tuple) — any direct/external callers must be updated.
4. Additive schema changes (`RuleKilnTask.clustering`, `CaseEvalResult.raw_content`/`recovered`, `QualityGateResult.paired_fixed`/`paired_broken`, `StrategyComparison.selection_table`) are backward compatible, but consumers of `strategy_comparison.json` or the eval-report JSON should expect new top-level fields.
5. **New provenance/promotion artifacts and fields are additive and optional.** `StrategyComparison` gains `strategy_provenance`, `artifact_hashes`, `provenance_validation`, `promotion_status` (all defaulted, so legacy `strategy_comparison.json` still parses); new runs also emit `metadata/{hashes,provenance_validation,promotion_status}.json` and `outputs/clustering/<algo>/*`. `metadata/manifest.json` keeps its `artifacts` list unchanged and adds an `entries` array. Historical runs are **not** backfilled — they render as `provenance_status: incomplete` / `reproducibility_status: unverified`. No metric definitions changed and quality gates were not loosened.
