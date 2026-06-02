## Plan: Phase 4 Rule Quality and Ablation

Deliver Phase 4 by adding three linked capabilities without introducing new architecture: (1) per-rule provenance artifacts, (2) optional smoke/small-run leave-one-rule-out ablation, and (3) a two-pass pruning optimizer with support_count / utility / utility_per_token modes. Reuse current worker orchestration, eval persistence, and artifact conventions; add only targeted schema/config/reporting extensions.

**Steps**
1. Phase 1 - Contracts and Config Surface (blocks all later phases)
   - Extend task/config inputs in src/rulekiln/schemas/task_case.py and src/rulekiln/config/settings.py:
   - Add rule_pruning_mode: Literal["support_count", "utility", "utility_per_token"] (default support_count).
   - Add rule_regression_penalty: float (used by utility score).
   - Add rule ablation controls: enable_rule_ablation, max_ablation_rules, small_run_case_threshold, ablation_min_changed_cases.
   - Extend pipeline schemas in src/rulekiln/schemas/pipeline.py for:
   - Rule provenance records (associative + optional causal fields, flags, notes).
   - Rule ablation report + per-rule classification.
   - Optional pruning_mode_comparison payload attached to strategy comparison artifacts.
   - Extend benchmark schemas in src/rulekiln/benchmarks/schemas.py to carry pruning_mode_comparison pass-through data for summary rendering.

2. Phase 2 - Rule Provenance Assembly and Export (depends on 1)
   - Add repository read helper in src/rulekiln/db/repositories/jobs.py to load rule clusters by job_id + strategy (for cluster_id lookup).
   - Add provenance builder logic (new helper module under src/rulekiln/pipeline/ or private worker helpers) that composes, for every final selected rule:
   - source_case_ids, cluster_id, support_count, support_ratio.
   - examples_fixed/examples_broken using associative attribution available in every run.
   - evaluator notes when present (paired-comparison notes, conflict summary, attribution metadata).
   - flags:
   - zero_validation_impact when no attributed fixed/broken evidence.
   - regression_flag when broken-attributed evidence exists.
   - Implement writer functions in src/rulekiln/artifacts/writer.py:
   - write_rule_provenance_json -> outputs/rule_provenance.json.
   - write_rule_provenance_markdown -> outputs/rule_provenance.md.
   - Integrate in src/rulekiln/workers/distillation_worker.py export stage and register in manifest entries.
   - Add artifact discoverability in src/rulekiln/ui/routes.py _KNOWN_ARTIFACT_PATTERNS.

3. Phase 3 - Leave-One-Rule-Out Ablation Engine (depends on 1; helper work parallel with 2)
   - In src/rulekiln/workers/distillation_worker.py aggregate path, after selected strategy is determined and before final artifact export:
   - Gate execution to small/smoke-style runs (case-count threshold + enable_rule_ablation flag).
   - For each selected rule candidate (deterministic order, capped):
   - Remove exactly one rule.
   - Recompile prompt.
   - Re-evaluate subset/split via existing _evaluate_prompt_strategy using unique strategy IDs per ablation.
   - Compute deltas vs full selected strategy using primary metric:
   - helpful if removing rule worsens score by > 0.005.
   - harmful if removing rule improves score by > 0.005.
   - neutral if absolute delta <= 0.005.
   - inconclusive if changed cases < 5 or evaluation failed/insufficient data.
   - Emit outputs/rule_ablation.json via new writer in src/rulekiln/artifacts/writer.py.
   - Add durable resume markers per ablated rule (artifact_type-based) to preserve idempotent retries.

4. Phase 4 - Two-Pass Prompt Budget Optimizer (depends on 2 and 3)
   - Refactor src/rulekiln/pipeline/rule_pruning.py to support ranking_mode and optional utility inputs while preserving current deterministic pruning order and hard token-budget enforcement.
   - Pass 1 (always): run existing support_count pruning/evaluation to establish baseline selected strategy and validation evidence.
   - Build per-rule utility signals:
   - Preferred: causal ablation evidence when available.
   - Fallback: associative provenance evidence.
   - Score formulas:
   - utility score = validation_fix_count - regression_penalty * validation_break_count.
   - utility_per_token score = utility score / estimated_rule_tokens.
   - Pass 2: re-prune, recompile, and re-evaluate selected distilled strategy under utility and utility_per_token.
   - Produce pruning_mode_comparison with evaluated mode rows only (no synthetic/unrun rows).
   - Respect max_prompt_tokens in every mode and pass.
   - Set final selected prompt/rules according to configured rule_pruning_mode.

5. Phase 5 - Report and Benchmark Surfacing (depends on 4)
   - Extend strategy comparison payloads produced in src/rulekiln/workers/distillation_worker.py and written by src/rulekiln/artifacts/writer.py:
   - Include pruning_mode_comparison, provenance summary flags, and ablation summary.
   - Ensure eval_report.json and strategy_comparison.json carry pruning-mode comparison blocks.
   - Extend benchmark report rendering in src/rulekiln/benchmarks/reporting.py and src/rulekiln/benchmarks/schemas.py:
   - Render a pruning-mode comparison table in summary.md when pipeline-emitted comparison data is available.
   - Keep benchmark logic as pass-through of pipeline data (no benchmark-only alternative scoring implementation).
   - If needed for ingestion, add minimal CLI plumbing in src/rulekiln/benchmarks/cli.py and src/rulekiln/benchmarks/banking77.py to accept pipeline comparison input and render it in benchmark summaries.

6. Phase 6 - Tests, Verification, and Documentation (depends on 2-5)
   - Unit tests:
   - Extend tests/unit/test_rule_pruning.py for support_count/utility/utility_per_token ordering and budget invariants.
   - Add tests/unit/test_rule_provenance_artifact.py for JSON/MD shape, flags, and attribution metadata labeling.
   - Add tests/unit/test_rule_ablation.py for leave-one-rule-out deltas and helpful/harmful/neutral/inconclusive thresholds.
   - Add tests for provenance/ablation helper builders (cluster mapping, causal-preferred merge).
   - Integration tests:
   - Extend tests/integration/test_offline_e2e_fake_providers.py artifact and manifest assertions for rule_provenance.json, rule_provenance.md, rule_ablation.json.
   - Add/extend smoke integration for ablation-enabled run idempotency and deterministic output keys.
   - Extend tests/integration/test_dbos_spike_workflow.py assertions for new ablation-related stage markers if introduced.
   - Extend tests/integration/test_benchmark_cli_smoke.py to verify pruning-mode table appears when comparison data is supplied.
   - UI artifact listing tests:
   - Extend tests/ui/test_artifacts.py for new artifact names.
   - Documentation:
   - Update README.md artifact layout and strategy comparison description.
   - Update examples/datasets/banking77/README.md snapshot/reporting section to include pruning-mode comparison semantics.

**Relevant files**
- /home/adam/Git/Rulekiln/src/rulekiln/schemas/task_case.py - task-level optimizer/ablation controls.
- /home/adam/Git/Rulekiln/src/rulekiln/config/settings.py - default thresholds and toggles.
- /home/adam/Git/Rulekiln/src/rulekiln/schemas/pipeline.py - provenance/ablation/comparison schema additions.
- /home/adam/Git/Rulekiln/src/rulekiln/pipeline/rule_pruning.py - ranking-mode logic and token-budget-safe selection.
- /home/adam/Git/Rulekiln/src/rulekiln/pipeline/failure_analysis.py - associative attribution improvements and per-rule mapping metadata.
- /home/adam/Git/Rulekiln/src/rulekiln/workers/distillation_worker.py - orchestration for provenance extraction, ablation, two-pass re-evaluation, and final mode selection.
- /home/adam/Git/Rulekiln/src/rulekiln/db/repositories/jobs.py - cluster retrieval helpers for provenance.
- /home/adam/Git/Rulekiln/src/rulekiln/artifacts/writer.py - new rule_provenance and rule_ablation writers plus payload extensions.
- /home/adam/Git/Rulekiln/src/rulekiln/ui/routes.py - artifact listing patterns for new outputs.
- /home/adam/Git/Rulekiln/src/rulekiln/benchmarks/schemas.py - pruning-mode comparison schema support.
- /home/adam/Git/Rulekiln/src/rulekiln/benchmarks/reporting.py - benchmark summary pruning-mode table rendering.
- /home/adam/Git/Rulekiln/src/rulekiln/benchmarks/banking77.py - optional pipeline-comparison pass-through to summary.
- /home/adam/Git/Rulekiln/src/rulekiln/benchmarks/cli.py - optional comparison input plumbing.
- /home/adam/Git/Rulekiln/tests/unit/test_rule_pruning.py - optimizer mode behavior and budget tests.
- /home/adam/Git/Rulekiln/tests/unit/test_rule_provenance_artifact.py - new provenance artifact tests.
- /home/adam/Git/Rulekiln/tests/unit/test_rule_ablation.py - new ablation classification tests.
- /home/adam/Git/Rulekiln/tests/integration/test_offline_e2e_fake_providers.py - artifact/manifest assertions.
- /home/adam/Git/Rulekiln/tests/integration/test_dbos_spike_workflow.py - resume/idempotency behavior with ablation markers.
- /home/adam/Git/Rulekiln/tests/integration/test_benchmark_cli_smoke.py - benchmark summary comparison checks.
- /home/adam/Git/Rulekiln/tests/ui/test_artifacts.py - UI artifact list visibility for new files.
- /home/adam/Git/Rulekiln/README.md - artifact layout and report semantics.
- /home/adam/Git/Rulekiln/examples/datasets/banking77/README.md - benchmark summary documentation updates.

**Verification**
1. Run targeted unit tests for pruning/provenance/ablation builders and writers.
2. Run targeted integration tests for offline e2e artifacts, DBOS idempotency, and benchmark smoke reporting.
3. Run uv run ruff check src/ tests/.
4. Run uv run pyright.
5. Run DATABASE_URL="sqlite+aiosqlite://" MLFLOW_TRACKING_URI="file:///tmp/mlflow-ci" uv run pytest -m "not external" --tb=short -q.
6. Manual artifact validation from one smoke-like run:
   - Confirm outputs/rule_provenance.json and outputs/rule_provenance.md exist and every final rule has source tracing + flags.
   - Confirm outputs/rule_ablation.json exists when ablation is enabled and includes helpful/harmful/neutral/inconclusive labels.
   - Confirm strategy_comparison.json/eval_report.json include pruning_mode_comparison with evaluated rows only.
   - Confirm prompt token budget never exceeded in any pruning mode.

**Decisions**
- Provenance attribution: hybrid.
- Always emit associative attribution fields; when ablation exists, add causal fields and prefer them in markdown/report summaries.
- Do not present associative attribution as causal; label method explicitly.
- Ablation thresholds:
- helpful if metric_delta_without_rule < -0.005.
- harmful if metric_delta_without_rule > 0.005.
- neutral if abs(delta) <= 0.005.
- inconclusive if changed_cases < 5 or data is insufficient.
- Primary metric for ablation classification: macro_f1 when available, accuracy fallback otherwise.
- Optimizer execution model: two-pass.
- Pass 1 support_count evaluation to generate validation evidence.
- Pass 2 re-prune/re-evaluate utility and utility_per_token.
- Do not mark non-re-evaluated modes as evaluated.
- Reporting scope: both worker outputs and BANKING77 benchmark summary, with benchmark reporting as pass-through of pipeline-emitted comparison data.

**Scope boundaries**
- Included:
- New artifacts: rule_provenance.json, rule_provenance.md, rule_ablation.json.
- Configurable pruning modes and two-pass evaluation-backed optimization.
- Benchmark summary pruning-mode comparison rendering when pipeline comparison data is supplied.
- Excluded:
- No retroactive backfill for historical runs.
- No new public API endpoints for provenance/ablation in this phase.
- No provider or model-API contract changes.

**Further Considerations**
1. If two-pass cost is too high on large jobs, add a hard cap for utility-mode pass-2 evaluations (for example, selected strategy only, never baseline variants), while still preserving smoke/small full coverage.
2. Keep stage-marker naming deterministic for ablation variants to avoid duplicate evaluations during resume/retry paths.