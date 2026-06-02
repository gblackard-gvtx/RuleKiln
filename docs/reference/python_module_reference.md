# Python Module Reference

This document describes each runtime Python module in RuleKiln, including what its models/classes represent and what its top-level functions are for.

Scope covered: `src/rulekiln/**`, `main.py`, and `migrations/**` (tests excluded).

## `main.py`

- **Module purpose:** CLI-style entrypoint that launches the RuleKiln API server.

- **Models / classes:** none.

- **Functions:**

  - `main()`: Process entrypoint that starts the configured application runtime.



## `migrations/__init__.py`

- **Module purpose:** Package marker module.

- **Models / classes:** none.

- **Functions:** none.



## `migrations/env.py`

- **Module purpose:** Alembic environment wiring for online/offline migration execution.

- **Models / classes:** none.

- **Functions:**

  - `run_migrations_offline()`: Configures Alembic to generate SQL migration scripts without opening a live database connection.

  - `run_migrations_online()`: Creates a database connection and applies migrations directly against the target database.



## `migrations/versions/0001_mvp_schema.py`

- **Module purpose:** Alembic revision that creates the initial MVP database schema.

- **Models / classes:** none.

- **Functions:**

  - `upgrade()`: Alembic migration step to upgrade schema changes for this revision.

  - `downgrade()`: Alembic migration step to downgrade schema changes for this revision.



## `migrations/versions/0002_add_selected_strategy.py`

- **Module purpose:** Alembic revision that adds selected-strategy persistence fields.

- **Models / classes:** none.

- **Functions:**

  - `upgrade()`: Alembic migration step to upgrade schema changes for this revision.

  - `downgrade()`: Alembic migration step to downgrade schema changes for this revision.



## `migrations/versions/0003_add_queue_and_rule_columns.py`

- **Module purpose:** Alembic revision that introduces queue/lease columns and additional rule metadata columns.

- **Models / classes:** none.

- **Functions:**

  - `upgrade()`: Alembic migration step to upgrade schema changes for this revision.

  - `downgrade()`: Alembic migration step to downgrade schema changes for this revision.



## `migrations/versions/0004_add_model_call_events.py`

- **Module purpose:** Alembic revision that adds model call event persistence and usage/cost summary columns.

- **Models / classes:** none.

- **Functions:**

  - `upgrade()`: Alembic migration step to upgrade schema changes for this revision.

  - `downgrade()`: Alembic migration step to downgrade schema changes for this revision.



## `migrations/versions/0005_add_eval_case_results_and_idempotency_keys.py`

- **Module purpose:** Alembic revision that adds durable per-case evaluation rows and model-call idempotency-key indexing.

- **Models / classes:** none.

- **Functions:**

  - `upgrade()`: Alembic migration step to upgrade schema changes for this revision.

  - `downgrade()`: Alembic migration step to downgrade schema changes for this revision.



## `src/rulekiln/__init__.py`

- **Module purpose:** Package marker module.

- **Models / classes:** none.

- **Functions:** none.



## `src/rulekiln/agents/__init__.py`

- **Module purpose:** Package marker module.

- **Models / classes:** none.

- **Functions:** none.



## `src/rulekiln/agents/rule_conflict_review.py`

- **Module purpose:** Static rule review agent — checks each synthesized rule for internal linguistic contradictions before evaluation. This is a pre-evaluation hygiene pass, not the paper's Phase 3 closed-loop conflict resolution.

- **Models / classes:** none.

- **Functions:**

  - `_build_conflict_review_prompt()`: Builds the static review prompt template used for model calls.

  - `async review_rule_for_conflicts()`: Static rule review: call the teacher to check a synthesized rule for linguistic conflicts. Performs no student inference and does not inspect case outcomes.



## `src/rulekiln/agents/rule_extraction.py`

- **Module purpose:** Rule-extraction agent logic that derives micro-rules from individual labeled cases.

- **Models / classes:** none.

- **Functions:**

  - `_build_extraction_prompt()`: Builds the extraction prompt template used for model calls.

  - `async extract_rules_for_case()`: Call the teacher model to extract micro-rules for a single case.



## `src/rulekiln/agents/rule_synthesis.py`

- **Module purpose:** Rule-synthesis agent logic that merges clustered micro-rules into consolidated synthesized rules.

- **Models / classes:** none.

- **Functions:**

  - `_build_synthesis_prompt()`: Builds the synthesis prompt template used for model calls.

  - `async synthesize_cluster()`: Call the teacher model to synthesize a cluster of micro-rules into one rule.



## `src/rulekiln/benchmarks/cli.py`

- **Module purpose:** CLI entrypoint (`rulekiln-benchmark`) for reproducible benchmark workflows.

- **Models / classes:** none.

- **Functions:**

  - `main()`: CLI main entrypoint. Subcommands: `banking77` (BANKING77 benchmark) and `refinement-ablation` (loop ON vs OFF comparison).

  - `_run_ablation_subcommand()`: Runs the `refinement-ablation` subcommand: reads eval artifacts from two pipeline runs and writes `refinement_ablation.json`.

  - `_count_refinement_iterations()`: Count completed refinement iteration artifacts for a given run directory.



## `src/rulekiln/benchmarks/refinement_ablation.py`

- **Module purpose:** Builds and writes `RefinementAblationArtifact` by comparing two pipeline eval results (loop ON vs loop OFF).

- **Models / classes:** none.

- **Functions:**

  - `build_refinement_ablation()`: Construct a `RefinementAblationArtifact` from two `EvalResult` objects and metadata.

  - `write_refinement_ablation_json()`: Write a `RefinementAblationArtifact` to disk.

  - `load_eval_result_from_artifact()`: Load an `EvalResult` from a strategy eval JSON artifact directory.



## `src/rulekiln/api/__init__.py`

- **Module purpose:** Package marker module.

- **Models / classes:** none.

- **Functions:** none.



## `src/rulekiln/api/app.py`

- **Module purpose:** FastAPI application factory that composes lifespan hooks, API routes, UI routes, and error handlers.

- **Models / classes:** none.

- **Functions:**

  - `create_app()`: Creates and returns the fully configured FastAPI application instance (lifespan, routes, and exception handlers).



## `src/rulekiln/api/errors.py`

- **Module purpose:** Exception handling and error response formatting for FastAPI endpoints.

- **Models / classes:**

  - `ErrorResponse` (model) _(bases: BaseModel)_: Pydantic model defining the standardized error payload returned by API exception handlers.

- **Functions:**

  - `register_exception_handlers()`: Attach global exception handlers to the FastAPI app.



## `src/rulekiln/api/lifespan.py`

- **Module purpose:** Application startup/shutdown lifecycle management, including config/provider validation.

- **Models / classes:** none.

- **Functions:**

  - `async lifespan()`: Validate settings and provider profiles on startup.



## `src/rulekiln/api/routes/__init__.py`

- **Module purpose:** Package marker module.

- **Models / classes:** none.

- **Functions:** none.



## `src/rulekiln/api/routes/distillation_jobs.py`

- **Module purpose:** FastAPI route handlers for distillation job lifecycle operations (create and status retrieval).

- **Models / classes:** none.

- **Functions:**

  - `async create_distillation_job()`: Creates a new distillation job from the request payload and triggers asynchronous execution/enqueueing.

  - `async get_distillation_job()`: Retrieves distillation job status and progress by job ID for API clients.



## `src/rulekiln/api/routes/distillation_outputs.py`

- **Module purpose:** FastAPI route handlers for retrieving completed job outputs (prompt, rules, and evaluation report).

- **Models / classes:**

  - `PromptResponse` (model) _(bases: BaseModel)_: Pydantic model that structures validated data exchanged in this module.

  - `RulesResponse` (model) _(bases: BaseModel)_: Pydantic model that structures validated data exchanged in this module.

  - `EvalReportResponse` (model) _(bases: BaseModel)_: Pydantic model that structures validated data exchanged in this module.

- **Functions:**

  - `async get_job_prompt()`: Returns job prompt for downstream use.

  - `async get_job_rules()`: Returns job rules for downstream use.

  - `async get_job_eval_report()`: Returns job eval report for downstream use.

  - `_assert_job_exists_and_done()`: Internal guard that verifies the requested job exists and is completed before output retrieval.



## `src/rulekiln/api/validators/__init__.py`

- **Module purpose:** Package marker module.

- **Models / classes:** none.

- **Functions:** none.



## `src/rulekiln/api/validators/request_shape.py`

- **Module purpose:** Post-schema validation rules for distillation requests (provider/profile and case-list constraints).

- **Models / classes:**

  - `RequestValidationError` (class) _(bases: ValueError)_: Custom exception type used for explicit failure signaling.

- **Functions:**

  - `validate_distillation_request()`: Validate provider profiles and case list beyond Pydantic schema checks.



## `src/rulekiln/artifacts/__init__.py`

- **Module purpose:** Package marker module.

- **Models / classes:** none.

- **Functions:** none.



## `src/rulekiln/artifacts/settings_snapshot.py`

- **Module purpose:** Builds and writes redacted runtime-settings snapshots for job artifacts.

- **Models / classes:** none.

- **Functions:**

  - `_redact_value()`: Internal helper function supporting the module implementation.

  - `_redact_dict()`: Internal helper function supporting the module implementation.

  - `build_settings_snapshot()`: Build a redacted snapshot of application settings for artifact export.

  - `write_settings_snapshot()`: Writes settings snapshot output to disk artifacts.



## `src/rulekiln/artifacts/writer.py`

- **Module purpose:** Filesystem artifact writer for task inputs, pipeline outputs, exports, and metadata manifests.

- **Models / classes:** none.

- **Functions:**

  - `job_artifact_root()`: Returns the root filesystem path where artifacts for a specific job are stored.

  - `write_task()`: Writes task output to disk artifacts.

  - `write_cases_normalized()`: Writes cases normalized output to disk artifacts.

  - `write_prompt()`: Writes prompt output to disk artifacts.

  - `write_selected_prompt()`: Writes selected prompt output to disk artifacts.

  - `write_rules()`: Writes rules output to disk artifacts.

  - `write_eval_report()`: Writes eval report output to disk artifacts.

  - `write_strategy_comparison()`: Writes strategy comparison output to disk artifacts.

  - `write_failure_jsonl()`: Writes failure jsonl output to disk artifacts.

  - `write_promptfoo_yaml()`: Writes promptfoo yaml output to disk artifacts.

  - `write_mlflow_run_id()`: Writes mlflow run id output to disk artifacts.

  - `write_manifest()`: Writes manifest output to disk artifacts.



## `src/rulekiln/config/__init__.py`

- **Module purpose:** Package marker module.

- **Models / classes:** none.

- **Functions:** none.



## `src/rulekiln/config/settings.py`

- **Module purpose:** Pydantic settings models and cached settings loading from environment variables/.env.

- **Models / classes:**

  - `ProviderProfile` (model) _(bases: BaseModel)_: Configuration for a named provider profile.

  - `QualityGateDefaults` (model) _(bases: BaseModel)_: Default quality gate thresholds; may be overridden per task.

  - `AppSettings` (class) _(bases: BaseSettings)_: Application-wide settings loaded from environment / .env file.

- **Functions:**

  - `get_settings()`: Return a cached singleton AppSettings instance.



## `src/rulekiln/db/__init__.py`

- **Module purpose:** Package marker module.

- **Models / classes:** none.

- **Functions:** none.



## `src/rulekiln/db/models.py`

- **Module purpose:** SQLAlchemy ORM model definitions for jobs, cases, rules, prompts, evaluations, and stage markers.

- **Models / classes:**

  - `Base` (model) _(bases: DeclarativeBase)_: SQLAlchemy ORM model used for database persistence.

  - `DistillationJob` (model) _(bases: Base)_: SQLAlchemy ORM model used for database persistence.

  - `Case` (model) _(bases: Base)_: SQLAlchemy ORM model used for database persistence.

  - `MicroRule` (model) _(bases: Base)_: SQLAlchemy ORM model used for database persistence.

  - `RuleCluster` (model) _(bases: Base)_: SQLAlchemy ORM model used for database persistence.

  - `SynthesizedRule` (model) _(bases: Base)_: SQLAlchemy ORM model used for database persistence.

  - `PromptVersion` (model) _(bases: Base)_: SQLAlchemy ORM model used for database persistence.

  - `EvalRun` (model) _(bases: Base)_: SQLAlchemy ORM model used for database persistence.

  - `EvalCaseResultRecord` (model) _(bases: Base)_: Durable per-case evaluation row used for resumable baseline/distilled evaluation.

  - `StageMarker` (model) _(bases: Base)_: Durable stage completion marker for idempotent resume semantics.

  - `ModelCallEvent` (model) _(bases: Base)_: Persisted model API event row including usage, cost, and idempotency metadata.

- **Functions:**

  - `_uuid()`: Internal helper function supporting the module implementation.



## `src/rulekiln/db/repositories/__init__.py`

- **Module purpose:** Package marker module.

- **Models / classes:** none.

- **Functions:** none.



## `src/rulekiln/db/repositories/eval_case_results.py`

- **Module purpose:** Async repository functions for idempotent per-case evaluation persistence and resume lookups.

- **Models / classes:**

  - `EvalCaseResultUpsert` (model) _(bases: BaseModel)_: Payload model used to insert or update one durable case-evaluation row.

- **Functions:**

  - `async upsert_eval_case_result()`: Insert or update one case-level evaluation row keyed by job/student/strategy/split/case.

  - `async get_eval_case_results()`: Load persisted per-case evaluation rows for one job/student/strategy/split tuple.

  - `async get_completed_eval_case_ids()`: Return completed case IDs to support resume-safe evaluation skipping.



## `src/rulekiln/db/repositories/jobs.py`

- **Module purpose:** Async repository functions for persisting and querying distillation pipeline/job state.

- **Models / classes:** none.

- **Functions:**

  - `async create_job()`: Creates job as part of the module workflow.

  - `async get_job()`: Returns job for downstream use.

  - `async list_recent_jobs()`: Return the most recent jobs ordered by created_at descending.

  - `async update_job_status()`: Updates job status in storage or runtime state.

  - `async set_mlflow_run_id()`: Updates the job record with the associated MLflow run ID.

  - `async bulk_insert_cases()`: Persists cases records.

  - `async get_cases_for_job()`: Returns cases for job for downstream use.

  - `async mark_stage_complete()`: Marks stage complete state transitions in persistence.

  - `async is_stage_complete()`: Checks whether stage complete is true.

  - `async bulk_insert_micro_rules()`: Persists micro rules records.

  - `async get_micro_rules_for_job()`: Returns micro rules for job for downstream use.

  - `async bulk_insert_rule_clusters()`: Persists rule clusters records.

  - `async bulk_insert_synthesized_rules()`: Persists synthesized rules records.

  - `async get_synthesized_rules_for_job()`: Returns synthesized rules for job for downstream use.

  - `async insert_prompt_version()`: Persists prompt version records.

  - `async get_selected_prompt_version()`: Returns selected prompt version for downstream use.

  - `async mark_prompt_version_selected()`: Marks prompt version selected state transitions in persistence.

  - `async insert_eval_run()`: Persists eval run records.

  - `async get_eval_runs_for_job()`: Returns eval runs for job for downstream use.

  - `async update_synthesized_rule_conflict()`: Updates synthesized rule conflict in storage or runtime state.

  - `async update_synthesized_rule_pruning()`: Updates synthesized rule pruning in storage or runtime state.

  - `async get_selected_synthesized_rules_for_job()`: Return only non-pruned synthesized rules for a given strategy.

  - `async claim_next_job()`: Claim the next pending job using FOR UPDATE SKIP LOCKED.

  - `async renew_lease()`: Extend the lease on a running job owned by worker_id.

  - `async complete_job()`: Mark a job as completed in the queue.

  - `async fail_job()`: Mark a job as permanently failed in the queue.

  - `async apply_job_failure_policy()`: Apply retry classification and transition to waiting-for-retry or terminal-failure states.

  - `async recover_expired_leases()`: Reset expired-lease jobs back to pending or fail them if max_attempts exceeded.



## `src/rulekiln/db/repositories/model_calls.py`

- **Module purpose:** Async repository functions for model-call event persistence, dedupe, and job usage summary updates.

- **Models / classes:** none.

- **Functions:**

  - `async bulk_insert_model_call_events()`: Persist model-call events with in-memory and DB-level idempotency-key dedupe.

  - `async update_job_usage_totals()`: Write aggregated token/cost usage totals back to a distillation job row.



## `src/rulekiln/db/session.py`

- **Module purpose:** Async SQLAlchemy engine/session factory management and FastAPI DB-session dependency.

- **Models / classes:** none.

- **Functions:**

  - `get_engine()`: Return the async engine, creating it on first call.

  - `get_session_factory()`: Return the session factory, creating it on first call.

  - `async get_db_session()`: FastAPI dependency that yields a database session.

  - `override_session_factory()`: Replace the global session factory (for testing only).



## `src/rulekiln/integrations/__init__.py`

- **Module purpose:** Package marker module.

- **Models / classes:** none.

- **Functions:** none.



## `src/rulekiln/integrations/mlflow_tracker.py`

- **Module purpose:** MLflow integration adapter for run creation, parameter/metric logging, and artifact uploads.

- **Models / classes:** none.

- **Functions:**

  - `_get_mlflow()`: Import mlflow lazily; raise a clear error if not installed.

  - `create_run()`: Create an MLflow run and return its run_id.

  - `log_params()`: Log key-value params to an existing run (batched).

  - `log_metrics()`: Log numeric metrics to an existing run.

  - `log_artifacts_dir()`: Upload an entire local directory to the MLflow run's artifact store.

  - `log_prompt_to_registry()`: Optionally log a prompt to the MLflow Prompt Registry (≥3.5).

  - `build_run_params()`: Builds run params derived from inputs.

  - `build_run_metrics()`: Builds run metrics derived from inputs.

  - `build_provider_params()`: Build per-role provider params dict suitable for mlflow.log_params.



## `src/rulekiln/observability/__init__.py`

- **Module purpose:** Package marker module.

- **Models / classes:** none.

- **Functions:** none.



## `src/rulekiln/observability/events.py`

- **Module purpose:** Structured observability events for stage timing, model-call telemetry, retries, and token budgets.

- **Models / classes:** none.

- **Functions:**

  - `async stage_timing()`: Context manager that logs start/end of a pipeline stage with elapsed time.

  - `log_model_call()`: Log a model invocation with token usage.

  - `log_retry()`: Log a retry attempt within a stage.

  - `log_token_budget()`: Log token budget status for a compiled prompt.



## `src/rulekiln/observability/logging.py`

- **Module purpose:** Structlog configuration and logger retrieval utilities.

- **Models / classes:** none.

- **Functions:**

  - `configure_logging()`: Configure structlog for structured JSON output in production, pretty output locally.

  - `get_logger()`: Return a named structlog logger.



## `src/rulekiln/observability/security.py`

- **Module purpose:** Security masking helpers for URLs and dictionary values in logs/artifacts.

- **Models / classes:** none.

- **Functions:**

  - `mask_url()`: Replace credential portions of a URL with masked placeholders.

  - `mask_dict_values()`: Return a copy of *data* with secret-looking values replaced by '***REDACTED***'.



## `src/rulekiln/pipeline/__init__.py`

- **Module purpose:** Package marker module.

- **Models / classes:** none.

- **Functions:** none.



## `src/rulekiln/pipeline/clustering.py`

- **Module purpose:** Rule-embedding clustering stage implementing DBSCAN and HDBSCAN grouping strategies.

- **Models / classes:** none.

- **Functions:**

  - `_cosine_distance_matrix()`: Internal helper function supporting the module implementation.

  - `_build_clusters()`: Internal helper function supporting the module implementation.

  - `cluster_dbscan()`: Cluster rules using DBSCAN on cosine distance.

  - `cluster_hdbscan()`: Cluster rules using HDBSCAN on cosine distance.



## `src/rulekiln/pipeline/evaluator.py`

- **Module purpose:** Prompt evaluation stage that scores student-model outputs against case assertions and rubric criteria.

- **Models / classes:**

  - `_StudentOutputSchema` (model) _(bases: BaseModel)_: Internal schema wrapper used to parse arbitrary structured outputs returned by the student model.

- **Functions:**

  - `get_primary_metric()`: Returns primary metric for downstream use.

  - `async _call_student()`: Return (parsed_output, is_malformed).

  - `_score_assertion()`: Score a single assertion: 1.0 pass, 0.0 fail.

  - `_score_case()`: Internal helper function supporting the module implementation.

  - `_compute_metrics()`: Internal helper function supporting the module implementation.

  - `async evaluate_prompt()`: Run the student model against every case and return aggregate metrics.



## `src/rulekiln/pipeline/failure_analysis.py`

- **Module purpose:** Failure-analysis stage that classifies per-case outcome changes and maps failures to rule IDs via outcome_conditions. Produces structured failure records used by the closed-loop refinement controller and provenance builder.

- **Models / classes:**

  - `FailureAnalysisResult` (class): Holds categorized case lists and structured failure records. Key methods: `violated_rule_summary()`, `build_utility_signals()`, `unattributed_fraction()`.

- **Functions:**

  - `analyze_failures()`: Compare baseline vs distilled per-case results; maps failed assertion_i keys to violated rule IDs via case assertion definitions and outcome_conditions. Accepts optional `cases` for real attribution.

  - `_build_outcome_to_rule_ids()`: Build outcome_label → [rule_id] index from rule.outcome_conditions.

  - `_add_structured_failure()`: Build a CaseEvaluationFailure with violated_rule_ids, matched_rule_ids, failed_assertion_types, and UNATTRIBUTED_RULE_ID sentinel.

  - `_case_entry()`: Internal helper function supporting the module implementation.



## `src/rulekiln/pipeline/prompt_compiler.py`

- **Module purpose:** Prompt-compilation stage that renders deterministic policy prompts from synthesized rules.

- **Models / classes:** none.

- **Functions:**

  - `_render_rule()`: Internal helper function supporting the module implementation.

  - `compile_prompt()`: Compile a deterministic system prompt from synthesized rules.

  - `count_tokens_approx()`: Returns an approximate token count using a conservative character-to-token ratio.



## `src/rulekiln/pipeline/quality_gates.py`

- **Module purpose:** Quality-gate stage that enforces threshold checks before a strategy can be selected.

- **Models / classes:** none.

- **Functions:**

  - `_get_threshold()`: Resolves the effective threshold by precedence order: per-task override, then environment/default settings, then hardcoded fallback.

  - `check_quality_gates()`: Validates evaluation outcomes against configured quality-gate thresholds and records gate decisions.



## `src/rulekiln/pipeline/rule_pruning.py`

- **Module purpose:** Rule-pruning stage that filters/sorts synthesized rules to meet support and token-budget constraints.

- **Models / classes:**

  - `PruningRecord` (class): Records the pruning decision for a single rule.

  - `PruningResult` (class): Output of the rule pruning service.

- **Functions:**

  - `prune_rules()`: Apply the full pruning pipeline to a list of synthesized rules.



## `src/rulekiln/pipeline/rule_refinement.py`

- **Module purpose:** Closed-loop conflict resolution (paper Phase 3, §3.3) — empirical teacher interaction that revises rules based on observed failures and successes. Distinct from the static `review_rule_for_conflicts` check: this module runs after evaluation and uses case outcomes.

- **Models / classes:**

  - `RevisedRuleEntry` (model) _(bases: BaseModel)_: A revised rule with its replacement rule ID and rationale.

  - `RefinementResult` (model) _(bases: BaseModel)_: Output from one refinement teacher call. Contains `revised_rules` list and `schema_version = "rulekiln.refinement_result.v1"`.

- **Functions:**

  - `_build_refinement_prompt()`: Builds the teacher prompt containing implicated rules, failure cases, and success cases.

  - `async refine_rules_with_teacher()`: Call the teacher to diagnose root causes and emit revised rules. Accepts failure/success cases from a FailureAnalysisResult; deterministic given seed. Works offline with FakeChatClient.

  - `apply_refinements()`: Replace rules by ID with revised versions; preserve rule IDs; leave unaffected rules unchanged.



## `src/rulekiln/pipeline/strategy_selection.py`

- **Module purpose:** Strategy-selection stage that compares candidate strategies and picks the final winner.

- **Models / classes:** none.

- **Functions:**

  - `_primary_score()`: Internal helper function supporting the module implementation.

  - `select_strategy()`: Selects the winning strategy using primary metric ranking and deterministic tie-break logic, returning both strategy and rationale.

  - `build_strategy_comparison()`: Builds strategy comparison derived from inputs.



## `src/rulekiln/providers/__init__.py`

- **Module purpose:** Package marker module.

- **Models / classes:** none.

- **Functions:** none.



## `src/rulekiln/providers/chat/__init__.py`

- **Module purpose:** Chat client resolver and optional rate-limited wrapper for configured provider profiles.

- **Models / classes:**

  - `_RateLimitedChatClient` (class) _(bases: ChatModelClient)_: Internal wrapper that adds provider rate limiting around any ChatModelClient implementation.

- **Functions:**

  - `get_chat_client()`: Return the ChatModelClient implementation for the given provider config.



## `src/rulekiln/providers/chat/anthropic_chat.py`

- **Module purpose:** Anthropic Claude chat adapter implementation.

- **Models / classes:**

  - `AnthropicChatClient` (class) _(bases: ChatModelClient)_: Chat adapter for Anthropic Claude models.

- **Functions:** none.



## `src/rulekiln/providers/chat/bedrock_chat.py`

- **Module purpose:** AWS Bedrock chat adapter implementation.

- **Models / classes:**

  - `BedrockChatClient` (class) _(bases: ChatModelClient)_: Chat adapter for AWS Bedrock via pydantic-ai.

- **Functions:** none.



## `src/rulekiln/providers/chat/fake.py`

- **Module purpose:** Deterministic fake chat adapter used for tests/offline execution.

- **Models / classes:**

  - `FakeChatClient` (class) _(bases: ChatModelClient)_: Returns deterministic stub responses populated from output_schema defaults.

- **Functions:** none.



## `src/rulekiln/providers/chat/openai_chat.py`

- **Module purpose:** OpenAI chat adapter implementation.

- **Models / classes:**

  - `OpenAIChatClient` (class) _(bases: ChatModelClient)_: Chat adapter for OpenAI models via pydantic-ai.

- **Functions:** none.



## `src/rulekiln/providers/chat/openai_compatible_chat.py`

- **Module purpose:** OpenAI-compatible chat adapter for custom base-URL endpoints.

- **Models / classes:**

  - `OpenAICompatibleChatClient` (class) _(bases: ChatModelClient)_: Chat adapter for OpenAI-compatible endpoints (custom base_url).

- **Functions:** none.



## `src/rulekiln/providers/chat/stubs.py`

- **Module purpose:** Stub chat adapters for provider types not yet implemented.

- **Models / classes:**

  - `VertexGeminiChatClient` (class) _(bases: ChatModelClient)_: Class used to organize related state and behavior in this module.

  - `AzureOpenAIChatClient` (class) _(bases: ChatModelClient)_: Class used to organize related state and behavior in this module.

  - `CustomChatClient` (class) _(bases: ChatModelClient)_: Class used to organize related state and behavior in this module.

- **Functions:** none.



## `src/rulekiln/providers/contracts.py`

- **Module purpose:** Provider contracts, config models, and typed interfaces for chat/embedding adapters.

- **Models / classes:**

  - `ProviderConfig` (model) _(bases: BaseModel)_: Resolved provider configuration for a single model call.

  - `ProviderNotImplementedError` (class) _(bases: NotImplementedError)_: Raised when a provider adapter is a stub and not yet implemented.

  - `ProviderNotConfiguredError` (class) _(bases: ValueError)_: Raised when a provider is configured but missing required runtime credentials.

  - `ChatModelClient` (class) _(bases: ABC)_: Abstract base for chat / completion providers.

  - `EmbeddingClient` (class) _(bases: ABC)_: Abstract base for text embedding providers.

- **Functions:** none.



## `src/rulekiln/providers/embedding/__init__.py`

- **Module purpose:** Embedding client resolver and optional rate-limited wrapper for configured provider profiles.

- **Models / classes:**

  - `_RateLimitedEmbeddingClient` (class) _(bases: EmbeddingClient)_: Internal wrapper that adds provider rate limiting around any EmbeddingClient implementation.

- **Functions:**

  - `get_embedding_client()`: Return the EmbeddingClient implementation for the given provider config.



## `src/rulekiln/providers/embedding/bedrock_embedding.py`

- **Module purpose:** AWS Bedrock embedding adapter implementation.

- **Models / classes:**

  - `BedrockEmbeddingClient` (class) _(bases: EmbeddingClient)_: Embedding adapter for AWS Bedrock Titan/Cohere embedding models.

- **Functions:** none.



## `src/rulekiln/providers/embedding/fake.py`

- **Module purpose:** Deterministic fake embedding adapter used for tests/offline execution.

- **Models / classes:**

  - `FakeEmbeddingClient` (class) _(bases: EmbeddingClient)_: Returns deterministic pseudo-embeddings derived from text hashes.

- **Functions:** none.



## `src/rulekiln/providers/embedding/openai_compatible_embedding.py`

- **Module purpose:** OpenAI-compatible embedding adapter for custom base-URL endpoints.

- **Models / classes:**

  - `OpenAICompatibleEmbeddingClient` (class) _(bases: EmbeddingClient)_: Embedding adapter for any OpenAI-compatible endpoint.

- **Functions:** none.



## `src/rulekiln/providers/embedding/openai_embedding.py`

- **Module purpose:** OpenAI embedding adapter implementation.

- **Models / classes:**

  - `OpenAIEmbeddingClient` (class) _(bases: EmbeddingClient)_: Embedding adapter for OpenAI text-embedding models.

- **Functions:** none.



## `src/rulekiln/providers/embedding/stubs.py`

- **Module purpose:** Stub embedding adapters for provider types not yet implemented.

- **Models / classes:**

  - `AnthropicEmbeddingClient` (class) _(bases: EmbeddingClient)_: Class used to organize related state and behavior in this module.

  - `VertexGeminiEmbeddingClient` (class) _(bases: EmbeddingClient)_: Class used to organize related state and behavior in this module.

  - `AzureOpenAIEmbeddingClient` (class) _(bases: EmbeddingClient)_: Class used to organize related state and behavior in this module.

  - `CustomEmbeddingClient` (class) _(bases: EmbeddingClient)_: Class used to organize related state and behavior in this module.

- **Functions:** none.



## `src/rulekiln/providers/rate_limiter.py`

- **Module purpose:** Sliding-window provider rate limiting (RPM/TPM) and concurrency control state management.

- **Models / classes:**

  - `_RpmWindow` (class): Internal sliding-window requests-per-minute tracker.

  - `_LimiterState` (class): Internal state container for per-provider limiter counters and synchronization primitives.

  - `ProviderRateLimiter` (class): Manages per-ProviderConfig rate limiter state.

- **Functions:**

  - `get_rate_limiter()`: Return the process-global ProviderRateLimiter, creating it on first call.



## `src/rulekiln/providers/resolver.py`

- **Module purpose:** Provider-profile resolution from settings into concrete provider configuration objects.

- **Models / classes:** none.

- **Functions:**

  - `normalize_profile_name()`: Normalize profile name to lowercase with underscores.

  - `resolve_provider_config()`: Resolve a named provider profile and model into a ProviderConfig.



## `src/rulekiln/schemas/__init__.py`

- **Module purpose:** Package marker module.

- **Models / classes:** none.

- **Functions:** none.



## `src/rulekiln/schemas/job.py`

- **Module purpose:** API-facing job request/response schemas for distillation job creation and status reporting.

- **Models / classes:**

  - `DistillationRequest` (model) _(bases: BaseModel)_: Strict canonical envelope for a distillation job submission.

  - `JobProgress` (model) _(bases: BaseModel)_: Pydantic model that structures validated data exchanged in this module.

  - `JobStatusResponse` (model) _(bases: BaseModel)_: Pydantic model that structures validated data exchanged in this module.

  - `CreateJobResponse` (model) _(bases: BaseModel)_: Pydantic model that structures validated data exchanged in this module.

- **Functions:** none.



## `src/rulekiln/schemas/pipeline.py`

- **Module purpose:** Pipeline data schemas for rules, clusters, evaluation results, failures, strategy comparison, refinement artifacts, and ablation artifacts.

- **Models / classes:**

  - `MicroRuleSchema` (model) _(bases: BaseModel)_: A single rule extracted from a teacher-model case response.

  - `ExtractionOutput` (model) _(bases: BaseModel)_: Structured output from the rule-extraction agent.

  - `OutcomeCondition` (model) _(bases: BaseModel)_: A single outcome condition within a synthesized rule; `outcome` holds the expected label string.

  - `SynthesizedRuleSchema` (model) _(bases: BaseModel)_: A synthesized rule derived from a cluster of micro-rules. `outcome_conditions: dict[str, OutcomeCondition]` keys on outcome label.

  - `SynthesisOutput` (model) _(bases: BaseModel)_: Structured output from the rule-synthesis agent.

  - `RuleConflictReview` (model) _(bases: BaseModel)_: Static conflict review result for a single synthesized rule.

  - `RuleClusterSchema` (model) _(bases: BaseModel)_: A cluster of micro-rule IDs produced by a clustering algorithm.

  - `CaseEvalResult` (model) _(bases: BaseModel)_: Evaluation result for a single case. `assertion_scores` keys are `assertion_{i}` (0-based evaluator index).

  - `EvalResult` (model) _(bases: BaseModel)_: Aggregate evaluation result for a prompt version on a split.

  - `CaseEvaluationFailure` (model) _(bases: BaseModel)_: Granular failure record with `violated_rule_ids`, `matched_rule_ids`, `failed_assertion_types`, and `failed_assertion_paths`.

  - `QualityGateResult` (model) _(bases: BaseModel)_: Result of a quality gate check for one strategy.

  - `StrategyComparison` (model) _(bases: BaseModel)_: Full comparison across strategies after evaluation and gate checks.

  - `RefinementIterationArtifact` (model) _(bases: BaseModel)_: Per-iteration artifact from the closed-loop refinement controller. `schema_version = "rulekiln.refinement_iteration.v1"`. Written to `outputs/refinement_iter_{n}.json`.

  - `RefinementAblationRow` (model) _(bases: BaseModel)_: One arm of the refinement ablation (loop_on or loop_off) with macro_f1, regression rate, token count, teacher cost.

  - `RefinementAblationArtifact` (model) _(bases: BaseModel)_: Loop ON vs OFF comparison artifact. `schema_version = "rulekiln.refinement_ablation.v1"`. Written to `refinement_ablation.json`.

- **Functions:** none.



## `src/rulekiln/schemas/task_case.py`

- **Module purpose:** Task/case schemas defining routes, evaluation expectations, and case payload structure.

- **Models / classes:**

  - `ModelRoute` (model) _(bases: BaseModel)_: A provider profile + model pair for a specific role.

  - `RuleKilnTask` (model) _(bases: BaseModel)_: Reusable task definition. Includes rule pruning config (`rule_pruning_mode`, `max_rules`, …), ablation config (`enable_rule_ablation`, …), and closed-loop refinement config (`enable_refinement_loop`, `refinement_max_iterations`, `refinement_epsilon`, `refinement_seed`, `refinement_max_failure_cases`, `refinement_max_success_cases`).

  - `EvaluationAssertion` (model) _(bases: BaseModel)_: Pydantic model that structures validated data exchanged in this module.

  - `RubricCriterion` (model) _(bases: BaseModel)_: Pydantic model that structures validated data exchanged in this module.

  - `EvaluationSpec` (model) _(bases: BaseModel)_: Pydantic model that structures validated data exchanged in this module.

  - `RuleKilnCase` (model) _(bases: BaseModel)_: A single training or evaluation case.

- **Functions:** none.



## `src/rulekiln/ui/__init__.py`

- **Module purpose:** Package marker module.

- **Models / classes:** none.

- **Functions:** none.



## `src/rulekiln/ui/forms.py`

- **Module purpose:** Multipart form parsing/validation helpers for the server-rendered job-creation UI.

- **Models / classes:**

  - `NewJobForm` (class): Dependency class that parses the new-job multipart form submission.

- **Functions:** none.



## `src/rulekiln/ui/routes.py`

- **Module purpose:** Server-rendered UI route handlers for job creation, monitoring, results, and artifact download.

- **Models / classes:** none.

- **Functions:**

  - `async _get_selected_strategy()`: Derive selected strategy from the selected PromptVersion, falling back to the job column.

  - `_safe_artifact_path()`: Resolve an artifact path; raise 400 on traversal or absolute paths.

  - `_ext_ok()`: Internal helper function supporting the module implementation.

  - `async ui_root()`: Redirects the UI root to the jobs dashboard entry point.

  - `async job_list()`: Renders the jobs dashboard list with recent distillation jobs.

  - `async new_job_form()`: Renders the server-side form for creating a new distillation job.

  - `async preview_job()`: Parses uploaded task/case files and renders a validation preview before job submission.

  - `async create_job_from_ui()`: Creates job from ui as part of the module workflow.

  - `async job_detail()`: Renders the primary detail page for a single distillation job.

  - `async job_status_fragment()`: Renders the HTMX status fragment used for live job-progress polling.

  - `async job_results()`: Renders summarized strategy-comparison results for a completed job.

  - `async job_prompt()`: Renders the selected compiled prompt view for a completed job.

  - `async job_rules()`: Renders synthesized rule output for a completed job.

  - `async job_eval_report()`: Renders the evaluation report view for a completed job.

  - `async job_failures()`: Renders categorized failure analysis details for baseline vs distilled behavior.

  - `async job_artifacts()`: Renders the artifact browser/list for downloadable job output files.

  - `async download_artifact()`: Streams an individual artifact file download after path and extension safety checks.



## `src/rulekiln/ui/view_models.py`

- **Module purpose:** Pydantic view models used to render UI pages and fragments.

- **Models / classes:**

  - `JobListItemView` (model) _(bases: BaseModel)_: Summary row shown in the job list dashboard.

  - `JobDetailView` (model) _(bases: BaseModel)_: Full detail for a single distillation job.

  - `ResultsSummaryView` (model) _(bases: BaseModel)_: Metric comparison across strategies for a completed job.

  - `ProviderRouteView` (model) _(bases: BaseModel)_: Provider profile + model pair for display.

  - `PreviewView` (model) _(bases: BaseModel)_: Parsed and validated job preview before final submission.

  - `ArtifactFileView` (model) _(bases: BaseModel)_: A single downloadable artifact file.

  - `ArtifactsView` (model) _(bases: BaseModel)_: Manifest of all artifact files for a job.

- **Functions:** none.



## `src/rulekiln/workers/__init__.py`

- **Module purpose:** Package marker module.

- **Models / classes:** none.

- **Functions:** none.



## `src/rulekiln/workers/distillation_worker.py`

- **Module purpose:** End-to-end distillation pipeline worker orchestration across all processing stages. Implements 21 `PipelineStage` values with idempotent resume semantics.

- **Models / classes:**

  - `PipelineStage` (class) _(bases: StrEnum)_: String enum for all 21 pipeline stages including `refining_rules` (closed-loop conflict resolution).

- **Functions:**

  - `async run_distillation_pipeline()`: Top-level full pipeline runner used by the DBOS execution path.

  - `async _run()`: Internal helper function supporting the module implementation.

  - `async _run_refinement_loop()`: Closed-loop conflict resolution controller. Iterates analyze → refine → re-prune → compile → evaluate until convergence. Emits `outputs/refinement_iter_{n}.json` per iteration; rolls back on regression; returns `(None, None)` if no improvement.

  - `async _set_stage()`: Internal helper function supporting the module implementation.

  - `_to_db_case()`: Internal helper function supporting the module implementation.

  - `_to_db_cluster()`: Internal helper function supporting the module implementation.

  - `_synth_to_db()`: Internal helper function supporting the module implementation.

  - `_db_synth_to_schema()`: Internal helper function supporting the module implementation.

  - `_eval_to_db()`: Internal helper function supporting the module implementation.



## `src/rulekiln/workers/dbos_runtime.py`

- **Module purpose:** Runtime guard helpers that make DBOS backend availability checks explicit.

- **Models / classes:** none.

- **Functions:**

  - `is_dbos_available()`: Return whether the DBOS package is importable in the current runtime.

  - `require_dbos_available()`: Raise a clear runtime error when `EXECUTION_BACKEND=dbos` is configured without DBOS installed.



## `src/rulekiln/workers/dbos_workflow.py`

- **Module purpose:** DBOS spike workflow implementation with durable step boundaries and case-level resume semantics.

- **Models / classes:** none.

- **Functions:**

  - `async run_dbos_spike_workflow()`: Execute the DBOS spike sequence (`validating_project`, `compiling_prompts`, `evaluating_baseline`) for one job.

  - `async _validate_project_step()`: Durable step to stage and persist source cases.

  - `async _compile_prompts_step()`: Durable step to compile or reuse the baseline prompt.

  - `async _evaluate_baseline_step()`: Durable step to evaluate baseline prompt with per-case upsert persistence.

  - `_eval_case_record_to_schema()`: Convert persisted case-eval rows into runtime evaluator schema.

  - `_build_eval_case_upsert_payload()`: Build a validated upsert payload for durable per-case evaluation writes.

  - `async _set_stage()`: Internal helper to set running status/stage for workflow progress.



## `src/rulekiln/workers/dbos_worker.py`

- **Module purpose:** DBOS backend queue worker loop that claims jobs and runs the full distillation pipeline.

- **Models / classes:** none.

- **Functions:**

  - `_install_signal_handlers()`: Internal helper function supporting the module implementation.

  - `async _lease_renewer()`: Periodically renews the job lease until stop_event is set.

  - `async worker_loop()`: Main loop: recover leases, claim jobs, run DBOS workflow, and apply failure policy.

  - `main()`: CLI entrypoint for the DBOS worker (`rulekiln-worker` and `rulekiln-dbos-worker`).



## `src/rulekiln/workers/error_classification.py`

- **Module purpose:** Exception classifier that separates retryable worker errors from terminal failures.

- **Models / classes:**

  - `ErrorClassification` (model) _(bases: BaseModel)_: Classification payload describing retryability and normalized error type.

- **Functions:**

  - `classify_worker_error()`: Classify one exception into retryable/terminal for worker failure-policy decisions.
