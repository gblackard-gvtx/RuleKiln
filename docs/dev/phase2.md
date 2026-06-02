# Phase 2 Strategies and Make Targets

This document describes the current Phase 2 runtime behavior for strategy evaluation and the local Make target surface.

## Scope

Phase 2 extends baseline evaluation and strategy comparison beyond the original baseline/DBSCAN/HDBSCAN-only flow.

## Strategy catalog

| Strategy | Category | Prompt source | Evaluator path |
|----------|----------|---------------|----------------|
| `baseline_scaffold` | Baseline | Deterministic scaffold from `compile_baseline_prompt` | Student chat model |
| `baseline_few_shot_k3` | Baseline few-shot | `baseline_scaffold` plus deterministic few-shot examples (`k=3`) | Student chat model |
| `baseline_few_shot_k5` | Baseline few-shot | `baseline_scaffold` plus deterministic few-shot examples (`k=5`) | Student chat model |
| `embedding_centroid` | Embedding baseline | No extra prompt content; prediction from nearest centroid | Embedding-only classifier |
| `embedding_knn_k1` | Embedding baseline | No extra prompt content; prediction from KNN (`k=1`) | Embedding-only classifier |
| `embedding_knn_k3` | Embedding baseline | No extra prompt content; prediction from KNN (`k=3`) | Embedding-only classifier |
| `embedding_knn_k5` | Embedding baseline | No extra prompt content; prediction from KNN (`k=5`) | Embedding-only classifier |
| `retrieval_few_shot_k5` | Retrieval baseline | Per-case prompt built from top-`k` retrieved neighbors (`k=5`) | Student chat model (single-case eval calls) |
| `dbscan` | Distilled | Distilled rules compiled into prompt | Student chat model |
| `hdbscan` | Distilled | Distilled rules compiled into prompt | Student chat model |

## Few-shot prompt construction

Few-shot prompts are built deterministically in code:

1. Select examples from training cases using stable ordering and label-aware selection.
2. Render examples into a fixed markdown template.
3. Append examples to baseline scaffold prompt.
4. Clip examples to `max_prompt_tokens` budget.

The teacher model does not generate few-shot prompt text. The student model only consumes the generated prompt during evaluation.

## Retrieval few-shot behavior

`retrieval_few_shot_k5` runs per case:

1. Embed the query case.
2. Retrieve nearest training examples.
3. Build a case-specific few-shot prompt.
4. Evaluate one case at a time.

This means tests that stub eval calls should expect multiple single-case invocations for retrieval few-shot rather than one batch invocation.

## Selection and comparison model

`outputs/strategy_comparison.json` is the canonical comparison artifact. It includes:

- `strategy_evals`
- `strategy_gates`
- `strategy_prompt_tokens`
- `strategy_metadata`
- `selected_strategy`
- `selection_reason`

Selection uses the generic tie-break flow:

1. Primary metric
2. Golden failures
3. Malformed output rate
4. Prompt token count
5. Deterministic fallback order

## Artifact outputs added in Phase 2

Phase 2 adds baseline and strategy-specific artifacts under `.rulekiln/runs/{job_id}/outputs/`.

Examples:

- `baseline_scaffold_prompt.md`
- `baseline_few_shot_k3_prompt.md`
- `baseline_few_shot_k5_prompt.md`
- `<strategy>_eval.json` for each evaluated strategy

`selected_distilled_prompt.md` may contain a non-distilled winner if that strategy is selected.

## Make target quick reference

| Target | Command | Purpose |
|--------|---------|---------|
| `make lint` | `uv run ruff check src/ tests/` | Lint checks |
| `make format` | `uv run ruff format src/ tests/` | Code formatting |
| `make typecheck` | `uv run pyright` | Static type checking |
| `make test` | `DATABASE_URL="sqlite+aiosqlite://" MLFLOW_TRACKING_URI="file:///tmp/mlflow-ci" uv run pytest -m "not external" --tb=short -q` | Full offline test run |
| `make test-ui` | `DATABASE_URL="sqlite+aiosqlite://" MLFLOW_TRACKING_URI="file:///tmp/mlflow-ci" uv run pytest tests/ui/ -m "not external" --tb=short -q` | UI-only offline tests |
| `make ci-local` | `make lint && make typecheck && make test` | Local CI gate |
| `make docker-up` | `./scripts/dev-up.sh` | Start Docker stack |
| `make docker-down` | `./scripts/dev-down.sh` | Stop Docker stack |
| `make benchmark-smoke` | fixture presence checks | Verify benchmark fixture files |

## Recommended local validation flow

1. `make lint`
2. `make typecheck`
3. `make test`
4. `make test-ui` (when UI-related files changed)
5. `make benchmark-smoke` (when touching benchmark assets)
