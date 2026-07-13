# Clustering provenance, reproducibility hashes & promotion status — change summary

_Working-tree changes on `allow-batching-via-ui`. Implements durable, end-to-end provenance
and explicit promotion metadata for every clustered strategy evaluation, in response to the
integrity review in `docs/investigations/contractnli_hdbscan_validation_win.md`._

## Goal

Make every clustered (DBSCAN/HDBSCAN) evaluation answer, from persisted artifacts alone:
what config was requested, what config actually ran, what was clustered, what clusters/prompts
were produced, can it be reproduced, was it *selected* vs. *promoted*, and what blocks
promotion. Selection is unchanged; promotion is a new, separate decision.

See the architecture references:
- `docs/architecture/clustering_provenance.md`
- `docs/architecture/strategy_promotion.md`

## New modules

- **`schemas/provenance.py`** — typed, versioned clustering-provenance models
  (`ClusteringProvenance` = `rulekiln.clustering_provenance.v1`, plus `ClusteringParameters`,
  `EmbeddingProvenance`, `ClusterAssignmentRecord`, `ClusterSummary`, `StrategyProvenanceRow`).
  Requested vs. effective parameters are distinct; irrelevant fields are explicit `null`.
  A leaf module (no import from `schemas.pipeline`) so `StrategyComparison` can embed rows.
- **`schemas/promotion.py`** — `PromotionStatus` (`…promotion_status.v1`), `PromotionPolicy`
  (`…promotion_policy.v1`, defaults preserve current behavior), `PromotionBlocker`,
  `ProvenanceFinding`/`ProvenanceValidationReport`, `ArtifactHashes`, `ApprovedStrategyConfig`.
- **`artifacts/canonical_hash.py`** — one SHA-256 canonicalization path (sorted keys,
  order-sensitive lists, explicit float/null handling, recorded `canonicalization_version`).
  `canonical_hash`, `hash_text`, `hash_matrix`.
- **`pipeline/clustering_provenance.py`** — `capture_clustering_provenance(...)`: builds the
  `(ClusteringProvenance, [ClusterAssignmentRecord], ClusterSummary)` triple from the exact
  runtime inputs/outputs at clustering time (never reconstructed later from `task.yaml`).
- **`pipeline/approved_configs.py`** — loads the approved/promoted config registry and
  recomputes its canonical hash under the current canonicalization version.
- **`pipeline/strategy_families.py`** — classifies strategies into comparison families.
- **`pipeline/clustering_diagnostics.py`** — cluster-health diagnostics used by the sweep CLI.
- **`pipeline/artifact_hashing.py`** — aggregates job-level (dataset/task/provider/parser/gate)
  + clustering + strategy hashes into `ArtifactHashes`, and builds per-strategy
  `StrategyProvenanceRow`s. Hashes the runtime clustering config in the **same shape** as the
  approved registry so the two are directly comparable.
- **`pipeline/provenance_validation.py`** — `validate_provenance(...)` cross-checks persisted
  artifacts and emits structured findings: `CROSS_JOB_ARTIFACT`,
  `CLUSTER_ASSIGNMENT_HASH_MISMATCH`, `PROMPT_HASH_MISMATCH`, `CASE_SET_MISMATCH` (errors) and
  `CLUSTER_CONFIG_MISMATCH` (warning). Errors block promotion, never selection.
- **`pipeline/promotion.py`** — `evaluate_promotion(...)` → deterministic `PromotionStatus`
  with ordered blockers (`BLOCKER_ORDER`). Never auto-promotes: `approved` means "ready for a
  human", `promoted_strategy` stays `null`.
- **`artifacts/manifest.py`** — `ManifestEntry`/`RunManifest`, `build_manifest_entries`
  (type, schema version, sha256, size, timestamp, strategy), and `verify_manifest`.
- **`config/approved_strategy_configs/hdbscan.json`** — the promoted `5/1 eom` config.

## Wiring & modified files

- **`workers/pipeline/early_stages.py`** — `ClusteringStageService` captures + persists
  clustering provenance/assignments/summary under `outputs/clustering/<algo>/` at clustering
  time.
- **`workers/pipeline/output_stage_services.py`** — new
  `ArtifactExportService._write_provenance_and_promotion(...)`: reads back the persisted
  clustering provenance, builds `StrategyProvenanceRow`s + `ArtifactHashes`, runs provenance
  validation, evaluates promotion, embeds all of it into `strategy_comparison.json`, and
  writes `metadata/{hashes,provenance_validation,promotion_status}.json`. Best-effort — never
  fails the job. Also discovers clustering artifacts (written in an earlier stage) so they are
  included in the manifest.
- **`artifacts/writer.py`** — `write_manifest(root, artifact_paths, entries=None)` now emits
  the legacy `artifacts` path list **and** rich `entries`; new `write_*` functions for the
  clustering/hash/validation/promotion artifacts.
- **`schemas/pipeline.py`** — `StrategyComparison` gains optional, defaulted
  `strategy_provenance`, `artifact_hashes`, `provenance_validation`, `promotion_status`
  (backward compatible: legacy `strategy_comparison.json` still parses).
- **`ui/routes.py`** — `_build_provenance_promotion_view(...)` assembles the view from
  persisted artifacts; passed into the eval report. Returns `None` for legacy runs.
- **`templates/jobs/eval_report.html`** — new "Provenance & Promotion" section: promotion
  status/blockers, run-provenance hashes, requested-vs-effective params, embedding + cluster
  stats, and per-strategy hashes. Distinct selected/candidate/promoted labels; legacy runs
  show "Provenance unavailable for this historical run."

## Artifacts produced (new runs)

```
outputs/clustering/<algo>/{clustering_provenance,cluster_assignments,cluster_summary,synthesized_rules}
metadata/hashes.json                  # rulekiln.artifact_hashes.v1
metadata/provenance_validation.json   # rulekiln.provenance_validation.v1
metadata/promotion_status.json        # rulekiln.promotion_status.v1
metadata/manifest.json                # legacy `artifacts` + rich `entries` (rulekiln.run_manifest.v1)
outputs/strategy_comparison.json      # + strategy_provenance / promotion_status / hashes
```

## Legacy behavior

Historical runs are never backfilled from the current `task.yaml`. They render
"Provenance unavailable…" and resolve to `provenance_status: incomplete` /
`reproducibility_status: unverified`. New comparison fields are optional; the manifest keeps
its legacy `artifacts` list.

## Tests

- `tests/unit/test_promotion_policy.py` — selected≠promoted, invalid-provenance/repeat/
  malformed-boundary/minority-regression/confidence blockers, determinism (11).
- `tests/unit/test_provenance_validation.py` — config mismatch, cross-job, prompt-hash,
  case-set, assignment-hash tamper, clean run (6).
- `tests/unit/test_artifact_hashing.py` — case-set sensitivity, prediction/evaluation hash
  changes, runtime-config == approved-config hash, strategy-row population (8).
- `tests/unit/test_run_manifest.py` — entry metadata, legacy `artifacts` preserved, tamper
  detection (3).
- `tests/ui/test_provenance_promotion_report.py` — new-run render, blockers, label separation,
  legacy unavailable (4).
- `tests/integration/test_offline_e2e_fake_providers.py::test_pipeline_writes_provenance_and_promotion_artifacts`
  — full clustered run writes artifacts, manifest verifies, references match, status
  round-trips, non-model hash recompute is stable.

**Results:** 43 new tests pass; full unit suite 555 passed; UI+integration 114 passed; ruff +
pyright clean on all new/modified files.

## Known limitations

- `decoding_config` hash is intentionally `null` (student decoding is provider-runtime, not
  persisted per run) — byte-for-byte prediction reproduction needs pinned greedy decoding.
- Reproducibility is `unverified` for single runs; the ≥3-repeat requirement is policy-gated
  but not yet automated.
