# Clustering provenance

Every clustered strategy evaluation (DBSCAN, HDBSCAN) captures a durable, typed provenance
record so a reader can answer, from persisted artifacts alone:

1. What clustering configuration was **requested**?
2. What configuration was **used at runtime**?
3. What embeddings and text representation were clustered?
4. What clusters and prompts were produced?
5. Can the evaluation be reproduced?

This document is the reference for the schema, lifecycle, hashing rules, artifact
locations, legacy behavior, and reproduction procedure. It was written in response to the
integrity review in `docs/investigations/contractnli_hdbscan_validation_win.md`, which found
that a "winning" HDBSCAN run had persisted only a partial config and no cluster assignments,
so its result could not be verified or reproduced.

## Schema

Typed models live in `src/rulekiln/schemas/provenance.py`. The top-level record is
`ClusteringProvenance` (`schema_version = "rulekiln.clustering_provenance.v1"`):

| Field | Meaning |
|---|---|
| `job_id` | Owning job (enables cross-job artifact detection). |
| `algorithm` | `dbscan` or `hdbscan`. |
| `text_variant` | `rule_only` \| `label_plus_rule` \| `label_plus_hypothesis_plus_rule`. |
| `requested_parameters` | `ClusteringParameters` mirroring what the task config asked for. |
| `effective_parameters` | `ClusteringParameters` after library-default resolution. |
| `embedding` | `EmbeddingProvenance`: provider profile, model, dimension, quantization, normalization, distance metric. |
| `input` | `ClusteringInput`: rule/duplicate counts, label counts, text variant. |
| `output` | `ClusteringOutput`: cluster count, noise stats, cluster-size stats. |
| `runtime` | `ClusteringRuntime`: library + version, python version, timing. |
| `source` | `ClusteringSource`: task-config path, job task-snapshot path, strategy. |
| `hashes` | `ClusteringHashes`: input text, embedding matrix, assignments, summary, synthesized rules. |

Irrelevant fields for a given algorithm stay **explicitly `null`**, never omitted (for
example, `eps` is `null` for HDBSCAN; `min_cluster_size` is `null` for DBSCAN). The schema is
JSON-serializable and versioned from the start.

### Requested versus effective parameters

The distinction the integrity review demanded:

* **Requested** parameters mirror the task config verbatim. HDBSCAN `min_samples` stays
  visibly `null` when the config left it unset.
* **Effective** parameters record the resolved library behavior. HDBSCAN's `min_samples`
  defaults to `min_cluster_size` when unset, so the effective record shows that resolved
  value.

This is what lets a reader see both "the config asked for `min_samples=null`" and "the
library actually used `min_samples=2`", instead of guessing.

## Lifecycle

1. **Capture (runtime).** In `ClusteringStageService` (`workers/pipeline/early_stages.py`),
   immediately after DBSCAN/HDBSCAN run, `capture_clustering_provenance`
   (`pipeline/clustering_provenance.py`) turns the *exact runtime inputs and outputs* into
   the provenance triple `(ClusteringProvenance, [ClusterAssignmentRecord], ClusterSummary)`.
   Values are never reconstructed later from `task.yaml`.
2. **Persist (clustering stage).** The triple is written under the job artifact root (see
   locations below). This happens in the compile phase, which may run in a different process
   than artifact export, so it is persisted at capture time.
3. **Reference (export stage).** `ArtifactExportService` reads the persisted provenance back,
   embeds per-strategy hash references into `strategy_comparison.json`, aggregates
   `metadata/hashes.json`, runs provenance validation, and evaluates promotion.
4. **Validate.** `validate_provenance` (`pipeline/provenance_validation.py`) cross-checks the
   persisted artifacts and records findings in `metadata/provenance_validation.json`.

Capture and export are both **best-effort**: a provenance failure is logged and never fails
an otherwise-good job.

## Artifact locations

```
outputs/clustering/hdbscan/clustering_provenance.json
outputs/clustering/hdbscan/cluster_assignments.jsonl
outputs/clustering/hdbscan/cluster_summary.json
outputs/clustering/hdbscan/synthesized_rules.json      (when synthesized rules are persisted)
outputs/clustering/dbscan/clustering_provenance.json
outputs/clustering/dbscan/cluster_assignments.jsonl
outputs/clustering/dbscan/cluster_summary.json
metadata/hashes.json                                   (job + strategy hashes)
metadata/provenance_validation.json                    (structured findings)
metadata/promotion_status.json                         (see strategy_promotion.md)
```

`cluster_assignments.jsonl` preserves deterministic ordering (by embedding row index). Each
record carries `rule_id`, `source_case_id`, `cluster_id`, `is_noise`, `label`,
`hypothesis_id`, `text_hash`, and `embedding_row_index` — enough to debug a cluster without
storing raw vectors.

Raw embedding vectors are **not** persisted in JSON. Instead an `embedding_matrix_hash`
(shape + values, canonicalized) records the matrix identity.

## Hashing rules

All reproducibility hashes flow through `src/rulekiln/artifacts/canonical_hash.py`
(`algorithm = sha256`, `canonicalization_version = v1`):

* Mappings are canonicalized by key (sorted); ordered lists stay order-sensitive.
* Strings are UTF-8 with JSON escaping.
* Floats use shortest round-trip `repr`; `-0.0` is normalized to `0.0`; NaN/Inf are rejected.
* Integers and booleans are distinct from floats (`2` ≠ `2.0`).
* `null` is preserved and is distinct from an omitted key.

Aggregation into `metadata/hashes.json` and per-strategy rows is in
`pipeline/artifact_hashing.py`. The runtime clustering config is hashed in the **same shape**
as the approved-config registry (`ApprovedClusteringConfig`), so a run's config hash is
directly comparable to the approved config hash (see `strategy_promotion.md`).

## Legacy behavior

Historical runs written before this feature have no clustering provenance and no
`metadata/*.json` provenance artifacts. They are **never** backfilled from the current
`task.yaml` — doing so would fabricate runtime values that were not actually recorded. Such
runs surface as `provenance_status: incomplete` / `reproducibility_status: unverified`, and
the UI shows "Provenance unavailable for this historical run."

## Reproduction procedure

1. Read `outputs/clustering/<algo>/clustering_provenance.json` for the requested + effective
   parameters, embedding model/dimension, text variant, and hashes.
2. Re-embed the same ordered rule text with the recorded embedding model; confirm
   `embedding_matrix_hash` matches.
3. Re-run clustering with the recorded **effective** parameters; confirm the recomputed
   `cluster_assignment_hash` matches `cluster_assignments.jsonl`.
4. Recompile the prompt; confirm `compiled_prompt_hash` in `strategy_comparison.json`.
5. Note: student **decoding** settings (temperature/top_p/seed) are provider-runtime and are
   not persisted per run, so byte-for-byte prediction reproduction requires pinning the
   student server to greedy decoding. `decoding_config` is intentionally `null` until that is
   captured.
