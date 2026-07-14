# Dataset split boundaries

RuleKiln can accept a single combined train/validation/test case file for a job. This
document defines the split-safety mechanisms that keep validation- and test-split cases from
influencing train-derived artifacts, the six gaps closed to make that guarantee hold, and the
leakage-validation pass that checks it against persisted artifacts.

## Split routing

`pipeline/split_policy.py` (`resolve_split_policy`) is the single source of truth for how a
job's `RuleKilnCase` list is routed:

* **Extraction** (rule discovery, micro-rule extraction, embeddings, clustering, synthesis,
  initial prompt compilation) uses **only** `train`-split cases.
* **Evaluation** uses `validation` cases when present, falling back to `train` and then `test`
  only when no better split exists (each fallback is recorded as a `fallback_warning`).
* `test`-split cases are reserved for a final, explicitly-gated test evaluation and must not be
  touched by any earlier stage.

`resolve_few_shot_source_cases` applies the same policy to few-shot exemplar selection: it
prefers `train`-split cases and fails closed (`FewShotExemplarLeakageError`) if none exist,
unless the operator opts into evaluation-case fallback via
`RuleKilnTask.allow_few_shot_exemplars_from_eval_split` (default `False`). This mirrors the
existing `allow_refinement_on_train_split` opt-in pattern used for the refinement loop.

## The six gaps closed

1. **Few-shot exemplar leak.** Baseline few-shot strategies previously fell back silently to
   evaluation cases when no train cases existed, embedding the exact cases being scored into
   the prompt used to score them. Fixed via `resolve_few_shot_source_cases` +
   `allow_few_shot_exemplars_from_eval_split`, wired into `PipelineContextFactory` (fails at
   job-creation time) and `BaselineEvaluationService`.
2. **`RuleKilnCase.split` had a silent `"train"` default.** A case file that omitted `split`
   was absorbed into train-split extraction without any signal. `split` is now a required
   `Literal["train", "validation", "test", "golden"]` field with no default — a case missing
   `split` fails Pydantic validation instead of parsing.
3. **No split-purity assertion in evaluation.** `evaluate_prompt` now raises `ValueError` if
   any case in its input list has a `.split` that disagrees with the `split` parameter it was
   called with, catching accidental cross-split case-list construction before a single model
   call is made.
4. **Missing `source_split` provenance on train-derived artifacts.** `ClusterAssignmentRecord`,
   `RuleProvenanceRecord`, and `SynthesizedRuleSchema` gained a `source_split: str = "train"`
   field, threaded end to end: `build_cluster_assignments`/`capture_clustering_provenance`
   accept a `split_by_rule` map populated by `ClusteringStageService`, and
   `build_provenance_records` copies `source_split` from the input `SynthesizedRuleSchema`.
   `source_split` defaults to `"train"` (rather than being required) on `SynthesizedRuleSchema`
   because that schema is also used directly as an LLM structured-output target — the model
   never emits this field, so a required field would break parsing of its own output.
5. **Validation-refined prompts weren't marked as such.** `RefinementIterationArtifact` gained
   `refined_using_split: str = "train"` and `validation_tuned: bool = False`, set by
   `_run_refinement_loop` from the split actually used for refinement-loop evaluation
   (`validation_tuned = (eval_split == "validation")`). The same two keys are written into the
   raw `outputs/rule_refinement_provenance_{iteration}.json` artifact.
6. **No cross-artifact leakage check.** See below.

## Split-leakage validation pass

`pipeline/split_leakage_validation.py` (`validate_split_leakage`) is a **reproducible-from-disk**
validator, run alongside `pipeline/provenance_validation.py` during artifact export
(`ArtifactExportService._write_provenance_and_promotion`). Where provenance validation asks "are
these artifacts internally hash-consistent", this pass asks "did a case from the wrong split
influence this artifact". It reads only persisted artifacts under `artifact_root`
(`cases.normalized.jsonl`, per-algorithm `cluster_assignments.jsonl`, `rule_provenance.json`,
`outputs/*_eval.json`, `outputs/paired_comparison/*.jsonl`,
`outputs/rule_refinement_provenance_*.json`), so it works uniformly for a fresh run and for
post-hoc re-validation of an older job directory.

### Finding codes

| Code | Severity | Meaning |
|---|---|---|
| `TRAIN_STAGE_NONTRAIN_CASE` | error | A train-only artifact (cluster assignment, rule provenance record) declares a `source_split` other than `"train"`. |
| `SPLIT_PROVENANCE_MISMATCH` | error | An artifact's declared `source_split` disagrees with the actual `.split` of the case(s) it names. |
| `TEST_CASE_LEAKAGE` | error | Any scanned artifact (cluster assignment, rule provenance, eval result, paired comparison) references a `test`-split case before final test evaluation. |
| `VALIDATION_EVAL_NONVALIDATION_CASE` | error | An `EvalResult` declared `split="validation"` contains a case whose actual split is not `validation`. |
| `VALIDATION_REFINEMENT_NOT_MARKED` | error | A refinement iteration's `validation_tuned` flag disagrees with `refined_using_split == "validation"`. |
| `SOURCE_DOCUMENT_SPLIT_OVERLAP` | warning | Two or more cases sharing `metadata.document_hash` appear in different splits. Not an error: the same source document can legitimately be split across cases/hypotheses. |

Pipeline contamination (the first five codes) is a **blocking error**; source-document overlap
across splits is a **warning** only, since it reflects dataset construction, not pipeline
behavior.

### Persisted report

`write_split_leakage_validation` writes `metadata/split_leakage_validation.json`
(`SplitLeakageValidationReport`, `schema_version = "rulekiln.split_leakage_validation.v1"`),
readable back via `read_split_leakage_validation`. Each `SplitLeakageFinding` carries `code`,
`severity`, `message`, `expected`/`actual` (structured, for programmatic diffing), `artifact`
(the relative path it was found in), and `case_ids` (the offending case IDs).

### Legacy-artifact compatibility

Every loader in this module tolerates a missing file (returns an empty result) and a malformed
or legacy-shaped file (logs a warning and skips it) rather than raising — a job directory that
predates `source_split`/`refined_using_split`/`validation_tuned` fields, or one missing an
artifact entirely, still produces a valid (possibly empty) report instead of failing artifact
export.

## Tests

* `tests/unit/test_split_safety_gaps.py` — one test group per closed gap (1–5 above).
* `tests/unit/test_split_leakage_validation.py` — one test per finding code, a clean-run
  no-findings case, write/read round-trip, and legacy-artifact tolerance (missing files,
  malformed JSON, pre-`source_split` rule provenance).
