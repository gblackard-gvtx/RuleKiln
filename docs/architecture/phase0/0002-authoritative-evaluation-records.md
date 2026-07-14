# ADR-0002: Per-Case Evaluation Records Are Authoritative

- **Status:** Proposed
- **Date:** 2026-07-14
- **Decision owner:** Maintainer

## Context

Evaluation systems commonly persist several representations of the same result:

- per-case predictions;
- aggregate metrics;
- strategy-comparison tables;
- UI summaries;
- database records;
- promotion artifacts.

RuleKiln has already encountered stale or internally inconsistent aggregate artifacts. When multiple summaries are treated as equally authoritative, ranking and reporting can diverge from the actual predictions.

Malformed-output handling can also alter denominators or accidentally introduce spurious labels if metric computation is not centralized.

## Decision

Authoritative evaluation state consists of:

1. the immutable ordered evaluation case set;
2. one per-case prediction record for every assigned case and strategy;
3. the parser/output-contract result for each prediction.

All aggregates are derived from those records.

The derivation order is:

```text
Per-case records
    → strict metrics
    → diagnostic valid-output metrics
    → paired comparisons
    → eligibility gates
    → strategy selection
    → package lifecycle assessment
```

## Classification metric rules

For the initial classification mode:

- strict macro-F1 is the primary metric;
- strict accuracy uses all assigned cases;
- malformed responses count as incorrect;
- malformed is not added as a fourth class;
- valid-output metrics may be shown diagnostically;
- paired comparisons join by stable case ID;
- every derived artifact records source hashes.

## Stale artifacts

A persisted aggregate is stale when:

- its evaluation hash does not match source per-case records;
- its ordered case-set hash differs;
- its prediction hash differs;
- its parser or output-schema hash differs;
- it references a different job or strategy.

Stale summaries:

- may remain available for historical inspection;
- must not influence current promotion;
- must be clearly labeled;
- must not be silently overwritten.

## Consequences

### Positive

- Every score is traceable.
- Malformed handling is consistent.
- Paired comparisons cannot depend on array order.
- Stale summaries cannot silently determine lifecycle decisions.
- Package evaluation evidence can be independently verified.

### Negative

- Per-case artifacts become mandatory.
- Recomputing aggregates may cost additional time.
- Legacy runs may be marked incomplete.

## Rejected alternatives

### Database metric row as authority

Rejected because a database aggregate cannot explain case-level discrepancies and may diverge from persisted artifacts.

### Strategy-comparison JSON as authority

Rejected because it is downstream of metrics and can become stale.

### UI recomputation without persisted hashes

Rejected because it produces unverifiable transient results.

## Follow-up

- Centralize strict metric derivation.
- Persist ordered case-set and prediction hashes.
- Add stale-artifact validation.
- Make package evaluation summaries point to authoritative source hashes.
