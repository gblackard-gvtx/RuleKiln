# Documentation Map

Use this index to find the right document quickly.

## Start Here

- Local setup and run commands: [dev/docker.md](dev/docker.md)
- Phase 2 strategy behavior and Make targets: [dev/phase2.md](dev/phase2.md)
- Closed-loop conflict resolution validation: [dev/loop_validation_report.md](dev/loop_validation_report.md)
- Current documentation validation snapshot: [mvp_validation.md](mvp_validation.md)

## Architecture

- Clustering provenance (versioned params, hashes, `outputs/clustering/<algo>/`): [architecture/clustering_provenance.md](architecture/clustering_provenance.md)
- Strategy promotion status and policy: [architecture/strategy_promotion.md](architecture/strategy_promotion.md)
- How RuleKiln implements the Prompt-Level Distillation paper, phase by phase: [architecture/pld-implementation-mapping.md](architecture/pld-implementation-mapping.md)
- Gaps remaining to reach PLD paper-level empirical parity: [architecture/pld-gap-analysis.md](architecture/pld-gap-analysis.md)
- Full file-by-file summary of the provenance/promotion change: [dev/clustering_provenance_promotion_changes.md](dev/clustering_provenance_promotion_changes.md)

## Investigations

- ContractNLI HDBSCAN malformed-output root cause (compound-label leakage): [investigations/contractnli_hdbscan_malformed_output.md](investigations/contractnli_hdbscan_malformed_output.md)
- ContractNLI strategy-selection regression (selected strategy worse than baseline): [investigations/contractnli_strategy_selection_regression.md](investigations/contractnli_strategy_selection_regression.md)
- ContractNLI HDBSCAN validation win, with provenance/promotion follow-up: [investigations/contractnli_hdbscan_validation_win.md](investigations/contractnli_hdbscan_validation_win.md)
- Few-shot k=5 echo-failure investigation: [dev/few_shot_k5_investigation.md](dev/few_shot_k5_investigation.md)
- Run 3 investigation: [dev/run3_investigation.md](dev/run3_investigation.md)

## Design Specs

- Spec directory: [specs/](specs/)
- Feature specs: [specs/features/](specs/features/)
- Classroom-aware benchmark phases 1–4: [specs/SPEC_benchmark_classroom_phases.md](specs/SPEC_benchmark_classroom_phases.md)
- Batch API support spec: [specs/SPEC_Batch_API_Support.md](specs/SPEC_Batch_API_Support.md)
- OpenAI batch execution feature spec: [specs/features/rulekiln_openai_batch_execution_feature_spec.md](specs/features/rulekiln_openai_batch_execution_feature_spec.md)

## Execution Task Lists

- Task directory: [tasks/](tasks/)
- Canonical MVP task list: [tasks/rulekiln_mvp_tasks_v3_with_docker.md](tasks/rulekiln_mvp_tasks_v3_with_docker.md)
- Historical task snapshots: [tasks/archive/](tasks/archive/)

## Technical Reference

- Python module reference: [reference/python_module_reference.md](reference/python_module_reference.md)

## Notes

- Several spec and task files are historical planning artifacts. Prefer the canonical task list above when you need an implementation checklist.
