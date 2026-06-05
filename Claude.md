# Claude Code workflow

Goal: minimize context, file reads, tool output, and unnecessary edits.

## Context discipline

* Do not import AGENTS.md automatically.
* Read AGENTS.md only when editing or reviewing Python code, changing project standards, or preparing a PR.
* Prefer semantic/code-graph/MCP tools before broad file reads.
* Read only the smallest relevant file sections.
* Avoid repeated full-file reads of files already inspected.
* Avoid broad Glob/Read sweeps unless explicitly required.
* Summarize findings before edits.

## Implementation discipline

* Read the relevant spec, issue, or task file first.
* Make the smallest change that satisfies the task.
* Do not refactor unrelated code.
* Do not introduce new architecture unless required for benchmark evidence, reproducibility, or operational correctness.
* Remove temporary files, debug prints, backup files, and experimental scripts before stopping.

## Checks

Before finishing, run only the relevant checks for the changed area when practical.

For Python PR-level work, use the checks from AGENTS.md:

* `uv run ruff check src/ tests/`
* `uv run pyright`
* `DATABASE_URL="sqlite+aiosqlite://" MLFLOW_TRACKING_URI="file:///tmp/mlflow-ci" uv run pytest -m "not external" --tb=short -q`

## Long tasks

After each phase, stop and summarize:

* files changed
* tests run
* remaining risks
* next recommended step

## Compaction

When compacting, preserve only:

* task goal
* files inspected
* files changed
* decisions made
* commands run
* failing test output
* remaining risks
* next step

Drop exploratory dead ends, verbose logs, repeated file contents, and obsolete plans.
