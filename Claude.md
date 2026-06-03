@AGENTS.md

## Claude Code workflow

Use the instructions in AGENTS.md as the canonical project coding standards.

Prefer semantic/code-graph/MCP tools before broad file reads.

When investigating:
1. Use symbol/reference/search/code-graph tools first.
2. Read only the smallest file sections needed.
3. Avoid repeated full-file reads of files already inspected.
4. Avoid broad Glob/Read sweeps unless explicitly required.
5. Summarize findings before making edits.

When implementing:
1. Read the relevant spec or task file first.
2. Make the smallest change that satisfies the task.
3. Do not refactor unrelated code.
4. Run the relevant checks from AGENTS.md before finishing when practical.
5. Remove temporary files, debug prints, and experimental scripts before stopping.

For long tasks, stop after each phase and summarize:
- files changed
- tests run
- remaining risks
- next recommended step