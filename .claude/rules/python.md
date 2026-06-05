---

paths:

* "**/*.py"
* "src/**/*.py"
* "tests/**/*.py"
* "pyproject.toml"
* "uv.lock"

---

# Python rules

Follow AGENTS.md for full standards when needed. Apply these rules by default for Python work.

## Core standards

* Python 3.13+ only.
* Use modern typing: `str | None`, `list[str]`, `dict[str, int]`.
* Use `collections.abc` for interfaces such as `Sequence` and `Mapping`.
* All function parameters and returns must be typed.
* No bare `dict`, `list`, `tuple`, or `set`.
* No `Any` to bypass typing.
* No generic `# type: ignore`.
* Pyright ignores must be rule-specific and justified.

## Data modeling

* Use `dict[str, primitive]` only for simple primitive key-value data.
* Use Pydantic models for nested data, validation, external API/JSON, business objects, or data passed across functions.
* Use `Field(default_factory=list)` or `Field(default_factory=dict)` for mutable defaults.
* Do not use dataclasses.
* Prefer object attributes over magic-string dictionary access.

## Security

* Never hardcode secrets.
* Never log passwords, tokens, API keys, credentials, or raw connection strings.
* Use structured logging with safe IDs and metadata.
* Use SQLAlchemy parameterized expressions, not SQL string concatenation.
* Never use `eval()` or `exec()` with user input.
* Validate external input and file uploads with Pydantic where applicable.

## Change discipline

* Make the smallest safe change.
* Do not refactor unrelated code.
* Do not introduce architecture unless tied to benchmark evidence, reproducibility, or operational correctness.
* Remove temporary scripts, debug prints, backup files, and unused imports before finishing.

## Python checks

For PR-level Python changes, run when practical:

```bash
uv run ruff check src/ tests/
uv run pyright
DATABASE_URL="sqlite+aiosqlite://" MLFLOW_TRACKING_URI="file:///tmp/mlflow-ci" uv run pytest -m "not external" --tb=short -q
```

For small edits, run the narrowest relevant test first.
