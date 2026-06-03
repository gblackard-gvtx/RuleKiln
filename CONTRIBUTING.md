# Contributing to RuleKiln

Thanks for your interest in contributing to RuleKiln.

RuleKiln is an open-source prompt-hardening framework for turning labeled examples into tested, auditable prompts for smaller/local/edge student models.

## Ways to Contribute

You can help by:

- Reporting bugs
- Improving documentation
- Adding dataset examples
- Improving provider integrations
- Adding evaluation metrics
- Improving prompt compilation
- Testing local/edge models
- Suggesting benchmark tasks

## Development Setup

1. Fork the repository.
2. Clone your fork.
3. Create a branch:

```bash
git checkout -b feature/my-change
```

4. Install dependencies using the project's documented setup.
5. Run tests before opening a PR.

## Pull Request Guidelines

Before opening a PR:

- Keep changes focused.
- Add or update tests when behavior changes.
- Update docs for user-facing changes.
- Include examples when adding task/case functionality.
- Avoid committing full benchmark datasets unless explicitly approved.
- Do not add new architecture unless tied to benchmark evidence, reproducibility, or operational correctness.

### Before opening a PR checklist (coding agents)

- [ ] Run `uv run ruff check src/ tests/`
- [ ] Run `uv run pyright`
- [ ] Run `DATABASE_URL="sqlite+aiosqlite://" MLFLOW_TRACKING_URI="file:///tmp/mlflow-ci" uv run pytest -m "not external" --tb=short -q`
- [ ] Confirm docs and command references are updated for user-facing setup or workflow changes.
- [ ] Confirm no placeholder URLs remain in edited docs.

For coding standards and typing/security conventions, follow [AGENTS.md](AGENTS.md).

## Commit Style

Use clear commit messages:

- fix: correct validation split routing
- feat: add invalid label rate metric
- docs: add BANKING77 benchmark notes

## Benchmark Contributions

Dataset or benchmark contributions should include:

- Dataset source
- License
- Task type
- Case conversion script
- Sample cases
- Expected metrics
- Known limitations

## Provider Integrations

Provider integrations should avoid hardcoding credentials.

- Use environment variables or provider profiles.
- Never commit API keys, tokens, secrets, or private model credentials.
