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
