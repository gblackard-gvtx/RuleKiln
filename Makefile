.PHONY: lint format typecheck test test-ui ci-local docker-up docker-down benchmark-smoke

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

typecheck:
	uv run pyright

test:
	DATABASE_URL="sqlite+aiosqlite://" MLFLOW_TRACKING_URI="file:///tmp/mlflow-ci" uv run pytest -m "not external" --tb=short -q

test-ui:
	DATABASE_URL="sqlite+aiosqlite://" MLFLOW_TRACKING_URI="file:///tmp/mlflow-ci" uv run pytest tests/ui/ -m "not external" --tb=short -q

ci-local:
	@status=0; \
	$(MAKE) lint || status=$$?; \
	$(MAKE) typecheck || status=$$?; \
	$(MAKE) test || status=$$?; \
	if [ $$status -eq 0 ]; then \
		echo "Local CI checks passed"; \
	fi; \
	exit $$status

docker-up:
	./scripts/dev-up.sh

docker-down:
	./scripts/dev-down.sh

benchmark-smoke:
	@test -s examples/datasets/banking77/task.yaml
	@test -s examples/datasets/banking77/cases.normalized.jsonl
	@echo "BANKING77 benchmark smoke checks passed"