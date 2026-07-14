.PHONY: lint format typecheck test test-ui test-cov test-cov-html badge ci-local docker-up docker-down benchmark-smoke

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

test-cov:
	DATABASE_URL="sqlite+aiosqlite://" MLFLOW_TRACKING_URI="file:///tmp/mlflow-ci" uv run pytest -m "not external" --tb=short -q --cov --cov-report=term-missing

test-cov-html:
	DATABASE_URL="sqlite+aiosqlite://" MLFLOW_TRACKING_URI="file:///tmp/mlflow-ci" uv run pytest -m "not external" --tb=short -q --cov --cov-report=html
	@echo "HTML report: htmlcov/index.html"

badge:
	DATABASE_URL="sqlite+aiosqlite://" MLFLOW_TRACKING_URI="file:///tmp/mlflow-ci" uv run pytest -m "not external" --tb=short -q --cov --cov-report=xml
	uv run genbadge coverage -i coverage.xml -o docs/assets/coverage-badge.svg
	@rm -f coverage.xml
	@echo "Updated docs/assets/coverage-badge.svg — commit it to refresh the README badge."

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