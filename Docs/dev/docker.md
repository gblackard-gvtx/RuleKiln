# Local Development Guide

RuleKiln supports two local workflows. Both are fully offline using the `fake` provider — no API keys or cloud credentials are required.

---

## Option A — Native Python

**Requirements**: Python 3.13+, [uv](https://docs.astral.sh/uv/), a running PostgreSQL 14+ instance, a running MLflow server.

### Setup

```bash
# Install dependencies
uv sync --extra dev

# Copy environment template
cp .env.example .env
# Edit .env — ensure DATABASE_URL points to your local Postgres instance

# Run database migrations
uv run alembic upgrade head

# Start the API with auto-reload
uv run uvicorn src.rulekiln.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Running a local MLflow server

```bash
mlflow server \
  --backend-store-uri ./mlruns \
  --default-artifact-root ./mlartifacts \
  --host 0.0.0.0 \
  --allowed-hosts localhost,127.0.0.1,mlflow,mlflow:5000 \
  --port 5000
```

Or use `file://` URI without a server by setting `MLFLOW_TRACKING_URI=file:///absolute/path/to/mlruns`.

### Running tests (native)

```bash
DATABASE_URL="sqlite+aiosqlite://" \
MLFLOW_TRACKING_URI="file:///tmp/mlflow-ci" \
uv run pytest -m "not external" --tb=short -q
```

---

## Option B — Docker Compose

**Requirements**: Docker Desktop ≥ 4.x or Docker Engine + Compose v2.

The Compose stack provides Postgres, MLflow, and the RuleKiln API in one command. No external credentials are needed for the default `fake` provider.

### Start the stack

```bash
./scripts/dev-up.sh
```

This script:
1. Copies `.env.example` → `.env` if no `.env` exists.
2. Runs `docker compose up -d --build`.

| Service | URL |
|---------|-----|
| RuleKiln API | <http://localhost:8000> |
| OpenAPI docs | <http://localhost:8000/docs> |
| MLflow UI | <http://localhost:5000> |
| PostgreSQL | `localhost:5432` (user/pass: `rulekiln/rulekiln`) |

### Stop the stack

```bash
./scripts/dev-down.sh
```

### Run migrations inside Docker

```bash
docker compose exec api uv run alembic upgrade head
```

### View API logs

```bash
docker compose logs -f api
```

### Rebuild after dependency changes

```bash
docker compose build api
docker compose up -d api
```

---

## Environment variable reference

Both workflows use the same `.env` contract. Key differences:

| Variable | Native value | Compose value |
|----------|-------------|---------------|
| `DATABASE_URL` | `postgresql+asyncpg://rulekiln:rulekiln@localhost:5432/rulekiln` | `postgresql+asyncpg://rulekiln:rulekiln@postgres:5432/rulekiln` |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` or `file:///...` | `http://mlflow:5000` |
| `MLFLOW_ALLOWED_HOSTS` | Optional host allowlist for MLflow server | `localhost,127.0.0.1,mlflow,mlflow:5000` |
| `ARTIFACT_ROOT` | `.rulekiln/runs` | `.rulekiln/runs` (mounted into container) |

The Compose stack sets `DATABASE_URL` and `MLFLOW_TRACKING_URI` directly in `docker-compose.yml`, overriding whatever is in `.env` for those two keys.

---

## Feature flags

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_PGVECTOR` | `false` | Enable pgvector for embedding storage. Requires the `pgvector` Postgres extension. |

pgvector is optional for MVP. Leave `ENABLE_PGVECTOR=false` for the default local setup.

---

## Adding provider credentials

To use real providers, add the relevant keys to `.env`. The fake provider continues to work without any keys for offline development.

```env
# OpenAI
OPENAI_API_KEY=sk-...

# AWS Bedrock (uses ambient IAM/credential chain — no key in .env needed)
PROVIDER_PROFILES__BEDROCK_PRIMARY__REGION=us-east-1
```

Never commit `.env` to version control. It is listed in `.gitignore`.
