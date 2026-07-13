# syntax=docker/dockerfile:1
FROM python:3.13-slim

WORKDIR /app

# Prebuilt uv binary (faster than `pip install uv`)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# uv build behavior:
# - copy link mode is required when the cache lives on a mount
# - compile bytecode for faster container startup
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# 1) Install ONLY third-party deps first, without the project or dev group.
#    Cached across source edits: only busts when pyproject.toml / uv.lock change.
#    The uv cache mount preserves downloaded/built wheels (e.g. hdbscan) across builds.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# 2) Copy source and install the project itself (editable). Source edits only
#    re-run this cheap step, not the full dependency install above.
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

COPY migrations/ migrations/
COPY alembic.ini ./
COPY main.py ./

# Non-root user for security
RUN adduser --disabled-password --gecos "" rulekiln && \
    chown -R rulekiln:rulekiln /app
USER rulekiln

EXPOSE 8000

CMD ["uvicorn", "src.rulekiln.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
