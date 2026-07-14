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

# Non-root user for security. Only chown the dirs the app actually writes to
# at runtime (everything else is world-readable, so no need to walk the full
# tree — a recursive chown over the venv's ~48k files added several minutes
# to the build).
RUN adduser --disabled-password --gecos "" rulekiln && \
    mkdir -p /app/.rulekiln /app/logs && \
    chown rulekiln:rulekiln /app /app/.rulekiln /app/logs
USER rulekiln

EXPOSE 8000

CMD ["uvicorn", "src.rulekiln.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
