FROM python:3.13-slim

WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy dependency metadata first for layer caching
COPY pyproject.toml ./

# Install all runtime dependencies (no dev extras)
RUN uv pip install --system --no-cache ".[dev]"

# Copy source
COPY src/ src/
COPY migrations/ migrations/
COPY alembic.ini ./
COPY main.py ./

# Non-root user for security
RUN adduser --disabled-password --gecos "" rulekiln && \
    chown -R rulekiln:rulekiln /app
USER rulekiln

EXPOSE 8000

CMD ["uvicorn", "src.rulekiln.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
