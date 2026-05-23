#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

if [[ ! -f .env ]]; then
  echo "No .env found — copying .env.example to .env"
  cp .env.example .env
fi

# Create required local directories before Docker Compose starts.
# .rulekiln/runs is the ARTIFACT_ROOT bind-mounted into the api and worker
# containers; creating it here ensures correct ownership (not root-owned).
echo "Creating required directories..."
mkdir -p .rulekiln/runs

echo "Starting RuleKiln local Docker Compose stack..."
docker compose up -d --build

echo ""
echo "Services:"
echo "  API:    http://localhost:8000"
echo "  MLflow: http://localhost:5000"
echo "  Docs:   http://localhost:8000/docs"
echo ""
echo "Run 'scripts/dev-down.sh' to stop."
