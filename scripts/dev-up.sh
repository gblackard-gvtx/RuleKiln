#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

API_PORT="${API_PORT:-8010}"
export API_PORT

if [[ ! -f .env ]]; then
  echo "No .env found — copying .env.example to .env"
  cp .env.example .env
fi

ARTIFACT_DIR="${PROJECT_ROOT}/.rulekiln"

ensure_artifact_dir_writable() {
  mkdir -p "${ARTIFACT_DIR}/runs"

  local probe_file
  probe_file="${ARTIFACT_DIR}/.write_probe"

  if touch "${probe_file}" >/dev/null 2>&1; then
    rm -f "${probe_file}"
    return
  fi

  echo "Artifact directory is not writable: ${ARTIFACT_DIR}"
  echo "Attempting permission repair using a temporary container..."
  docker run --rm -v "${ARTIFACT_DIR}:/data" alpine sh -c "chown -R $(id -u):$(id -g) /data && chmod -R u+rwX,g+rwX /data" >/dev/null

  if touch "${probe_file}" >/dev/null 2>&1; then
    rm -f "${probe_file}"
    echo "Artifact directory permissions repaired."
    return
  fi

  echo "Could not repair permissions for ${ARTIFACT_DIR}."
  echo "Run: sudo chown -R $(id -u):$(id -g) ${ARTIFACT_DIR}"
  exit 1
}

ensure_artifact_dir_writable

echo "Starting RuleKiln local Docker Compose stack..."
docker compose up -d --build

echo ""
echo "Services:"
echo "  API:    http://localhost:${API_PORT}"
echo "  MLflow: http://localhost:5000"
echo "  Docs:   http://localhost:${API_PORT}/docs"
echo ""
echo "Run 'scripts/dev-down.sh' to stop."
