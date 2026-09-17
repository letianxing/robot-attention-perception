#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOLANDS_ROOT:-$HOME/Golands}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"

cd "$ROOT/vision-detection"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ "${PYTHON_BIN}" == "python3" && -x ".venv312/bin/python" ]]; then
  PYTHON_BIN=".venv312/bin/python"
elif [[ "${PYTHON_BIN}" == "python3" && -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

exec "$PYTHON_BIN" scripts/local_vision_dashboard.py --host "$HOST" --port "$PORT"
