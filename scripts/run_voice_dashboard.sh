#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOLANDS_ROOT:-$HOME/Golands}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8090}"

cd "$ROOT/voice-detection"
exec "${PYTHON_BIN:-python3}" scripts/local_voice_dashboard.py --host "$HOST" --port "$PORT"
