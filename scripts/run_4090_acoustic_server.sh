#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOLANDS_ROOT:-$HOME/Golands}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-9097}"

cd "$ROOT/voice-detection"
exec python3 -m voice_detection.cli run-remote-server --host "$HOST" --port "$PORT"
