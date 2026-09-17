#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOLANDS_ROOT:-$HOME/Golands}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8788}"
DATA_DIR="${DATA_DIR:-./data}"

cd "$ROOT/hri-memory-service"
exec cargo run -- --host "$HOST" --port "$PORT" --data-dir "$DATA_DIR"
