#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOLANDS_ROOT:-$HOME/Golands}"
HOST="${HOST:-127.0.0.1}"
MEMORY_PORT="${MEMORY_PORT:-8788}"
VOICE_PORT="${VOICE_PORT:-8090}"
ATTENTION_PORT="${ATTENTION_PORT:-8091}"
VISION_PORT="${VISION_PORT:-8080}"
START_VISION="${START_VISION:-1}"

PIDS=()

cleanup() {
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}

trap cleanup EXIT INT TERM

start_service() {
  local name="$1"
  shift
  echo "starting ${name}: $*"
  "$@" &
  PIDS+=("$!")
}

start_service "hri-memory-service" \
  env HOST="$HOST" PORT="$MEMORY_PORT" "$ROOT/robot-attention-perception/scripts/run_memory.sh"

start_service "voice dashboard" \
  env HOST="$HOST" PORT="$VOICE_PORT" "$ROOT/robot-attention-perception/scripts/run_voice_dashboard.sh"

start_service "attention dashboard" \
  env HOST="$HOST" PORT="$ATTENTION_PORT" "$ROOT/robot-attention-perception/scripts/run_attention_dashboard.sh"

if [[ "$START_VISION" == "1" ]]; then
  start_service "vision dashboard" \
    env HOST="$HOST" PORT="$VISION_PORT" "$ROOT/robot-attention-perception/scripts/run_vision_dashboard.sh"
fi

echo "memory:    http://${HOST}:${MEMORY_PORT}/v1/health"
echo "voice:     http://${HOST}:${VOICE_PORT}"
echo "attention: http://${HOST}:${ATTENTION_PORT}"
if [[ "$START_VISION" == "1" ]]; then
  echo "vision:    http://${HOST}:${VISION_PORT}"
fi
echo "press Ctrl-C to stop all started services"

while true; do
  for pid in "${PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"
      exit $?
    fi
  done
  sleep 1
done
