#!/usr/bin/env bash
set -euo pipefail

ROOT="${GOLANDS_ROOT:-$HOME/Golands}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8091}"
SCENARIO="${SCENARIO:-fixtures/cocktail_effect_three_people.json}"

cd "$ROOT/robot-attention-perception"
exec python3 -m robot_attention_perception.cli dashboard \
  --host "$HOST" \
  --port "$PORT" \
  --scenario "$SCENARIO"
