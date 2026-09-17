#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "${REPO_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_DIR}/.venv-mac/bin/python"
HOST="${HOST:-127.0.0.1}"
VISION_PORT="${VISION_PORT:-8080}"
VOICE_PORT="${VOICE_PORT:-8090}"
CONSOLE_PORT="${CONSOLE_PORT:-8092}"
MEMORY_PORT="${MEMORY_PORT:-8788}"
ROSBRIDGE_PORT="${ROSBRIDGE_PORT:-9090}"
SESSION_ID="${SESSION_ID:-mac-first-test}"
OPEN_BROWSER="${OPEN_BROWSER:-1}"
RUNTIME_DIR="/tmp/robot-attention-perception-mac"
ASR_MODEL="${ASR_MODEL:-${WORKSPACE_DIR}/voice-detection/weights/ggml-base.bin}"
ASR_COMMAND="${VOICE_ASR_COMMAND:-}"
SHERPA_MODEL="${VOICE_ASR_SHERPA_MODEL:-${WORKSPACE_DIR}/voice-detection/weights/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30}"
VOSK_MODEL_CANDIDATE="${VOICE_ASR_VOSK_MODEL:-${WORKSPACE_DIR}/voice-detection/weights/vosk-model-small-cn-0.22}"
VOSK_MODEL=""
if [[ -d "${VOSK_MODEL_CANDIDATE}" ]]; then
  VOSK_MODEL="${VOSK_MODEL_CANDIDATE}"
fi

if [[ -z "${ASR_COMMAND}" && -x "/opt/homebrew/bin/whisper-cli" && -s "${ASR_MODEL}" ]]; then
  ASR_COMMAND="/opt/homebrew/bin/whisper-cli -m ${ASR_MODEL} -f {wav} -l zh -nt -np -sns"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "run scripts/setup_mac_first_test.sh first" >&2
  exit 1
fi
if [[ "${VOICE_AUDIO_SCENE:-1}" == "1" ]]; then
  for model_dir in "${VOICE_AUDIO_MODEL:-${WORKSPACE_DIR}/voice-detection/weights/ast-audioset}" "${VOICE_GENRE_MODEL:-${WORKSPACE_DIR}/voice-detection/weights/music-genre}"; do
    if [[ ! -s "${model_dir}/model.safetensors" ]]; then
      echo "audio model missing: ${model_dir}; run ${PYTHON_BIN} ${WORKSPACE_DIR}/voice-detection/scripts/setup_audio_perception.py" >&2
      exit 1
    fi
  done
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker Desktop is not running" >&2
  exit 1
fi

PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  docker compose -f "${REPO_DIR}/docker-compose.mac.yml" down >/dev/null 2>&1 || true
  rm -f "${RUNTIME_DIR}/launcher.pid"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir -p "${RUNTIME_DIR}"
echo "$$" > "${RUNTIME_DIR}/launcher.pid"

ROSBRIDGE_PORT="${ROSBRIDGE_PORT}" docker compose -f "${REPO_DIR}/docker-compose.mac.yml" up -d --build

(cd "${WORKSPACE_DIR}/hri-memory-service" && cargo run -- --host "${HOST}" --port "${MEMORY_PORT}" --data-dir ./data) &
PIDS+=("$!")

"${PYTHON_BIN}" "${WORKSPACE_DIR}/vision-detection/scripts/local_vision_dashboard.py" \
  --host "${HOST}" --port "${VISION_PORT}" --camera-index 0 \
  --emotion-backend ferplus --stream-fps 20 --inference-fps 8 &
PIDS+=("$!")

env VOICE_ASR_COMMAND="${ASR_COMMAND}" VOICE_ASR_VOSK_MODEL="${VOSK_MODEL}" VOICE_ASR_SHERPA_MODEL="${SHERPA_MODEL}" \
VOICE_ASR_LANGUAGE="zh-CN" VOICE_ASR_SAMPLE_RATE="16000" \
VOICE_ROSBRIDGE_URL="ws://${HOST}:${ROSBRIDGE_PORT}" \
"${PYTHON_BIN}" "${WORKSPACE_DIR}/voice-detection/scripts/local_voice_dashboard.py" \
  --host "${HOST}" --port "${VOICE_PORT}" &
PIDS+=("$!")

CONSOLE_ARGS=(
  -m robot_attention_perception.live_dashboard
  --host "${HOST}" --port "${CONSOLE_PORT}"
  --vision-url "http://${HOST}:${VISION_PORT}"
  --voice-url "http://${HOST}:${VOICE_PORT}"
  --memory-url "http://${HOST}:${MEMORY_PORT}"
  --rosbridge-url "ws://${HOST}:${ROSBRIDGE_PORT}"
  --session-id "${SESSION_ID}"
)
if [[ "${ATTENTION_S2S_ENABLED:-0}" == "1" ]] && [[ -n "${ASR_COMMAND}" || -d "${VOSK_MODEL}" || -d "${SHERPA_MODEL}" ]]; then
  CONSOLE_ARGS+=(--s2s)
fi
if [[ -n "${S2S_LLM_URL:-}" ]]; then
  CONSOLE_ARGS+=(--llm-url "${S2S_LLM_URL}" --llm-model "${S2S_LLM_MODEL:-local-model}")
fi
"${PYTHON_BIN}" "${CONSOLE_ARGS[@]}" &
PIDS+=("$!")

"${PYTHON_BIN}" - "${HOST}" "${VISION_PORT}" "${VOICE_PORT}" "${CONSOLE_PORT}" "${MEMORY_PORT}" "${ROSBRIDGE_PORT}" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
ports = [int(value) for value in sys.argv[2:]]
deadline = time.monotonic() + 180
pending = set(ports)
while pending and time.monotonic() < deadline:
    for port in list(pending):
        try:
            with socket.create_connection((host, port), timeout=0.3):
                pending.remove(port)
        except OSError:
            pass
    time.sleep(0.2)
if pending:
    raise SystemExit(f"services did not become ready: {sorted(pending)}")
PY

URL="http://${HOST}:${CONSOLE_PORT}"
echo "perception console: ${URL}"
echo "ROS bridge: ws://${HOST}:${ROSBRIDGE_PORT}"
echo "press Ctrl-C to stop native services and the ROS2 container"
if [[ "${OPEN_BROWSER}" == "1" ]]; then
  open "${URL}"
fi

while true; do
  for pid in "${PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"
      exit $?
    fi
  done
  sleep 1
done
