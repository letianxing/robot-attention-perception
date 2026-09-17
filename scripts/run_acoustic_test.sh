#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "${REPO_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_DIR}/.venv-mac/bin/python"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8090}"
OPEN_BROWSER="${OPEN_BROWSER:-1}"
VOSK_MODEL="${VOICE_ASR_VOSK_MODEL:-${WORKSPACE_DIR}/voice-detection/weights/vosk-model-small-cn-0.22}"
ASR_MODEL="${ASR_MODEL:-${WORKSPACE_DIR}/voice-detection/weights/ggml-base.bin}"
ASR_COMMAND="${VOICE_ASR_COMMAND:-}"
SHERPA_MODEL="${VOICE_ASR_SHERPA_MODEL:-${WORKSPACE_DIR}/voice-detection/weights/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30}"

if [[ -z "${ASR_COMMAND}" && -x "/opt/homebrew/bin/whisper-cli" && -s "${ASR_MODEL}" ]]; then
  ASR_COMMAND="/opt/homebrew/bin/whisper-cli -m ${ASR_MODEL} -f {wav} -l zh -nt -np -sns"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "run ${REPO_DIR}/scripts/setup_mac_first_test.sh first" >&2
  exit 1
fi

URL="http://${HOST}:${PORT}"
if [[ "${OPEN_BROWSER}" == "1" ]]; then
  (sleep 1; open "${URL}") &
fi

echo "acoustic test: ${URL}"
echo "select mac_builtin + 1ch for Mac mic, or sipeed_6_plus_1_usb_array + 8ch for Sipeed"
echo "press Ctrl-C to stop microphone capture and dashboard"

cd "${WORKSPACE_DIR}/voice-detection"
exec env \
  VOICE_ASR_COMMAND="${ASR_COMMAND}" \
  VOICE_ASR_SHERPA_MODEL="${SHERPA_MODEL}" \
  VOICE_ASR_VOSK_MODEL="${VOSK_MODEL}" \
  VOICE_ASR_LANGUAGE="zh-CN" \
  VOICE_ASR_SAMPLE_RATE="16000" \
  "${PYTHON_BIN}" scripts/local_voice_dashboard.py --host "${HOST}" --port "${PORT}"
