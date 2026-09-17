#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "${REPO_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_DIR}/.venv-mac/bin/python"
DEVICE="${1:-}"
DURATION_SEC="${2:-10}"
OUTPUT="${3:-${WORKSPACE_DIR}/voice-detection/data/sipeed_8ch_48k_test.wav}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "run ${REPO_DIR}/scripts/setup_mac_first_test.sh first" >&2
  exit 1
fi

cd "${WORKSPACE_DIR}/voice-detection"
if [[ -z "${DEVICE}" ]]; then
  "${PYTHON_BIN}" -m voice_detection.cli list-devices
  echo "usage: $0 <Sipeed device index> [seconds] [output.wav]" >&2
  exit 2
fi

"${PYTHON_BIN}" -m voice_detection.cli record-live \
  --profile sipeed_6_plus_1_usb_array \
  --device "${DEVICE}" --seconds "${DURATION_SEC}" --output "${OUTPUT}"

echo "saved: ${OUTPUT}"
