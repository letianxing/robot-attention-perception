#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "${REPO_DIR}/.." && pwd)"
VENV_DIR="${REPO_DIR}/.venv-mac"
PYTHON_CMD="${PYTHON_CMD:-python3.12}"

if ! command -v "${PYTHON_CMD}" >/dev/null 2>&1; then
  echo "missing ${PYTHON_CMD}; install Python 3.12 first" >&2
  exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_CMD}" -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install \
  numpy \
  opencv-contrib-python \
  aiohttp \
  sounddevice \
  websocket-client \
  "mediapipe==0.10.32" \
  vosk
"${VENV_DIR}/bin/python" -m pip install -e "${WORKSPACE_DIR}/voice-detection[live,scene]"
"${VENV_DIR}/bin/python" -m pip install -e "${REPO_DIR}"

"${VENV_DIR}/bin/python" - <<'PY'
import aiohttp
import cv2
import numpy
import sounddevice
import websocket
print("Mac perception Python dependencies are ready")
PY

echo "environment: ${VENV_DIR}"
