#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "${REPO_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_DIR}/.venv-mac/bin/python"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
CAMERA_INDEX="${CAMERA_INDEX:-0}"
SOURCE_TYPE="${SOURCE_TYPE:-opencv}"
SOURCE_ID="${SOURCE_ID:-${CAMERA_INDEX}}"
OPEN_BROWSER="${OPEN_BROWSER:-1}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "run ${REPO_DIR}/scripts/setup_mac_first_test.sh first" >&2
  exit 1
fi

URL="http://${HOST}:${PORT}"
if [[ "${OPEN_BROWSER}" == "1" ]]; then
  (sleep 2; open "${URL}") &
fi

echo "visual test: ${URL}"
echo "source=${SOURCE_TYPE}:${SOURCE_ID}; press Ctrl-C to release the camera"

cd "${WORKSPACE_DIR}/vision-detection"
exec "${PYTHON_BIN}" scripts/local_vision_dashboard.py \
  --host "${HOST}" --port "${PORT}" \
  --camera-index "${CAMERA_INDEX}" \
  --source-type "${SOURCE_TYPE}" --source-id "${SOURCE_ID}" \
  --emotion-backend ferplus --stream-fps 20 --inference-fps 8
