#!/usr/bin/env bash
set -euo pipefail

PID_FILE="/tmp/robot-attention-perception-mac/launcher.pid"
if [[ ! -f "${PID_FILE}" ]]; then
  echo "Mac perception stack is not running"
  exit 0
fi

PID="$(tr -cd '0-9' < "${PID_FILE}")"
if [[ -z "${PID}" ]] || ! kill -0 "${PID}" 2>/dev/null; then
  rm -f "${PID_FILE}"
  echo "Removed stale launcher PID"
  exit 0
fi

COMMAND="$(ps -p "${PID}" -o command=)"
if [[ "${COMMAND}" != *"run_mac_first_test.sh"* ]]; then
  echo "Refusing to stop unrelated PID ${PID}: ${COMMAND}" >&2
  exit 1
fi

kill -INT "${PID}"
echo "Stopping Mac perception stack (PID ${PID})"
