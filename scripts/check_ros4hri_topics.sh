#!/usr/bin/env bash
set -euo pipefail

command -v ros2 >/dev/null || { echo "ros2 is not on PATH; source /opt/ros/<jazzy|humble>/setup.bash first" >&2; exit 2; }

echo "ROS_DISTRO=${ROS_DISTRO:-unknown}"
echo "--- required ROS4HRI topics ---"
topics=(
  /humans/faces/tracked
  /humans/bodies/tracked
  /humans/voices/tracked
  /humans/persons/tracked
  /humans/candidate_matches
  /perception/vision/people
  /perception/voice/tracks
  /perception/attention/state
  /attention/look_at
)
available="$(ros2 topic list 2>/dev/null || true)"
missing=0
for topic in "${topics[@]}"; do
  if grep -Fxq "${topic}" <<<"${available}"; then
    echo "[OK]   ${topic}"
  else
    echo "[MISS] ${topic}"
    missing=1
  fi
done

echo "--- active nodes ---"
ros2 node list | grep -E 'attention|person|ingress|face|emotion|body' || true
exit "${missing}"
