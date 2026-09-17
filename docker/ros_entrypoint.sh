#!/usr/bin/env bash
set -eo pipefail
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source /ros2_ws/install/setup.bash
set -u
export PYTHONPATH="/workspace/robot-attention-perception:${PYTHONPATH:-}"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

ros2 launch rosbridge_server rosbridge_websocket_launch.xml port:=9090 &
PIDS+=("$!")
python3 /workspace/robot-attention-perception/scripts/ros4hri_native_ingress.py &
PIDS+=("$!")
ros2 run attention_stack attention_stack_node --ros-args \
  --params-file /ros2_ws/install/share/attention_stack/config/attention_stack.yaml &
PIDS+=("$!")

while true; do
  for pid in "${PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"
      exit $?
    fi
  done
  sleep 1
done
