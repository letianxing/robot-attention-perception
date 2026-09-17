ARG ROS_BASE_IMAGE=pacific-rim-ros2:humble
ARG ROS_DISTRO=humble
FROM ${ROS_BASE_IMAGE}
ARG ROS_DISTRO=humble
ENV ROS_DISTRO=${ROS_DISTRO}

USER root
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -o Acquire::Retries=5 update \
    && apt-get install -y --no-install-recommends \
       python3-pip \
       python3-colcon-common-extensions \
       git \
       ros-${ROS_DISTRO}-hri-msgs \
       ros-${ROS_DISTRO}-rosbridge-server \
       ros-${ROS_DISTRO}-tf2-ros \
    && rm -rf /var/lib/apt/lists/*

COPY ros2/perception_interfaces /ros2_ws/src/perception_interfaces
COPY ros2/attention_stack /ros2_ws/src/attention_stack
# Consume the existing authoritative package; do not duplicate .msg sources in this repository.
COPY --from=brain_audio_interfaces / /ros2_ws/src/audio_msgs
COPY scripts /ros2_ws/scripts
RUN . /opt/ros/${ROS_DISTRO}/setup.sh \
    && cd /ros2_ws \
    && colcon build --merge-install

COPY docker/ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
