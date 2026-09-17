"""ROS4HRI-first launch for the attention perception skeleton.

The detector backends remain configurable (the local vision/voice projects can
publish into the ingress topics), while the official ROS4HRI person manager,
engagement node and visualizers are enabled when installed in the workspace.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _include(package: str, launch_file: str, condition):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare(package), "launch", launch_file])
        ),
        condition=condition,
    )


def generate_launch_description():
    use_person_manager = LaunchConfiguration("use_person_manager")
    use_engagement = LaunchConfiguration("use_engagement")
    use_visualization = LaunchConfiguration("use_visualization")
    use_pyhri_adapter = LaunchConfiguration("use_pyhri_adapter")
    use_official_detectors = LaunchConfiguration("use_official_detectors")
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_person_manager", default_value="true"),
            DeclareLaunchArgument("use_engagement", default_value="false"),
            DeclareLaunchArgument("use_visualization", default_value="false"),
            DeclareLaunchArgument("use_pyhri_adapter", default_value="true"),
            DeclareLaunchArgument("use_official_detectors", default_value="false"),
            _include("hri_face_detect", "hri_face_detect.launch.py", IfCondition(use_official_detectors)),
            _include("hri_emotion_recognizer", "emotion_recognizer.launch.py", IfCondition(use_official_detectors)),
            _include("hri_body_detect", "hri_body_detect.launch.py", IfCondition(use_official_detectors)),
            _include("hri_person_manager", "person_manager.launch.py", IfCondition(use_person_manager)),
            _include("hri_engagement", "monitor.launch.py", IfCondition(use_engagement)),
            Node(
                package="attention_stack",
                executable="attention_stack_node",
                name="attention_stack",
                output="screen",
            ),
            Node(
                package="attention_stack",
                executable="pyhri_people_adapter.py",
                name="pyhri_people_adapter",
                output="screen",
                condition=IfCondition(use_pyhri_adapter),
            ),
            # hri_visualization and rqt_human_radar are GUI tools with
            # package-specific entry points. They are documented and launched
            # separately when use_visualization is enabled; keeping them out
            # of this launch avoids making headless deployments depend on Qt.
        ]
    )
