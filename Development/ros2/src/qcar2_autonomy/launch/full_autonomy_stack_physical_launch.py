# 2026-05-27: one-shot launch for the physical-robot autonomy stack.
#
# Mirrors full_autonomy_stack_launch.py but for the PHYSICAL QCar 2 on the
# Jetson AGX Orin (native ~/ros2 workspace, not the laptop dev container).
#
# Spawns:
#   - qcar2_perception perception_core_physical.launch.py
#       (D435 source + YOLO + semantic landmarks + TF, mode=internal by
#        default — the recommended "everything on the Jetson" flow per
#        CLAUDE.md §4)
#   - qcar2_autonomy path_follower
#       (pure pursuit, defaults to publishing /cmd_vel_path)
#   - qcar2_perception lane_lanenet_stanley_launch.py
#       (lane_detector + lane_stanley_controller + cmd_vel_blender,
#        camera_source=d435 default which is correct for physical)
#
# Does NOT include:
#   - AMCL stack (run carto_to_amcl.sh physical or amcl_load.sh physical first)
#   - The "node_values" arm command (set manually after launch)
#
# Usage (ON the QCar 2 Jetson, after sync + native build):
#   cd ~/ros2 && source install/setup.bash && export ROS_DOMAIN_ID=69
#   ros2 launch qcar2_autonomy full_autonomy_stack_physical_launch.py
#   # then in another terminal:
#   ros2 param set /path_follower node_values "[0, 6, 8]"
#
# Optional launch args (passthrough to perception_core_physical):
#   mode:=internal | external                   default: internal
#   source_only:=true | false                   default: false
#   enable_landmark_correction:=true | false    default: false

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    perception_share = get_package_share_directory("qcar2_perception")

    mode = LaunchConfiguration("mode")
    source_only = LaunchConfiguration("source_only")
    enable_landmark_correction = LaunchConfiguration("enable_landmark_correction")

    perception_core = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(perception_share, "launch",
                         "perception_core_physical.launch.py")
        ),
        launch_arguments={
            "mode": mode,
            "source_only": source_only,
            "enable_landmark_correction": enable_landmark_correction,
        }.items(),
    )

    path_follower = Node(
        package="qcar2_autonomy",
        executable="path_follower",
        name="path_follower",
        output="screen",
    )

    lane_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(perception_share, "launch",
                         "lane_lanenet_stanley_launch.py")
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "mode",
            default_value="internal",
            choices=["internal", "external"],
            description=(
                "Where perception runs. 'internal' (default, recommended): "
                "full stack on QCar2 Jetson. 'external': D435 source must be "
                "running elsewhere (paired with internal source_only:=true)."
            ),
        ),
        DeclareLaunchArgument(
            "source_only",
            default_value="false",
            choices=["true", "false"],
            description=(
                "When mode=internal: 'true' starts only the D435 source on the "
                "Jetson (and YOLO/landmarks run external). Default 'false' = "
                "Jetson runs everything."
            ),
        ),
        DeclareLaunchArgument(
            "enable_landmark_correction",
            default_value="false",
            description=(
                "Phase-4 landmark→EKF correction publishing. Default OFF — see "
                "CLAUDE.md §7.9 for the three prerequisites before enabling."
            ),
        ),
        perception_core,
        path_follower,
        lane_stack,
    ])
