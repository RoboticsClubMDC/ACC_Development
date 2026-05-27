# 2026-05-27: one-shot launch for the user-facing autonomy stack.
#
# Spawns, in one terminal, everything Terminal-B / C / D used to spawn:
#   - qcar2_perception perception_core_virtual.launch.py
#       (D435 source + YOLO + semantic landmarks + TF)
#   - qcar2_autonomy path_follower
#       (pure pursuit, defaults to publishing /cmd_vel_path)
#   - qcar2_perception lane_lanenet_stanley_launch.py
#       (lane_detector + lane_stanley_controller + cmd_vel_blender)
#
# Does NOT include:
#   - AMCL stack (run carto_to_amcl.sh or amcl_load.sh first)
#   - The "node_values" arm command (set manually after launch)
#
# Usage (after AMCL is up in another terminal):
#   ros2 launch qcar2_autonomy full_autonomy_stack_launch.py
#   ros2 param set /path_follower node_values "[0, 6, 8]"

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    perception_share = get_package_share_directory("qcar2_perception")

    # 2026-05-27: VIRTUAL stack runs on the laptop dev container which
    # typically does NOT have CUDA passthrough. Force CPU for YOLO and
    # the D435 source so torch doesn't waste time probing for an absent
    # GPU and falling back. Read by semantic_yolo_detector.py:42 and
    # d435_aligned_source.py:14. LaneNet stays on HSV backend (default
    # in lane_lanenet_stanley_launch.py) which doesn't need CUDA.
    force_cpu = SetEnvironmentVariable(name="QCAR2_FORCE_CPU", value="1")

    perception_core = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(perception_share, "launch",
                         "perception_core_virtual.launch.py")
        )
    )

    path_follower = Node(
        package="qcar2_autonomy",
        executable="path_follower",
        name="path_follower",
        output="screen",
    )

    # 2026-05-27: trip_planner is now bundled. It polls services lazily so
    # startup order within the launch group doesn't matter.
    trip_planner = Node(
        package="qcar2_autonomy",
        executable="trip_planner",
        name="trip_planner",
        output="screen",
    )

    lane_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(perception_share, "launch",
                         "lane_lanenet_stanley_launch.py")
        )
    )

    return LaunchDescription([
        force_cpu,
        perception_core,
        path_follower,
        trip_planner,
        lane_stack,
    ])
