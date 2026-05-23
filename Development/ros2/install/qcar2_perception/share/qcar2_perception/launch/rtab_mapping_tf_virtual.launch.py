import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("qcar2_perception")

    semantic_tf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(package_share, "launch", "semantic_tf_launch.py")
        )
    )

    lidar_tf = Node(
        package="qcar2_nodes",
        executable="fixed_lidar_frame_virtual",
        name="fixed_lidar_frame",
        output="screen",
    )

    return LaunchDescription([
        semantic_tf,
        lidar_tf,
    ])
