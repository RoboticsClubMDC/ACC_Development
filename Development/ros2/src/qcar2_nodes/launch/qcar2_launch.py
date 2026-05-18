# This is the launch file that starts up the basic QCar2 nodes

import subprocess

from launch import LaunchDescription
from launch.actions import (ExecuteProcess, LogInfo, RegisterEventHandler,
                             OpaqueFunction, TimerAction, DeclareLaunchArgument)
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration, PythonExpression
from launch.event_handlers import (OnProcessExit, OnProcessStart)
from launch.conditions import IfCondition

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # --- 2026-05-14: camera source selector (single-owner architecture) ---
    # 'depth_aligned' (default) -> qcar2_camera_bridge owns the RealSense via
    #                              the Quanser depth-align runtime and
    #                              publishes aligned 32FC1 depth and bgr8 RGB
    #                              on /camera/depth_image and /camera/color_image.
    # 'rgbd'                    -> legacy rgbd.cpp owns the RealSense and
    #                              publishes MONO16 raw depth.
    # Subscribers see the same topic names in either case.
    camera_source_arg = DeclareLaunchArgument(
        'camera_source',
        default_value='depth_aligned',
        description=("RealSense owner: 'depth_aligned' (qcar2_camera_bridge, "
                     "aligned 32FC1 meters) or 'rgbd' (legacy rgbd.cpp, "
                     "MONO16 raw)."))
    camera_source = LaunchConfiguration('camera_source')

    lidar_node = Node(
            package='qcar2_nodes',
            executable='lidar',
            name='Lidar'
        )

    # Legacy RealSense owner. Active only when camera_source == 'rgbd'.
    realsense_camera_node = Node(
            package='qcar2_nodes',
            executable='rgbd',
            name='RealsenseCamera',
            condition=IfCondition(
                PythonExpression(["'", camera_source, "' == 'rgbd'"]))
        )

    # Single-owner bridge — Python implementation in qcar2_autonomy.
    # Uses pit.YOLO.utils.QCar2DepthAligned directly (same class
    # legacy yolo_detector.py used) which achieves ~30 Hz on this
    # Jetson. A C++ counterpart exists in qcar2_nodes/src/
    # qcar2_camera_bridge.cpp but was pinned at ~0.3 fps by
    # kernel-level constraints we couldn't reach from pure C; see the
    # 2026-05-14 entries in VO_CHANGELOG.md for the full forensic.
    camera_bridge_node = Node(
            package='qcar2_autonomy',
            executable='camera_bridge',
            name='RealsenseCamera',
            condition=IfCondition(
                PythonExpression(["'", camera_source, "' == 'depth_aligned'"]))
        )

    csi_camera_node = Node(
            package='qcar2_nodes',
            executable='csi',
            name='csi_camera'
        )

    qcar2_hardware = Node(
            package='qcar2_nodes',
            executable='qcar2_hardware',
            name='qcar2_hardware',
        )

    return LaunchDescription([
        camera_source_arg,
        lidar_node,
        realsense_camera_node,
        camera_bridge_node,
        csi_camera_node,
        qcar2_hardware,
    ])
