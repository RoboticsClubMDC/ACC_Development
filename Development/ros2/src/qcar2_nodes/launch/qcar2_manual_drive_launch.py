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

    # See qcar2_launch.py for the camera_source rationale.
    camera_source_arg = DeclareLaunchArgument(
        'camera_source',
        default_value='depth_aligned',
        description=("RealSense owner: 'depth_aligned' (qcar2_camera_bridge) "
                     "or 'rgbd' (legacy rgbd.cpp)."))
    camera_source = LaunchConfiguration('camera_source')

    lidar_node = Node(
            package='qcar2_nodes',
            executable='lidar',
            name='Lidar'
        )

    realsense_camera_node = Node(
            package='qcar2_nodes',
            executable='rgbd',
            name='RealsenseCamera',
            condition=IfCondition(
                PythonExpression(["'", camera_source, "' == 'rgbd'"]))
        )

    # Python bridge in qcar2_autonomy (see qcar2_launch.py for rationale).
    camera_bridge_node = Node(
            package='qcar2_autonomy',
            executable='camera_bridge',
            name='RealsenseCamera',
            condition=IfCondition(
                PythonExpression(["'", camera_source, "' == 'depth_aligned'"]))
        )

    downward_facing_camera_node = Node(
            package='qcar2_nodes',
            executable='csi',
            name='csi_camera',
            parameters=[{'camera_num': 3}]
        )

    qcar2_hardware = Node(
            package='qcar2_nodes',
            executable='qcar2_hardware',
            name='qcar2_hardware',
        )

    joystick_command = Node(
        package = 'qcar2_nodes',
        executable = 'command',
        name = 'joystick_command'
    )
    return LaunchDescription([
        camera_source_arg,
        lidar_node,
        realsense_camera_node,
        camera_bridge_node,
        downward_facing_camera_node,
        qcar2_hardware,
        joystick_command
    ])
