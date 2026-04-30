# Keyboard-drive base stack for physical QCar2.
# This launch intentionally excludes the joystick command node and includes
# nav2_qcar2_converter so /cmd_vel_nav (Twist) can drive qcar2_motor_speed_cmd.

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    lidar_node = Node(
        package='qcar2_nodes',
        executable='lidar',
        name='Lidar'
    )

    realsense_camera_node = Node(
        package='qcar2_nodes',
        executable='rgbd',
        name='RealsenseCamera'
    )

    downward_facing_camera_node = Node(
        package='qcar2_nodes',
        executable='csi',
        name='csi_camera'
    )

    qcar2_hardware = Node(
        package='qcar2_nodes',
        executable='qcar2_hardware',
        name='qcar2_hardware',
    )

    nav2_qcar2_converter = Node(
        package='qcar2_nodes',
        executable='nav2_qcar2_converter',
        name='nav2_qcar2_converter',
    )

    return LaunchDescription([
        lidar_node,
        realsense_camera_node,
        downward_facing_camera_node,
        qcar2_hardware,
        nav2_qcar2_converter,
    ])
