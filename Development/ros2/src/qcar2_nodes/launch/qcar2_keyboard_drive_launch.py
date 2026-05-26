# Keyboard-drive base stack for physical QCar2.
# This launch intentionally excludes the joystick command node and includes
# nav2_qcar2_converter so /cmd_vel_nav (Twist) can drive qcar2_motor_speed_cmd.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node


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

    nav2_qcar2_converter = Node(
        package='qcar2_nodes',
        executable='nav2_qcar2_converter',
        name='nav2_qcar2_converter',
    )

    return LaunchDescription([
        camera_source_arg,
        lidar_node,
        realsense_camera_node,
        camera_bridge_node,
        downward_facing_camera_node,
        qcar2_hardware,
        nav2_qcar2_converter,
    ])
