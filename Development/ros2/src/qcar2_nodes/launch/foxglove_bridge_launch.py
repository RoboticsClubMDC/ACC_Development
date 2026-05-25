"""
Foxglove bridge launch — Foxglove is the principal watcher/dashboard for the
QCar2 stack, so any always-on diagnostic node that produces data exclusively
for Foxglove visualization is launched alongside the bridge here.

Currently bundled:
  - foxglove_bridge   — WebSocket bridge on ws://localhost:8765
  - controller_watchdog — publishes /nav/controller_health for an Indicator panel

If the QCar2 ever feels strained, the watchdog can be commented out
without affecting the controller — it's a pure observer.
"""

from ament_index_python.packages import PackageNotFoundError, get_package_prefix
from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node

def package_available(package_name):
    try:
        get_package_prefix(package_name)
        return True
    except PackageNotFoundError:
        return False

def generate_launch_description():
    actions = []

    if package_available('foxglove_bridge'):
        actions.append(Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
            parameters=[{
                'port': 8765,
                'address': '0.0.0.0',
                'use_compression': True,
                'send_buffer_limit': 10_000_000,
            }],
        ))
    else:
        actions.append(LogInfo(
            msg=(
                "foxglove_bridge package is not installed in this ROS overlay. "
                "Run this launch inside the Isaac ROS container, or install/build "
                "foxglove_bridge in this native ROS overlay before using it here. "
                "Skipping controller_watchdog because this launch is only useful "
                "when the bridge is available."
            )
        ))
        return LaunchDescription(actions)

    controller_watchdog = Node(
        package='qcar2_autonomy',
        executable='controller_watchdog',
        name='controller_watchdog',
        output='screen',
    )

    actions.append(controller_watchdog)

    return LaunchDescription(actions)
