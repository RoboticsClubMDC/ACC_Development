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

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    foxglove_bridge = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        parameters=[{
            'port': 8765,
            'address': '0.0.0.0',
            'use_compression': True,
            'send_buffer_limit': 10_000_000,
        }],
    )

    controller_watchdog = Node(
        package='qcar2_autonomy',
        executable='controller_watchdog',
        name='controller_watchdog',
        output='screen',
    )

    return LaunchDescription([
        foxglove_bridge,
        controller_watchdog,
    ])