"""
This is just a file for the connection of foxglove studio as a bridge
it will subscribe as a node, and inspect everything easier than usin rviz2
and rqt, and another tools by separated. If Qcar2 is bothered by this I would
recommend getting this off maybe.
"""

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
            parameters=[{
                'port': 8765,
                'address': '0.0.0.0',
                'use_compression': True,
                'send_buffer_limit': 10_000_000,
            }],
        ),
    ])