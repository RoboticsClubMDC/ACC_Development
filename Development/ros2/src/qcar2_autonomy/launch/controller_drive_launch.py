from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    controller_drive = Node(
        package='qcar2_autonomy',
        executable='controller_drive',
        name='controller_drive',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'cmd_topic':             '/cmd_vel_nav',
            'max_speed':             0.2,
            'max_turn':              2.0,
            'deadzone':              0.08,
            'publish_hz':            30.0,
            'speed_slew':            1.0,
            'steer_slew':            3.0,
            'require_enable_button': True,
            'enable_button':         4,    # LB
            'reverse_button':        0,    # A
            'stop_button':           1,    # B
            'steer_axis':            0,    # left stick X
            'trigger_axis':          5,    # right trigger RT
            'use_trigger':           True,
        }],
    )

    return LaunchDescription([controller_drive])
