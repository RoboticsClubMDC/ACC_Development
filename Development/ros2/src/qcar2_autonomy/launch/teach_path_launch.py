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
            'steer_slew':            12.0,
            'require_enable_button': True,
            'enable_button':         4,    # LB
            'reverse_button':        0,    # A
            'stop_button':           1,    # B
            'steer_axis':            0,    # left stick X
            'trigger_axis':          5,    # right trigger RT
            'use_trigger':           True,
            'undo_button':           3,    # Y
            'node_delete_wait':      30.0,
        }],
    )

    path_teacher = Node(
        package='qcar2_autonomy',
        executable='path_teacher',
        name='path_teacher',
        output='screen',
        parameters=[{
            'global_frame':             'map',
            'node_interval_sec':        0.1,
            'min_node_spacing_m':       0.05,
            'min_node_yaw_change_rad':  0.10,
            'line_marker_width':        0.025,
            'node_delete_wait':         30.0,
        }],
    )

    return LaunchDescription([
        controller_drive,
        path_teacher,
    ])
