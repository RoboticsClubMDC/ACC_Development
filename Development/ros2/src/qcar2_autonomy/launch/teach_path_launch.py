from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # manual_drive = Node(
    #     package='qcar2_autonomy',
    #     executable='manual_drive',
    #     name='manual_drive',
    #     output='screen',
    #     emulate_tty=True,
    # )

    path_teacher = Node(
        package='qcar2_autonomy',
        executable='path_teacher',
        name='path_teacher',
        output='screen',
        parameters=[{
            'global_frame': 'map',
            'node_interval_sec': 1.0,
            'min_node_spacing_m': 0.10,
            'min_node_yaw_change_rad': 0.20,
        }],
    )

    return LaunchDescription([
        # manual_drive,
        path_teacher,
    ])
