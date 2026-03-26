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
            'node_interval_sec': 1.0,
        }],
    )

    return LaunchDescription([
        # manual_drive,
        path_teacher,
    ])
