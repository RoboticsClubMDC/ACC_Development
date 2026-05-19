from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    path_follower = Node(
        package='qcar2_autonomy',
        executable='path_follower',
        name='path_follower',
        parameters=[{
            'route_frame':      'map',
            'desired_speed':    [0.15],   # slow for accurate mapping
            'min_lookahead_m':  0.20,     # must be << 0.678 m roundabout radius
            'lookahead_gain':   1.0,      # keeps effective lookahead near min_lookahead_m
            'finish_progress_window': 80,
            'assume_start_at_origin': True,
            'steering_sign':    1.0,
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
            'mapping_enabled':          True,
            'node_delete_wait':         30.0,
            'assume_start_at_origin':   True,
        }],
    )

    auto_teach_mapper = Node(
        package='qcar2_autonomy',
        executable='auto_teach_mapper',
        name='auto_teach_mapper',
        output='screen',
        parameters=[{
            'route_frame':              'map',
            'waypoint_spacing':         0.03,
            'publish_delay_sec':        3.0,
            'spawn_xy':                 [0.0, 0.0],
            'spawn_yaw_deg':            -33.0,
            'align_start_yaw_to_robot': True,
            'anchor_start_to_spawn':    True,
            'align_start_yaw_to_spawn': True,
        }],
    )

    return LaunchDescription([
        path_follower,
        path_teacher,
        auto_teach_mapper,
    ])
