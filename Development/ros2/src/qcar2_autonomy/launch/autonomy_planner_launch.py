import subprocess

from launch import LaunchDescription
from launch_ros.actions import Node


# SDCS big-map, right-hand traffic node 10 in PNG/reference coordinates.
# trip_planner converts these goal coordinates into the live ROS map frame.
SDCS_TAXI_HUB_XY = [-1.28205, -0.45991]


def generate_launch_description():

    path_follower = Node(
        package='qcar2_autonomy',
        executable='path_follower',
        name='path_follower',
        parameters=[{
            'visualize_pose': [True],
            'route_frame':    'map',
            'progress_search_max_step': 25,
            'curve_lookahead_min_m':    0.18,
            'curve_lookahead_max_m':    0.70,
            'curvature_lookahead_gain': 1.0,
            'min_curve_speed':          0.16,
            'curvature_speed_gain':     1.5,
            'lateral_error_slowdown_threshold_m': 0.12,
        }],
    )

    recorded_map_visualizer = Node(
        package='qcar2_autonomy',
        executable='recorded_map_visualizer',
        name='recorded_map_visualizer',
        output='screen',
        parameters=[{
            'route_frame': 'map',
            'line_width':  0.025,
            'node_size':   0.08,
        }],
    )

    traffic_system_detector = Node(
        package ='qcar2_autonomy',
        executable='yolo_detector',
        name = 'qcar2_yolo_detector'
    )
    
    trip_planner = Node(
    package='qcar2_autonomy',
    executable='trip_planner',
    name='trip_planner',
    parameters=[{
        'route_frame': 'map',
        'hub_xy': SDCS_TAXI_HUB_XY,
        'goals_are_sdcs_frame': True,
        'one_way_route': True,
        'allow_route_wrap': True,
        'heading_aware_start': True,
    }],
    # parameters=[{
    #     'taxi_node': [10],
    #     'trip_nodes': [2, 4, 14, 20, 22, 10],
    # }]
)
    
    lane_detection = Node(
        package='qcar2_autonomy',
        executable='lane_detection',
        name='lane_detection',
    )


    lane_stanley_node = Node(
        package='qcar2_autonomy',
        executable='lane_stanley_node',
        name='lane_stanley_node',
    )

    bev_csi_node = Node(
        package='qcar2_autonomy',
        executable='bev_csi_node',
        name='bev_csi_node',
    )

    sidewalk_detection = Node(
        package='qcar2_autonomy',
        executable='sidewalk_detection',
        name='sidewalk_detection',
    )

    bev_csi_seg = Node(
        package='qcar2_autonomy',
        executable='bev_csi_seg',
        name='bev_csi_seg',
    )

    ''' TODO: Once finished this launch file must also include
    - Lane detector to help smooth out tracking of lanes while driving
    - Planner server to coordinate which LEDs on the QCar should be on based on trip logic
    '''

    return LaunchDescription([
        path_follower,
        recorded_map_visualizer,
        traffic_system_detector,
        trip_planner,
        bev_csi_node,
        lane_detection,
        lane_stanley_node,
        sidewalk_detection,
        bev_csi_seg,
        ]
    )
