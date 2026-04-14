import subprocess

from launch import LaunchDescription
from launch_ros.actions import Node


# SDCS big-map, right-hand traffic node 10.
# We keep using the recorded map only as route geometry in this same frame.
SDCS_TAXI_HUB_XY = [-1.28205, -0.45991]


def generate_launch_description():

    path_follower = Node(
        package='qcar2_autonomy',
        executable='path_follower',
        name='path_follower',
        parameters=[{
            'visualize_pose': [True],
            'route_frame': 'map',
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
    }],
    # parameters=[{
    #     'taxi_node': [10],
    #     'trip_nodes': [2, 4, 14, 20, 22, 10],
    # }]
)
    
    yolo_detector = Node(
        package='qcar2_autonomy',
        executable='yolo_detector',
        name='yolo_detector',

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
        traffic_system_detector,
        trip_planner,
        bev_csi_node,
        lane_detection,
        lane_stanley_node,
        yolo_detector,
        sidewalk_detection,
        bev_csi_seg,
        ]
    )
