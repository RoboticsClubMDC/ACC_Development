import subprocess

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    path_follower = Node(
        package='qcar2_autonomy',
        executable='path_follower',
        name='path_follower',
        parameters=[{
            'start_path': [False],          # IMPORTANT: don't move immediately
            'node_values': [2, 4, 8, 6, 8],
        }]
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
        'taxi_node': [10],
        'trip_nodes': [2, 4, 8, 6, 8],
        'initial_start_node': 10,
        'initial_end_at_taxi': True,
    }]
)

    ''' TODO: Once finished this launch file must also include
    - Lane detector to help smooth out tracking of lanes while driving
    - Planner server to coordinate which LEDs on the QCar should be on based on trip logic
    '''

    return LaunchDescription([
        path_follower,
        # traffic_system_detector,
        trip_planner
        ]
    )