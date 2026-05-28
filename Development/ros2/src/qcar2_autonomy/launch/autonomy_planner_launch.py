import subprocess

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    path_follower = Node(
        package='qcar2_autonomy',
        executable='path_follower',
        name='path_follower',
        parameters=[{'visualize_pose': [True]}], 
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

    # sidewalk_detection = Node(
    #     package='qcar2_autonomy',
    #     executable='sidewalk_detection',
    #     name='sidewalk_detection',
    # )

    ''' TODO: Once finished this launch file must also include
    - Lane detector to help smooth out tracking of lanes while driving
    - Planner server to coordinate which LEDs on the QCar should be on based on trip logic
    '''

    return LaunchDescription([
        path_follower,
        traffic_system_detector,
        trip_planner,
        # lane_detection,
        # lane_stanley_node,
        yolo_detector,
        #sidewalk_detection,
        ]
    )