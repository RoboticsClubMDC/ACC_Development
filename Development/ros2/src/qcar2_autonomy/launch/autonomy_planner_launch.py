import subprocess

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    path_follower = Node(
        package='qcar2_autonomy',
        executable='path_follower',
        name='path_follower',
        output='screen'
    )

    lane_detection = Node(
        package='qcar2_autonomy',
        executable='lane_detection',
        name='lane_detection',
        output='screen'
    )

    sidewalk_detection = Node(
        package='qcar2_autonomy',
        executable='sidewalk_detection',
        name='sidewalk_detection',
        output='screen'
    )

    lane_keeping = Node(
        package='qcar2_autonomy',
        executable='lane_keeping',
        name='lane_keeping',
        output='screen'
    )
    
    traffic_system_detector = Node(
        package ='qcar2_autonomy',
        executable='yolo_detector',
        name = 'qcar2_yolo_detector'
    )
    
    
    # trip_planner = Node(
    # package='qcar2_autonomy',
    # executable='trip_planner',
    # name='trip_planner',
    # )
    


    return LaunchDescription([
        path_follower,
        lane_detection,
        sidewalk_detection,
        lane_keeping,
        traffic_system_detector,
        # trip_planner,
        ]
    )