# This is the launch file that starts up the basic QCar2 nodes

import subprocess

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, LogInfo, RegisterEventHandler, OpaqueFunction, TimerAction)
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.event_handlers import (OnProcessExit, OnProcessStart)

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    lidar_publish_rate = LaunchConfiguration('lidar_publish_rate')
    lidar_min_distance = LaunchConfiguration('lidar_min_distance')
    lidar_max_distance = LaunchConfiguration('lidar_max_distance')
        
    lidar_node = Node(
            package='qcar2_nodes',
            executable='lidar',
            name='Lidar',
            parameters=[{
                "device_type": "virtual",
                "publish_rate": ParameterValue(lidar_publish_rate, value_type=float),
                "min_distance": ParameterValue(lidar_min_distance, value_type=float),
                "max_distance": ParameterValue(lidar_max_distance, value_type=float),
            }]
        )
    
    realsense_camera_node = Node(
            package='qcar2_nodes',
            executable='rgbd',
            name='RealsenseCamera',
            parameters=[{"device_type":"virtual"},
                        {"frame_width_rgb":640},
                        {"frame_height_rgb":480},
                        {"frame_width_depth":640},
                        {"frame_height_depth":480}]
        )
    
    csi_camera_node = Node(
            package='qcar2_nodes',
            executable='csi',
            name='csi_camera',
            parameters=[{"device_type":"virtual"},
                        {"frame_width":820},
                        {"frame_height":410},
                        {"frame_rate":15.0},
                        {"camera_num":3}]
        )
    
    qcar2_hardware = Node(
            package='qcar2_nodes',
            executable='qcar2_hardware',
            name='qcar2_hardware',
            parameters=[{"device_type":"virtual"}]

        )

    qcar2_odometry = Node(
            package='qcar2_nodes',
            executable='qcar2_odometry',
            name='qcar2_odometry',
        )

    return LaunchDescription([
        DeclareLaunchArgument('lidar_publish_rate', default_value='20.0'),
        DeclareLaunchArgument('lidar_min_distance', default_value='0.0'),
        DeclareLaunchArgument('lidar_max_distance', default_value='10.0'),
        lidar_node,
        realsense_camera_node,
        csi_camera_node,
        qcar2_hardware,
        #qcar2_odometry, -> DIsable for cartographer testing 2026-05-21

    ])
