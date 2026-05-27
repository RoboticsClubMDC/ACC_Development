import subprocess

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    path_follower = Node(
        package='qcar2_autonomy',
        executable='path_follower',
        name='path_follower',
        parameters=[{
            'cmd_topic': '/cmd_vel_path',
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
    # parameters=[{
    #     'taxi_node': [10],
    #     'trip_nodes': [2, 4, 14, 20, 22, 10],
    # }]
)
    
    lane_detector = Node(
        package='qcar2_autonomy',
        executable='lane_detector',
        name='lane_detector',
        parameters=[{
            'image_topic': '/camera/csi_image',
            'detector_backend': 'lanenet',
            'publish_debug_images': True,
            'use_centerline_skeleton': True,
            'intersection_branch': 'straight',
            'lane_width_m': 0.254,
            'car_center_offset_m': -0.52,
            'front_axle_offset_m': 0.256,
            'heading_offset_deg': 3.3,
            'tracking_roi_top': 200,
            'tracking_roi_bottom': 400,
            'debug_crop_overlay': True,
            'debug_crop_top': 200,
            'debug_crop_bottom': 400,
            'lookahead_distance_m': 0.05,
            'heading_segment_m': 0.05,
        }],
    )

    lane_stanley_controller = Node(
        package='qcar2_autonomy',
        executable='lane_stanley_controller',
        name='lane_stanley_controller',
        parameters=[{
            'cmd_topic': '/cmd_vel_lane',
            'speed_mps': 0.20,
            'stanley_gain': 0.5,
            'heading_gain': 1.0,
            'max_steer_rad': 0.35,
            'publish_stop_when_lost': False,
        }],
    )

    cmd_vel_blender = Node(
        package='qcar2_autonomy',
        executable='cmd_vel_blender',
        name='cmd_vel_blender',
        parameters=[{
            'path_cmd_topic': '/cmd_vel_path',
            'lane_cmd_topic': '/cmd_vel_lane',
            'cmd_topic': '/cmd_vel_nav',
            'lane_weight': 0.60,
            'path_weight': 0.40,
            'linear_source': 'path',
        }],
    )

    Planner_server = Node(
        package='qcar2_autonomy',
        executable='Planner_server',
        name='Planner_server',
        
    )

    ''' TODO: Once finished this launch file must also include
    - Lane detector to help smooth out tracking of lanes while driving
    - Planner server to coordinate which LEDs on the QCar should be on based on trip logic
    '''

    return LaunchDescription([
        path_follower,
        traffic_system_detector,
        trip_planner,
        lane_detector,
        lane_stanley_controller,
        cmd_vel_blender,
        Planner_server

        ]
    )
