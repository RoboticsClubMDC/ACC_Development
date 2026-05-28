from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    # NOTE: the old `yolo_detector` (autonomy.yolo_detector_MARKERS_CPU_ABC)
    # was retired 2026-05-24. Semantic detection now lives in qcar2_perception
    # via `semantic_yolo_detector`. The traffic_system_detector that used to
    # alias the old yolo node has been removed from this launch too.

    path_follower = Node(
        package='qcar2_autonomy',
        executable='path_follower',
        name='path_follower',
        parameters=[{
            'visualize_pose': [True],
            'lane_bias_gain': 0.70,
            'lane_bias_max': 0.10,
            'lane_bias_turn_start': 0.18,
            'lane_bias_turn_end': 0.38,
            'stanley_trust_min': 0.35,
            'stanley_timeout_sec': 0.40,
        }],
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

    lane_detection = Node(
        package='qcar2_autonomy',
        executable='lane_detection',
        name='lane_detection',
        parameters=[{
            'image_topic': '/camera/csi_image',
            'config_file': 'csi_front_config.json',
        }],
    )

    lane_keeping = Node(
        package='qcar2_autonomy',
        executable='lane_keeping',
        name='lane_keeping',
        parameters=[{
            'input_cmd_topic': '/cmd_vel_nav',
            'mask_topic': '/lane_detection/road_mask',
            'publish_cmd': False,
            'k_cte': 1.15,
            'k_heading': 0.95,
            'max_steering': 0.52,
        }],
    )

    sidewalk_detection = Node(
        package='qcar2_autonomy',
        executable='sidewalk_detection',
        name='sidewalk_detection',
    )

    ''' TODO: Once finished this launch file must also include
    - Planner server to coordinate which LEDs on the QCar should be on based on trip logic
    - Hook into qcar2_perception's semantic_yolo_detector for traffic-light/stop-sign awareness
    '''

    return LaunchDescription([
        path_follower,
        trip_planner,
        lane_detection,
        lane_keeping,
        sidewalk_detection,
    ])
