import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def package_launch(package_name, launch_file):
    return PythonLaunchDescriptionSource(
        os.path.join(
            get_package_share_directory(package_name),
            'launch',
            launch_file,
        )
    )


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
            'stop_required_topic': '/perception/stop_required',
            'enable_stop_sign_object_stop': False,
            'stop_sign_objects_topic': '/perception/objects_3d',
            'stop_sign_trigger_distance_m': 0.20,
            'stop_sign_hold_seconds': 3.0,
            'stop_sign_cooldown_seconds': 10.0,
            'stop_sign_min_confidence': 0.30,
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
            'output_cmd_topic': '/cmd_vel_nav',
            'mask_topic': '/lane_detection/road_mask',
            'publish_cmd': False,
            'band_y0_frac': 0.58,
            'band_y1_frac': 0.96,
            'n_strips': 12,
            'min_lane_px': 350,
            'min_strip_px': 12,
            'target_x_frac': 0.50,
            'target_offset_px': 0.0,
            'k_cte': 1.15,
            'k_heading': 0.95,
            'speed_softening': 0.06,
            'max_steering': 0.55,
            'steer_output_rate': 5.0,
            'cte_alpha': 0.35,
            'heading_alpha': 0.35,
            'mask_timeout_sec': 0.40,
            'require_forward_motion': True,
            'publish_debug': True,
            'use_lane_assist_toggle': False,
            'enabled_on_start': True,
        }],
    )

    sidewalk_detection = Node(
        package='qcar2_autonomy',
        executable='sidewalk_detection',
        name='sidewalk_detection',
        parameters=[{
            'device': 'cpu',
        }],
    )

    localization_stack = IncludeLaunchDescription(
        package_launch('qcar2_nodes', 'qcar2_cartographer_launch.py'),
    )

    perception_core = IncludeLaunchDescription(
        package_launch('qcar2_perception', 'perception_core_physical.launch.py'),
        launch_arguments={
            'mode': 'internal',
            'source_only': 'false',
        }.items(),
    )

    return LaunchDescription([
        LogInfo(msg='autonomy_planner_launch: starting localization + perception landmarks + autonomy.'),
        localization_stack,
        perception_core,
        path_follower,
        trip_planner,
        lane_detection,
        lane_keeping,
        sidewalk_detection,
    ])
