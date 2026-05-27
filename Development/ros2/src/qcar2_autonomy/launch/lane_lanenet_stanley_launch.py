from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    front_csi_camera = Node(
        package='qcar2_nodes',
        executable='csi',
        name='front_csi_camera',
        parameters=[{
            'camera_num': 3,
            'frame_width': 820,
            'frame_height': 410,
            'frame_rate': 30.0,
        }],
        remappings=[
            ('camera/csi_image', '/camera/front_csi_image'),
        ],
    )

    lane_detector = Node(
        package='qcar2_autonomy',
        executable='lane_detector',
        name='lane_detector',
        parameters=[{
            'image_topic': '/camera/front_csi_image',
            'detector_backend': 'lanenet',
            'use_centerline_skeleton': True,
            'intersection_branch': 'straight',
            'car_center_offset_m': 0.0,
            'front_axle_offset_m': 0.256,
            'tracking_roi_top': 200,
            'tracking_roi_bottom': 400,
            'debug_crop_overlay': True,
            'debug_crop_top': 200,
            'debug_crop_bottom': 400,
            'lookahead_distance_m': 0.1,
            'heading_segment_m': 0.30,
            'heading_offset_deg': 0.0,
        }],
    )

    lane_stanley_controller = Node(
        package='qcar2_autonomy',
        executable='lane_stanley_controller',
        name='lane_stanley_controller',
        parameters=[{
            'speed_mps': 0.20,
            'stanley_gain': 1.0,
            'max_steer_rad': 0.35,
            'publish_stop_when_lost': False,
        }],
    )

    return LaunchDescription([
        front_csi_camera,
        lane_detector,
        lane_stanley_controller,
    ])
