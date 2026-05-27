from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # Lane-only test speed. Override from the command line, for example:
    # ros2 launch qcar2_autonomy lane_lanenet_stanley_launch.py lane_speed_mps:=0.10
    lane_speed_arg = DeclareLaunchArgument(
        'lane_speed_mps',
        default_value='0.20',
        description='Forward speed used by lane_stanley_controller.')
    lane_speed = LaunchConfiguration('lane_speed_mps')

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
            'speed_mps': ParameterValue(lane_speed, value_type=float),
            'stanley_gain': 1.0,
            'max_steer_rad': 0.35,
            'publish_stop_when_lost': False,
        }],
    )

    return LaunchDescription([
        lane_speed_arg,
        front_csi_camera,
        lane_detector,
        lane_stanley_controller,
    ])
