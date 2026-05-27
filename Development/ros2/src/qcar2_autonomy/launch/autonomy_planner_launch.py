import subprocess

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    # 2026-05-26 PM: yolo_model_path launch arg lets us swap the YOLO
    # backend without editing this file. Empty string (default) =
    # Quanser PIT seg model (existing behavior). Set to a .pt path to
    # use the custom-trained ultralytics model (e.g. road_signs_4class).
    yolo_model_path_arg = DeclareLaunchArgument(
        'yolo_model_path',
        default_value='',
        description=("Path to a .pt YOLO model file inside the dev "
                     "container. Empty = Quanser PIT seg model (default). "
                     "Set to e.g. /workspaces/isaac_ros-dev/ros2/src/"
                     "qcar2_autonomy/models/road_signs_4class_yolov8s.pt "
                     "to use the custom-trained ultralytics model."))
    yolo_model_path = LaunchConfiguration('yolo_model_path')

    # 2026-05-26 PM-7: tunable knobs for the v3.5 yolo_detector,
    # exposed as launch args so the user can copy-paste-test variations
    # via `ros2 launch ... <name>:=<value>` syntax. Each forwards to the
    # corresponding ROS parameter on the yolo_detector node.
    crop_bottom_px_arg = DeclareLaunchArgument(
        'crop_bottom_px', default_value='24',
        description='Bottom rows to crop before YOLO inference (hides CSI bumper).')
    tl_color_history_size_arg = DeclareLaunchArgument(
        'tl_color_history_size', default_value='8',
        description='TL color majority-vote window (frames).')
    tl_color_min_v_arg = DeclareLaunchArgument(
        'tl_color_min_v', default_value='90',
        description='HSV V (brightness) floor for TL color check, 0-255.')
    tl_color_min_s_arg = DeclareLaunchArgument(
        'tl_color_min_s', default_value='70',
        description='HSV S (saturation) floor for TL color check, 0-255.')
    tl_pass_line_height_px_arg = DeclareLaunchArgument(
        'tl_pass_line_height_px', default_value='100',
        description='bbox height beyond which NEW TL brakes are blocked.')
    tl_hold_s_arg = DeclareLaunchArgument(
        'tl_hold_s', default_value='0.60',
        description='How long the TL brake stays engaged per refresh (seconds).')
    # 2026-05-26 PM-8: TL state-machine knobs.
    tl_fsm_lost_frames_to_reset_arg = DeclareLaunchArgument(
        'tl_fsm_lost_frames_to_reset', default_value='15',
        description='Frames (at 30fps) of no TL detection before FSM resets to IDLE.')
    tl_fsm_green_frames_to_release_arg = DeclareLaunchArgument(
        'tl_fsm_green_frames_to_release', default_value='3',
        description='Consecutive green-effective-color frames to transition COMMIT_STOP -> COMMIT_GO.')

    crop_bottom_px         = LaunchConfiguration('crop_bottom_px')
    tl_color_history_size  = LaunchConfiguration('tl_color_history_size')
    tl_color_min_v         = LaunchConfiguration('tl_color_min_v')
    tl_color_min_s         = LaunchConfiguration('tl_color_min_s')
    tl_pass_line_height_px = LaunchConfiguration('tl_pass_line_height_px')
    tl_hold_s              = LaunchConfiguration('tl_hold_s')
    tl_fsm_lost_frames_to_reset    = LaunchConfiguration('tl_fsm_lost_frames_to_reset')
    tl_fsm_green_frames_to_release = LaunchConfiguration('tl_fsm_green_frames_to_release')

    from_frame_arg = DeclareLaunchArgument(
        'from_frame',
        default_value='map',
        description=("Localization source frame for nav_to_pose. Default "
                     "'map' (cartographer). Set to 'odom' to run without SLAM."))
    from_frame = LaunchConfiguration('from_frame')

    sdcs_map_image_arg = DeclareLaunchArgument(
        'sdcs_map_image',
        default_value=('/workspaces/isaac_ros-dev/backup/Quanser_Academic_Resources/'
                       '6_teaching/4_Autonomous_Systems/SDCS/skills_activities/'
                       '04-vehicle_control/cityscape.png'),
        description=('Path inside the container to the SDCS course PNG that '
                     'will be republished on /sdcs_map_image for RViz. '
                     'Override with sdcs_map_image:=/abs/path/to/file.png.'))
    sdcs_map_image_path = LaunchConfiguration('sdcs_map_image')

    path_follower = Node(
        package='qcar2_autonomy',
        executable='path_follower',
        name='path_follower',
        parameters=[{
            'cmd_topic': '/cmd_vel_path',
            'from_frame': from_frame,
        }],
    )

    cmd_vel_blender = Node(
        package='qcar2_autonomy',
        executable='cmd_vel_blender',
        name='cmd_vel_blender',
        parameters=[{
            'path_cmd_topic': '/cmd_vel_path',
            'lane_cmd_topic': '/cmd_vel_lane',
            'cmd_topic':      '/cmd_vel_nav',
            'path_weight':    0.40,
            'lane_weight':    0.60,
            'linear_source':  'path',
            'stop_topic':     '/car_stop',
        }],
    )

    traffic_system_detector = Node(
        package='qcar2_autonomy',
        executable='yolo_detector',
        name='qcar2_yolo_detector',
        parameters=[{
            # String param — passed through as-is.
            'model_path':             yolo_model_path,
            # Int / float params need ParameterValue with explicit type
            # so the LaunchConfiguration string gets coerced correctly.
            'crop_bottom_px':         ParameterValue(crop_bottom_px,         value_type=int),
            'tl_color_history_size':  ParameterValue(tl_color_history_size,  value_type=int),
            'tl_color_min_v':         ParameterValue(tl_color_min_v,         value_type=int),
            'tl_color_min_s':         ParameterValue(tl_color_min_s,         value_type=int),
            'tl_pass_line_height_px': ParameterValue(tl_pass_line_height_px, value_type=int),
            'tl_hold_s':              ParameterValue(tl_hold_s,              value_type=float),
            'tl_fsm_lost_frames_to_reset':    ParameterValue(tl_fsm_lost_frames_to_reset,    value_type=int),
            'tl_fsm_green_frames_to_release': ParameterValue(tl_fsm_green_frames_to_release, value_type=int),
        }],
    )

    trip_planner = Node(
        package='qcar2_autonomy',
        executable='trip_planner',
        name='trip_planner',
         parameters=[{
            'taxi_node': [10],
        #     'trip_nodes': [2, 4, 14, 20, 22, 10],
         }]
    )

    lane_detector = Node(
        package='qcar2_autonomy',
        executable='lane_detector',
        name='lane_detector',
    )

    Planner_server = Node(
        package='qcar2_autonomy',
        executable='Planner_server',
        name='Planner_server',
    )

    sdcs_map_publisher = Node(
        package='qcar2_autonomy',
        executable='sdcs_map_publisher',
        name='sdcs_map_publisher',
        parameters=[{
            'image_path': sdcs_map_image_path,
        }],
        output='screen',
    )

    return LaunchDescription([
        from_frame_arg,
        sdcs_map_image_arg,
        yolo_model_path_arg,
        crop_bottom_px_arg,
        tl_color_history_size_arg,
        tl_color_min_v_arg,
        tl_color_min_s_arg,
        tl_pass_line_height_px_arg,
        tl_hold_s_arg,
        tl_fsm_lost_frames_to_reset_arg,
        tl_fsm_green_frames_to_release_arg,
        path_follower,
        cmd_vel_blender,
        traffic_system_detector,
        trip_planner,
        lane_detector,
        Planner_server,
        sdcs_map_publisher,
    ])
