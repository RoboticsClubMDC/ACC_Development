import subprocess

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node


# SDCS big-map, right-hand traffic node 10 in PNG/reference coordinates.
# trip_planner converts these goal coordinates into the live ROS map frame.
SDCS_TAXI_HUB_XY = [-1.28205, -0.45991]


def generate_launch_description():
    use_cuda = LaunchConfiguration('use_cuda')
    cuda_device = LaunchConfiguration('cuda_device')
    use_tensorrt = LaunchConfiguration('use_tensorrt')
    allow_cpu_fallback = LaunchConfiguration('allow_cpu_fallback')

    path_follower = Node(
        package='qcar2_autonomy',
        executable='path_follower',
        name='path_follower',
        parameters=[{
            'visualize_pose': [True],
            'route_frame':    'map',
            'progress_search_max_step': 25,
            'control_hz': 30.0,
            'curve_lookahead_min_m':    0.18,
            'curve_lookahead_max_m':    0.70,
            'curvature_lookahead_gain': 1.0,
            'min_curve_speed':          0.16,
            'curvature_speed_gain':     1.5,
            'lateral_error_slowdown_threshold_m': 0.12,
        }],
    )

    recorded_map_visualizer = Node(
        package='qcar2_autonomy',
        executable='recorded_map_visualizer',
        name='recorded_map_visualizer',
        output='screen',
        parameters=[{
            'route_frame': 'map',
            'line_width':  0.025,
            'node_size':   0.08,
        }],
    )

    traffic_system_detector = Node(
        package ='qcar2_autonomy',
        executable='yolo_detector',
        name = 'qcar2_yolo_detector',
        parameters=[{
            'use_cuda':    ParameterValue(use_cuda, value_type=bool),
            'device':      ParameterValue(cuda_device, value_type=int),
            'use_tensorrt': ParameterValue(use_tensorrt, value_type=bool),
            'allow_cpu_fallback': ParameterValue(allow_cpu_fallback, value_type=bool),
            'inference_hz': 10.0,
            'motion_flag_hz': 30.0,
        }],
    )
    
    trip_planner = Node(
    package='qcar2_autonomy',
    executable='trip_planner',
    name='trip_planner',
    parameters=[{
        'route_frame': 'map',
        'hub_xy': SDCS_TAXI_HUB_XY,
        'goals_are_sdcs_frame': True,
        'one_way_route': True,
        'allow_route_wrap': True,
        'heading_aware_start': True,
    }],
    # parameters=[{
    #     'taxi_node': [10],
    #     'trip_nodes': [2, 4, 14, 20, 22, 10],
    # }]
)
    
    ''' TODO: Once finished this launch file must also include
    - Planner server to coordinate which LEDs on the QCar should be on based on trip logic
    '''

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_cuda',
            default_value='true',
            description='Prefer CUDA for YOLO/lane/sidewalk inference nodes'),
        DeclareLaunchArgument(
            'cuda_device',
            default_value='0',
            description='CUDA device index for GPU-capable inference nodes'),
        DeclareLaunchArgument(
            'use_tensorrt',
            default_value='false',
            description='Convert/use TensorRT engine for traffic YOLO when available'),
        DeclareLaunchArgument(
            'allow_cpu_fallback',
            default_value='false',
            description='Allow AI inference nodes to fall back to CPU if CUDA fails'),
        SetEnvironmentVariable('QCAR2_USE_CUDA', use_cuda),
        SetEnvironmentVariable('QCAR2_CUDA_DEVICE', cuda_device),
        SetEnvironmentVariable('QCAR2_ALLOW_CPU_FALLBACK', allow_cpu_fallback),
        SetEnvironmentVariable('CUDA_VISIBLE_DEVICES', cuda_device),
        path_follower,
        recorded_map_visualizer,
        traffic_system_detector,
        trip_planner,
        ]
    )
