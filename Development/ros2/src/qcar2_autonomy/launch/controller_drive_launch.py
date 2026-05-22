from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    use_cuda = LaunchConfiguration('use_cuda')
    cuda_device = LaunchConfiguration('cuda_device')
    allow_cpu_fallback = LaunchConfiguration('allow_cpu_fallback')

    controller_drive = Node(
        package='qcar2_autonomy',
        executable='controller_drive',
        name='controller_drive',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'cmd_topic':             '/cmd_vel_raw',  # → lane_assist_blend → /cmd_vel_nav
            'max_speed':             0.2,
            'max_turn':              2.0,
            'deadzone':              0.08,
            'publish_hz':            30.0,
            'speed_slew':            1.0,
            'steer_slew':            12.0,
            'require_enable_button': True,
            'enable_button':         4,    # LB  — hold to drive
            'reverse_button':        0,    # A
            'stop_button':           1,    # B
            'steer_axis':            0,    # left stick X
            'trigger_axis':          5,    # right trigger RT
            'use_trigger':           True,
            'lane_assist_button':    5,    # RB  — press to toggle lane assist on/off
        }],
    )

    lane_detection = Node(
        package='qcar2_autonomy',
        executable='lane_detection',
        name='lane_detection',
        output='screen',
        parameters=[{
            'image_topic': '/camera/csi_image',
            'imgsz':       640,
            'device':      ParameterValue(cuda_device, value_type=int),
            'use_cuda':    ParameterValue(use_cuda, value_type=bool),
            'allow_cpu_fallback': ParameterValue(allow_cpu_fallback, value_type=bool),
        }],
    )

    lane_stanley = Node(
        package='qcar2_autonomy',
        executable='lane_stanley_node',
        name='lane_stanley_node',
        output='screen',
        parameters=[{
            'band_y0_frac': 0.65,
            'band_y1_frac': 0.95,
            'min_lane_px':  400,
            'cte_alpha':    0.2,
            'head_alpha':   0.2,
            'n_strips':     10,
            'k_cte':        0.35,
            'k_head':       0.65,
            'max_steer':    0.25,
        }],
    )

    lane_assist_blend = Node(
        package='qcar2_autonomy',
        executable='lane_assist_blend',
        name='lane_assist_blend',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'input_cmd_topic':   '/cmd_vel_raw',
            'output_cmd_topic':  '/cmd_vel_nav',
            'gain':              3.0,   # tune if correction feels too strong/weak
            'max_turn':          2.0,
            'trust_threshold':   0.5,
            'delta_timeout_sec': 0.3,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_cuda',
            default_value='true',
            description='Prefer CUDA for lane detection'),
        DeclareLaunchArgument(
            'cuda_device',
            default_value='0',
            description='CUDA device index for lane detection'),
        DeclareLaunchArgument(
            'allow_cpu_fallback',
            default_value='false',
            description='Allow lane detection to fall back to CPU if CUDA fails'),
        SetEnvironmentVariable('QCAR2_USE_CUDA', use_cuda),
        SetEnvironmentVariable('QCAR2_CUDA_DEVICE', cuda_device),
        SetEnvironmentVariable('QCAR2_ALLOW_CPU_FALLBACK', allow_cpu_fallback),
        SetEnvironmentVariable('CUDA_VISIBLE_DEVICES', cuda_device),
        controller_drive,
        lane_detection,
        lane_stanley,
        lane_assist_blend,
    ])
