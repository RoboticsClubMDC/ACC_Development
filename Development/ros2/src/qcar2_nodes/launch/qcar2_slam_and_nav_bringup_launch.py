import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction,
                            IncludeLaunchDescription, SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.actions import PushRosNamespace
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    # Get the launch directory
    bringup_dir = get_package_share_directory('qcar2_nodes')
    nav2_dir = get_package_share_directory('nav2_bringup')
    launch_dir = os.path.join(nav2_dir, 'launch')

    # Include the qbot_platform cartographer launch files
    qcar2_cartographer_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('qcar2_nodes'), 'launch', 'qcar2_cartographer_launch.py')]
        )
    )

    # Create the launch configuration variables
    namespace = LaunchConfiguration('namespace')
    use_namespace = LaunchConfiguration('use_namespace')
    slam = LaunchConfiguration('slam')
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    autostart = LaunchConfiguration('autostart')
    use_composition = LaunchConfiguration('use_composition')
    use_respawn = LaunchConfiguration('use_respawn')
    log_level = LaunchConfiguration('log_level')
    use_cuda = LaunchConfiguration('use_cuda')
    cuda_device = LaunchConfiguration('cuda_device')

    remappings = [('/tf', 'tf'),
                  ('/tf_static', 'tf_static')]

    # Create our own temporary YAML files that include substitutions
    param_substitutions = {
        'use_sim_time': use_sim_time}

    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key=namespace,
        param_rewrites=param_substitutions,
        convert_types=True)

    stdout_linebuf_envvar = SetEnvironmentVariable(
        'RCUTILS_LOGGING_BUFFERED_STREAM', '1')
    qcar2_use_cuda_envvar = SetEnvironmentVariable(
        'QCAR2_USE_CUDA', use_cuda)
    qcar2_cuda_device_envvar = SetEnvironmentVariable(
        'QCAR2_CUDA_DEVICE', cuda_device)
    cuda_visible_devices_envvar = SetEnvironmentVariable(
        'CUDA_VISIBLE_DEVICES', cuda_device)

    declare_namespace_cmd = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Top-level namespace')

    declare_use_namespace_cmd = DeclareLaunchArgument(
        'use_namespace',
        default_value='false',
        description='Whether to apply a namespace to the navigation stack')

    declare_slam_cmd = DeclareLaunchArgument(
        'slam',
        default_value='False',
        description='Whether run a SLAM')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true')

    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(bringup_dir, 'config', 'qcar2_slam_and_nav.yaml'),
        description='Full path to the ROS2 parameters file to use for all launched nodes')

    declare_autostart_cmd = DeclareLaunchArgument(
        'autostart', default_value='true',
        description='Automatically startup the nav2 stack')

    declare_use_composition_cmd = DeclareLaunchArgument(
        'use_composition', default_value='True',
        description='Whether to use composed bringup')

    declare_use_respawn_cmd = DeclareLaunchArgument(
        'use_respawn', default_value='False',
        description='Whether to respawn if a node crashes. Applied when composition is disabled.')

    declare_log_level_cmd = DeclareLaunchArgument(
        'log_level', default_value='warn',
        description='log level')

    declare_use_cuda_cmd = DeclareLaunchArgument(
        'use_cuda', default_value='true',
        description='Prefer CUDA for any GPU-capable QCar2 perception nodes launched under this bringup')

    declare_cuda_device_cmd = DeclareLaunchArgument(
        'cuda_device', default_value='0',
        description='CUDA device index for GPU-capable QCar2 perception nodes')

    # Specify the actions
    bringup_cmd_group = GroupAction([
        PushRosNamespace(
            condition=IfCondition(use_namespace),
            namespace=namespace),

        Node(
            condition=IfCondition(use_composition),
            name='nav2_container',
            package='rclcpp_components',
            executable='component_container_isolated',
            parameters=[configured_params, {'autostart': autostart}],
            arguments=['--ros-args', '--log-level', log_level],
            remappings=remappings,
            output='screen'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(launch_dir, 'slam_launch.py')),
            condition=IfCondition(slam),
            launch_arguments={'namespace': namespace,
                              'use_sim_time': use_sim_time,
                              'autostart': autostart,
                              'use_respawn': use_respawn,
                              'params_file': params_file}.items()),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(launch_dir, 'navigation_launch.py')),
            launch_arguments={'namespace': namespace,
                              'use_sim_time': use_sim_time,
                              'autostart': autostart,
                              'params_file': params_file,
                              'use_composition': use_composition,
                              'use_respawn': use_respawn,
                              'container_name': 'nav2_container'}.items()),
    ])
    
    qcar2_nav2_converter = Node(
    package='qcar2_nodes',
    executable='nav2_qcar2_converter',
    name='nav2_qcar2_converter',
    )

    # Create the launch description and populate
    ld = LaunchDescription([qcar2_nav2_converter])

    # Load the QBot Platform Cartographer launch file
    ld.add_action(qcar2_cartographer_launch)

    # Declare the launch options
    ld.add_action(declare_namespace_cmd)
    ld.add_action(declare_use_namespace_cmd)
    ld.add_action(declare_slam_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_params_file_cmd)
    ld.add_action(declare_autostart_cmd)
    ld.add_action(declare_use_composition_cmd)
    ld.add_action(declare_use_respawn_cmd)
    ld.add_action(declare_log_level_cmd)
    ld.add_action(declare_use_cuda_cmd)
    ld.add_action(declare_cuda_device_cmd)

    # Set environment variables after their launch arguments exist.
    ld.add_action(stdout_linebuf_envvar)
    ld.add_action(qcar2_use_cuda_envvar)
    ld.add_action(qcar2_cuda_device_envvar)
    ld.add_action(cuda_visible_devices_envvar)

    # Add the actions to launch all of the navigation nodes
    ld.add_action(bringup_cmd_group)

    return ld
