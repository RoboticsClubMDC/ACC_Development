# This is the launch file that starts up the basic QBot Platform nodes,
# plus the TF node. Then start the cartographer.

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (IncludeLaunchDescription, DeclareLaunchArgument)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (PathJoinSubstitution, LaunchConfiguration)

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    use_sim = LaunchConfiguration('use_sim')
    # Phase-4 (added 2026-05-25): when true, ekf_fusor's correction_source is
    # set to 'landmark' so it consumes /perception/landmark_pose_correction
    # from the semantic_landmark_mapper. ALSO requires the mapper to be
    # launched with enable_landmark_correction:=true. Default OFF so the
    # existing tf-only flow is unchanged.
    use_landmark_correction = LaunchConfiguration('use_landmark_correction')

    cartographer_config_dir = PathJoinSubstitution(
        [
            FindPackageShare('qcar2_nodes'),
            'config',
        ]
    )
    configuration_basename = LaunchConfiguration('configuration_basename')
    resolution = LaunchConfiguration('resolution')
    publish_period_sec = LaunchConfiguration('publish_period_sec', default='1.0')
    
    qcar2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('qcar2_nodes'), 'launch', 'qcar2_launch.py')]
        )
    )

    qcar2_to_lidar_tf_node = Node(
            package='qcar2_nodes',
            executable='fixed_lidar_frame',
            name='fixed_lidar_frame')

    pose_estimator = Node(
            package='qcar2_autonomy',
            executable='pose_estimator',
            name='pose_estimator',
            output='screen',
            parameters=[{'use_sim_time': use_sim}])

    # ekf_fusor consumes pose_estimator's /odom + Cartographer's map → base_link
    # TF and publishes the fused pose on /qcar2_pose_fused. This is the single
    # source of truth that path_follower (and any other autonomy consumer)
    # should read. Bundled here so launching cartographer always gives you the
    # fused pose topic too.
    # If `use_landmark_correction:=true` is passed, switch ekf_fusor's
    # correction source from 'tf' (Cartographer) to 'landmark'. We use a
    # PythonExpression so the value is resolved at launch time.
    from launch.substitutions import PythonExpression
    correction_source_expr = PythonExpression([
        "'landmark' if '", use_landmark_correction, "' == 'true' else 'tf'",
    ])
    ekf_fusor = Node(
            package='qcar2_autonomy',
            executable='ekf_fusor',
            name='ekf_fusor',
            output='screen',
            parameters=[{'use_sim_time': use_sim,
                         'correction_source': correction_source_expr}])

    configuration_basename_la = DeclareLaunchArgument(
            'configuration_basename',
            default_value='qcar2_2d.lua',
            description='Name of LUA file for cartographer')

    use_sim_la = DeclareLaunchArgument(
            'use_sim',
            default_value='false',
            description='Start robot in Gazebo simulation')
    
    resolution_la = DeclareLaunchArgument(
            'resolution',
            default_value='0.05',
            description='Resolution of a grid cell of occupancy grid')
    
    publish_period_sec_la = DeclareLaunchArgument(
            'publish_period_sec',
            default_value='1.0',
            description='Publishing period')

    use_landmark_correction_la = DeclareLaunchArgument(
            'use_landmark_correction',
            default_value='false',
            description=(
                'Phase-4: set ekf_fusor correction_source to "landmark" so it '
                'consumes /perception/landmark_pose_correction from the mapper. '
                'Requires the mapper to be launched with '
                'enable_landmark_correction:=true. DO NOT enable until stable '
                'landmarks have been validated across multiple drives.'
            ))

    cartographer_node = Node(
            package='cartographer_ros',
            executable='cartographer_node',
            output='screen',
            parameters=[{'use_sim_time': use_sim}],
            arguments=['-configuration_directory', cartographer_config_dir,
                       '-configuration_basename', configuration_basename])

    cartographer_occupancy_grid_node = Node(
            package='cartographer_ros',
            executable='cartographer_occupancy_grid_node',
            output='screen',
            parameters=[{'use_sim_time': use_sim}],
            arguments=['-resolution', resolution, '-publish_period_sec', publish_period_sec])
    qcar2_nav2_converter = Node(
       package='qcar2_nodes',
       executable='nav2_qcar2_converter',
       name='nav2_qcar2_converter',
       output = 'screen'
       )
    return LaunchDescription([
        configuration_basename_la,
        use_sim_la,
        resolution_la,
        publish_period_sec_la,
        use_landmark_correction_la,
        qcar2_launch,
        qcar2_nav2_converter,
        pose_estimator,
        ekf_fusor,
        cartographer_node,
        cartographer_occupancy_grid_node,
        qcar2_to_lidar_tf_node,
    ])
