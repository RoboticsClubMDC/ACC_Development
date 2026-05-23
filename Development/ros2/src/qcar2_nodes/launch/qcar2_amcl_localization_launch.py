import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    map_yaml = LaunchConfiguration("map")
    set_initial_pose = LaunchConfiguration("set_initial_pose")
    initial_x = LaunchConfiguration("initial_x")
    initial_y = LaunchConfiguration("initial_y")
    initial_yaw = LaunchConfiguration("initial_yaw")
    global_localization = LaunchConfiguration("global_localization")
    laser_max_range = LaunchConfiguration("laser_max_range")
    lidar_publish_rate = LaunchConfiguration("lidar_publish_rate")
    lidar_min_distance = LaunchConfiguration("lidar_min_distance")
    lidar_max_distance = LaunchConfiguration("lidar_max_distance")

    qcar2_physical = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("qcar2_nodes"),
                "launch",
                "qcar2_launch.py",
            )
        ),
        launch_arguments={
            "lidar_publish_rate": lidar_publish_rate,
            "lidar_min_distance": lidar_min_distance,
            "lidar_max_distance": lidar_max_distance,
        }.items(),
    )

    lidar_tf = Node(
        package="qcar2_nodes",
        executable="fixed_lidar_frame",
        name="fixed_lidar_frame",
        output="screen",
    )

    pose_estimator = Node(
        package="qcar2_autonomy",
        executable="pose_estimator",
        name="pose_estimator",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
        ],
    )

    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "yaml_filename": map_yaml,
        }],
    )

    amcl = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "global_frame_id": "map",
            "odom_frame_id": "odom",
            "base_frame_id": "base_link",
            "scan_topic": "/scan",
            "transform_tolerance": 0.5,
            "min_particles": 500,
            "max_particles": 4000,
            "max_beams": 120,
            "update_min_d": 0.05,
            "update_min_a": 0.05,
            "resample_interval": 1,
            "laser_model_type": "likelihood_field",
            "laser_min_range": 0.12,
            "laser_max_range": ParameterValue(laser_max_range, value_type=float),
            "laser_likelihood_max_dist": 0.5,
            "z_hit": 0.85,
            "z_rand": 0.15,
            "set_initial_pose": ParameterValue(set_initial_pose, value_type=bool),
            "initial_pose": {
                "x": ParameterValue(initial_x, value_type=float),
                "y": ParameterValue(initial_y, value_type=float),
                "z": 0.0,
                "yaw": ParameterValue(initial_yaw, value_type=float),
            },
        }],
    )

    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_localization",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "autostart": True,
            "node_names": ["map_server", "amcl"],
        }],
    )

    global_localization_request = TimerAction(
        period=4.0,
        actions=[
            ExecuteProcess(
                condition=IfCondition(global_localization),
                cmd=[
                    "ros2",
                    "service",
                    "call",
                    "/reinitialize_global_localization",
                    "std_srvs/srv/Empty",
                    "{}",
                ],
                output="screen",
            )
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use simulation time.",
        ),
        DeclareLaunchArgument(
            "map",
            default_value=os.path.expanduser("~/qcar2_maps/competition_map.yaml"),
            description="Saved occupancy map YAML generated after the Cartographer lap.",
        ),
        DeclareLaunchArgument(
            "set_initial_pose",
            default_value="true",
            description="Let AMCL initialize from the launch-file initial pose.",
        ),
        DeclareLaunchArgument(
            "initial_x",
            default_value="0.0",
            description="Initial AMCL x pose in the saved map frame.",
        ),
        DeclareLaunchArgument(
            "initial_y",
            default_value="0.0",
            description="Initial AMCL y pose in the saved map frame.",
        ),
        DeclareLaunchArgument(
            "initial_yaw",
            default_value="0.0",
            description="Initial AMCL yaw pose in radians.",
        ),
        DeclareLaunchArgument(
            "global_localization",
            default_value="false",
            description="Ask AMCL to spread particles across the whole map after startup.",
        ),
        DeclareLaunchArgument(
            "laser_max_range",
            default_value="4.5",
            description="Maximum scan range AMCL scores against the saved map.",
        ),
        DeclareLaunchArgument(
            "lidar_publish_rate",
            default_value="20.0",
            description="Target /scan publish frequency in Hz.",
        ),
        DeclareLaunchArgument(
            "lidar_min_distance",
            default_value="0.0",
            description="Minimum range kept in the published /scan.",
        ),
        DeclareLaunchArgument(
            "lidar_max_distance",
            default_value="10.0",
            description="Maximum range kept in the published /scan.",
        ),
        qcar2_physical,
        lidar_tf,
        pose_estimator,
        map_server,
        amcl,
        lifecycle_manager,
        global_localization_request,
    ])
