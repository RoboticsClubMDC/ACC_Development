import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    map_yaml = LaunchConfiguration("map")

    qcar2_virtual = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("qcar2_nodes"),
                "launch",
                "qcar2_virtual_launch.py",
            )
        )
    )

    lidar_tf = Node(
        package="qcar2_nodes",
        executable="fixed_lidar_frame_virtual",
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
            # --- Scan-match tightness (the "anchor" knobs) ---
            "laser_model_type": "likelihood_field",
            "laser_min_range": 0.12,
            "laser_max_range": 10.0,
            "laser_likelihood_max_dist": 0.30,   # was 0.15 — wider basin of attraction so we can recover
            "max_beams": 180,                     # keep — high-impact, low-risk
            "z_hit": 0.90,                        # was 0.95 — leave some room for noise
            "z_rand": 0.05,
            "z_max": 0.05,
            "z_short": 0.00,
            "sigma_hit": 0.20,                    # was 0.10 — broader likelihood surface
            # --- Particle filter size ---
            "min_particles": 2000,
            "max_particles": 5000,
            "pf_err": 0.05,
            "pf_z": 0.99,
            # --- Update cadence ---
            "update_min_d": 0.02,    # keep tight — more correction opportunities is good
            "update_min_a": 0.02,
            "resample_interval": 1,
            # --- Recovery (kidnapped-robot detection) ---
            "recovery_alpha_slow": 0.001,
            "recovery_alpha_fast": 0.20,    # keep — helps re-spread when match deteriorates
            # --- Motion model noise (REVERTED: tight values caused brittle basin) ---
            "robot_model_type": "nav2_amcl::DifferentialMotionModel",
            "alpha1": 0.2,    # back to default — particles need to spread on rotation
            "alpha2": 0.2,
            "alpha3": 0.2,    # back to default — particles need to spread on translation
            "alpha4": 0.2,
            "alpha5": 0.2,
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

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
        ),
        DeclareLaunchArgument(
            "map",
            default_value=os.path.expanduser("~/qcar2_maps/competition_map.yaml"),
        ),
        qcar2_virtual,
        lidar_tf,
        pose_estimator,
        map_server,
        amcl,
        lifecycle_manager,
    ])