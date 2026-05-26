import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def package_launch(package_name, launch_file):
    return PythonLaunchDescriptionSource(
        os.path.join(
            get_package_share_directory(package_name),
            "launch",
            launch_file,
        )
    )


def generate_launch_description():
    start_hardware = LaunchConfiguration("start_hardware")
    start_perception = LaunchConfiguration("start_perception")
    start_autonomy = LaunchConfiguration("start_autonomy")
    perception_mode = LaunchConfiguration("perception_mode")
    perception_source_only = LaunchConfiguration("perception_source_only")

    qcar2_base = IncludeLaunchDescription(
        package_launch("qcar2_nodes", "qcar2_launch.py"),
        condition=IfCondition(start_hardware),
    )

    perception_core = IncludeLaunchDescription(
        package_launch("qcar2_perception", "perception_core_physical.launch.py"),
        condition=IfCondition(start_perception),
        launch_arguments={
            "mode": perception_mode,
            "source_only": perception_source_only,
        }.items(),
    )

    autonomy_planner = IncludeLaunchDescription(
        package_launch("qcar2_autonomy", "autonomy_planner_launch.py"),
        condition=IfCondition(start_autonomy),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "start_hardware",
            default_value="true",
            choices=["true", "false"],
            description="Start qcar2_nodes hardware/camera/CSI launch.",
        ),
        DeclareLaunchArgument(
            "start_perception",
            default_value="true",
            choices=["true", "false"],
            description="Start qcar2_perception YOLO/depth/landmark/behavior stack.",
        ),
        DeclareLaunchArgument(
            "start_autonomy",
            default_value="true",
            choices=["true", "false"],
            description="Start qcar2_autonomy path follower, trip planner, lane/sidewalk nodes.",
        ),
        DeclareLaunchArgument(
            "perception_mode",
            default_value="internal",
            choices=["internal", "external"],
            description=(
                "Pass-through for perception_core_physical.launch.py. "
                "Use internal on the QCar; external in laptop Docker consuming QCar D435 topics."
            ),
        ),
        DeclareLaunchArgument(
            "perception_source_only",
            default_value="false",
            choices=["true", "false"],
            description="Pass-through for perception_core_physical.launch.py source_only.",
        ),
        LogInfo(msg="luigi_bbl: launching QCar2 hardware + perception landmarks + autonomy."),
        qcar2_base,
        perception_core,
        autonomy_planner,
    ])
