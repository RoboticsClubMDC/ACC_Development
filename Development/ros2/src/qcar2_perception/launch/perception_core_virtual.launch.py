import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def semantic_map_path():
    override = os.environ.get("QCAR2_SEMANTIC_MAP_PATH", "").strip()
    if override:
        return override

    candidates = [
        Path("/workspaces/isaac_ros-dev/ros2/src/qcar2_perception/maps/semantic_map.json"),
        Path("/workspaces/isaac_ros-dev/Development/ros2/src/qcar2_perception/maps/semantic_map.json"),
        Path.home() / "Documents/GitHub/ACC_Development/Development/ros2/src/qcar2_perception/maps/semantic_map.json",
        Path.cwd() / "src/qcar2_perception/maps/semantic_map.json",
    ]

    for candidate in candidates:
        if candidate.parent.exists():
            return str(candidate)

    return str(candidates[0])


def generate_launch_description():
    package_share = get_package_share_directory("qcar2_perception")

    semantic_tf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(package_share, "launch", "semantic_tf_launch.py")
        )
    )

    d435_aligned_source = Node(
        package="qcar2_perception",
        executable="d435_aligned_source",
        name="d435_aligned_source",
        parameters=[{
            "is_physical": False,
            "distance_scale": 0.1,
        }],
        output="screen",
    )

    semantic_yolo_detector = Node(
        package="qcar2_perception",
        executable="semantic_yolo_detector",
        name="semantic_yolo_detector",
        output="screen",
    )

    object_3d_estimator = Node(
        package="qcar2_perception",
        executable="object_3d_estimator",
        name="object_3d_estimator",
        parameters=[{
            "depth_crop_ratio": 0.30,
            "min_quality": 0.0,
            "marker_lifetime_sec": 2.0,
            "max_uncertainty_radius": 0.75,
            "max_marker_radius": 0.45,
        }],
        output="screen",
    )

    semantic_landmark_mapper = Node(
        package="qcar2_perception",
        executable="semantic_landmark_mapper",
        name="semantic_landmark_mapper",
        parameters=[{
            "semantic_map_path": semantic_map_path(),
            # Added 2026-05-20 16:08:30 EDT:
            # Keep stable map markers separate from hypotheses/current sightings.
            "markers_topic": "/perception/semantic_landmark_markers",
            "hypothesis_markers_topic": "/perception/semantic_hypothesis_markers",
            "current_markers_topic": "/perception/semantic_current_markers",
            "reset_map_on_start": True,
            "association_radius": 0.80,
            "association_use_xy_only": True,
            "confirmed_seen_count": 3,
            "stable_seen_count": 8,
            "permanent_statuses": ["stable"],
            "load_only_permanent": True,
            "visible_timeout_sec": 2.0,
            "max_marker_radius": 0.50,
        }],
        output="screen",
    )

    perception_behavior_interface = Node(
        package="qcar2_perception",
        executable="perception_behavior_interface",
        name="perception_behavior_interface",
        parameters=[{
            "pose_topic": "/qcar2_pose_fused",
            "track_classes": ["stop sign", "traffic light"],
            "active_statuses": ["confirmed", "stable"],
            "max_event_distance_m": 1.50,
            "forward_fov_deg": 100.0,
        }],
        output="screen",
    )

    semantic_consistency_monitor = Node(
        package="qcar2_perception",
        executable="semantic_consistency_monitor",
        name="semantic_consistency_monitor",
        parameters=[{
            "match_radius": 0.75,
            "small_residual": 0.20,
            "medium_residual": 0.45,
        }],
        output="screen",
    )

    return LaunchDescription([
        semantic_tf,
        d435_aligned_source,
        semantic_yolo_detector,
        object_3d_estimator,
        semantic_landmark_mapper,
        semantic_consistency_monitor,
        perception_behavior_interface,
    ])
