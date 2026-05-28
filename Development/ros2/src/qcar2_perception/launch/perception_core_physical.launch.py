import os
import platform
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def semantic_map_path():
    override = os.environ.get("QCAR2_SEMANTIC_MAP_PATH", "").strip()
    if override:
        return override

    candidates = [
        Path("/workspaces/isaac_ros-dev/ros2/src/qcar2_perception/maps/semantic_map.json"),
        Path("/workspaces/isaac_ros-dev/Development/ros2/src/qcar2_perception/maps/semantic_map.json"),
        Path.home() / "Documents/ACC_Development_luigi/Development/ros2/src/qcar2_perception/maps/semantic_map.json",
        Path.cwd() / "src/qcar2_perception/maps/semantic_map.json",
    ]

    for candidate in candidates:
        if candidate.parent.exists():
            return str(candidate)

    return str(candidates[0])


def jetson_torch_env():
    libgomp = Path("/lib/aarch64-linux-gnu/libgomp.so.1")
    if platform.machine() in ("aarch64", "arm64") and libgomp.exists():
        return {"LD_PRELOAD": str(libgomp)}
    return {}


def launch_setup(context, *args, **kwargs):
    mode = LaunchConfiguration("mode").perform(context).strip().lower()
    source_only = LaunchConfiguration("source_only").perform(context).strip().lower() == "true"
    if mode not in ("internal", "external"):
        raise RuntimeError(
            "perception_core_physical.launch.py mode must be 'internal' or 'external'. "
            "Use mode:=internal on the QCar, or mode:=external in laptop Docker."
        )

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
            "is_physical": True,
            "distance_scale": 1.0,
        }],
        output="screen",
    )

    semantic_yolo_detector = Node(
        package="qcar2_perception",
        executable="semantic_yolo_detector",
        name="semantic_yolo_detector",
        # Jetson/Focal can fail importing torch with:
        # "libgomp.so.1: cannot allocate memory in static TLS block".
        # Preload only on the ARM QCar, not in x86 laptop containers.
        additional_env=jetson_torch_env(),
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

    actions = []
    if mode == "internal":
        actions.extend([
            LogInfo(msg="perception_core_physical mode=internal: starting QCar D435 source."),
            semantic_tf,
            d435_aligned_source,
        ])
    else:
        actions.extend([
            LogInfo(
                msg=(
                    "perception_core_physical mode=external: not starting D435 hardware source; "
                    "subscribing to /perception/d435/* from the physical QCar over DDS."
                )
            ),
            semantic_tf,
        ])

    if mode == "internal" and source_only:
        actions.append(LogInfo(
            msg="perception_core_physical source_only=true: camera topics only; skipping YOLO/landmark compute."
        ))
        return actions

    actions.extend([
        semantic_yolo_detector,
        object_3d_estimator,
        semantic_landmark_mapper,
        semantic_consistency_monitor,
        perception_behavior_interface,
    ])

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "mode",
            default_value="internal",
            choices=["internal", "external"],
            description=(
                "internal = run on the QCar and start the physical D435 source; "
                "external = run in laptop Docker and listen to QCar-published D435 topics"
            ),
        ),
        DeclareLaunchArgument(
            "source_only",
            default_value="false",
            choices=["true", "false"],
            description=(
                "When mode:=internal, publish physical D435 topics only and skip YOLO/landmark compute. "
                "Default is false: the QCar Jetson runs the full perception stack (D435 + YOLO + "
                "landmarks). If you ever set source_only:=true, you MUST run mode:=external on the "
                "laptop to provide the compute, and you MUST NOT run mode:=internal source_only:=false "
                "on the QCar at the same time (that double-publishes every perception topic). "
                "The default-stack-on-Jetson flow is the supported one; use Foxglove on the laptop "
                "but only subscribe to lightweight diagnostic/marker topics — pulling raw D435 images "
                "over the AP saturates the link and stalls Cartographer."
            ),
        ),
        OpaqueFunction(function=launch_setup),
    ])
