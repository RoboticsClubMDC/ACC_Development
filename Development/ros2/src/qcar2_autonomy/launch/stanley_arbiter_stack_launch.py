# 2026-05-28: Stanley test stack built around motion_arbiter (NOT the blender),
# so Stanley can be GATED OFF on no-lane legs (e.g. the HUB curves 8→10, 10→1).
#
# Topology:
#   d435_aligned_source ─→ /perception/d435/rgb/image_raw
#        ↓
#   lane_detector ─→ /lane_keeping/* ─→ lane_stanley_controller ─→ /cmd_vel_lane ┐
#                                                                                 │
#   path_follower (apply_gyro_damping:=false) ─→ /cmd_vel_path ──────────────────┤
#        └─→ /nav/lane_gate (full-leg gating from no_lane_legs)                   │
#                                                                                 ▼
#                                                          motion_arbiter ─→ /cmd_vel_nav
#                          (lane-primary open road; PATH-ONLY when gate closed)
#
# Does NOT start a base / AMCL / converter — run carto_to_amcl.sh or
# amcl_load.sh (with NAV_TO_HUB=0) first for localization + /cmd_vel_nav→motors.
# Only the D435 source is started (no YOLO/landmarks) to keep it light for a
# pure lane test.
#
# Usage (after AMCL is up):
#   ros2 launch qcar2_autonomy stanley_arbiter_stack_launch.py
#   ros2 param set /path_follower node_values "[-1, <destination_node>]"
#
# Watch: /nav/lane_gate (true=lane on, false=Stanley off), /arbiter/mode,
#        /cmd_vel_lane, /lane_keeping/lane_detected.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            OpaqueFunction, SetEnvironmentVariable)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context):
    perception_share = get_package_share_directory("qcar2_perception")

    physical = LaunchConfiguration("physical").perform(context).strip().lower() \
        in ("true", "1", "yes")
    camera_source = LaunchConfiguration("camera_source").perform(context)
    detector_backend = LaunchConfiguration("detector_backend")

    actions = []

    # On the VIRTUAL laptop container (no CUDA passthrough) force CPU; on the
    # physical Jetson leave QCAR2_FORCE_CPU UNSET so the Orin GPU is used.
    if not physical:
        actions.append(SetEnvironmentVariable(name="QCAR2_FORCE_CPU", value="1"))

    # D435 source. PHYSICAL: real D435 backend + metric depth (scale 1.0).
    # VIRTUAL: QLabs backend + 10× world → distance_scale 0.1.
    d435_aligned_source = Node(
        package="qcar2_perception",
        executable="d435_aligned_source",
        name="d435_aligned_source",
        parameters=[{
            "is_physical": physical,
            "distance_scale": 1.0 if physical else 0.1,
        }],
        output="screen",
    )

    # lane_detector + lane_stanley_controller, blender OMITTED (arbiter owns bus).
    lane_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(perception_share, "launch",
                         "lane_lanenet_stanley_launch.py")
        ),
        launch_arguments={
            "camera_source": camera_source,
            "detector_backend": detector_backend.perform(context),
            "run_blender": "false",
        }.items(),
    )

    motion_arbiter = Node(
        package="qcar2_autonomy",
        executable="motion_arbiter",
        name="motion_arbiter",
        output="screen",
        # 2026-05-28: more path bias than the arbiter's 0.75/0.25 default while
        # we tune Stanley out of its overshoot — lane still leads, path steadies.
        # /nav/lane_gate still flips to JUNCTION_PATH_ONLY on the gated legs.
        parameters=[{
            # 2026-05-28 (rev2): lane leads more now that CTE is clean — the
            # extra path bias was masking the late-steering symptom.
            "lane_weight": 0.7,   # 0.75→0.6→0.7
            "path_weight": 0.3,   # 0.25→0.4→0.3
            # rev3: path_follower runs apply_gyro_damping:=false here, so the
            # ARBITER is the ONLY damper. 0.20 under-damped raw PP → big swings
            # (PP-only mode self-damped and drove clean). Bump the single D-gain.
            "kd_steering": 0.40,  # was 0.20
        }],
    )

    path_follower = Node(
        package="qcar2_autonomy",
        executable="path_follower",
        name="path_follower",
        output="screen",
        parameters=[{
            # Publish to the arbiter, NOT directly to /cmd_vel_nav.
            "cmd_topic": "/cmd_vel_path",
            # Arbiter is the single gyro D-damp authority — path emits raw kp·pp.
            "apply_gyro_damping": False,
            # Full-leg no-Stanley zones near the HUB: 8→10 and 10→1.
            "no_lane_legs": [8, 10, 10, 1],
            "leg_node_radius": 0.40,
            # rev3: smoother PP (longer lookahead, gentler gain) to stop the big
            # swings on the way to the HUB; wider align trigger so the seat
            # reliably fires once near node 10 (fixes stuck-at-HUB).
            "lookahead_dist_floor": 0.50,          # was 0.30 — less oscillation
            "kp_steering": 0.90,                   # was 1.10 — gentler raw PP
            "arrival_align_trigger_radius": 0.55,  # was 0.35 — fire align sooner
            # Physical RPLidar is mounted 180° yaw; virtual is 0. Only used if
            # align_wall_detect is turned on.
            "lidar_yaw_offset_deg": 180.0 if physical else 0.0,
        }],
    )

    actions += [d435_aligned_source, lane_stack, motion_arbiter, path_follower]
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("camera_source", default_value="d435",
                              choices=["csi", "d435"]),
        DeclareLaunchArgument("detector_backend", default_value="hsv",
                              choices=["hsv", "lanenet"]),
        DeclareLaunchArgument(
            "physical", default_value="false", choices=["true", "false"],
            description=("true → real D435 backend (is_physical, depth scale "
                         "1.0), GPU enabled, lidar 180° yaw. false → QLabs "
                         "virtual (sim backend, depth 0.1, force CPU).")),
        OpaqueFunction(function=_setup),
    ])
