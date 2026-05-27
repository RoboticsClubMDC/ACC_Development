"""
2026-05-27 Lane stack: BEV + LaneNet/HSV + Stanley + cmd_vel blender.

Migrated from origin/i-hate-gabriel (Arturo's branch). Replaces the previous
qcar2_perception lane_assist.launch.py which ran our HSV-pixel-Stanley stack
that we could not stabilize. See Easy_Start.md change log 2026-05-27 for the
full rationale.

Topology (NO topic fights — each controller writes its OWN topic):

  /camera/csi_image
        ↓
    lane_detector (BEV homography + LaneNet OR HSV)
        ↓ /lane_keeping/{cross_track_error, heading_error, lane_detected}
        ↓
    lane_stanley_controller (30 Hz timer)
        ↓ /cmd_vel_lane
                          \\
                           cmd_vel_blender (60% lane / 40% path) → /cmd_vel_nav
                          /
  path_follower → /cmd_vel_path
                          ↑
                    /car_stop (emergency stop topic, optional)

PRE-REQS:
- A csi_camera node must be publishing /camera/csi_image (from
  qcar2_cartographer_*_launch.py or qcar2_launch.py).
- For LaneNet backend: pit.LaneNet package must be importable from one of
  the candidate paths in lane_detector._resolve_lanenet_import. The LaneNet
  TensorRT engine file 'lanenet.engine' must exist (set lanenet_model_path
  param or place at the default pretrained_models location).
- Homography source points (src_top_left/right/bottom_left/right) are
  scene-specific. The defaults are from Arturo's QCar setup. RECALIBRATE
  for QLabs by laying a known rectangle on the road and reading off pixel
  coords (a simple OpenCV picker script helps).

LAUNCH ARGS:
  detector_backend:=hsv | lanenet      (default: hsv — safer until calibrated)
  publish_cmd:=true                    (lane_stanley publishes /cmd_vel_lane)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    detector_backend = LaunchConfiguration("detector_backend")
    lane_weight = LaunchConfiguration("lane_weight")
    path_weight = LaunchConfiguration("path_weight")

    lane_detector = Node(
        package="qcar2_perception",
        executable="lane_detector",
        name="lane_detector",
        parameters=[{
            "image_topic": "/camera/csi_image",
            "detector_backend": detector_backend,
            "publish_debug_images": True,
            "use_centerline_skeleton": True,
            "intersection_branch": "straight",
            "lane_width_m": 0.254,           # 10-inch lane (SDCS)
            # 2026-05-27 (v5): back to Arturo's -0.40. With the empirical
            # wide-trapezoid scaling (bev_world_width_m=1.5 is empirical,
            # not literal), -0.40 is the matching empirical bias that
            # places car_x at the BEV position where Stanley's CTE settles
            # correctly. Pinhole-math 0.0 only works with the matching
            # mathematically-correct narrow trapezoid (v4), which we
            # abandoned because it didn't give enough near-field detail
            # on our hardware-limited CSI.
            "car_center_offset_m": -0.40,
            # 2026-05-27: front_axle_offset_m is "how far ahead of BEV
            # bottom to evaluate CTE." Was 0.256 (wheelbase) → CTE at
            # 0.756m forward (too far for chasing). Now 0.10 → CTE at
            # 0.60m forward — Stanley reacts to near-field lane geometry
            # promptly without over-anticipating.
            "front_axle_offset_m": 0.10,
            # 2026-05-27 (v6): MAXIMUM near-field, MINIMUM waste.
            # Building on v5 (Arturo's empirical wide trapezoid) but
            # pushed further per user feedback: "what if we sacrifice
            # more, why does my camera see the walls?"
            #
            # Changes from v5:
            #   - Bottom edge: 0 → 820 (FULL image width) instead of 26-794
            #     → captures the lane line even when it curves into the
            #       very edge of the image at the bumper
            #   - Top edge: y=171 (above horizon, sky) → y=215 (just below
            #     horizon, ground ~3-4m forward)
            #     → no more sky pixels in the trapezoid
            #     → top of BEV no longer fills with distorted sky/wall
            #       content that the row scanner had to skip anyway
            #
            # Trapezoid shape: very wide at bottom (full image), narrower
            # at top (just below horizon). Heavy perspective stretch but
            # ALL pixels are useful (none are sky or far-walls).
            "src_top_left":     [310, 215],
            "src_top_right":    [510, 215],
            "src_bottom_right": [820, 410],
            "src_bottom_left":  [0,   410],
            "bev_world_width_m": 1.5,         # empirical, may need retune
            "bev_width":  400,
            "bev_height": 400,
            "heading_offset_deg": 0.0,       # calibrate per scene
            "tracking_roi_top": 200,
            "tracking_roi_bottom": 400,
            "debug_crop_overlay": True,
            "debug_crop_top": 200,
            "debug_crop_bottom": 400,
            # 2026-05-27: lookahead and heading-span TIGHTENED.
            # Stanley needs near-field info to compute CTE at the front
            # axle / just ahead of the bumper. With BEV bottom = 0.5m
            # forward, front_axle_offset_m=0.256 was putting CTE eval at
            # 0.756m forward — too far for low-speed lane chasing; if
            # mask is thin there → LOST → car stops mid-chase.
            # Now CTE evaluated at 0.6m forward (0.10m ahead of BEV bottom).
            # Heading span 0.15m covers BEV 0.6m → 0.75m — tight local
            # tangent estimate that responds well to curves.
            "lookahead_distance_m": 0.05,
            "heading_segment_m": 0.15,
        }],
        output="screen",
    )

    lane_stanley_controller = Node(
        package="qcar2_perception",
        executable="lane_stanley_controller",
        name="lane_stanley_controller",
        parameters=[{
            "cmd_topic": "/cmd_vel_lane",
            "speed_mps": 0.20,
            "stanley_gain": 0.5,
            "heading_gain": 1.0,
            "max_steer_rad": 0.35,
            "publish_stop_when_lost": False,
        }],
        output="screen",
    )

    cmd_vel_blender = Node(
        package="qcar2_perception",
        executable="cmd_vel_blender",
        name="cmd_vel_blender",
        parameters=[{
            "path_cmd_topic": "/cmd_vel_path",
            "lane_cmd_topic": "/cmd_vel_lane",
            "cmd_topic": "/cmd_vel_nav",
            "lane_weight": lane_weight,
            "path_weight": path_weight,
            "linear_source": "path",   # speed from path_follower
        }],
        output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "detector_backend",
            default_value="hsv",
            choices=["hsv", "lanenet"],
            description=(
                "Lane detector backend. 'hsv' is the classical color-threshold "
                "fallback (no model needed). 'lanenet' uses Quanser PIT's "
                "neural net — requires the lanenet.engine file."
            ),
        ),
        DeclareLaunchArgument(
            "lane_weight",
            default_value="0.60",
            description="Blend weight for lane Stanley δ in cmd_vel_blender.",
        ),
        DeclareLaunchArgument(
            "path_weight",
            default_value="0.40",
            description="Blend weight for path_follower δ in cmd_vel_blender.",
        ),
        lane_detector,
        lane_stanley_controller,
        cmd_vel_blender,
    ])
