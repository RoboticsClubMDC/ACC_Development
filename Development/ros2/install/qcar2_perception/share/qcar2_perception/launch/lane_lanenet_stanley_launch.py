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
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# ──────────────────────────────────────────────────────────────────────────
# 2026-05-27: camera_source switch (csi | d435). D435 is the new default
# — the CSI's wide FOV + distortion was hurting Stanley's CTE quality on
# the SDCS lane. D435 has a narrower, rectified, higher-mounted RGB feed.
#
# Geometry differences between cameras (per QCar 2 User Manual + CLAUDE.md
# §14):
#   CSI front:  approx (X=0.18, Y=0.00, Z=0.13)  body frame, wide FOV
#   D435 RGB:   (X=0.095, Y=0.032, Z=0.172) body frame, ~69° HFOV, looking
#               forward. 3.2 cm LEFT of centerline (asymmetric).
#
# Two consequences for Stanley:
#   1. BEV "bottom row" is ~0.44 m ahead of D435 mount = ~0.54 m ahead of
#      body origin. That's already well ahead of the front axle (0.13 m).
#      So `front_axle_offset_m` can be small (we want CTE close to the
#      nose, not 0.5 m past it).
#   2. The car center is 3.2 cm RIGHT of the camera optical axis in body
#      frame. `car_center_offset_m` should reflect that asymmetry so
#      "CTE = 0" means "car centerline is on the lane center," not
#      "camera optical axis is on the lane center."
#
# The D435 src_* trapezoid below is derived analytically assuming ZERO
# camera pitch + zero roll on a 640×480 RGB image with fx=fy=615, cx=320,
# cy=240. If the physical D435 is tilted (almost certainly yes), the
# trapezoid will look skewed in the BEV debug overlay and you'll need to
# empirically refine src_top_left/right and src_bottom_left/right exactly
# as we did for the CSI. See change-log 2026-05-27 BEV iterations v1–v6.
# ──────────────────────────────────────────────────────────────────────────


CSI_PARAMS = {
    "image_topic": "/camera/csi_image",
    "undistort_enabled": True,
    "camera_matrix_fx": 318.86,
    "camera_matrix_fy": 312.14,
    "camera_matrix_cx": 401.34,
    "camera_matrix_cy": 201.50,
    # Arturo's empirical wide trapezoid (v6) — see change-log 2026-05-27.
    "src_top_left":     [310, 215],
    "src_top_right":    [510, 215],
    "src_bottom_right": [820, 410],
    "src_bottom_left":  [0,   410],
    "bev_world_width_m": 1.5,
    "car_center_offset_m": -0.40,   # empirical bias for CSI distortion
    "front_axle_offset_m": 0.10,
    "tracking_roi_top": 200,
    "tracking_roi_bottom": 400,
    "debug_crop_top": 200,
    "debug_crop_bottom": 400,
    "heading_offset_deg": 0.0,
}


D435_PARAMS = {
    "image_topic": "/perception/d435/rgb/image_raw",
    # D435 RGB is already rectified by the realsense / Quanser driver.
    "undistort_enabled": False,
    # 2026-05-27 (v2): REAL D435 intrinsics from
    # d435_aligned_source.py:55-58 (NOT Intel factory defaults).
    # The previous fx=615 was wrong — actual fx=455 means the D435 sees
    # a WIDER FOV than I credited. All trapezoid coords scaled up
    # accordingly.
    "camera_matrix_fx": 455.20,
    "camera_matrix_fy": 459.43,
    "camera_matrix_cx": 308.53,
    "camera_matrix_cy": 213.56,
    # Trapezoid v4 (2026-05-27): "slash image in half, focus on bottom"
    # per user instinct. v3's true ground rectangle was geometrically
    # clean but too narrow (±0.26 m laterally) AND its top row at v=246
    # was just below horizon → BEV's upper portion was filled with
    # far-field background (walls / horizon haze ~2.4 m ahead) rather
    # than usable road. v4 trades geometric cleanliness for visible road.
    #
    # Trapezoid lives entirely in the LOWER HALF of the source image
    # (v=300 → v=470). At zero pitch:
    #   v=300 → ground 0.91 m ahead of camera (= 1.00 m ahead of body)
    #   v=470 → ground 0.31 m ahead of camera (= 0.40 m ahead of body)
    #
    # Width chosen for max FOV coverage at each row, NOT for a true
    # ground rectangle (the world-frame shape is a trapezoid wider at
    # top than at bottom — like CSI's BEV did). Accept slight perspective
    # bias in CTE in exchange for actually-useful pixels.
    #
    #   top   u=[80, 560]  at v=300 → world Y ∈ [-0.47, +0.49] m
    #   bot   u=[ 0, 639]  at v=470 → world Y ∈ [-0.19, +0.24] m
    "src_top_left":     [ 80, 300],
    "src_top_right":    [560, 300],
    "src_bottom_right": [639, 470],
    "src_bottom_left":  [  0, 470],
    # BEV lateral coverage. Set to roughly match the WIDER end of the
    # trapezoid (top row ~0.96 m total) so far-field lane lines aren't
    # squashed in the BEV. CTE at the bottom row will have a slight
    # scale-bias (real ~0.43 m mapped to BEV's 1.0 m) — Stanley
    # tolerates that as long as gains are tuned post-calibration.
    "bev_world_width_m": 1.0,
    # D435 is 3.2 cm LEFT of centerline (Y_body = +0.032 m). The BEV's
    # vertical center pixel maps to a point 3.2 cm LEFT of the car
    # centerline, so the car centerline appears at a BEV column shifted
    # RIGHT by 3.2 cm. car_center_offset_m corrects CTE accordingly.
    # Sign tracks the convention "CTE_corrected = CTE_observed + offset".
    "car_center_offset_m": 0.032,
    # BEV bottom is ~0.40 m ahead of body origin (i.e., ~0.27 m ahead of
    # front axle, ~0.20 m ahead of front bumper). Keep front_axle_offset
    # small so Stanley's CTE evaluation point is near the nose, not way
    # out in front.
    "front_axle_offset_m": 0.05,
    "tracking_roi_top": 200,
    "tracking_roi_bottom": 400,
    "debug_crop_top": 200,
    "debug_crop_bottom": 400,
    "heading_offset_deg": 0.0,
}


def _launch_setup(context):
    cam = LaunchConfiguration("camera_source").perform(context).strip().lower()
    if cam == "csi":
        cam_params = CSI_PARAMS
    elif cam == "d435":
        cam_params = D435_PARAMS
    else:
        raise RuntimeError(
            f"camera_source must be 'csi' or 'd435' (got '{cam}')")

    detector_backend = LaunchConfiguration("detector_backend")
    lane_weight = LaunchConfiguration("lane_weight")
    path_weight = LaunchConfiguration("path_weight")

    common_params = {
        "detector_backend": detector_backend,
        "publish_debug_images": True,
        "use_centerline_skeleton": True,
        "intersection_branch": "straight",
        "lane_width_m": 0.254,           # 10-inch lane (SDCS)
        "bev_width":  400,
        "bev_height": 400,
        "debug_crop_overlay": True,
        "lookahead_distance_m": 0.05,
        "heading_segment_m": 0.15,
        # 2026-05-27: aggressive centerline skeletonization for DASHED
        # lane lines. Keep smaller component fragments (dash chunks),
        # widen the row-scan to bridge gaps between dashes, and lower
        # the minimum-valid-rows so a partial dash run still counts.
        "min_lane_component_area": 12,     # was 30 — keep small dash fragments
        "centerline_search_margin_px": 200,  # was 120 — bridge dash gaps
        "min_valid_rows": 8,               # was 15 — accept thinner runs
        "min_row_pixels": 2,               # was 3 — sensitive to thin lines
        # 2026-05-27: HSV defaults tuned for WHITE lane lines (the SDCS
        # dashed center line). The lane_detector's internal defaults are
        # for YELLOW which leaves the mask empty when the visible lane
        # is white — Stanley then sits silent because no centroids pass
        # min_valid_rows. "Bright + desaturated" = white:
        #   - any hue (white has no defined hue)
        #   - low saturation (desaturated)
        #   - high brightness (not the dark road)
        # To switch back to yellow tracking at runtime:
        #   ros2 param set /lane_detector hsv_h_low 18
        #   ros2 param set /lane_detector hsv_h_high 40
        #   ros2 param set /lane_detector hsv_s_low 80
        #   ros2 param set /lane_detector hsv_s_high 255
        #   ros2 param set /lane_detector hsv_v_low 120
        "hsv_h_low":  0,
        "hsv_h_high": 180,
        "hsv_s_low":  0,
        "hsv_s_high": 60,
        "hsv_v_low":  180,
        "hsv_v_high": 255,
    }

    lane_detector = Node(
        package="qcar2_perception",
        executable="lane_detector",
        name="lane_detector",
        parameters=[{**common_params, **cam_params}],
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
            "require_path": True,      # path_follower is the ignition (2026-05-27)
        }],
        output="screen",
    )

    return [lane_detector, lane_stanley_controller, cmd_vel_blender]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "camera_source",
            default_value="d435",
            choices=["csi", "d435"],
            description=(
                "Which camera feeds the lane detector. 'd435' (default, "
                "2026-05-27): rectified RGB, narrower FOV, mounted higher. "
                "'csi': legacy wide-FOV front CSI; needs distortion correction."
            ),
        ),
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
        OpaqueFunction(function=_launch_setup),
    ])
