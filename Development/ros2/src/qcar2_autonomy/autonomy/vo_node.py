"""VO redundancy monitor built on depth-based visual odometry.

This node runs the depth-based VO engine and compares the resulting pose
against the current Cartographer `map -> base_link` transform.

Input modes:
1) `ros_rgbd` (default):
   - subscribes to `/camera/color_image` and `/camera/depth_image`
   - expects raw MONO16 depth from `rgbd.cpp`
2) `pit_aligned`:
   - reads aligned RGB+Depth directly from `QCar2DepthAligned`
   - uses depth already aligned to color pixels (meters)

Published topics:
    /vo/fault_status   — Full diagnostic string
    /vo/delta_trans     — Residual rho (Float64)
    /vo/vo_weight       — Trust weight 0-1 (Float64)
    /vo/healthy         — Health flag (Bool)
    /vo/state_id        — Numeric state (Int32)
    /vo/vo_x,y,psi      — VO pose in map frame (Float64 each)
    /vo/vo_psi_shadow   — Camera-only yaw integrator, ignores force_cart_yaw
                          (Float64, radians)
    /vo/cart_x,y,psi    — Cartographer pose (Float64 each)
    /vo/frame_dx,dy,dpsi — Per-frame delta (Float64 each)
    /vo/inliers         — Inlier count (Int32)
    /vo/confidence      — Confidence ratio 0-1 (Float64)
    /vo/spread          — Inlier pixel spread (Float64)
"""

import threading
import time
import os
import sys
from pathlib import Path
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import (
    MutuallyExclusiveCallbackGroup,
    ReentrantCallbackGroup,
)
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float64, Bool, Int32
from cv_bridge import CvBridge
import tf2_ros
from rcl_interfaces.msg import ParameterDescriptor

# Import the depth-based VO engine
from autonomy.visual_odometry import VisualOdometryDepth


def _load_qcar2_depth_aligned(custom_path):
    """Resolve and import PIT QCar2DepthAligned helper.

    Returns:
        (QCar2DepthAligned class, resolved_path_or_note)
    Raises:
        ImportError if PIT helper cannot be imported.
    """
    # Try current environment first.
    try:
        from pit.YOLO.utils import QCar2DepthAligned  # type: ignore
        return QCar2DepthAligned, "sys.path(existing)"
    except Exception:
        pass

    candidates = []
    if custom_path:
        candidates.append(custom_path)

    env_path = os.getenv('MDC_PYTHON_PATH', '').strip()
    if env_path:
        candidates.extend([p for p in env_path.split(':') if p.strip()])

    home = Path.home()
    candidates.extend([
        str(home / "Documents/ACC_Development/Development/MDC_libraries/python"),
        "/home/nvidia/Documents/ACC_Development/Development/MDC_libraries/python",
        "/workspaces/isaac_ros-dev/MDC_libraries/python",
    ])

    for path in candidates:
        if not path:
            continue
        if not Path(path).exists():
            continue
        if path not in sys.path:
            sys.path.insert(0, path)
        try:
            from pit.YOLO.utils import QCar2DepthAligned  # type: ignore
            return QCar2DepthAligned, path
        except Exception:
            continue

    raise ImportError(
        "Could not import pit.YOLO.utils.QCar2DepthAligned. "
        "Set parameter 'pit_python_path' or env 'MDC_PYTHON_PATH' to the "
        "directory containing the 'pit' package."
    )


class VONodeDepth(Node):

    def __init__(self):
        super().__init__('vo_redundancy_monitor_depth')

        # ── Parameters: VO engine ───────────────────────────────
        self.declare_parameter('n_features', 800)
        self.declare_parameter('match_ratio', 0.75)
        self.declare_parameter('ransac_iterations', 300)
        self.declare_parameter('ransac_threshold', 0.05)
        self.declare_parameter('min_inliers', 8)
        self.declare_parameter('negate_deltas', True)
        self.declare_parameter('max_translation', 0.20)
        self.declare_parameter('max_rotation_deg', 15.0)
        # 2026-05-15: ORB ROI mask. Top fraction of the image is excluded
        # from feature detection. 0.0 = full frame (legacy). Typical
        # productive values 0.30-0.40 to skip sky/horizon and focus on
        # ground-level features with usable depth.
        self.declare_parameter('roi_top_fraction', 0.0)
        # 2026-05-15: RANSAC minimum-sample size. 2 = legacy minimum for
        # the 3-DOF rigid transform; 3 = robust over-constrained sample
        # that resists degenerate consensus on low-texture/bare-wall
        # scenes. Clamped to >= 2 in VisualOdometryDepth.
        self.declare_parameter('ransac_sample_size', 2)
        # 2026-05-18: Step 1 — depth-weighted Procrustes power.
        # Each matched correspondence is weighted by 1/range**power in
        # the SVD rigid fit (range = camera depth of the noisier
        # endpoint, meters in physical mode). 0.0 = OFF = the prior
        # unweighted estimator (Test 6 control). 2.0 = gentle
        # (weight ∝ 1/Z^2). 4.0 = full RealSense model (sigma_Z ∝ Z^2
        # -> weight ∝ 1/Z^4). Clamped to [0, 8] in VisualOdometryDepth.
        self.declare_parameter('depth_weight_power', 0.0)
        # 2026-05-18: Step 1b — HARD max-depth cutoff (operator idea;
        # Quanser precedent: pit/YOLO/nets.py clips depth past
        # clippingDistance). A matched feature farther than this (camera
        # range, metres in physical mode) is dropped from VO motion
        # entirely. 0.0 = OFF (prior behavior). Composes with
        # depth_weight_power. Typical test value 2.5-3.0.
        self.declare_parameter('max_vo_feature_depth_m', 0.0)
        # Official short-term policy for competition:
        # use Cartographer yaw for VO heading while keeping camera-derived
        # translation increments.
        self.declare_parameter('force_cart_yaw', True)

        # Depth-specific parameters
        # `depth_scale <= 0` means "use the mode-specific default from
        # visual_odometry.py" instead of forcing the virtual divisor into
        # physical runs.
        self.declare_parameter('depth_scale', 0.0)
        # `alignment_mode` controls whether VO should use depth-to-color
        # alignment:
        #   auto: virtual=True, physical=False (conservative physical default)
        #   on:   always warp
        #   off:  never warp
        # Allow CLI inputs like `-p alignment_mode:=off` which ROS may coerce
        # to bool False; we normalize downstream.
        self.declare_parameter(
            'alignment_mode',
            'auto',
            ParameterDescriptor(dynamic_typing=True),
        )

        # Redundancy monitor params (same as homography version)
        self.declare_parameter('trans_thresh_m', 0.35)
        self.declare_parameter('n_consecutive', 3)
        self.declare_parameter('window_seconds', 2.0)
        self.declare_parameter('turn_gate_deg', 5.0)
        self.declare_parameter('min_vo_weight', 0.30)
        self.declare_parameter('spread_bad', 55.0)
        self.declare_parameter('spread_good', 120.0)
        self.declare_parameter('warmup_after_reanchor_s', 1.0)
        self.declare_parameter('reanchor_cooldown_s', 8.0)
        self.declare_parameter('reanchor_min_dist', 0.25)

        # Optional RGB intrinsics override
        self.declare_parameter('fx_override', 0.0)
        self.declare_parameter('fy_override', 0.0)

        # Camera mode: 'virtual' (FAQ intrinsics) or 'physical' (calibrated intrinsics)
        self.declare_parameter('camera_mode', 'virtual')
        # Camera input source:
        #   ros_rgbd    -> /camera/color_image + /camera/depth_image topics
        #   pit_aligned -> PIT QCar2DepthAligned runtime stream
        self.declare_parameter('input_source', 'ros_rgbd')
        self.declare_parameter('color_topic', '/camera/color_image')
        self.declare_parameter('depth_topic', '/camera/depth_image')
        self.declare_parameter('pit_python_path', '')
        self.declare_parameter('pit_ip', 'localhost')
        self.declare_parameter('pit_port', 18777)
        self.declare_parameter('pit_manual_start', False)
        self.declare_parameter('pit_non_blocking', True)

        # Read params
        n_features = self.get_parameter('n_features').value
        match_ratio = self.get_parameter('match_ratio').value
        ransac_iter = self.get_parameter('ransac_iterations').value
        ransac_thresh = self.get_parameter('ransac_threshold').value
        min_inliers = self.get_parameter('min_inliers').value
        negate = self.get_parameter('negate_deltas').value
        max_trans = self.get_parameter('max_translation').value
        max_rot = self.get_parameter('max_rotation_deg').value
        roi_top_fraction = float(
            self.get_parameter('roi_top_fraction').value)
        ransac_sample_size = int(
            self.get_parameter('ransac_sample_size').value)
        depth_weight_power = float(
            self.get_parameter('depth_weight_power').value)
        max_vo_feature_depth_m = float(
            self.get_parameter('max_vo_feature_depth_m').value)
        self.force_cart_yaw = bool(self.get_parameter('force_cart_yaw').value)
        camera_mode = self.get_parameter('camera_mode').value
        if camera_mode not in ('virtual', 'physical'):
            self.get_logger().warn(
                f"Invalid camera_mode '{camera_mode}', defaulting to 'virtual'")
            camera_mode = 'virtual'
        self.input_source = str(
            self.get_parameter('input_source').value).strip().lower()
        if self.input_source not in ('ros_rgbd', 'pit_aligned'):
            self.get_logger().warn(
                f"Invalid input_source '{self.input_source}', defaulting to "
                "'ros_rgbd'")
            self.input_source = 'ros_rgbd'

        depth_scale_param = float(self.get_parameter('depth_scale').value)
        depth_scale = depth_scale_param if depth_scale_param > 0.0 else None

        alignment_mode = str(self.get_parameter('alignment_mode').value).strip().lower()
        if alignment_mode == 'auto':
            use_align = (camera_mode == 'virtual')
        elif alignment_mode in ('on', 'true', '1', 'yes'):
            use_align = True
        elif alignment_mode in ('off', 'false', '0', 'no'):
            use_align = False
        else:
            self.get_logger().warn(
                f"Invalid alignment_mode '{alignment_mode}', defaulting to 'auto'")
            alignment_mode = 'auto'
            use_align = (camera_mode == 'virtual')
        # `depth_on_color_grid` governs which intrinsics K_inv is used when
        # backprojecting matched ORB features. PIT delivers depth that is
        # already aligned to the color pixel grid even though no software
        # warp is applied — force the flag True for that case so RGB K_inv
        # is used. For ros_rgbd we default it to `use_align` (the legacy
        # meaning of the flag).
        depth_on_color_grid = bool(use_align)
        if self.input_source == 'pit_aligned':
            if use_align:
                self.get_logger().warn(
                    "input_source=pit_aligned already provides aligned depth."
                    " Overriding alignment OFF for VO projector.")
                use_align = False
            depth_on_color_grid = True

        self.trans_thresh = self.get_parameter('trans_thresh_m').value
        self.n_consecutive = self.get_parameter('n_consecutive').value
        self.window_sec = self.get_parameter('window_seconds').value
        self.turn_gate_rad = np.radians(
            self.get_parameter('turn_gate_deg').value)
        self.min_vo_weight = self.get_parameter('min_vo_weight').value
        self.spread_bad = self.get_parameter('spread_bad').value
        self.spread_good = self.get_parameter('spread_good').value
        self.warmup_s = self.get_parameter('warmup_after_reanchor_s').value
        self.reanchor_cool = self.get_parameter('reanchor_cooldown_s').value
        self.reanchor_min_dist = self.get_parameter('reanchor_min_dist').value

        fx_ov = self.get_parameter('fx_override').value
        fy_ov = self.get_parameter('fy_override').value
        fx = fx_ov if fx_ov > 1.0 else None
        fy = fy_ov if fy_ov > 1.0 else None

        # ── VO Engine (depth-based) ─────────────────────────────
        self.vo = VisualOdometryDepth(
            img_width=640, img_height=480,
            fx_rgb=fx, fy_rgb=fy,
            depth_scale=depth_scale,
            use_alignment=use_align,
            depth_on_color_grid=depth_on_color_grid,
            camera_mode=camera_mode,
            n_features=n_features,
            match_ratio=match_ratio,
            ransac_threshold=ransac_thresh,
            ransac_iterations=ransac_iter,
            min_inliers=min_inliers,
            max_translation=max_trans,
            max_rotation_deg=max_rot,
            negate_deltas=negate,
            roi_top_fraction=roi_top_fraction,
            ransac_sample_size=ransac_sample_size,
            depth_weight_power=depth_weight_power,
            max_vo_feature_depth_m=max_vo_feature_depth_m,
        )
        self.vo_lock = threading.Lock()

        if (camera_mode == 'physical'
                and self.input_source == 'ros_rgbd'
                and depth_scale is None):
            self.get_logger().warn(
                "Physical mode is using the placeholder MONO16 divisor from "
                "visual_odometry.py until rgbd.cpp publishes metric depth or "
                "a measured physical depth_scale is provided.")
        if camera_mode == 'physical' and alignment_mode == 'auto' and not use_align:
            self.get_logger().warn(
                "Physical mode defaulted alignment OFF in auto mode. "
                "Use '-p alignment_mode:=on' only when validating projective "
                "depth->color alignment behavior.")
        if camera_mode == 'physical' and use_align:
            self.get_logger().info(
                "Physical mode depth alignment is ON "
                "(projective depth->color path).")
        if self.force_cart_yaw:
            self.get_logger().warn(
                "force_cart_yaw is ON: VO heading will be pinned to "
                "Cartographer yaw.")

        # ── State ───────────────────────────────────────────────
        self.bridge = CvBridge()
        self.color_img = None
        self.color_stamp = None
        self.depth_img = None
        self.state_lock = threading.Lock()

        self.cart_x = self.cart_y = self.cart_psi = 0.0
        self.has_cart = False

        self.eval_window = deque(maxlen=2000)
        self.agree_count = 0
        self.odom_suspect_count = 0
        self.vo_suspect_count = 0
        self.confirmed_state = 'init'
        self.healthy = True

        self.last_reanchor_time = 0.0
        self.last_reanchor_xy = np.array([0.0, 0.0])
        self.warmup_until = 0.0
        self.vo_anchored = False
        self.last_color_stamp = None

        # Per-frame diagnostics
        self.last_frame_dx = 0.0
        self.last_frame_dy = 0.0
        self.last_frame_dpsi = 0.0
        self.last_inliers = 0
        self.last_confidence = 0.0
        self.last_spread = 0.0

        # Shadow camera-only yaw: independent integrator of the raw per-frame
        # dpsi from the engine. Tracks what `vo_psi` would be if force_cart_yaw
        # were OFF, so we can compare against the pinned/cart yaw without
        # disturbing the active pipeline.
        self.vo_psi_shadow = 0.0

        # Depth diagnostics (for verifying depth_scale)
        self._depth_diag_counter = 0
        self._pit_camera = None
        # Set to True the first time we see aligned float depth on
        # /camera/depth_image (encoding=32FC1 from qcar2_camera_bridge).
        # This both prevents repeated log-spam and is the gate for flipping
        # depth_on_color_grid on the projector at runtime.
        self._aligned_depth_locked = False

        # ── TF ──────────────────────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Input setup ─────────────────────────────────────────
        if self.input_source == 'pit_aligned':
            self._init_pit_camera(camera_mode)
        else:
            color_topic = self.get_parameter('color_topic').value
            depth_topic = self.get_parameter('depth_topic').value
            cam_cb_group = ReentrantCallbackGroup()
            # 2026-05-15: sensor_data QoS (BEST_EFFORT, depth=5) to match
            # the bridge publishers. Reliable QoS on high-rate large Image
            # messages was causing half-drop at the DDS layer (visible as
            # ~17 fps on `ros2 topic hz` vs 30 fps on the bridge heartbeat).
            self.color_sub = self.create_subscription(
                Image, color_topic, self._color_cb, qos_profile_sensor_data,
                callback_group=cam_cb_group)
            # Depth from rgbd.cpp: MONO16 encoding, uint16 per pixel
            self.depth_sub = self.create_subscription(
                Image, depth_topic, self._depth_cb, qos_profile_sensor_data,
                callback_group=cam_cb_group)

        # ── Timer ───────────────────────────────────────────────
        vo_cb_group = MutuallyExclusiveCallbackGroup()
        self.vo_timer = self.create_timer(
            0.05, self._vo_tick, callback_group=vo_cb_group)

        # ── Publishers (same topics as homography version) ──────
        self.pub_fault = self.create_publisher(String, '/vo/fault_status', 5)
        self.pub_delta = self.create_publisher(Float64, '/vo/delta_trans', 5)
        self.pub_weight = self.create_publisher(Float64, '/vo/vo_weight', 5)
        self.pub_healthy = self.create_publisher(Bool, '/vo/healthy', 5)
        self.pub_state_id = self.create_publisher(Int32, '/vo/state_id', 5)
        self.pub_vo_x = self.create_publisher(Float64, '/vo/vo_x', 5)
        self.pub_vo_y = self.create_publisher(Float64, '/vo/vo_y', 5)
        self.pub_vo_psi = self.create_publisher(Float64, '/vo/vo_psi', 5)
        self.pub_cart_x = self.create_publisher(Float64, '/vo/cart_x', 5)
        self.pub_cart_y = self.create_publisher(Float64, '/vo/cart_y', 5)
        self.pub_cart_psi = self.create_publisher(Float64, '/vo/cart_psi', 5)
        self.pub_frame_dx = self.create_publisher(Float64, '/vo/frame_dx', 5)
        self.pub_frame_dy = self.create_publisher(Float64, '/vo/frame_dy', 5)
        self.pub_frame_dpsi = self.create_publisher(
            Float64, '/vo/frame_dpsi', 5)
        self.pub_inliers = self.create_publisher(Int32, '/vo/inliers', 5)
        self.pub_conf = self.create_publisher(Float64, '/vo/confidence', 5)
        self.pub_spread = self.create_publisher(Float64, '/vo/spread', 5)
        # Shadow yaw — camera-only yaw integrated independent of force_cart_yaw
        self.pub_vo_psi_shadow = self.create_publisher(
            Float64, '/vo/vo_psi_shadow', 5)

        # ── Startup banner ──────────────────────────────────────
        k = self.vo.projector.K_rgb
        mode_str = f"{camera_mode.upper()} (FAQ intrinsics)" if camera_mode == 'virtual' else f"{camera_mode.upper()} (calibrated)"
        source_str = "PIT_ALIGNED" if self.input_source == 'pit_aligned' else "ROS_RGBD"
        self.get_logger().info(
            f"\n╔══════════════════════════════════════════════╗\n"
            f"║  VO REDUNDANCY MONITOR — DEPTH-BASED         ║\n"
            f"╠══════════════════════════════════════════════╣\n"
            f"║  Camera mode: {mode_str:<21}║\n"
            f"║  Input source: {source_str:<20}║\n"
            f"║  RGB K: fx={k[0,0]:.1f} fy={k[1,1]:.1f}"
            f" cx={k[0,2]:.1f} cy={k[1,2]:.1f}\n"
            f"║  Depth scale: {self.vo.projector.depth_scale:.0f}"
            f" ({self.vo.projector.depth_units_label})\n"
            f"║  Alignment: {'ON' if self.vo.projector.use_alignment else 'OFF':<21}║\n"
            f"║  negate_deltas: {negate!s:<16}║\n"
            f"║  features={n_features}  match_ratio={match_ratio}\n"
            f"║  depth_weight_power={depth_weight_power} "
            f"({'ON' if depth_weight_power > 0.0 else 'OFF (=Test 6)'})\n"
            f"║  max_vo_feature_depth_m={max_vo_feature_depth_m} "
            f"({'ON' if max_vo_feature_depth_m > 0.0 else 'OFF'})\n"
            f"╚══════════════════════════════════════════════╝")
        if depth_weight_power > 0.0:
            self.get_logger().warn(
                f"Step 1 depth-weighted Procrustes ACTIVE "
                f"(depth_weight_power={depth_weight_power}); the SVD fit "
                "down-weights far/noisy depth points. Set "
                "depth_weight_power:=0.0 for the unweighted Test 6 control.")
        if max_vo_feature_depth_m > 0.0:
            self.get_logger().warn(
                f"Step 1b hard max-depth cutoff ACTIVE "
                f"(max_vo_feature_depth_m={max_vo_feature_depth_m} m); "
                "matched features farther than this are dropped from VO "
                "motion. Set max_vo_feature_depth_m:=0.0 to disable.")

    # ── Callbacks ───────────────────────────────────────────────

    def _color_cb(self, msg):
        with self.state_lock:
            self.color_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.color_stamp = msg.header.stamp

    def _depth_cb(self, msg):
        """Decode the depth image from whichever publisher is active.

        Two upstream publishers are supported:

        - `rgbd.cpp` (legacy / virtual default):
            encoding `MONO16`, uint16 raw counts. Engine divides by
            `depth_scale` to convert to mode units.

        - `qcar2_camera_bridge` (physical default, single-owner pattern):
            encoding `32FC1`, float depth in meters, ALREADY aligned to
            the color pixel grid. On first sight of this encoding we flip
            `depth_on_color_grid=True` on the projector so the K_inv fix
            (use RGB intrinsics for backprojection) takes effect.

        The PAL library (vision.py) uses UINT8 instead, with a different
        representation. The two paths are not interchangeable.
        """
        encoding = getattr(msg, 'encoding', '') or ''
        is_aligned_float = (encoding == '32FC1')

        with self.state_lock:
            try:
                if is_aligned_float:
                    depth_f = self.bridge.imgmsg_to_cv2(msg, '32FC1')
                    if len(depth_f.shape) == 3:
                        depth_f = depth_f[:, :, 0]
                    self.depth_img = depth_f.astype(np.float32, copy=False)
                    if not self._aligned_depth_locked:
                        self.vo.projector.depth_on_color_grid = True
                        self._aligned_depth_locked = True
                        self.get_logger().info(
                            "Detected aligned 32FC1 depth on "
                            f"{msg.header.frame_id or '/camera/depth_image'}; "
                            "set depth_on_color_grid=True. VO will use RGB "
                            "intrinsics (K_rgb_inv) for backprojection.")
                else:
                    # Legacy MONO16 path (rgbd.cpp / virtual). Preserve
                    # the existing uint16 + depth_scale behavior.
                    depth_16 = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
                    if len(depth_16.shape) == 3:
                        depth_16 = depth_16[:, :, 0]
                    self.depth_img = depth_16.astype(np.uint16)
            except Exception as e:
                self.get_logger().warn(
                    f"Depth decode error: {e}", throttle_duration_sec=5.0)
                return

        # Periodic depth diagnostic: branch on whichever path is active.
        self._depth_diag_counter += 1
        if self._depth_diag_counter % 200 == 1:  # every ~8 seconds at 25Hz
            if is_aligned_float:
                H, W = self.depth_img.shape
                center_m = float(self.depth_img[H // 2, W // 2])
                self.get_logger().info(
                    f"[DEPTH DIAG ALIGNED] center={center_m:.3f} m (32FC1)")
            else:
                H, W = self.depth_img.shape
                raw_center = int(self.depth_img[H // 2, W // 2])
                projector = self.vo.projector
                scale = projector.depth_scale
                depth_val = raw_center / scale if scale > 0 else 0.0
                physical_m = depth_val * projector.physical_m_per_unit
                self.get_logger().info(
                    f"[DEPTH DIAG] center raw={raw_center}, "
                    f"/ {scale:.0f} = {depth_val:.3f} "
                    f"{projector.depth_units_label} "
                    f"({physical_m:.4f} m physical)")

    def _init_pit_camera(self, camera_mode):
        pit_python_path = str(self.get_parameter('pit_python_path').value).strip()
        pit_ip = str(self.get_parameter('pit_ip').value).strip()
        pit_port = int(self.get_parameter('pit_port').value)
        pit_manual_start = bool(self.get_parameter('pit_manual_start').value)
        pit_non_blocking = bool(self.get_parameter('pit_non_blocking').value)

        try:
            QCar2DepthAligned, pit_path_used = _load_qcar2_depth_aligned(
                pit_python_path)
            self._pit_camera = QCar2DepthAligned(
                ip=pit_ip,
                nonBlocking=pit_non_blocking,
                manualStart=pit_manual_start,
                port=str(pit_port),
                isPhyscial=(camera_mode == 'physical'),
            )
            self.get_logger().info(
                "PIT aligned source initialized "
                f"(python_path={pit_path_used}, ip={pit_ip}, port={pit_port})")
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize input_source=pit_aligned. "
                "Check pit_python_path/MDC_PYTHON_PATH and PIT runtime setup."
            ) from exc

    def _read_pit_frame(self):
        if self._pit_camera is None:
            return None, None, None
        try:
            new = self._pit_camera.read()
        except Exception as exc:
            self.get_logger().warn(
                f"PIT camera read failed: {exc}", throttle_duration_sec=5.0)
            return None, None, None
        if not new:
            return None, None, None

        img = np.ascontiguousarray(self._pit_camera.rgb)
        depth = self._pit_camera.depth
        if depth is None:
            return None, None, None
        depth = np.asarray(depth)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        depth = np.ascontiguousarray(depth.astype(np.float32, copy=False))
        ts = self.get_clock().now().nanoseconds * 1e-9

        self._depth_diag_counter += 1
        if self._depth_diag_counter % 200 == 1:
            H, W = depth.shape
            center_m = float(depth[H // 2, W // 2])
            self.get_logger().info(
                f"[DEPTH DIAG PIT] center={center_m:.3f} m")

        return img, depth, ts

    def _get_cart_pose(self):
        """Read Cartographer pose from TF (map → base_link)."""
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
            self.cart_x = t.transform.translation.x
            self.cart_y = t.transform.translation.y
            q = t.transform.rotation
            self.cart_psi = np.arctan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            self.has_cart = True
        except Exception:
            pass

    # ── Main VO tick ────────────────────────────────────────────

    def _vo_tick(self):
        self._get_cart_pose()

        if self.input_source == 'pit_aligned':
            img, depth, ts = self._read_pit_frame()
            if img is None or depth is None or ts is None:
                return
        else:
            with self.state_lock:
                img = self.color_img
                stamp = self.color_stamp
                depth = self.depth_img

            if img is None or stamp is None:
                self._publish_all(0.0, 0.0, 'init', True, 0.0, 0.0, 0.0)
                return

            if depth is None:
                self._publish_all(0.0, 0.0, 'init', True, 0.0, 0.0, 0.0)
                return
            ts = stamp.sec + stamp.nanosec * 1e-9

        if not self.has_cart:
            self._publish_all(0.0, 0.0, 'init', True, 0.0, 0.0, 0.0)
            return

        # Duplicate frame detection
        if self.last_color_stamp is not None and ts == self.last_color_stamp:
            return
        self.last_color_stamp = ts

        # Anchor on first valid frame
        if not self.vo_anchored:
            with self.vo_lock:
                self.vo.reset(self.cart_x, self.cart_y, self.cart_psi)
                self.vo.update(img, depth, ts)
            self.vo_anchored = True
            self.vo_psi_shadow = self.cart_psi
            self.eval_window.clear()
            self.warmup_until = ts + self.warmup_s
            self.last_reanchor_time = ts
            self.last_reanchor_xy = np.array([self.cart_x, self.cart_y])
            return

        # Run VO with color + depth
        with self.vo_lock:
            # Inject cart yaw before update so dx/dy map projection uses
            # the trusted heading during this competition phase.
            if self.force_cart_yaw and self.vo_anchored:
                self.vo.pose[2] = self.cart_psi
            result = self.vo.update(img, depth, ts)
            if self.force_cart_yaw:
                # Keep heading pinned after update as well.
                self.vo.pose[2] = self.cart_psi
                pose = result['pose'].copy()
                pose[2] = self.cart_psi
                result['pose'] = pose

        # Store per-frame diagnostics (already negated inside engine)
        if result['delta_pose'] is not None:
            dp = result['delta_pose']
            self.last_frame_dx = dp[0]
            self.last_frame_dy = dp[1]
            self.last_frame_dpsi = dp[2]
            # Integrate the camera-only frame yaw into the shadow accumulator.
            # This stays independent of force_cart_yaw so we can compare what
            # the VO heading would look like before flipping that flag off.
            raw_dpsi = float(dp[2])
            self.vo_psi_shadow = np.arctan2(
                np.sin(self.vo_psi_shadow + raw_dpsi),
                np.cos(self.vo_psi_shadow + raw_dpsi))
        else:
            self.last_frame_dx = 0.0
            self.last_frame_dy = 0.0
            self.last_frame_dpsi = 0.0

        self.last_inliers = result['inlier_count']
        self.last_confidence = result['confidence']
        self.last_spread = result.get('inlier_spread', 0.0)

        vo_x, vo_y, vo_psi = result['pose']
        vo_valid = result['valid']
        vo_conf = result['confidence']
        inliers = result['inlier_count']
        spread = result.get('inlier_spread', 0.0)

        # Warmup guard
        now = ts
        if now < self.warmup_until:
            self._publish_all(0.0, 0.0, 'warming', True, vo_x, vo_y, vo_psi)
            return

        # Evaluation window
        self.eval_window.append((
            now, self.cart_x, self.cart_y, self.cart_psi,
            vo_x, vo_y, vo_psi,
            vo_valid, vo_conf, inliers, spread
        ))

        cutoff = now - self.window_sec
        while self.eval_window and self.eval_window[0][0] < cutoff:
            self.eval_window.popleft()

        self._evaluate(now, vo_x, vo_y, vo_psi)

    # ── Evaluation (identical logic to homography version) ──────

    def _evaluate(self, now, vo_x, vo_y, vo_psi):
        if len(self.eval_window) < 3:
            self._publish_all(0.0, 0.0, 'warming', True, vo_x, vo_y, vo_psi)
            return

        anchor = None
        for s in self.eval_window:
            if s[7]:
                anchor = s
                break
        if anchor is None:
            self._push_decision('vo_suspect', now)
            self._publish_all(0.0, 0.0, self.confirmed_state, self.healthy,
                              vo_x, vo_y, vo_psi)
            return

        latest = self.eval_window[-1]

        dx_cart = latest[1] - anchor[1]
        dy_cart = latest[2] - anchor[2]
        dx_vo = latest[4] - anchor[4]
        dy_vo = latest[5] - anchor[5]

        rx = dx_vo - dx_cart
        ry = dy_vo - dy_cart
        rho = np.sqrt(rx**2 + ry**2)

        dpsi_cart = abs(latest[3] - anchor[3])
        if dpsi_cart > np.pi:
            dpsi_cart = 2 * np.pi - dpsi_cart

        vo_valid = latest[7]
        conf = latest[8]
        inliers = latest[9]
        spread = latest[10]

        if not vo_valid or conf < 0.15 or inliers < 20:
            self._push_decision('vo_suspect', now)
            self._publish_all(rho, 0.0, self.confirmed_state, self.healthy,
                              vo_x, vo_y, vo_psi)
            return

        q_conf = np.clip(conf, 0.0, 1.0)
        q_inl = np.clip((inliers - 20) / 80.0, 0.0, 1.0)
        q_spread = np.clip(
            (spread - self.spread_bad) / (self.spread_good - self.spread_bad),
            0.0, 1.0)
        weight = (q_conf * q_inl * q_spread) ** (1.0 / 3.0)

        if weight < self.min_vo_weight:
            self._push_decision('vo_suspect', now)
            self._publish_all(rho, weight, self.confirmed_state, self.healthy,
                              vo_x, vo_y, vo_psi)
            return

        # Turn gate: depth version may handle turns better than homography,
        # but we keep the gate as a safety measure. Can be loosened later
        # once depth heading is validated.
        if dpsi_cart > self.turn_gate_rad:
            self._push_decision('vo_suspect', now)
            self._publish_all(rho, weight, self.confirmed_state, self.healthy,
                              vo_x, vo_y, vo_psi)
            return

        if rho <= self.trans_thresh:
            self._push_decision('agree', now)
        else:
            self._push_decision('odom_suspect', now)

        self._publish_all(rho, weight, self.confirmed_state, self.healthy,
                          vo_x, vo_y, vo_psi)

    def _push_decision(self, decision, now):
        if decision == 'agree':
            self.agree_count += 1
            self.odom_suspect_count = 0
            self.vo_suspect_count = 0
            if self.agree_count >= self.n_consecutive:
                self.confirmed_state = 'agree'
                self.healthy = True
                self._try_reanchor(now)
        elif decision == 'odom_suspect':
            self.odom_suspect_count += 1
            self.agree_count = 0
            self.vo_suspect_count = 0
            if self.odom_suspect_count >= self.n_consecutive:
                self.confirmed_state = 'odom_suspect'
                self.healthy = False
        elif decision == 'vo_suspect':
            self.vo_suspect_count += 1
            self.agree_count = 0
            self.odom_suspect_count = 0
            if self.vo_suspect_count >= self.n_consecutive:
                self.confirmed_state = 'vo_suspect'
                self.healthy = True

    def _try_reanchor(self, now):
        if now - self.last_reanchor_time < self.reanchor_cool:
            return
        dist = np.sqrt((self.cart_x - self.last_reanchor_xy[0])**2 +
                        (self.cart_y - self.last_reanchor_xy[1])**2)
        if dist < self.reanchor_min_dist:
            return
        with self.vo_lock:
            self.vo.soft_reset(self.cart_x, self.cart_y, self.cart_psi)
        self.vo_psi_shadow = self.cart_psi
        self.eval_window.clear()
        self.agree_count = 0
        self.warmup_until = now + self.warmup_s
        self.last_reanchor_time = now
        self.last_reanchor_xy = np.array([self.cart_x, self.cart_y])

    # ── Publishing ──────────────────────────────────────────────

    def _publish_all(self, rho, weight, state, healthy,
                     vo_x=0.0, vo_y=0.0, vo_psi=0.0):
        STATE_MAP = {'init': 0, 'warming': 1, 'agree': 2,
                     'vo_suspect': 3, 'odom_suspect': 4}

        msg = String()
        msg.data = (
            f"{state} rho={rho:.3f} w={weight:.2f} | "
            f"vo({vo_x:.2f},{vo_y:.2f},{np.degrees(vo_psi):.0f}) "
            f"ct({self.cart_x:.2f},{self.cart_y:.2f},"
            f"{np.degrees(self.cart_psi):.0f}) | "
            f"dx={self.last_frame_dx:.4f} dy={self.last_frame_dy:.4f} "
            f"dpsi={np.degrees(self.last_frame_dpsi):.1f} "
            f"inl={self.last_inliers} sp={self.last_spread:.0f} "
            f"psi_raw={np.degrees(self.vo_psi_shadow):.0f}"
        )
        self.pub_fault.publish(msg)

        def _f(val):
            m = Float64(); m.data = float(val); return m

        d = Float64(); d.data = float(rho)
        self.pub_delta.publish(d)
        w = Float64(); w.data = float(weight)
        self.pub_weight.publish(w)
        h = Bool(); h.data = bool(healthy)
        self.pub_healthy.publish(h)
        s = Int32(); s.data = STATE_MAP.get(state, 0)
        self.pub_state_id.publish(s)

        self.pub_vo_x.publish(_f(vo_x))
        self.pub_vo_y.publish(_f(vo_y))
        self.pub_vo_psi.publish(_f(vo_psi))
        self.pub_cart_x.publish(_f(self.cart_x))
        self.pub_cart_y.publish(_f(self.cart_y))
        self.pub_cart_psi.publish(_f(self.cart_psi))

        self.pub_frame_dx.publish(_f(self.last_frame_dx))
        self.pub_frame_dy.publish(_f(self.last_frame_dy))
        self.pub_frame_dpsi.publish(_f(self.last_frame_dpsi))
        i = Int32(); i.data = int(self.last_inliers)
        self.pub_inliers.publish(i)
        self.pub_conf.publish(_f(self.last_confidence))
        self.pub_spread.publish(_f(self.last_spread))
        self.pub_vo_psi_shadow.publish(_f(self.vo_psi_shadow))

    def destroy_node(self):
        if self._pit_camera is not None:
            try:
                self._pit_camera.terminate()
            except Exception:
                pass
            self._pit_camera = None
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VONodeDepth()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, Exception):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
