"""VO redundancy monitor built on depth-based visual odometry.

This node subscribes to `/camera/color_image` and `/camera/depth_image`,
runs the depth-based VO engine, and compares the resulting pose against the
current Cartographer `map -> base_link` transform.

It expects raw MONO16 depth from `rgbd.cpp`. The scale and alignment behavior
come from the calibration constants configured in `autonomy.visual_odometry`.

Published topics:
    /vo/fault_status   — Full diagnostic string
    /vo/delta_trans     — Residual rho (Float64)
    /vo/vo_weight       — Trust weight 0-1 (Float64)
    /vo/healthy         — Health flag (Bool)
    /vo/state_id        — Numeric state (Int32)
    /vo/vo_x,y,psi      — VO pose in map frame (Float64 each)
    /vo/cart_x,y,psi    — Cartographer pose (Float64 each)
    /vo/frame_dx,dy,dpsi — Per-frame delta (Float64 each)
    /vo/inliers         — Inlier count (Int32)
    /vo/confidence      — Confidence ratio 0-1 (Float64)
    /vo/spread          — Inlier pixel spread (Float64)
"""

import threading
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import (
    MutuallyExclusiveCallbackGroup,
    ReentrantCallbackGroup,
)
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float64, Bool, Int32
from cv_bridge import CvBridge
import tf2_ros

# Import the depth-based VO engine
from autonomy.visual_odometry import VisualOdometryDepth


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
        #   auto: virtual=True, physical=True (projective for physical mode)
        #   on:   always warp
        #   off:  never warp
        self.declare_parameter('alignment_mode', 'auto')

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

        # Read params
        n_features = self.get_parameter('n_features').value
        match_ratio = self.get_parameter('match_ratio').value
        ransac_iter = self.get_parameter('ransac_iterations').value
        ransac_thresh = self.get_parameter('ransac_threshold').value
        min_inliers = self.get_parameter('min_inliers').value
        negate = self.get_parameter('negate_deltas').value
        max_trans = self.get_parameter('max_translation').value
        max_rot = self.get_parameter('max_rotation_deg').value
        self.force_cart_yaw = bool(self.get_parameter('force_cart_yaw').value)
        camera_mode = self.get_parameter('camera_mode').value
        if camera_mode not in ('virtual', 'physical'):
            self.get_logger().warn(
                f"Invalid camera_mode '{camera_mode}', defaulting to 'virtual'")
            camera_mode = 'virtual'

        depth_scale_param = float(self.get_parameter('depth_scale').value)
        depth_scale = depth_scale_param if depth_scale_param > 0.0 else None

        alignment_mode = str(self.get_parameter('alignment_mode').value).strip().lower()
        if alignment_mode == 'auto':
            use_align = True
        elif alignment_mode in ('on', 'true', '1', 'yes'):
            use_align = True
        elif alignment_mode in ('off', 'false', '0', 'no'):
            use_align = False
        else:
            self.get_logger().warn(
                f"Invalid alignment_mode '{alignment_mode}', defaulting to 'auto'")
            alignment_mode = 'auto'
            use_align = (camera_mode == 'virtual')

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
            camera_mode=camera_mode,
            n_features=n_features,
            match_ratio=match_ratio,
            ransac_threshold=ransac_thresh,
            ransac_iterations=ransac_iter,
            min_inliers=min_inliers,
            max_translation=max_trans,
            max_rotation_deg=max_rot,
            negate_deltas=negate,
        )
        self.vo_lock = threading.Lock()

        if camera_mode == 'physical' and depth_scale is None:
            self.get_logger().warn(
                "Physical mode is using the placeholder MONO16 divisor from "
                "visual_odometry.py until rgbd.cpp publishes metric depth or "
                "a measured physical depth_scale is provided.")
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

        # Depth diagnostics (for verifying depth_scale)
        self._depth_diag_counter = 0

        # ── TF ──────────────────────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Subscribers ─────────────────────────────────────────
        cam_cb_group = ReentrantCallbackGroup()
        self.color_sub = self.create_subscription(
            Image, '/camera/color_image', self._color_cb, 5,
            callback_group=cam_cb_group)
        # Depth from rgbd.cpp: MONO16 encoding, uint16 per pixel
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth_image', self._depth_cb, 5,
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

        # ── Startup banner ──────────────────────────────────────
        k = self.vo.projector.K_rgb
        mode_str = f"{camera_mode.upper()} (FAQ intrinsics)" if camera_mode == 'virtual' else f"{camera_mode.upper()} (calibrated)"
        self.get_logger().info(
            f"\n╔══════════════════════════════════════════════╗\n"
            f"║  VO REDUNDANCY MONITOR — DEPTH-BASED         ║\n"
            f"╠══════════════════════════════════════════════╣\n"
            f"║  Camera mode: {mode_str:<21}║\n"
            f"║  RGB K: fx={k[0,0]:.1f} fy={k[1,1]:.1f}"
            f" cx={k[0,2]:.1f} cy={k[1,2]:.1f}\n"
            f"║  Depth scale: {self.vo.projector.depth_scale:.0f}"
            f" ({self.vo.projector.depth_units_label})\n"
            f"║  Alignment: {'ON' if self.vo.projector.use_alignment else 'OFF':<21}║\n"
            f"║  negate_deltas: {negate!s:<16}║\n"
            f"║  features={n_features}  match_ratio={match_ratio}\n"
            f"╚══════════════════════════════════════════════╝")

    # ── Callbacks ───────────────────────────────────────────────

    def _color_cb(self, msg):
        with self.state_lock:
            self.color_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.color_stamp = msg.header.stamp

    def _depth_cb(self, msg):
        """Decode MONO16 depth image from rgbd.cpp.

        rgbd.cpp publishes depth as:
            - CV_16UC1 (16-bit unsigned, 1 channel)
            - ROS encoding: sensor_msgs::image_encodings::MONO16
            - Each pixel is a uint16 raw value
            - Convert to active mode units: raw / depth_scale

        The PAL library (vision.py) uses UINT8 instead, with a different
        representation. The two paths are not interchangeable.
        """
        with self.state_lock:
            # Decode as uint16 — passthrough preserves original encoding
            try:
                depth_16 = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
                # Ensure we have a 2D array (H, W) not (H, W, 1)
                if len(depth_16.shape) == 3:
                    depth_16 = depth_16[:, :, 0]
                self.depth_img = depth_16.astype(np.uint16)
            except Exception as e:
                self.get_logger().warn(
                    f"Depth decode error: {e}", throttle_duration_sec=5.0)
                return

        # Periodic depth diagnostic: print center pixel raw value
        # so user can verify depth_scale against known distance.
        self._depth_diag_counter += 1
        if self._depth_diag_counter % 200 == 1:  # every ~8 seconds at 25Hz
            H, W = self.depth_img.shape
            raw_center = int(self.depth_img[H // 2, W // 2])
            projector = self.vo.projector
            scale = projector.depth_scale
            depth_val = raw_center / scale if scale > 0 else 0.0
            physical_m = depth_val * projector.physical_m_per_unit
            self.get_logger().info(
                f"[DEPTH DIAG] center raw={raw_center}, "
                f"/ {scale:.0f} = {depth_val:.3f} {projector.depth_units_label} "
                f"({physical_m:.4f} m physical)")

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

        if not self.has_cart:
            self._publish_all(0.0, 0.0, 'init', True, 0.0, 0.0, 0.0)
            return

        # Duplicate frame detection
        ts = stamp.sec + stamp.nanosec * 1e-9
        if self.last_color_stamp is not None and ts == self.last_color_stamp:
            return
        self.last_color_stamp = ts

        # Anchor on first valid frame
        if not self.vo_anchored:
            with self.vo_lock:
                self.vo.reset(self.cart_x, self.cart_y, self.cart_psi)
                self.vo.update(img, depth, ts)
            self.vo_anchored = True
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
            f"inl={self.last_inliers} sp={self.last_spread:.0f}"
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
