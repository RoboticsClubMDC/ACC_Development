# #!/usr/bin/env python3
# """vo_node.py - Standalone VO with redundancy v5 (Part 2).

# v5 CHANGES (Part 2 — weighted redundancy):
#   1. Vector residual: compares displacement DIRECTION (Δx,Δy), not
#      just distance traveled. Catches "same distance, wrong direction"
#      failures that scalar comparison misses.
#   2. VO trust weight: combines confidence, inlier count, and inlier
#      spread into a single 0..1 score via geometric mean. If ANY
#      component collapses (e.g. spread drops during turns), the whole
#      weight drops — preventing VO from blaming odom when VO itself
#      is geometrically degenerate.
#   3. Improved fault isolation: even if VO passes basic quality gates
#      (conf ≥ threshold, inliers ≥ threshold), a low trust weight
#      (degenerate geometry) triggers vo_suspect instead of odom_suspect.
#      Only when VO is truly strong (high weight) AND residual is large
#      do we blame odom.
#   4. Window now stores inlier_spread for future analysis.
#   5. Publishes vo_weight in fault_status for dashboard visibility.

# v4.1 fixes preserved:
#   1. Stamp-based duplicate frame gating
#   2. SCALE mode depth defaults (shift=0, scale=15.7)
#   3. Color/depth timestamp sync guard
#   4. inlier_spread field from visual_odometry.py
# v4 fixes preserved:
#   1. MultiThreadedExecutor (3 threads)
#   2. ReentrantCallbackGroup for subs
#   3. Camera stale timeout, Queue depth 10
# v3 fixes preserved:
#   A. Clear window on re-anchor
#   B. Yaw diagnostic-only (translation-only decisions)
#   C. TF dropout tolerance
#   D. Increased re-anchor guards (min_dist=0.25, cooldown=8.0)
#   E. negate_deltas toggle

# SETUP.PY:  'vo_node=autonomy.vo_node:main'
# """

# import time
# import math
# import threading
# from collections import deque

# import numpy as np
# import rclpy
# from rclpy.node import Node
# from rclpy.executors import MultiThreadedExecutor
# from rclpy.callback_groups import (
#     MutuallyExclusiveCallbackGroup,
#     ReentrantCallbackGroup,
# )
# from sensor_msgs.msg import Image
# from std_msgs.msg import String, Float32, UInt8, Bool
# from cv_bridge import CvBridge
# from scipy.spatial.transform import Rotation as R

# from tf2_ros import TransformException
# from tf2_ros.buffer import Buffer
# from tf2_ros.transform_listener import TransformListener

# from autonomy.visual_odometry_old import VisualOdometry


# def _wrap(a):
#     return math.atan2(math.sin(a), math.cos(a))


# class VONode(Node):

#     def __init__(self):
#         super().__init__('vo_node')

#         # ── Parameters ──
#         # v4.1 FIX #2: defaults changed to SCALE mode (Part 1 conclusion)
#         #   WAS: depth_scale=5.5, depth_shift_bits=8, depth_unit_m=0.0163
#         #   NOW: depth_scale=15.7 → divisor=15700, shift_bits=0
#         #   This means launching with NO -p overrides now gives you the
#         #   correct SCALE decode, instead of the deprecated HIGH_BYTE mode.
#         self.declare_parameter('depth_scale', 15.7)
#         self.declare_parameter('depth_shift_bits', 0)
#         self.declare_parameter('depth_unit_m', 0.001)
#         self.declare_parameter('vo_rate_hz', 20.0)
#         self.declare_parameter('target_frame', 'base_link')

#         # VO estimator tuning (A/B testable from command line)
#         self.declare_parameter('n_features', 800)
#         self.declare_parameter('match_ratio', 0.75)
#         self.declare_parameter('depth_ratio_max', 3.0)

#         self.declare_parameter('window_seconds', 2.0)
#         self.declare_parameter('min_window_samples', 15)
#         self.declare_parameter('n_consecutive', 3)

#         self.declare_parameter('min_vo_conf', 0.15)
#         # v5.1: default changed from 40 to 20 (QLabs baseline —
#         # typical valid frames produce 20-35 inliers with 800 features)
#         self.declare_parameter('min_vo_inliers', 20)

#         # v5.1: default changed from 0.25 to 0.35 (QLabs baseline —
#         # vector residual over 2s window needs more headroom than
#         # scalar distance comparison)
#         self.declare_parameter('trans_thresh_m', 0.35)
#         self.declare_parameter('yaw_thresh_deg', 15.0)

#         self.declare_parameter('reanchor_on_agree', True)
#         self.declare_parameter('min_reanchor_dist_m', 0.25)
#         self.declare_parameter('reanchor_cooldown_s', 8.0)
#         self.declare_parameter('warmup_after_reanchor_s', 1.0)

#         self.declare_parameter('tf_stale_timeout_s', 0.5)
#         self.declare_parameter('camera_stale_timeout_s', 1.0)

#         # v5.1: default changed from False to True (QLabs baseline —
#         # SVD Procrustes computes feature motion which is inverse of
#         # vehicle motion, so sign flip is needed)
#         self.declare_parameter('negate_deltas', True)

#         # Camera intrinsic overrides for calibration.
#         # Default fx=483.671 fy=483.579 from D435 spec sheet.
#         # If QLabs virtual camera has different FOV, these need to
#         # be adjusted. Smaller fx/fy = wider FOV = larger 3D motion.
#         # Use vo_scale_test=1.0 and sweep fx_override to find the
#         # value where alpha ≈ 1.0 on straight segments.
#         self.declare_parameter('fx_override', 0.0)  # 0 = use default
#         self.declare_parameter('fy_override', 0.0)  # 0 = use default

#         # TEMPORARY: scale test parameter for diagnosing VO translation
#         # compression. Set to 1.0 for normal operation. Set to ~3.0 to
#         # test whether VO is systematically under-measuring translation.
#         # Scales VO displacement relative to re-anchor point, so anchor
#         # alignment stays correct.
#         self.declare_parameter('vo_scale_test', 1.0)

#         # v4.1 FIX #3: max allowed timestamp skew between color & depth
#         self.declare_parameter('max_sync_skew_s', 0.05)

#         # v5 Part 2: minimum VO trust weight before classifying as
#         # degenerate. Below this, VO is vo_suspect even if conf/inliers
#         # pass the basic gate. Prevents blaming odom during turns when
#         # VO features are geometrically clustered.
#         self.declare_parameter('min_vo_weight', 0.3)

#         # Spread thresholds for VO weight calculation.
#         # SPREAD_BAD: below this, q_spread=0 → weight collapses.
#         # SPREAD_GOOD: above this, q_spread=1 → full trust.
#         # Raised from 40→55 after fx=161 calibration revealed that
#         # spread 43-53 (turn territory) was passing the weight gate
#         # and causing false odom_suspect with rho up to 1.8m.
#         self.declare_parameter('spread_bad', 55.0)
#         self.declare_parameter('spread_good', 120.0)

#         # Turn gate: if Cartographer yaw change over the evaluation
#         # window exceeds this threshold, force vo_suspect regardless
#         # of VO quality signals. VO translation estimates are
#         # geometrically unreliable during turns even when feature
#         # count and spread look healthy. With fx=161, directionally
#         # wrong turn translations accumulate into meters of VO pose
#         # drift that triggers false odom_suspect.
#         self.declare_parameter('turn_gate_deg', 8.0)

#         # ── Read params ──
#         depth_scale = float(self.get_parameter('depth_scale').value)
#         depth_shift = int(self.get_parameter('depth_shift_bits').value)
#         depth_unit = float(self.get_parameter('depth_unit_m').value)
#         vo_rate = float(self.get_parameter('vo_rate_hz').value)
#         negate = bool(self.get_parameter('negate_deltas').value)
#         n_features = int(self.get_parameter('n_features').value)
#         match_ratio = float(self.get_parameter('match_ratio').value)
#         depth_ratio_max = float(self.get_parameter(
#             'depth_ratio_max').value)

#         self._target_frame = self.get_parameter(
#             'target_frame').get_parameter_value().string_value
#         self._win_sec = float(self.get_parameter('window_seconds').value)
#         self._min_samples = int(self.get_parameter(
#             'min_window_samples').value)
#         self._n_consec = int(self.get_parameter('n_consecutive').value)
#         self._min_conf = float(self.get_parameter('min_vo_conf').value)
#         self._min_inl = int(self.get_parameter('min_vo_inliers').value)
#         self._trans_thresh = float(self.get_parameter(
#             'trans_thresh_m').value)
#         self._yaw_thresh = float(self.get_parameter(
#             'yaw_thresh_deg').value)
#         self._do_reanchor = bool(self.get_parameter(
#             'reanchor_on_agree').value)
#         self._min_ra_dist = float(self.get_parameter(
#             'min_reanchor_dist_m').value)
#         self._ra_cooldown = float(self.get_parameter(
#             'reanchor_cooldown_s').value)
#         self._warmup_s = float(self.get_parameter(
#             'warmup_after_reanchor_s').value)
#         self._tf_stale_timeout = float(self.get_parameter(
#             'tf_stale_timeout_s').value)
#         self._camera_stale_timeout = float(self.get_parameter(
#             'camera_stale_timeout_s').value)
#         self._max_sync_skew = float(self.get_parameter(
#             'max_sync_skew_s').value)
#         self._min_vo_weight = float(self.get_parameter(
#             'min_vo_weight').value)
#         self._spread_bad = float(self.get_parameter(
#             'spread_bad').value)
#         self._spread_good = float(self.get_parameter(
#             'spread_good').value)
#         self._turn_gate_rad = math.radians(float(
#             self.get_parameter('turn_gate_deg').value))
#         self._vo_scale_test = float(self.get_parameter(
#             'vo_scale_test').value)

#         # ── TF ──
#         self.tf_buffer = Buffer()
#         self.tf_listener = TransformListener(self.tf_buffer, self)

#         # ── VO engine ──
#         fx_over = float(self.get_parameter('fx_override').value)
#         fy_over = float(self.get_parameter('fy_override').value)
#         self.vo = VisualOdometry(
#             img_width=640, img_height=480,
#             use_depth=True, n_features=n_features,
#             match_ratio=match_ratio, ransac_threshold=0.05,
#             depth_scale=depth_scale,
#             depth_shift_bits=depth_shift,
#             depth_unit_m=depth_unit,
#             depth_ratio_max=depth_ratio_max,
#             negate_deltas=negate,
#             fx_override=fx_over if fx_over > 0 else None,
#             fy_override=fy_over if fy_over > 0 else None)

#         self.bridge = CvBridge()

#         # v4: separate callback groups
#         # Subs: reentrant so image callbacks always fire
#         # VO timer: mutually exclusive so two _vo_tick never overlap
#         #   (overlapping ticks cause race on _prev_kp → IndexError)
#         self._sub_cb_group = ReentrantCallbackGroup()
#         self._vo_cb_group = MutuallyExclusiveCallbackGroup()

#         # v4: lock to protect shared state between threads
#         self._lock = threading.Lock()
#         # v4: dedicated lock for VO engine (update + soft_reset)
#         self._vo_lock = threading.Lock()

#         # ── State ──
#         self._color = None
#         self._depth = None
#         self._color_last_time = 0.0
#         self._depth_last_time = 0.0
#         self._have_color = False       # once True, never False
#         self._last_good_color = None   # cached last valid frame
#         self._last_good_depth = None
#         self._cart_x = 0.0
#         self._cart_y = 0.0
#         self._cart_psi = 0.0
#         self._tf_ok = False
#         self._tf_last_good_time = 0.0

#         # v4.1 FIX #1: image header timestamps for duplicate detection
#         self._color_stamp = None          # float seconds from msg header
#         self._depth_stamp = None          # float seconds from msg header
#         self._last_processed_stamp = None # last color stamp VO ran on
#         self._skipped_dup_count = 0       # diagnostic counter
#         # v4.1 FIX #3: sync guard diagnostic counter
#         self._depth_sync_skip_count = 0

#         self._vo_pose = np.zeros(3)
#         self._vo_anchored = False
#         self._vo_dist = 0.0

#         self._window = deque(maxlen=2000)
#         self._c_agree = 0
#         self._c_odom_sus = 0
#         self._c_vo_sus = 0

#         self._flag = 'init'
#         self._healthy = True
#         self._dt_trans = 0.0
#         self._dt_yaw = 0.0
#         self._decision_msg = 'warming up'
#         self._reanchor_count = 0

#         self._last_reanchor_time = 0.0
#         self._last_reanchor_cart = np.zeros(3)
#         self._warmup_until = 0.0

#         self._valid_frames = 0
#         self._reject_frames = 0
#         self._no_camera_count = 0
#         self._stale_camera_count = 0

#         # v4.1 FIX #4: inlier spread (forward-compatible placeholder).
#         # Will be populated by visual_odometry.py when it computes the
#         # bounding-box or std-dev of inlier pixel locations. Until then
#         # it reads 0.0 from the result dict. This becomes the key Part 2
#         # weighting signal: high inliers + low spread = turn degeneracy.
#         self._last_inlier_spread = 0.0

#         # v5 Part 2: VO trust weight (0..1), published for dashboard
#         self._last_vo_weight = 0.0

#         # ── ROS: subs + timers use callback group ──
#         # v4: subs use reentrant (always receive frames)
#         self.create_subscription(
#             Image, '/camera/color_image', self._color_cb, 10,
#             callback_group=self._sub_cb_group)
#         self.create_subscription(
#             Image, '/camera/depth_image', self._depth_cb, 10,
#             callback_group=self._sub_cb_group)
#         self._pub = self.create_publisher(
#             String, '/vo/fault_status', 1)
#         # v5.2: Numeric topics for rqt_plot / supervisor node
#         # These publish the same data as fault_status string but as
#         # plottable numeric values for demo visualization.
#         self._pub_rho = self.create_publisher(
#             Float32, '/vo/delta_trans', 1)
#         self._pub_weight = self.create_publisher(
#             Float32, '/vo/vo_weight', 1)
#         self._pub_state = self.create_publisher(
#             UInt8, '/vo/state_id', 1)
#         self._pub_healthy = self.create_publisher(
#             Bool, '/vo/healthy', 1)
#         # State ID encoding:
#         #   0 = init, 1 = warming, 2 = vo_suspect,
#         #   3 = agree, 4 = odom_suspect
#         self._STATE_IDS = {
#             'init': 0, 'warming': 1, 'vo_suspect': 2,
#             'agree': 3, 'odom_suspect': 4,
#         }
#         self._tf_tmr = self.create_timer(
#             0.025, self._tf_tick,
#             callback_group=self._sub_cb_group)
#         # v4: VO timer uses exclusive group — prevents two
#         # _vo_tick from overlapping (race on _prev_kp)
#         self._vo_tmr = self.create_timer(
#             1.0 / max(vo_rate, 1.0), self._vo_tick,
#             callback_group=self._vo_cb_group)

#         self.get_logger().info('=' * 60)
#         self.get_logger().info(
#             'VO NODE v5 (Part 2: vector residual + weighted trust)')
#         self.get_logger().info(
#             '  depth: shift=%d unit=%.4f scale=%.1f  -> %s mode'
#             % (depth_shift, depth_unit, depth_scale,
#                'SCALE' if depth_shift == 0 else 'HIGH_BYTE'))
#         self.get_logger().info(
#             '  window=%.1fs  trans_thresh=%.2fm  '
#             'yaw_thresh=%.1fdeg(diag-only)  consec=%d'
#             % (self._win_sec, self._trans_thresh,
#                self._yaw_thresh, self._n_consec))
#         self.get_logger().info(
#             '  re-anchor: on=%s  min_dist=%.2fm  '
#             'cooldown=%.1fs  warmup=%.1fs'
#             % (self._do_reanchor, self._min_ra_dist,
#                self._ra_cooldown, self._warmup_s))
#         self.get_logger().info(
#             '  tf_stale=%.2fs  cam_stale=%.2fs  '
#             'negate_deltas=%s  sync_skew=%.3fs'
#             % (self._tf_stale_timeout,
#                self._camera_stale_timeout, negate,
#                self._max_sync_skew))
#         self.get_logger().info(
#             '  vo_tuning: features=%d  match_ratio=%.2f  '
#             'depth_ratio_max=%.1f'
#             % (n_features, match_ratio, depth_ratio_max))
#         self.get_logger().info(
#             '  Part2: min_vo_weight=%.2f  '
#             'min_conf=%.2f  min_inliers=%d  '
#             'spread_bad=%.0f  spread_good=%.0f  '
#             'turn_gate=%.1fdeg'
#             % (self._min_vo_weight, self._min_conf, self._min_inl,
#                self._spread_bad, self._spread_good,
#                math.degrees(self._turn_gate_rad)))
#         K = self.vo.projector.K
#         self.get_logger().info(
#             '  intrinsics: fx=%.1f  fy=%.1f  cx=%.1f  cy=%.1f'
#             % (K[0,0], K[1,1], K[0,2], K[1,2]))
#         if self._vo_scale_test != 1.0:
#             self.get_logger().warn(
#                 '  *** SCALE TEST ACTIVE: vo_scale_test=%.2f ***'
#                 % self._vo_scale_test)
#         self.get_logger().info('=' * 60)

#     # ══════════════════════════════════════════════════════
#     # CALLBACKS (run on separate threads via ReentrantCBG)
#     # ══════════════════════════════════════════════════════

#     def _color_cb(self, msg):
#         try:
#             img = self.bridge.imgmsg_to_cv2(
#                 msg, desired_encoding='bgr8')
#             # v4.1: extract header timestamp for duplicate detection
#             stamp = (msg.header.stamp.sec
#                      + 1e-9 * msg.header.stamp.nanosec)
#             with self._lock:
#                 self._color = img
#                 self._last_good_color = img
#                 self._color_last_time = time.time()
#                 self._color_stamp = stamp
#                 self._have_color = True
#                 # One-shot: log actual image dimensions on first frame
#                 if not hasattr(self, '_logged_shape'):
#                     self._logged_shape = True
#                     self.get_logger().info(
#                         'color image shape=%s dtype=%s'
#                         % (img.shape, img.dtype))
#         except Exception:
#             pass

#     def _depth_cb(self, msg):
#         try:
#             img = self.bridge.imgmsg_to_cv2(
#                 msg, desired_encoding='passthrough')
#             # v4.1: extract header timestamp for sync guard
#             stamp = (msg.header.stamp.sec
#                      + 1e-9 * msg.header.stamp.nanosec)
#             with self._lock:
#                 self._depth = img
#                 self._last_good_depth = img
#                 self._depth_last_time = time.time()
#                 self._depth_stamp = stamp
#         except Exception:
#             pass

#     def _tf_tick(self):
#         try:
#             t = self.tf_buffer.lookup_transform(
#                 'map', self._target_frame, rclpy.time.Time())
#             with self._lock:
#                 self._cart_x = float(t.transform.translation.x)
#                 self._cart_y = float(t.transform.translation.y)
#                 q = [t.transform.rotation.x,
#                      t.transform.rotation.y,
#                      t.transform.rotation.z,
#                      t.transform.rotation.w]
#                 _, _, self._cart_psi = R.from_quat(q).as_euler(
#                     'xyz')
#                 self._tf_ok = True
#                 self._tf_last_good_time = time.time()
#         except TransformException:
#             with self._lock:
#                 if (time.time() - self._tf_last_good_time
#                         < self._tf_stale_timeout):
#                     pass  # keep last TF
#                 else:
#                     self._tf_ok = False

#     # ══════════════════════════════════════════════════════
#     # MAIN VO TICK
#     # ══════════════════════════════════════════════════════

#     def _vo_tick(self):
#         now = time.time()

#         # v4: snapshot ONLY last-good frames (never _color directly)
#         with self._lock:
#             have_color = self._have_color
#             color = self._last_good_color
#             depth = self._last_good_depth
#             color_age = (now - self._color_last_time
#                          if self._color_last_time > 0 else 999.0)
#             # v4.1: also snapshot stamps for duplicate + sync checks
#             color_stamp = self._color_stamp
#             depth_stamp = self._depth_stamp
#             tf_ok = self._tf_ok
#             cart_x = self._cart_x
#             cart_y = self._cart_y
#             cart_psi = self._cart_psi

#         # If we have NEVER received a frame, this is real startup
#         if not have_color:
#             self._no_camera_count += 1
#             self._flag = 'init'
#             self._healthy = True
#             self._dt_trans = 0.0
#             self._dt_yaw = 0.0
#             self._publish_status(reason='no_camera_yet')
#             return

#         # After streaming started, never publish "no_camera_yet".
#         # If we somehow still have no frame, just skip silently.
#         if color is None:
#             return

#         # Stale check (camera genuinely stopped publishing)
#         if color_age > self._camera_stale_timeout:
#             self._stale_camera_count += 1
#             self._flag = 'vo_suspect'
#             self._healthy = True
#             self._dt_trans = 0.0
#             self._dt_yaw = 0.0
#             self._publish_status(
#                 reason='camera_stale_%.2fs' % color_age)
#             return

#         # ── v4.1 FIX #1: Duplicate frame gating ──
#         # The VO timer fires at 20Hz but the camera publishes at ~15Hz.
#         # Without this check, ~25-33% of ticks reprocess the SAME image,
#         # producing dx=dy=0 with high inliers. That's not just wasted
#         # compute — it creates false "VO says no motion" windows that
#         # can bias the redundancy comparator into thinking odom is wrong
#         # when in reality VO just didn't have a new frame to process.
#         # Fix: compare the image header stamp to what we last processed.
#         # If identical, this is a duplicate — skip silently (no publish,
#         # no state change, no window entry).
#         if color_stamp is not None:
#             if self._last_processed_stamp == color_stamp:
#                 self._skipped_dup_count += 1
#                 return
#         self._last_processed_stamp = color_stamp

#         if not tf_ok:
#             self._flag = 'vo_suspect'
#             self._healthy = True
#             self._dt_trans = 0.0
#             self._dt_yaw = 0.0
#             self._publish_status(reason='no_tf')
#             return

#         # ── v4.1 FIX #3: Color/depth sync guard ──
#         # If the depth frame timestamp is too far from the color frame
#         # timestamp, the 3D back-projection uses depth from a different
#         # camera viewpoint than the features were detected in. This
#         # causes wrong 3D points and biased rigid fits.
#         # Fix: if skew > threshold, pass depth=None so VO falls back to
#         # ground-plane projection (which already exists in the pipeline).
#         use_depth = depth
#         if (depth is not None
#                 and color_stamp is not None
#                 and depth_stamp is not None):
#             if abs(color_stamp - depth_stamp) > self._max_sync_skew:
#                 use_depth = None
#                 self._depth_sync_skip_count += 1

#         # ── Run VO (under dedicated lock) ──
#         # v4.1: use image timestamp (not wall clock) for VO dt calc.
#         # This gives consistent dt tied to actual frame timing, even
#         # if the timer callback fires slightly late.
#         vo_timestamp = color_stamp if color_stamp else now
#         with self._vo_lock:
#             # Feed cartographer yaw into VO so translation
#             # rotation uses correct heading (VO yaw unreliable)
#             self.vo.pose[2] = cart_psi
#             res = self.vo.update(color, vo_timestamp, use_depth)

#         vo_valid = bool(res.get('valid', False))
#         vo_conf = float(res.get('confidence', 0.0))
#         inliers = int(res.get('inlier_count', 0))
#         reason = res.get('rejected_reason', '') or 'ok'

#         # v4.1 FIX #4: read inlier spread from VO result dict.
#         # Currently visual_odometry.py doesn't compute this yet, so
#         # it returns 0.0. When we add the computation (Part 2), it
#         # flows through automatically. The metric measures geometric
#         # degeneracy: high inlier count + low spread = features are
#         # clustered in one image region (classic turn problem).
#         self._last_inlier_spread = float(
#             res.get('inlier_spread', 0.0))

#         if vo_valid:
#             self._valid_frames += 1
#         elif reason != 'init':
#             self._reject_frames += 1

#         # Initial anchor (hard reset, only once)
#         if not self._vo_anchored:
#             if abs(cart_x) > 1e-3 or abs(cart_y) > 1e-3:
#                 with self._vo_lock:
#                     self.vo.reset(cart_x, cart_y, cart_psi)
#                 self._vo_pose = np.array([
#                     cart_x, cart_y, cart_psi])
#                 self._vo_anchored = True
#                 self._last_reanchor_time = now
#                 self._last_reanchor_cart = np.array([
#                     cart_x, cart_y, cart_psi])
#                 self._warmup_until = now + self._warmup_s
#                 # v5.1: clear any pre-anchor samples/counters so the
#                 # evaluator starts clean (same as re-anchor cleanup)
#                 self._window.clear()
#                 self._c_agree = 0
#                 self._c_odom_sus = 0
#                 self._c_vo_sus = 0
#                 self._dt_trans = 0.0
#                 self._dt_yaw = 0.0
#                 self._last_vo_weight = 0.0
#                 self.get_logger().info(
#                     'VO anchored at cart=(%.3f, %.3f, %.1fdeg)'
#                     % (cart_x, cart_y,
#                        math.degrees(cart_psi)))

#         if self._vo_anchored and vo_valid:
#             raw_pose = res['pose'].copy()
#             raw_pose[2] = cart_psi  # keep yaw from cartographer

#             # TEMPORARY SCALE TEST: scale VO translation relative to
#             # the last re-anchor point. At re-anchor, VO == Cart, so
#             # scaling the displacement from that point preserves the
#             # alignment while stretching the VO translation by the
#             # test factor. If vo_scale_test=3.0 collapses rho, that
#             # confirms systematic VO translation compression.
#             if self._vo_scale_test != 1.0:
#                 anchor_xy = self._last_reanchor_cart[:2]
#                 disp = raw_pose[:2] - anchor_xy
#                 raw_pose[:2] = anchor_xy + disp * self._vo_scale_test

#             self._vo_pose = raw_pose

#         dp = res.get('delta_pose')
#         dx = float(dp[0]) if dp is not None else 0.0
#         dy = float(dp[1]) if dp is not None else 0.0
#         dpsi = float(dp[2]) if dp is not None else 0.0

#         d = math.sqrt(dx**2 + dy**2)
#         if d > 1e-7:
#             self._vo_dist += d

#         vel = res.get('velocity', np.zeros(2))
#         vo_speed = float(np.linalg.norm(vel))

#         self._window.append((
#             now,
#             cart_x, cart_y, cart_psi,
#             float(self._vo_pose[0]),
#             float(self._vo_pose[1]),
#             float(self._vo_pose[2]),
#             vo_valid, vo_conf, inliers,
#             self._last_inlier_spread))  # v5: 11th element

#         cutoff = now - self._win_sec * 3.0
#         while self._window and self._window[0][0] < cutoff:
#             self._window.popleft()

#         self._evaluate(
#             now, vo_valid, vo_conf, inliers, reason,
#             cart_x, cart_y, cart_psi)
#         self._publish_full(
#             vo_valid, vo_conf, inliers, reason,
#             dx, dy, dpsi, vo_speed,
#             cart_x, cart_y, cart_psi)

#     # ══════════════════════════════════════════════════════
#     # VO TRUST WEIGHT (Part 2)
#     # ══════════════════════════════════════════════════════

#     def _vo_weight(self, vo_conf, inliers, spread):
#         """Compute a 0..1 trust weight for VO from quality signals.

#         Uses geometric mean so that if ANY component collapses
#         (low confidence, few inliers, or clustered features),
#         the overall weight drops sharply.

#         When spread is 0.0 (not yet computed, or rejected frame),
#         we fall back to a 2-component weight using only conf and
#         inliers. This avoids marking all early frames as degenerate.
#         """
#         # Normalize confidence: 0 at min_conf, 1 at 1.0
#         q_conf = ((vo_conf - self._min_conf)
#                   / max(1e-6, 1.0 - self._min_conf))
#         q_conf = float(np.clip(q_conf, 0.0, 1.0))

#         # Normalize inliers: 0 at min_inl, 1 at 2× min_inl
#         inl_hi = self._min_inl * 2.0
#         q_inl = ((inliers - self._min_inl)
#                  / max(1e-6, inl_hi - self._min_inl))
#         q_inl = float(np.clip(q_inl, 0.0, 1.0))

#         # If spread not available (0.0), use 2-component weight
#         if spread < 1.0:
#             w = (q_conf * q_inl) ** 0.5
#             return float(np.clip(w, 0.0, 1.0))

#         # Normalize spread: 0 at spread_bad, 1 at spread_good
#         # spread < spread_bad:  heavily clustered (turn degeneracy)
#         # spread > spread_good: well-distributed (straight/gentle curve)
#         # Default: spread_bad=55 spread_good=120 (tuned for fx=161 QLabs)
#         q_spread = ((spread - self._spread_bad)
#                     / max(1e-6, self._spread_good - self._spread_bad))
#         q_spread = float(np.clip(q_spread, 0.0, 1.0))

#         # Geometric mean of all three components
#         w = (q_conf * q_inl * q_spread) ** (1.0 / 3.0)
#         return float(np.clip(w, 0.0, 1.0))

#     # ══════════════════════════════════════════════════════
#     # EVALUATOR — Part 2: VECTOR RESIDUAL + WEIGHTED TRUST
#     # ══════════════════════════════════════════════════════

#     def _evaluate(self, now, vo_valid, vo_conf, inliers, reason,
#                   cart_x, cart_y, cart_psi):
#         if not self._vo_anchored:
#             self._flag = 'init'
#             self._healthy = True
#             self._dt_trans = 0.0
#             self._dt_yaw = 0.0
#             self._decision_msg = 'not_anchored'
#             return

#         if now < self._warmup_until:
#             self._flag = 'warming'
#             self._healthy = True
#             self._dt_trans = 0.0
#             self._dt_yaw = 0.0
#             self._decision_msg = 'warmup_%.1fs_left' % (
#                 self._warmup_until - now)
#             return

#         if len(self._window) < self._min_samples:
#             self._flag = 'warming'
#             self._healthy = True
#             self._dt_trans = 0.0
#             self._dt_yaw = 0.0
#             self._decision_msg = 'collecting_%d/%d' % (
#                 len(self._window), self._min_samples)
#             return

#         # ── Find anchor and latest samples in the window ──
#         # v5.1 FIX: require vo_valid (s[7]) on anchor so we don't
#         # compare against a stale VO pose from an invalid frame.
#         t_now = self._window[-1][0]
#         t_cutoff = t_now - self._win_sec
#         anchor = None
#         for s in self._window:
#             if s[0] >= t_cutoff and s[7]:  # s[7] = vo_valid
#                 anchor = s
#                 break

#         # v5.1 FIX: if no valid VO sample in window, treat as
#         # vo_suspect (not silent warming). This prevents the node
#         # from sitting at "warming" indefinitely when the camera
#         # goes dark — the consecutive counter keeps ticking so
#         # the system eventually escalates to confirmed vo_suspect.
#         if anchor is None:
#             self._c_vo_sus += 1
#             self._c_agree = 0
#             self._c_odom_sus = 0
#             self._last_vo_weight = 0.0
#             self._dt_trans = 0.0
#             self._dt_yaw = 0.0
#             if self._c_vo_sus >= self._n_consec:
#                 self._flag = 'vo_suspect'
#                 self._healthy = True
#                 self._decision_msg = (
#                     'VO_suspect_no_valid_anchor_in_window')
#             else:
#                 self._flag = 'vo_suspect'
#                 self._healthy = True
#                 self._decision_msg = (
#                     'VO_no_anchor_pending_%d/%d'
#                     % (self._c_vo_sus, self._n_consec))
#             return

#         latest = self._window[-1]

#         # v5: unpack 11-element tuples (added spread at index 10)
#         (_, c0x, c0y, c0p, v0x, v0y, v0p,
#          _, _, _, _) = anchor
#         (_, c1x, c1y, c1p, v1x, v1y, v1p,
#          _, _, _, _) = latest

#         # ── Step B: Vector residual ──
#         # Compare displacement VECTORS, not just scalar distances.
#         # This catches "same distance, wrong direction" failures.
#         # r(t) = Δp_vo - Δp_cart
#         dx_cart = c1x - c0x
#         dy_cart = c1y - c0y
#         dx_vo = v1x - v0x
#         dy_vo = v1y - v0y

#         rx = dx_vo - dx_cart
#         ry = dy_vo - dy_cart
#         rho = math.sqrt(rx * rx + ry * ry)

#         # Store for publishing (compatible with dashboard)
#         self._dt_trans = rho

#         # Yaw residual (diagnostic only, not used for decisions)
#         dpsi_cart = _wrap(c1p - c0p)
#         dpsi_vo = _wrap(v1p - v0p)
#         self._dt_yaw = abs(math.degrees(
#             _wrap(dpsi_vo - dpsi_cart)))

#         if reason == 'init':
#             self._flag = 'warming'
#             self._healthy = True
#             self._decision_msg = 'VO_initializing'
#             return

#         # ── Step C: Compute VO trust weight ──
#         w_vo = self._vo_weight(
#             vo_conf, inliers, self._last_inlier_spread)
#         self._last_vo_weight = w_vo

#         # ── Step D: Decision tree with weighted isolation ──
#         #
#         # Gate 1: Basic quality check (binary — same as v4)
#         #   VO must be valid, and have minimum conf + inliers.
#         #   If it fails this, VO is suspect regardless of weight.
#         #
#         # Gate 2: Degeneracy check (NEW in v5)
#         #   VO passes basic gate, but if weight is too low (e.g.
#         #   features are clustered during a turn), we still mark
#         #   VO as suspect. This prevents blaming odom when VO is
#         #   geometrically fragile even with high inlier count.
#         #
#         # Gate 3: Residual threshold (upgraded to vector in v5)
#         #   Only reached when VO is both valid AND trustworthy.
#         #   rho <= threshold → agree (both channels consistent)
#         #   rho >  threshold → odom_suspect (VO is strong but
#         #   disagrees with odom → something wrong with odom)

#         vo_good = (vo_valid
#                    and vo_conf >= self._min_conf
#                    and inliers >= self._min_inl)

#         # Gate 1: basic quality
#         if not vo_good:
#             self._c_vo_sus += 1
#             self._c_agree = 0
#             self._c_odom_sus = 0
#             if self._c_vo_sus >= self._n_consec:
#                 self._flag = 'vo_suspect'
#                 self._healthy = True
#                 self._decision_msg = (
#                     'VO_suspect_conf=%.2f_inl=%d_w=%.2f_%s'
#                     % (vo_conf, inliers, w_vo, reason))
#             else:
#                 self._flag = 'vo_suspect'
#                 self._healthy = True  # v5.1: explicit healthy on pending
#                 self._decision_msg = (
#                     'VO_suspect_pending_%d/%d_%s'
#                     % (self._c_vo_sus, self._n_consec, reason))
#             return

#         # Gate 2: degeneracy check (v5 Part 2)
#         if w_vo < self._min_vo_weight:
#             self._c_vo_sus += 1
#             self._c_agree = 0
#             self._c_odom_sus = 0
#             if self._c_vo_sus >= self._n_consec:
#                 self._flag = 'vo_suspect'
#                 self._healthy = True
#                 self._decision_msg = (
#                     'VO_degenerate_w=%.2f<%.2f_spread=%.0f'
#                     % (w_vo, self._min_vo_weight,
#                        self._last_inlier_spread))
#             else:
#                 self._flag = 'vo_suspect'
#                 self._healthy = True  # v5.1: explicit healthy on pending
#                 self._decision_msg = (
#                     'VO_degen_pending_%d/%d_w=%.2f'
#                     % (self._c_vo_sus, self._n_consec, w_vo))
#             return

#         # Gate 2.5: Turn gate (NEW — post fx=161 calibration)
#         # During turns, the SVD rigid-body assumption breaks down:
#         # matched features don't move rigidly when the camera rotates
#         # through a scene with depth variation. The SVD still produces
#         # a "confident" result (high inliers, high spread, high weight)
#         # but the translation direction is wrong. With corrected
#         # intrinsics (fx=161), these wrong-direction translations are
#         # at full metric scale, so they accumulate into meters of VO
#         # pose drift within a single turn. The worst case in testing
#         # was 5+ meters of VO drift through a 90° turn.
#         #
#         # Fix: if the Cartographer yaw has changed by more than
#         # turn_gate_deg over the evaluation window, the car is
#         # turning and VO translation should not be trusted. Force
#         # vo_suspect so the system never blames odom during turns.
#         # dpsi_cart is already computed above from the window anchor.
#         cart_turn = abs(dpsi_cart)
#         if cart_turn > self._turn_gate_rad:
#             self._c_vo_sus += 1
#             self._c_agree = 0
#             self._c_odom_sus = 0
#             if self._c_vo_sus >= self._n_consec:
#                 self._flag = 'vo_suspect'
#                 self._healthy = True
#                 self._decision_msg = (
#                     'VO_turning_dpsi=%.1fdeg>%.1fdeg_w=%.2f'
#                     % (math.degrees(cart_turn),
#                        math.degrees(self._turn_gate_rad),
#                        w_vo))
#             else:
#                 self._flag = 'vo_suspect'
#                 self._healthy = True
#                 self._decision_msg = (
#                     'VO_turn_pending_%d/%d_dpsi=%.1fdeg'
#                     % (self._c_vo_sus, self._n_consec,
#                        math.degrees(cart_turn)))
#             return

#         # Gate 3: vector residual threshold
#         # VO is valid AND trustworthy. Compare against odom.
#         trans_agree = (rho <= self._trans_thresh)

#         if trans_agree:
#             self._c_agree += 1
#             self._c_odom_sus = 0
#             self._c_vo_sus = 0
#             if self._c_agree >= self._n_consec:
#                 self._flag = 'agree'
#                 self._healthy = True
#                 self._decision_msg = (
#                     'AGREE_rho=%.3fm_w=%.2f_dpsi=%.1fdeg(diag)'
#                     % (rho, w_vo, self._dt_yaw))
#                 if self._do_reanchor:
#                     self._try_reanchor(
#                         now, cart_x, cart_y, cart_psi)
#             else:
#                 self._flag = 'agree'
#                 self._healthy = True  # v5.1: explicit healthy on pending
#                 self._decision_msg = (
#                     'agree_pending_%d/%d_rho=%.3f'
#                     % (self._c_agree, self._n_consec, rho))
#             return

#         # VO is strong but disagrees → odom suspect
#         self._c_odom_sus += 1
#         self._c_agree = 0
#         self._c_vo_sus = 0
#         if self._c_odom_sus >= self._n_consec:
#             self._flag = 'odom_suspect'
#             self._healthy = False
#             self._decision_msg = (
#                 'ODOM_SUSPECT_rho=%.3fm_w=%.2f_'
#                 'dpsi=%.1fdeg(diag)'
#                 % (rho, w_vo, self._dt_yaw))
#         else:
#             self._flag = 'odom_suspect'
#             self._healthy = True  # v5.2: pending should NOT be unhealthy
#             self._decision_msg = (
#                 'odom_suspect_pending_%d/%d_rho=%.3f'
#                 % (self._c_odom_sus, self._n_consec, rho))

#     def _try_reanchor(self, now, c_x, c_y, c_psi):
#         dt = now - self._last_reanchor_time
#         if dt < self._ra_cooldown:
#             self._decision_msg += (
#                 '_cooldown_%.1f/%.1fs' % (dt, self._ra_cooldown))
#             return

#         dx = c_x - self._last_reanchor_cart[0]
#         dy = c_y - self._last_reanchor_cart[1]
#         cart_moved = math.sqrt(dx**2 + dy**2)
#         if cart_moved < self._min_ra_dist:
#             self._decision_msg += (
#                 '_motion_%.3f/%.2fm'
#                 % (cart_moved, self._min_ra_dist))
#             return

#         with self._vo_lock:
#             self.vo.soft_reset(c_x, c_y, c_psi)
#         self._vo_pose = np.array([c_x, c_y, c_psi])
#         self._c_agree = 0
#         self._reanchor_count += 1
#         self._last_reanchor_time = now
#         self._last_reanchor_cart = np.array([c_x, c_y, c_psi])
#         self._warmup_until = now + self._warmup_s

#         # Clear window on re-anchor
#         self._window.clear()
#         self._c_odom_sus = 0
#         self._c_vo_sus = 0

#         self.get_logger().info(
#             'Re-anchored VO #%d at (%.3f,%.3f,%.1fdeg) '
#             'moved=%.3fm valid=%d reject=%d '
#             'nocam=%d stale=%d dup_skip=%d dsync=%d'
#             % (self._reanchor_count, c_x, c_y,
#                math.degrees(c_psi), cart_moved,
#                self._valid_frames, self._reject_frames,
#                self._no_camera_count,
#                self._stale_camera_count,
#                self._skipped_dup_count,
#                self._depth_sync_skip_count))

#     # ══════════════════════════════════════════════════════
#     # PUBLISHERS
#     # ══════════════════════════════════════════════════════

#     def _publish_numeric(self):
#         """v5.2: Publish plottable numeric topics alongside the
#         string-based fault_status. These are consumed by rqt_plot
#         and the supervisor node for real-time visualization."""
#         rho_msg = Float32()
#         rho_msg.data = float(self._dt_trans)
#         self._pub_rho.publish(rho_msg)

#         w_msg = Float32()
#         w_msg.data = float(self._last_vo_weight)
#         self._pub_weight.publish(w_msg)

#         state_msg = UInt8()
#         state_msg.data = self._STATE_IDS.get(self._flag, 0)
#         self._pub_state.publish(state_msg)

#         h_msg = Bool()
#         h_msg.data = bool(self._healthy)
#         self._pub_healthy.publish(h_msg)

#     def _publish_status(self, reason='waiting'):
#         """Publish status during early-return paths (no_camera, no_tf, stale).
#         v5.1: uses current _flag/_healthy instead of hardcoded 'init'.
#         """
#         msg = String()
#         msg.data = (
#             "flag=%s healthy=%s delta_trans=%.4f "
#             "delta_yaw=%.2f vo_valid=False vo_conf=0.00 inliers=0 "
#             "inlier_spread=0.0 vo_weight=0.00 "
#             "reason=%s "
#             "vo_x=%.5f vo_y=%.5f vo_psi=%.5f "
#             "dx=0 dy=0 dpsi=0 "
#             "cart_x=%.5f cart_y=%.5f cart_psi=%.5f "
#             "vo_speed=0 vo_dist=%.3f "
#             "reanchors=%d decision=%s_%s"
#             % (self._flag, self._healthy,
#                self._dt_trans, self._dt_yaw,
#                reason,
#                self._vo_pose[0], self._vo_pose[1],
#                self._vo_pose[2],
#                self._cart_x, self._cart_y, self._cart_psi,
#                self._vo_dist, self._reanchor_count,
#                self._flag, reason))
#         self._pub.publish(msg)
#         self._publish_numeric()

#     def _publish_full(self, vo_valid, vo_conf, inliers, reason,
#                       dx, dy, dpsi, vo_speed,
#                       cart_x, cart_y, cart_psi):
#         msg = String()
#         msg.data = (
#             "flag=%s "
#             "healthy=%s "
#             "delta_trans=%.4f "
#             "delta_yaw=%.2f "
#             "vo_valid=%s "
#             "vo_conf=%.2f "
#             "inliers=%d "
#             "inlier_spread=%.1f "
#             "vo_weight=%.2f "
#             "reason=%s "
#             "vo_x=%.5f vo_y=%.5f vo_psi=%.5f "
#             "dx=%.6f dy=%.6f dpsi=%.6f "
#             "cart_x=%.5f cart_y=%.5f cart_psi=%.5f "
#             "vo_speed=%.4f "
#             "vo_dist=%.3f "
#             "reanchors=%d "
#             "decision=%s"
#             % (self._flag, self._healthy,
#                self._dt_trans, self._dt_yaw,
#                vo_valid, vo_conf, inliers,
#                self._last_inlier_spread,
#                self._last_vo_weight,
#                reason,
#                self._vo_pose[0], self._vo_pose[1],
#                self._vo_pose[2],
#                dx, dy, dpsi,
#                cart_x, cart_y, cart_psi,
#                vo_speed, self._vo_dist,
#                self._reanchor_count,
#                self._decision_msg))
#         self._pub.publish(msg)
#         self._publish_numeric()


# def main(args=None):
#     rclpy.init(args=args)
#     node = VONode()
#     try:
#         # v4: MultiThreadedExecutor so image callbacks
#         # aren't blocked by VO compute (RANSAC 30-50ms)
#         executor = MultiThreadedExecutor(num_threads=3)
#         executor.add_node(node)
#         executor.spin()
#     except KeyboardInterrupt:
#         node.get_logger().info('VO node shutting down...')
#     finally:
#         try:
#             node.destroy_node()
#         except Exception:
#             pass
#         try:
#             rclpy.shutdown()
#         except Exception:
#             pass


# if __name__ == '__main__':
#     main()