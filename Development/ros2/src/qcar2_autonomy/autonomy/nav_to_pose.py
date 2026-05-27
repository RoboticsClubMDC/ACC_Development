#! /usr/bin/env python3

from hal.products.mats import SDCSRoadMap
from pal.utilities.math import wrap_to_pi


# 2026-05-27: Helper to MIRROR node 8's directional edges in the SDCSRoadMap.
# Edge list extracted from hal/products/mats.py (lines 148-194 for right-hand
# traffic, 69-116 for left-hand). For each edge touching node 8, we:
#   1. Locate it in roadmap.edges by fromNode/toNode identity
#   2. Remove from edges list AND from per-node inEdges/outEdges
#   3. Add a new edge with from/to swapped (reverses direction)
# Result: same connectivity topology, opposite direction. Pure pursuit's
# path planner (A*) sees node 8 with reversed incoming/outgoing edges.
def _mirror_node_8_edges(roadmap, leftHandTraffic, useSmallMap):
    scale = 0.002035
    innerLaneRadius = 305.5 * scale
    outerLaneRadius = 438.0 * scale
    oneWayStreetRadius = 350.0 * scale
    if leftHandTraffic:
        edges = [(6, 8, innerLaneRadius),
                 (8, 2, innerLaneRadius),
                 (8, 10, 0.0)]
        if not useSmallMap:
            edges += [(8, 12, outerLaneRadius),
                      (14, 8, outerLaneRadius)]
    else:
        edges = [(1, 8, outerLaneRadius),
                 (6, 8, 0.0),
                 (8, 10, oneWayStreetRadius)]
        if not useSmallMap:
            edges += [(8, 23, innerLaneRadius),
                      (12, 8, innerLaneRadius)]

    flipped_count = 0
    for (a, b, r) in edges:
        na = roadmap.nodes[a]
        nb = roadmap.nodes[b]
        matching = [e for e in roadmap.edges
                    if e.fromNode is na and e.toNode is nb]
        if not matching:
            continue
        e = matching[0]
        if e in na.outEdges: na.outEdges.remove(e)
        if e in nb.inEdges:  nb.inEdges.remove(e)
        if e in roadmap.edges: roadmap.edges.remove(e)
        roadmap.add_edge(b, a, r)  # reversed!
        flipped_count += 1
    return flipped_count

import os
import select
import sys
import termios
import threading
import time
import tty
import numpy as np
import scipy.signal as signal
from scipy.spatial.transform import Rotation as R
from pal.utilities.scope import MultiScope


# ─── Keystroke helpers folded in from manual_drive.py ─────────────────────
# Used by the path_follower's "manual" control mode so the same node owns
# /cmd_vel_nav in all modes (no double-publishing fight). Falls back
# gracefully if stdin isn't a TTY (e.g., when launched via launch files).

_TTY_ATTRS_BACKUP = None


def _stdin_is_tty():
    return sys.stdin is not None and sys.stdin.isatty()


def _setup_terminal():
    global _TTY_ATTRS_BACKUP
    if not _stdin_is_tty():
        return False
    fd = sys.stdin.fileno()
    _TTY_ATTRS_BACKUP = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    return True


def _restore_terminal():
    global _TTY_ATTRS_BACKUP
    if _TTY_ATTRS_BACKUP is None or not _stdin_is_tty():
        return
    fd = sys.stdin.fileno()
    termios.tcsetattr(fd, termios.TCSADRAIN, _TTY_ATTRS_BACKUP)
    _TTY_ATTRS_BACKUP = None


def _get_key(timeout_s=0.05):
    if not _stdin_is_tty():
        return ""
    r, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not r:
        return ""
    return os.read(sys.stdin.fileno(), 1).decode(errors="ignore")

from rclpy.duration import Duration
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from rclpy.node import Node
from nav_msgs.msg import Path
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Imu, JointState
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Bool, Float32

# Pose source: subscribe to /qcar2_pose_fused, produced by the standalone
# ekf_fusor node. The embedded QcarEKF/GyroKF instances that used to live in
# this file have been retired — ekf_fusor owns the EKF math now, and the
# filtered pose comes in via topic. See Easy_Start.md change log.


class PathFollower(Node):

    def __init__(self):
        super().__init__('path_follower')
        self.get_logger().info("BP00 TEST: THIS IS THE FILE ROS IS RUNNING")
        self.declare_parameter('node_values', [0, 8, 10])
        self.waypoints = list(self.get_parameter('node_values').get_parameter_value().integer_array_value)

        self.declare_parameter('desired_speed', [0.4])
        self.desired_speed = list(self.get_parameter('desired_speed').get_parameter_value().double_array_value)

        self.declare_parameter('visualize_pose', [True])
        self.pose_visualize_flag = True

        self.scale = 1.0

        self.declare_parameter('rotation_offset', [83.0])
        self.rotation_offset = list(self.get_parameter('rotation_offset').get_parameter_value().double_array_value)

        self.declare_parameter('translation_offset', [0.0, 0.0])
        self.translation_offset = list(self.get_parameter('translation_offset').get_parameter_value().double_array_value)

        # Unified control mode — ONE node owns /cmd_vel_nav, no double-publish.
        #   idle        → publish NOTHING (bus is free; external nodes can drive)
        #   manual      → WASD keystrokes from this terminal publish /cmd_vel_nav
        #   autonomous  → pure-pursuit driving (existing behavior)
        #
        # Default = idle so spawning path_follower doesn't immediately fight
        # anything. To enter a driving mode:
        #   * ros2 param set /path_follower control_mode "manual"
        #   * ros2 param set /path_follower control_mode "autonomous"
        #   * publish /cmd_waypoints  (auto-switches to "autonomous")
        #   * update node_values param (auto-switches to "autonomous")
        self.declare_parameter('control_mode', 'idle')
        self.control_mode = str(self.get_parameter('control_mode').value).lower()
        if self.control_mode not in ('idle', 'manual', 'autonomous'):
            self.control_mode = 'idle'
        # Legacy start_path: kept for compatibility but is now derived from
        # control_mode. Setting start_path=True is equivalent to control_mode=autonomous.
        self.declare_parameter('start_path', [False])
        self.path_execute_flag = (self.control_mode == 'autonomous')

        # ─── Manual-mode WASD config (folded in from manual_drive.py) ───
        self.declare_parameter('manual_forward_speed', 0.25)
        self.declare_parameter('manual_reverse_speed', 0.20)
        self.declare_parameter('manual_turn_rate',     0.50)
        self.declare_parameter('manual_speed_step',    0.05)
        self.manual_forward_speed = float(self.get_parameter('manual_forward_speed').value)
        self.manual_reverse_speed = float(self.get_parameter('manual_reverse_speed').value)
        self.manual_turn_rate     = float(self.get_parameter('manual_turn_rate').value)
        self.manual_speed_step    = float(self.get_parameter('manual_speed_step').value)
        self.manual_linear  = 0.0
        self.manual_angular = 0.0
        self._manual_thread = None
        self._manual_thread_stop = threading.Event()

        self.declare_parameter('mission_pickup_xy',  [0.0, 0.0])
        self.declare_parameter('mission_dropoff_xy', [0.0, 0.0])
        self.declare_parameter('mission_enable',     [False])

        # PD gains for pure-pursuit steering. Live-tunable via Foxglove
        # Parameters panel or `ros2 param set /path_follower kp_steering <v>`.
        # Defaults 2026-05-24 (Option B from BO post-analysis):
        #   Kp=1.10, Kd=0.20 — "robust cluster" middle. BO's literal best was
        #   Kp=1.08, Kd=0.08 but that's effectively undamped; the BO test
        #   path didn't exercise tight corners. Kd=0.20 gives a safety margin
        #   for real competition driving without sacrificing responsiveness.
        self.declare_parameter('kp_steering', 1.100)
        # 2026-05-27: kd_steering 0.20 → 0.10 to match Gabriel's effective
        # damping (his Kd=5 with implicit ×π/180 ≈ 0.087). User testing
        # showed our 0.20 made the controller feel sluggish/laggy compared
        # to his. 0.10 is the responsive-but-still-damped sweet spot.
        self.declare_parameter('kd_steering', 0.20)
        # 2026-05-27: cos^N(steering) speed reduction. Default 0.0 → DISABLED
        # (matches Gabriel — constant speed through curves). Set to 2.0 to
        # re-enable cos² speed cut for more conservative cornering.
        self.declare_parameter('steering_speed_exponent', 0.0)
        # 2026-05-27: NEW runtime-tunable lookahead params.
        # lookahead_dist = max(v_eff × lookahead_dist_multiplier, lookahead_dist_floor)
        # Defaults match Gabriel's (1.7, 0.30). Raise multiplier to ~2.5 and
        # floor to ~0.50 if PP wobbles through tight intersection curvature
        # — bigger lookahead = smoother tracking but less responsive to real
        # sharp turns.
        self.declare_parameter('lookahead_dist_multiplier', 1.7)
        self.declare_parameter('lookahead_dist_floor', 0.30)
        # 2026-05-27: WaypointDist floor for the PP atan2 formula.
        # Prevents saturation when waypoints cluster at intersections. 0.05
        # was too tight; 0.20 is the safe minimum on dense planner output.
        self.declare_parameter('waypoint_dist_floor', 0.20)
        # 2026-05-27: SDCSRoadMap traffic-direction toggle.
        # Originally hardcoded False after we found Quanser's default True
        # produced wrong paths on our right-hand-traffic scene.
        # But user observed paths still take "long way around" — possible
        # the right answer is True after all (Quanser PNG label might be
        # mislabeled). Made tunable for A/B testing.
        # CHANGES TAKE EFFECT ON NEXT `node_values` SET (path is regenerated).
        self.declare_parameter('left_hand_traffic', False)
        # 2026-05-27: SDCSRoadMap small-map toggle. Quanser provides two
        # map variants: the large SDCS (default, useSmallMap=False) and
        # a smaller variant with different node connectivity. If routes
        # on the large map have unavoidable detours, the small map might
        # have a more direct graph (or vice versa). A/B testable.
        self.declare_parameter('use_small_map', False)
        # 2026-05-27 DIAGNOSTIC — node-8 EDGE/POSITION mirror test.
        # The SDCSRoadMap library is closed-source. We CAN access
        # roadmap.nodes[i] (per Quanser's path_planning_example.py:
        # `for i, node in enumerate(roadmap.nodes): print(node.pose)`).
        # So we can MIRROR JUST NODE 8 in the in-memory graph:
        #   - Negate node 8's pose X coord BEFORE calling generate_path
        # If the library computes edge connectivity from poses dynamically,
        # this will reroute the planner via different (potentially
        # symmetric to node 6's) edges.
        # If the library has pre-cached edges from init, this only changes
        # the path's final waypoint position — still informative.
        # USAGE: set mirror_node_8=true → re-set node_values to regen path.
        self.declare_parameter('mirror_node_8', False)
        # Kept from previous attempt — path-level mirror (less targeted).
        # Leave this False; mirror_node_8 above is what user actually wants.
        self.declare_parameter('mirror_path_x', False)
        # 2026-05-27: tunable end-of-path stop precision.
        # `final_target_index_back` — how many waypoints from the end to
        #   clamp wpi at. Smaller value = PP targets closer to actual
        #   final waypoint = car stops closer to the actual node.
        #   Too small (0-1) re-introduces the "instant complete" wobble.
        # `final_stop_dist` — distance from current target (wp[wpi after
        #   clamp]) at which to declare path_complete. Smaller value =
        #   car drives closer before stopping.
        # Effective max gap from actual node: ~final_target_index_back × 1cm
        #   + final_stop_dist. With defaults: ~3cm + 0.15m = 0.18m.
        # Was previously Arturo's defaults (5 and 0.4 → up to 45cm short).
        self.declare_parameter('final_target_index_back', 3)
        self.declare_parameter('final_stop_dist', 0.15)
        # 2026-05-27: approach slowdown distance. When `dist_to_final` (the
        # Euclidean distance from current pose to wp[-1]) drops below this,
        # speed_command is capped at 0.2 m/s for a smooth final approach.
        # Was hardcoded `wpi > N-100` which is fragile on paths that double
        # back (1m of PATHWISE distance ≠ 1m of Euclidean distance when the
        # path bends). Now distance-based and tunable. 0.5 m is a tight
        # slowdown zone for low-speed driving; raise to 1.0 if you want a
        # gentler approach.
        self.declare_parameter('approach_slowdown_dist', 0.5)
        # 2026-05-27: cluster-skip toggle. When True, after the standard
        # wpi += 1 advance, keep skipping waypoints that are still inside
        # lookahead_dist. EMPIRICAL FINDING: with the planner's ~1cm
        # waypoint spacing along smooth paths, this loop advanced 100+
        # waypoints per tick, causing PP to target a waypoint so far
        # ahead that path curvature between car and target didn't match
        # PP's straight-line assumption → high-frequency zig-zag (~30-50
        # cm wavelength) throughout the path. DEFAULT FALSE until we
        # build an angular-aware version that doesn't skip past curves.
        self.declare_parameter('cluster_skip_enabled', False)
        self.kp_steering = float(self.get_parameter('kp_steering').value)
        self.kd_steering = float(self.get_parameter('kd_steering').value)
        self.steering_speed_exponent = float(
            self.get_parameter('steering_speed_exponent').value)

        # Stanley blend is kept at 0 — pure pursuit only
        self.stanley_blend     = 0.3
        self.stanley_trust_min = 999.0

        self.add_on_set_parameters_callback(self.parameter_update_callback)

        self.target_frame = self.declare_parameter(
            'target_frame', 'base_link').get_parameter_value().string_value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.dt = 1 / 80

        # Fused-pose cache, populated by /qcar2_pose_fused subscriber. Until
        # the first message arrives these stay None and the path planner falls
        # back to raw TF (see _read_pose_for_planner).
        self.fused_pose_x = None
        self.fused_pose_y = None
        self.fused_pose_yaw = None
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/qcar2_pose_fused',
            self.fused_pose_cb,
            10,
        )

        self.yaw = 0
        self.cutoff_frequency_filter = 15.0
        self.a1, self.b1 = self.filter_coefficients(self.cutoff_frequency_filter, self.dt)

        self.path_control_timer = self.create_timer(self.dt, self.path_planner)
        self.timer = self.create_timer(self.dt, self.tf_timer)
        self.translation = [0, 0, 0]
        self.rotation = [0, 0, 0]
        _lht = bool(self.get_parameter('left_hand_traffic').value)
        _usm = bool(self.get_parameter('use_small_map').value)
        _roadmap_init = SDCSRoadMap(leftHandTraffic=_lht, useSmallMap=_usm)
        # 2026-05-27 DIAGNOSTIC — mirror node 8's directional edges in the
        # roadmap graph BEFORE path generation. Keeps node 8's position the
        # same; only flips the direction of every edge touching it.
        if bool(self.get_parameter('mirror_node_8').value):
            n = _mirror_node_8_edges(_roadmap_init, _lht, _usm)
            self.get_logger().warn(
                f'MIRROR_NODE_8: flipped {n} edges touching node 8.')
        _init_wp = _roadmap_init.generate_path(self.waypoints)
        if _init_wp is None:
            self.get_logger().error(
                f'generate_path({self.waypoints}) returned None at startup '
                f'(no legal A* path with current settings). Using empty path; '
                f'set valid node_values via ros2 param set to get going.')
            self.wp = np.zeros((3, 1))  # 3xN empty-ish placeholder
        else:
            self.wp = _init_wp * self.scale
        # Legacy path-level mirror (set mirror_path_x=true to enable).
        if bool(self.get_parameter('mirror_path_x').value):
            self.wp[0, :] = -self.wp[0, :]
            self.get_logger().warn(
                f'mirror_path_x=True — X coords of {self.wp.shape[1]} '
                f'waypoints negated.')
        # -------------ADDED N_1
        self.auto_align_start = True
        self.auto_aligned = False
        #----------
        self.N = len(self.wp[0, :])
        self.wpi = 0
        self.wp_prior = []
        self.current_steering = 0
        self._wp_in_ros_frame = False

        self.stanley_delta = 0.0
        self.stanley_trust = 0.0
        self.pp_delta_raw  = 0.0

        # 2026-05-27: cmd_topic is now parametrizable so path_follower can
        # publish to /cmd_vel_path when running under cmd_vel_blender (which
        # owns /cmd_vel_nav and blends path + lane). Default keeps legacy
        # behavior (publish directly to /cmd_vel_nav).
        self.declare_parameter('cmd_topic', '/cmd_vel_nav')
        cmd_topic = self.get_parameter('cmd_topic').get_parameter_value().string_value
        self.publisher = self.create_publisher(Twist, cmd_topic, 1)
        self.get_logger().info(f'path_follower publishing to {cmd_topic}')
        self.cyclic = False
        # 2026-05-27: 0.52 → 0.55. ±0.52 rad = ±30° is the hardware spec
        # (Table 11). Gabriel used 0.60 (above spec) for harder cornering;
        # we settle on 0.55 (~31.5°) — a small overshoot above nominal spec
        # but still within the servo's mechanical range, gives Stanley
        # margin without exceeding hardware safety.
        self.max_steering_angle = 0.55

        self.joint_state_subscriber = self.create_subscription(
            JointState, '/qcar2_joint', self.joint_state_callback, 1)
        self.qcar2_measurred_speed = 0

        self.object_detection_flag = self.create_subscription(
            Bool, '/motion_enable', self.object_detector_callback, 1)
        self.motion_flag = True
        self.path_complete = False

        self.imu_subscrition = self.create_subscription(
            Imu, '/qcar2_imu', self.imu_callback, 10)
        self.gyroscope = [0, 0, 0]

        self.path_publisher_topic  = self.create_publisher(Path, '/planned_path', 1)
        self.path_status_publisher = self.create_publisher(Bool, '/path_status', 1)
        self.robot_pose_publisher  = self.create_publisher(PoseStamped, '/robot_pose', 10)

        self.create_subscription(Path, '/cmd_waypoints', self._cmd_waypoints_cb, 1)

        # Foxglove-friendly live tuning: publish a Float32 to these topics from
        # a Foxglove "Publish" panel slider to update Kp/Kd in real time. This
        # exists in addition to the ROS parameters (both routes work).
        self.create_subscription(Float32, '/nav/kp_steering_set', self._kp_set_cb, 5)
        self.create_subscription(Float32, '/nav/kd_steering_set', self._kd_set_cb, 5)

        self.pub_pp_delta      = self.create_publisher(Float32, '/nav/pp_delta',      2)
        self.pub_stanley_delta = self.create_publisher(Float32, '/nav/stanley_delta', 2)
        self.pub_blended_delta = self.create_publisher(Float32, '/nav/blended_delta', 2)
        self.pub_blend_alpha   = self.create_publisher(Float32, '/nav/blend_alpha',   2)

        # --- Controller diagnostics (Foxglove plot targets) ---
        # All Float32 for easy Foxglove Plot panel binding via `.data`.
        self.pub_dist_wp           = self.create_publisher(Float32, '/nav/distance_to_waypoint', 2)
        self.pub_dist_final        = self.create_publisher(Float32, '/nav/distance_to_final',    2)
        self.pub_psi               = self.create_publisher(Float32, '/nav/psi',                  2)
        self.pub_steering_saturation = self.create_publisher(Float32, '/nav/steering_saturation_rate', 2)
        self.pub_speed_cmd         = self.create_publisher(Float32, '/nav/speed_cmd',            2)
        self.pub_yaw_rate          = self.create_publisher(Float32, '/nav/yaw_rate_imu',         2)
        self.pub_progress_rate     = self.create_publisher(Float32, '/nav/progress_rate',        2)
        self.pub_wpi               = self.create_publisher(Float32, '/nav/wpi',                  2)
        self.pub_controller_mode   = self.create_publisher(Float32, '/nav/controller_mode',      2)

        # Saturation rate uses a moving window of the last N steering commands.
        self._sat_window = []
        self._sat_window_size = 80   # ~1s at 80 Hz
        # For progress_rate, remember the previous distance-to-final.
        self._prev_dist_final = None
        self._prev_progress_time = None

        self.t0 = time.time()
        self.t_plot = 0
        self.plot_visualized = False
        self.scopeTimer = self.create_timer(0.1, self.scopeDataTimer)

        self.get_logger().info(
            f'PathFollower ready | PURE PURSUIT BASELINE | '
            f'stanley_blend={self.stanley_blend} trust_min={self.stanley_trust_min}')

    def _cmd_waypoints_cb(self, msg: Path):
        if not msg.poses:
            self.get_logger().warn('/cmd_waypoints empty — ignoring')
            return
        xs = [p.pose.position.x for p in msg.poses]
        ys = [p.pose.position.y for p in msg.poses]
        self.wp              = np.array([xs, ys])
        self.N               = self.wp.shape[1]
        self.wpi             = 0
        self.path_complete   = False
        self._wp_in_ros_frame = True
        # Auto-switch to autonomous mode (BO/trip_planner/etc. sending a path
        # is an explicit driving intent). Stops the manual thread if it was up.
        self._switch_mode('autonomous')
        self.get_logger().info(f'/cmd_waypoints received N={self.N}')

    def _kp_set_cb(self, msg: Float32):
        new_kp = float(msg.data)
        # Soft clamp to safe range; the real hard limit is steering saturation
        # downstream, but very large values cause instant divergence.
        new_kp = float(np.clip(new_kp, 0.0, 3.0))
        if abs(new_kp - self.kp_steering) > 1e-4:
            self.kp_steering = new_kp
            self.get_logger().info(f'kp_steering (topic) -> {self.kp_steering:.3f}')

    def _kd_set_cb(self, msg: Float32):
        new_kd = float(msg.data)
        new_kd = float(np.clip(new_kd, 0.0, 2.0))
        if abs(new_kd - self.kd_steering) > 1e-4:
            self.kd_steering = new_kd
            self.get_logger().info(f'kd_steering (topic) -> {self.kd_steering:.3f}')

    def _stanley_delta_cb(self, msg: Float32):
        self.stanley_delta = float(msg.data)

    def _stanley_trust_cb(self, msg: Float32):
        self.stanley_trust = float(msg.data)

    def _blend_steering(self, pp_delta: float) -> float:
        self.pp_delta_raw = pp_delta

        if self.stanley_trust < self.stanley_trust_min:
            alpha = 0.0
        else:
            alpha = np.clip(self.stanley_blend * self.stanley_trust, 0.0, 1.0)

        blended = (1.0 - alpha) * pp_delta + alpha * self.stanley_delta
        blended = float(np.clip(blended, -self.max_steering_angle, self.max_steering_angle))

        def f32(v):
            m = Float32(); m.data = float(v); return m

        self.pub_pp_delta.publish(f32(pp_delta))
        self.pub_stanley_delta.publish(f32(self.stanley_delta))
        self.pub_blended_delta.publish(f32(blended))
        self.pub_blend_alpha.publish(f32(alpha))

        return blended

    def parameter_update_callback(self, params):
        for param in params:
            if param.name == 'node_values' and param.type_ == param.Type.INTEGER_ARRAY:
                self.waypoints = list(param.value)
                _lht2 = bool(self.get_parameter('left_hand_traffic').value)
                _usm2 = bool(self.get_parameter('use_small_map').value)
                _roadmap2 = SDCSRoadMap(leftHandTraffic=_lht2, useSmallMap=_usm2)
                if bool(self.get_parameter('mirror_node_8').value):
                    n = _mirror_node_8_edges(_roadmap2, _lht2, _usm2)
                    self.get_logger().warn(
                        f'MIRROR_NODE_8 (param-update): flipped {n} edges.')
                _new_wp = _roadmap2.generate_path(self.waypoints)
                if _new_wp is None:
                    # 2026-05-27: SDCSRoadMap.generate_path returns None when
                    # A* can't find any legal directed path. Keep the OLD wp
                    # so the node doesn't crash; warn so user knows the
                    # node_values change had no effect.
                    self.get_logger().error(
                        f'generate_path({self.waypoints}) returned None — '
                        f'no legal A* path exists in the (possibly mirrored) '
                        f'graph. Keeping previous path. Try different '
                        f'node_values or set mirror_node_8=false.')
                else:
                    self.wp = _new_wp * 0.975
                if bool(self.get_parameter('mirror_path_x').value):
                    self.wp[0, :] = -self.wp[0, :]
                self.N = len(self.wp[0, :])
                self.wpi = 0
                self.previous_steering_value = 0
                self.path_complete = False
                self._wp_in_ros_frame = False
                # Auto-switch into autonomous: setting node_values is an
                # explicit "I want to drive" intent. Stops the manual thread
                # if it was running.
                self._switch_mode('autonomous')
                self.get_logger().info('nodes updated! → autonomous mode')
            elif param.name == 'control_mode' and param.type_ == param.Type.STRING:
                requested = str(param.value).lower()
                if requested in ('idle', 'manual', 'autonomous'):
                    self._switch_mode(requested)
                else:
                    self.get_logger().warn(
                        f"control_mode '{param.value}' not in "
                        f"['idle', 'manual', 'autonomous'] — ignoring"
                    )
            elif param.name == 'desired_speed' and param.type_ == param.Type.DOUBLE_ARRAY:
                self.desired_speed = list(param.value)
                self.get_logger().info('new desired speed...')
            elif param.name == 'rotation_offset' and param.type_ == param.Type.DOUBLE_ARRAY:
                self.rotation_offset = list(param.value)
            elif param.name == 'translation_offset' and param.type_ == param.Type.DOUBLE_ARRAY:
                self.translation_offset = list(param.value)
            elif param.name == 'start_path' and param.type_ == param.Type.BOOL_ARRAY:
                self.path_execute_flag = list(param.value)[0]
                self.get_logger().info('path status changed!')
            elif param.name == 'stanley_blend' or param.name == 'stanley_trust_min':
                self.get_logger().warn(f'{param.name} is hardcoded — ignoring')
            elif param.name == 'mission_pickup_xy' and param.type_ == param.Type.DOUBLE_ARRAY:
                self.get_logger().info(f'mission_pickup_xy updated: {list(param.value)}')
            elif param.name == 'mission_dropoff_xy' and param.type_ == param.Type.DOUBLE_ARRAY:
                self.get_logger().info(f'mission_dropoff_xy updated: {list(param.value)}')
            elif param.name == 'mission_enable' and param.type_ == param.Type.BOOL_ARRAY:
                self.get_logger().info(f'mission_enable updated: {list(param.value)}')
            elif param.name == 'kp_steering' and param.type_ == param.Type.DOUBLE:
                self.kp_steering = float(param.value)
                self.get_logger().info(f'kp_steering -> {self.kp_steering:.3f}')
            elif param.name == 'kd_steering' and param.type_ == param.Type.DOUBLE:
                self.kd_steering = float(param.value)
                self.get_logger().info(f'kd_steering -> {self.kd_steering:.3f}')
            elif param.name == 'visualize_pose' and param.type_ == param.Type.BOOL_ARRAY:
                self.pose_visualize_flag = list(param.value)[0]
                if self.pose_visualize_flag and not self.plot_visualized:
                    self.get_logger().info('Pose visualization enabled...')
                    tf = 200
                    self.steeringScope = MultiScope(rows=4, cols=1,
                                                    title='Vehicle Steering Control', fps=10)
                    self.steeringScope.addAxis(row=0, col=0, timeWindow=tf,
                                               yLabel='x Position [m]', yLim=(-2.5, 2.5))
                    self.steeringScope.axes[0].attachSignal(name='x_meas')
                    self.steeringScope.axes[0].attachSignal(name='x_ekf')
                    self.steeringScope.addAxis(row=1, col=0, timeWindow=tf,
                                               yLabel='y Position [m]', yLim=(-1, 6))
                    self.steeringScope.axes[1].attachSignal(name='y_meas')
                    self.steeringScope.axes[1].attachSignal(name='y_ekf')
                    self.steeringScope.addAxis(row=2, col=0, timeWindow=tf,
                                               yLabel='steering cmd [rad]', yLim=(-0.6, 0.6))
                    self.steeringScope.axes[2].attachSignal(name='pp_delta')
                    self.steeringScope.axes[2].attachSignal(name='stanley_delta')
                    self.steeringScope.axes[2].attachSignal(name='blended_delta')
                    self.steeringScope.axes[2].attachSignal(name='blend_alpha')
                    self.steeringScope.addAxis(row=3, col=0, timeWindow=tf,
                                               yLabel='heading', yLim=(-np.pi, np.pi))
                    self.steeringScope.axes[3].attachSignal(name='theta_meas')
                    self.steeringScope.axes[3].attachSignal(name='theta_EKF_sf')
                    self.plot_visualized = True
                elif self.pose_visualize_flag and self.plot_visualized:
                    self.get_logger().info('visualization running...')
                elif not self.pose_visualize_flag and self.plot_visualized:
                    self.plot_visualized = False
        return SetParametersResult(successful=True)

    def filter_coefficients(self, freq, dt):
        nyq_freq = 0.5 * (1 / dt)
        norm_cut = freq / nyq_freq
        b, a = signal.butter(2, norm_cut)
        self.hist = {'gyro': {'in': [0.0] * 3, 'out': [0.0] * 3}}
        return a, b

    def apply_filter(self, key, new_input, a, b):
        h = self.hist[key]
        h['in'] = [new_input] + h['in'][:2]
        y = (b[0]*h['in'][0] + b[1]*h['in'][1] + b[2]*h['in'][2]
             - a[1]*h['out'][0] - a[2]*h['out'][1])
        h['out'] = [y] + h['out'][:2]
        return y

    def object_detector_callback(self, msg):
        self.motion_flag = msg.data

    def joint_state_callback(self, msg):
        # 2026-05-27: GEAR RATIO FIX. Was (70.0 * 30.0) = Gabriel's bug
        # carried over from the Quanser original. Per the QCar 2 User
        # Manual hardware spec (Easy_Start §16, CLAUDE.md §5.1): the
        # denominator is 37, NOT 30 — 23% wrong otherwise.
        # Already fixed in pose_estimator.py and ekf_fusor.py but
        # was missed here. Bug manifestation: v_eff reads 23% too high,
        # lookahead_dist becomes 23% too long, waypoint advances trigger
        # early, controller cuts corners and creates "instant-complete"
        # geometry at end-of-path.
        self.qcar2_measurred_speed = (msg.velocity[0] / (720.0 * 4.0)) * ((13.0 * 19.0) / (70.0 * 37.0)) * (2.0 * np.pi) * 0.033

    def imu_callback(self, msg):
        self.gyroscope = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]



#----------------Change N2 auto_align_roadmap_to_current_pose----------------
    def auto_align_roadmap_to_current_pose(self):
        """
        Align SDCSRoadMap coordinates to the current Cartographer map frame.

        It assumes:
        - The QCar is physically placed at the first waypoint of the current route.
        - Cartographer map->base_link is already available.
        - self.wp has shape (2, N), where row 0 = x and row 1 = y.
        """

        try:
            # Current QCar pose in Cartographer map frame
            tf_msg = self.tf_buffer.lookup_transform(
                'map',
                self.target_frame,   # usually base_link
                rclpy.time.Time()
            )

            current_xy = np.array([
                tf_msg.transform.translation.x,
                tf_msg.transform.translation.y
            ])

            q = tf_msg.transform.rotation
            current_yaw = np.arctan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )

            # First point of SDCSRoadMap path
            raw_start = np.array([
                self.wp[0, 0],
                self.wp[1, 0]
            ])

            # Estimate ideal path heading using a point slightly ahead
            lookahead_i = min(20, self.wp.shape[1] - 1)
            raw_next = np.array([
                self.wp[0, lookahead_i],
                self.wp[1, lookahead_i]
            ])

            raw_vec = raw_next - raw_start
            raw_heading = np.arctan2(raw_vec[1], raw_vec[0])

            # Required global rotation from SDCS path frame to Cartographer map frame
            angle_rad = np.arctan2(
                np.sin(current_yaw - raw_heading),
                np.cos(current_yaw - raw_heading)
            )

            angle_deg = angle_rad * 180.0 / np.pi

            # Match the same rotation convention used in path_publisher/path_planner
            R_QLabs_ROS = np.array([
                [np.cos(-angle_deg * np.pi / 180.0), -np.sin(-angle_deg * np.pi / 180.0)],
                [np.sin(-angle_deg * np.pi / 180.0),  np.cos(-angle_deg * np.pi / 180.0)]
            ])

            # Existing code does:
            # wp_map = (wp_raw + translation_offset) @ R
            # Solve:
            # translation_offset = current_xy @ R.T - raw_start
            translation_offset = current_xy @ R_QLabs_ROS.T - raw_start

            self.rotation_offset = [float(angle_deg)]
            self.translation_offset = [
                float(translation_offset[0]),
                float(translation_offset[1])
            ]

            self.get_logger().info(
                f"Auto-aligned roadmap: rotation_offset={self.rotation_offset}, "
                f"translation_offset={self.translation_offset}"
            )

            return True

        except Exception as e:
            self.get_logger().warn(f"Roadmap auto-align waiting for TF map->{self.target_frame}: {e}")
            return False
    #----------------End of auto_align_roadmap_to_current_pose----------------N2

    def path_publisher(self):
        # 2026-05-27 BUG FIX: was publishing only `range(self.wpi)` — i.e.,
        # ONLY waypoints the car had already passed (a breadcrumb trail).
        # Now publishes the FULL planned path so Foxglove + /planned_path
        # echo show what the controller will actually steer toward.
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        for i in range(self.N):
            if i >= self.N:
                i = self.N - 1
            pose = PoseStamped()
            angle_offset = self.rotation_offset[0]
            R_QLabs_ROS = np.array([
                [np.cos(-angle_offset * np.pi / 180), -np.sin(-angle_offset * np.pi / 180)],
                [np.sin(-angle_offset * np.pi / 180),  np.cos(-angle_offset * np.pi / 180)]
            ])
            t = np.array([self.translation_offset[0], self.translation_offset[1]])
            if self._wp_in_ros_frame:
                wp_1_mod = np.array([self.wp[0, i], self.wp[1, i]])
            else:
                wp_1_mod = ([self.wp[0, i], self.wp[1, i]] + t) @ R_QLabs_ROS
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = 'map'
            pose.pose.position.x = wp_1_mod[0]
            pose.pose.position.y = wp_1_mod[1]
            path_msg.poses.append(pose)
        self.path_publisher_topic.publish(path_msg)

    def path_planner(self):
        max_speed = 1.5
        enable = 1
        speed_command = self.desired_speed[0]
        skip_index = 1

        self.t_plot = time.time() - self.t0
        # ekf_filter_timer() removed: prediction is now done by the external
        # ekf_fusor node; this node only consumes the fused pose.

        #----CHANGE N3--------------
        if self.auto_align_start and not self.auto_aligned:
            self.auto_aligned = self.auto_align_roadmap_to_current_pose()
            if not self.auto_aligned:
                return

        #--END OF CHANGE N3---------------------

        if round(self.t_plot) % 2 == 0:
            self.path_publisher()

        try:
            if not self.path_complete:
                wp_1 = np.array(self.wp[:, self.wpi])

                angle_offset = self.rotation_offset[0]
                R_QLabs_ROS = np.array([
                    [np.cos(-angle_offset * np.pi / 180), -np.sin(-angle_offset * np.pi / 180)],
                    [np.sin(-angle_offset * np.pi / 180),  np.cos(-angle_offset * np.pi / 180)]
                ])
                t = np.array([self.translation_offset[0], self.translation_offset[1]])
                if self._wp_in_ros_frame:
                    wp_1_mod = wp_1
                else:
                    wp_1_mod = (wp_1 + t) @ R_QLabs_ROS

                L = 0.256

                # Pose source priority: fused (ekf_fusor) → raw TF → origin.
                # Once /qcar2_pose_fused starts publishing (after bootstrap)
                # we use it as the controller's reference frame, since it
                # carries the EKF correction and outlier rejection.
                p, th = self._read_pose_for_planner()

                v = [wp_1_mod[0] - p[0], wp_1_mod[1] - p[1]]
                Rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                v_car = v @ Rot

                # 2026-05-27: WaypointDist floor is now a runtime param
                # (waypoint_dist_floor, default 0.20). PP atan2 denominator
                # bound — prevents saturation when waypoints cluster at
                # intersections (planner returns 5-10 cm spaced waypoints).
                waypoint_dist_floor = float(
                    self.get_parameter('waypoint_dist_floor').value)
                WaypointDist = max(np.linalg.norm(v_car), waypoint_dist_floor)
                psi = np.arctan2(v_car[1], v_car[0])

                pp_delta = np.arctan2(2 * L * np.sin(psi), WaypointDist)

                dist = np.linalg.norm([p[0] - wp_1_mod[0], p[1] - wp_1_mod[1]])

                # 2026-05-27: lookahead now uses runtime-tunable multiplier
                # and floor params. Bigger lookahead = PP looks further
                # ahead, smooths out tight curvature; smaller = more
                # responsive to sharp turns. Sweet spot per scene.
                lookahead_mul = float(
                    self.get_parameter('lookahead_dist_multiplier').value)
                lookahead_floor = float(
                    self.get_parameter('lookahead_dist_floor').value)
                v_eff = max(self.qcar2_measurred_speed, 0.05)
                lookahead_dist = max(v_eff * lookahead_mul, lookahead_floor)

                # 2026-05-27: SKIP CLUSTERED WAYPOINTS in one tick. Previous
                # behavior advanced wpi by 1 per tick — at intersections
                # where the planner clusters waypoints 5-10 cm apart, the
                # controller spent multiple ticks crawling through the
                # cluster while the car physically blew past it → loop.
                # Now: while the NEXT waypoint is also within advance_thresh,
                # keep incrementing. Single-tick jump-past-cluster.
                if dist < lookahead_dist:
                    if self.wpi < self.N - int(self.get_parameter('final_target_index_back').value):
                        self.wpi += skip_index
                        # Cluster-skip — runtime-toggleable so we can A/B
                        # against the simple single-step advance.
                        if bool(self.get_parameter('cluster_skip_enabled').value):
                            while self.wpi < self.N - int(self.get_parameter('final_target_index_back').value):
                                next_wp = self.wp[:, self.wpi]
                                next_wp_mod = (
                                    next_wp if self._wp_in_ros_frame
                                    else (next_wp + t) @ R_QLabs_ROS
                                )
                                next_dist = np.linalg.norm(
                                    [p[0] - next_wp_mod[0], p[1] - next_wp_mod[1]])
                                if next_dist >= lookahead_dist:
                                    break
                                self.wpi += 1

                self.wpi = np.clip(self.wpi, 0, self.N - int(self.get_parameter('final_target_index_back').value))

                # 2026-05-27: speed slowdown near end of path. Used to be
                # `wpi > N-100` (waypoint-count based) which is fragile on
                # paths that double back — geographically the car is
                # mid-map but PATHWISE 1m from end → unexpected slowdown
                # at "non-node" locations.
                # Now: trigger when dist_to_final < approach_slowdown_dist
                # (default 0.5 m). Euclidean distance from current pose to
                # the LAST waypoint, regardless of path shape.
                # NOTE: we use wp_final / dist_to_final computed below;
                # need to compute them early enough. Doing it here.
                wp_final_early = np.array(self.wp[:, -1])
                wp_final_mod_early = (
                    wp_final_early if self._wp_in_ros_frame
                    else (wp_final_early + t) @ R_QLabs_ROS
                )
                dist_to_final_early = np.linalg.norm(
                    [p[0] - wp_final_mod_early[0],
                     p[1] - wp_final_mod_early[1]])
                approach_slowdown_dist = float(
                    self.get_parameter('approach_slowdown_dist').value)
                if dist_to_final_early < approach_slowdown_dist:
                    speed_command = min(speed_command, 0.2)

                # 2026-05-27: stop condition rewritten to match Arturo
                # (origin/i-hate-gabriel nav_to_pose.py lines 558-564).
                # PREVIOUS bug:
                #   dist_to_final (to wp[-1]) < 0.40 triggered the stop.
                #   But wpi is clamped at N-5 so PP only targets wp[N-5],
                #   not wp[-1]. The car would reach wp[N-5], overshoot it
                #   (still has speed), and then PP would compute a ψ
                #   pointing BACKWARD (target now behind/sideways) →
                #   wobble while waiting for the LAST waypoint distance
                #   to drop below 0.40.
                # FIX: stop when (a) wpi has reached its clamp AND
                #      (b) we're within 0.4 m of the CURRENT TARGET (wp[wpi]).
                # That triggers the moment the car arrives at the
                # targetable waypoint — no overshoot, no wobble.
                # Keep dist_to_final calc for diagnostics only.
                wp_final = np.array(self.wp[:, -1])
                wp_final_mod = wp_final if self._wp_in_ros_frame else (wp_final + t) @ R_QLabs_ROS
                dist_to_final = np.linalg.norm([p[0] - wp_final_mod[0],
                                                p[1] - wp_final_mod[1]])
                final_stop_dist = float(
                    self.get_parameter('final_stop_dist').value)
                if (self.wpi >= self.N - int(self.get_parameter('final_target_index_back').value)
                        and dist < final_stop_dist):
                    speed_command        = 0.0
                    pp_delta             = 0.0
                    self.current_steering = 0.0
                    self.wp_prior        = self.wp
                    self.path_complete   = True

                # Live-tunable Kp/Kd (parameters, see __init__). Defaults
                # Kp=1.0, Kd=0.3. Adjust from Foxglove Parameters panel or:
                #   ros2 param set /path_follower kp_steering 1.2
                #   ros2 param set /path_follower kd_steering 0.4
                gyro_filtered = self.apply_filter('gyro', self.gyroscope[2], self.a1, self.b1)

                pp_delta_damped = np.clip(
                    self.kp_steering * pp_delta - self.kd_steering * gyro_filtered,
                    -self.max_steering_angle,
                    self.max_steering_angle)

                # ----- Publish controller diagnostics for Foxglove -----
                self._publish_controller_diagnostics(
                    dist=dist,
                    dist_to_final=dist_to_final,
                    psi=psi,
                    steering_cmd=float(pp_delta_damped),
                    speed_cmd=float(speed_command),
                )

                self.pp_delta_raw  = float(pp_delta_damped)
                self.stanley_delta = 0.0
                self.stanley_trust = 0.0

                steering = float(np.clip(pp_delta_damped,
                                         -self.max_steering_angle,
                                         self.max_steering_angle))
                self.current_steering = steering

                def f32(v):
                    m = Float32(); m.data = float(v); return m
                self.pub_pp_delta.publish(f32(self.pp_delta_raw))
                self.pub_stanley_delta.publish(f32(0.0))
                self.pub_blended_delta.publish(f32(self.current_steering))
                self.pub_blend_alpha.publish(f32(0.0))

                if int(self.t_plot * 5) != int((self.t_plot - self.dt) * 5):
                    self.get_logger().info(
                        f"wpi={self.wpi} dist={dist:.2f} "
                        f"pp={pp_delta_damped:.3f} "
                        f"steering={steering:.3f} "
                        f"v={speed_command:.2f}")

        except KeyboardInterrupt:
            speed_command = 0.0

        enable = float(
            self.path_execute_flag and
            self.motion_flag and
            not self.path_complete
        )

        self.nav_command(enable, speed_command)
        self.path_status()

    def nav_command(self, enable, speed_command):
        # CRITICAL: do NOT publish in non-autonomous modes. If we published
        # zeros here while manual_drive (or the manual-mode thread) was also
        # publishing, the bus would tug-of-war and stiffen the wheels.
        if self.control_mode != 'autonomous':
            return
        QCarCommands = Twist()
        # 2026-05-27: cos^N(steering) speed reduction made tunable via
        # steering_speed_exponent param. Default 0.0 (disabled, matches
        # Gabriel — constant speed through curves). Set to 2.0 to re-enable
        # the cos² conservative cornering speed cut.
        if self.steering_speed_exponent > 0.0:
            cos_steer = float(np.cos(self.current_steering))
            speed_factor = max(0.0, cos_steer) ** self.steering_speed_exponent
        else:
            speed_factor = 1.0
        QCarCommands.linear.x = enable * np.clip(
            speed_command * speed_factor, 0.0, 0.7)
        QCarCommands.angular.z = enable * self.current_steering
        self.publisher.publish(QCarCommands)

    def path_status(self):
        msg = Bool()
        msg.data = self.path_complete
        self.path_status_publisher.publish(msg)

    def tf_timer(self):
        """Look up map -> base_link, cache translation/yaw, publish /robot_pose.

        EKF correction calls used to live here. They were removed when
        ekf_fusor took over the EKF math; this method now only handles raw TF
        caching (still needed as a fallback for the path planner before
        /qcar2_pose_fused starts publishing) and the /robot_pose echo.
        """
        from_frame_rel = 'map'
        to_frame_rel = self.target_frame
        try:
            t = self.tf_buffer.lookup_transform(from_frame_rel, to_frame_rel, rclpy.time.Time())
            self.translation = t.transform.translation
            rotation = [t.transform.rotation.x, t.transform.rotation.y,
                        t.transform.rotation.z, t.transform.rotation.w]
            roll, pitch, self.yaw = R.from_quat(rotation).as_euler('xyz')

            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'map'
            pose_msg.pose.position.x = float(self.translation.x)
            pose_msg.pose.position.y = float(self.translation.y)
            pose_msg.pose.orientation.x = t.transform.rotation.x
            pose_msg.pose.orientation.y = t.transform.rotation.y
            pose_msg.pose.orientation.z = t.transform.rotation.z
            pose_msg.pose.orientation.w = t.transform.rotation.w
            self.robot_pose_publisher.publish(pose_msg)
        except TransformException as ex:
            self.get_logger().info(f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')

    def _switch_mode(self, new_mode):
        """Transition the controller into a new mode. Manages the manual
        keystroke thread + the autonomous publish gate (path_execute_flag).
        Safe to call from the parameter callback or from internal callbacks."""
        if new_mode == self.control_mode:
            return
        self.get_logger().info(
            f"control_mode: {self.control_mode} → {new_mode}"
        )
        # Tear down current mode
        if self.control_mode == 'manual':
            self._stop_manual_thread()

        # Establish new mode
        self.control_mode = new_mode
        if new_mode == 'autonomous':
            self.path_execute_flag = True
        elif new_mode == 'manual':
            self.path_execute_flag = False
            self._start_manual_thread()
        else:  # idle
            self.path_execute_flag = False

    # ─── Manual-mode keystroke thread (folded in from manual_drive) ─────

    def _start_manual_thread(self):
        if self._manual_thread is not None and self._manual_thread.is_alive():
            return
        if not _stdin_is_tty():
            self.get_logger().warn(
                "manual mode requested but stdin is not a TTY — keystrokes won't work. "
                "Run path_follower in a foreground terminal to use manual mode."
            )
            return
        self._manual_thread_stop.clear()
        self.manual_linear = 0.0
        self.manual_angular = 0.0
        _setup_terminal()
        self._manual_thread = threading.Thread(
            target=self._manual_keyboard_loop, daemon=True)
        self._manual_thread.start()
        self.get_logger().info(
            "Manual mode ON — WASD steers, space/x stops, +/- speed, [/] turn rate. "
            "Press 'q' to return to idle."
        )

    def _stop_manual_thread(self):
        if self._manual_thread is None:
            return
        self._manual_thread_stop.set()
        self._manual_thread.join(timeout=1.0)
        self._manual_thread = None
        _restore_terminal()
        # Publish a single stop so the car comes to rest if it was moving.
        self.manual_linear = 0.0
        self.manual_angular = 0.0
        stop = Twist()
        self.publisher.publish(stop)

    def _manual_publish(self):
        msg = Twist()
        msg.linear.x = float(self.manual_linear)
        msg.angular.z = float(self.manual_angular)
        self.publisher.publish(msg)

    def _manual_keyboard_loop(self):
        """Runs in a daemon thread while control_mode == 'manual'.
        Reads stdin in cbreak mode, maps WASD to Twist, publishes via
        self.publisher (the SAME publisher autonomous mode uses)."""
        while not self._manual_thread_stop.is_set():
            # Heartbeat publish so the nav2_qcar_command_convert 0.25 s
            # safety timeout doesn't zero throttle between keystrokes.
            key = _get_key(timeout_s=0.05).lower()
            if not key:
                self._manual_publish()
                continue

            if key == 'q':
                # Exit manual mode → switch back to idle (no driving).
                self.manual_linear = 0.0
                self.manual_angular = 0.0
                self._manual_publish()
                # Update the parameter so external observers see the change.
                self.set_parameters([
                    rclpy.parameter.Parameter(
                        'control_mode',
                        rclpy.parameter.Parameter.Type.STRING, 'idle')
                ])
                break
            elif key == 'w':
                self.manual_linear = self.manual_forward_speed
                self.manual_angular = 0.0
            elif key == 's':
                self.manual_linear = -self.manual_reverse_speed
                self.manual_angular = 0.0
            elif key == 'a':
                self.manual_angular = self.manual_turn_rate
            elif key == 'd':
                self.manual_angular = -self.manual_turn_rate
            elif key in (' ', 'x'):
                self.manual_linear = 0.0
                self.manual_angular = 0.0
            elif key in ('+', '='):
                self.manual_forward_speed = min(1.0, self.manual_forward_speed + self.manual_speed_step)
                self.manual_reverse_speed = min(1.0, self.manual_reverse_speed + self.manual_speed_step)
                self.get_logger().info(
                    f"forward={self.manual_forward_speed:.2f} reverse={self.manual_reverse_speed:.2f}"
                )
                continue
            elif key in ('-', '_'):
                self.manual_forward_speed = max(0.05, self.manual_forward_speed - self.manual_speed_step)
                self.manual_reverse_speed = max(0.05, self.manual_reverse_speed - self.manual_speed_step)
                self.get_logger().info(
                    f"forward={self.manual_forward_speed:.2f} reverse={self.manual_reverse_speed:.2f}"
                )
                continue
            elif key in ('{', '['):
                self.manual_turn_rate = max(0.10, self.manual_turn_rate - self.manual_speed_step)
                self.get_logger().info(f"turn_rate={self.manual_turn_rate:.2f}")
                continue
            elif key in ('}', ']'):
                self.manual_turn_rate = min(1.0, self.manual_turn_rate + self.manual_speed_step)
                self.get_logger().info(f"turn_rate={self.manual_turn_rate:.2f}")
                continue
            else:
                continue

            self._manual_publish()

    def _publish_controller_diagnostics(self, dist, dist_to_final, psi,
                                         steering_cmd, speed_cmd):
        """Publish the 9 controller-health topics for Foxglove plots.

        Designed to be called once per `path_planner` tick after a steering
        command has been computed. All scalars are Float32 with `.data`
        being the value — easy to add to a Plot panel.

        Topics published (see __init__ for the publisher instances):
          /nav/distance_to_waypoint     m, distance to current target waypoint
          /nav/distance_to_final         m, distance to last waypoint of path
          /nav/psi                       rad, angle from car nose to target
          /nav/steering_saturation_rate  0..1, fraction of last ~1s at the limit
          /nav/speed_cmd                 m/s, commanded forward speed
          /nav/yaw_rate_imu              rad/s, raw gyro (sanity-vs-controller)
          /nav/progress_rate             m/s, d(dist_to_final)/dt (negative = approaching)
          /nav/wpi                       float-encoded current waypoint index
          /nav/controller_mode           0=PP, 1=blended, 2=stopping, 3=complete
        """
        # Saturation: 1 if at the limit (within 1e-3 rad), 0 otherwise.
        sat = 1.0 if abs(abs(steering_cmd) - self.max_steering_angle) < 1e-3 else 0.0
        self._sat_window.append(sat)
        if len(self._sat_window) > self._sat_window_size:
            self._sat_window.pop(0)
        sat_rate = float(sum(self._sat_window) / len(self._sat_window))

        # Progress rate: change in dist_to_final per second.
        now = time.time()
        if self._prev_dist_final is not None and self._prev_progress_time is not None:
            dt = now - self._prev_progress_time
            if dt > 1e-3:
                progress_rate = (dist_to_final - self._prev_dist_final) / dt
            else:
                progress_rate = 0.0
        else:
            progress_rate = 0.0
        self._prev_dist_final = dist_to_final
        self._prev_progress_time = now

        # Controller mode encoding:
        if self.path_complete:
            mode = 3.0
        elif dist_to_final < 0.5:
            mode = 2.0
        elif self.stanley_trust > self.stanley_trust_min:
            mode = 1.0
        else:
            mode = 0.0

        # IMU yaw rate — read latest stored gyro (filtered version is used in
        # controller; here we publish RAW for diagnostic clarity).
        try:
            yaw_rate = float(self.gyroscope[2])
        except (IndexError, TypeError):
            yaw_rate = 0.0

        def f(v): m = Float32(); m.data = float(v); return m
        self.pub_dist_wp.publish(f(dist))
        self.pub_dist_final.publish(f(dist_to_final))
        self.pub_psi.publish(f(psi))
        self.pub_steering_saturation.publish(f(sat_rate))
        self.pub_speed_cmd.publish(f(speed_cmd))
        self.pub_yaw_rate.publish(f(yaw_rate))
        self.pub_progress_rate.publish(f(progress_rate))
        self.pub_wpi.publish(f(float(self.wpi)))
        self.pub_controller_mode.publish(f(mode))

    def fused_pose_cb(self, msg):
        """Cache the latest fused pose from ekf_fusor for the planner to read."""
        self.fused_pose_x = float(msg.pose.pose.position.x)
        self.fused_pose_y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.fused_pose_yaw = float(np.arctan2(siny_cosp, cosy_cosp))

    def _read_pose_for_planner(self):
        """Return (p, th) for pure-pursuit, preferring the EKF-fused pose.

        Priority:
          1. /qcar2_pose_fused (best — EKF-corrected, outlier-gated)
          2. Raw map->base_link TF (fallback while ekf_fusor warms up)
          3. Origin (fallback before any source has produced data)
        """
        if self.fused_pose_x is not None:
            return [self.fused_pose_x, self.fused_pose_y], self.fused_pose_yaw
        try:
            return [self.translation.x, self.translation.y], self.yaw
        except AttributeError:
            return [0.0, 0.0], 0.0

    def scopeDataTimer(self):
        if not getattr(self, 'pose_visualize_flag', False):
            return
        if not hasattr(self, 'steeringScope') or self.steeringScope is None:
            return
        if self.pose_visualize_flag:
            # Read fused pose (or zeros if not yet received) for visualization.
            p = [
                self.fused_pose_x if self.fused_pose_x is not None else 0.0,
                self.fused_pose_y if self.fused_pose_y is not None else 0.0,
                self.fused_pose_yaw if self.fused_pose_yaw is not None else 0.0,
            ]
            if self.t_plot > 200:
                self.t0 = time.time()
                for ax in self.steeringScope.axes:
                    ax.clean()
                MultiScope.refreshAll()
            try:
                x_ref = self.translation.x
                y_ref = self.translation.y
            except AttributeError:
                x_ref = 0
                y_ref = 0
            self.steeringScope.axes[0].sample(self.t_plot, [x_ref, p[0]])
            self.steeringScope.axes[1].sample(self.t_plot, [y_ref, p[1]])
            self.steeringScope.axes[2].sample(self.t_plot, [
                self.pp_delta_raw,
                0.0,
                self.current_steering,
                0.0
            ])
            fused_yaw_plot = self.fused_pose_yaw if self.fused_pose_yaw is not None else 0.0
            self.steeringScope.axes[3].sample(self.t_plot, [self.yaw, fused_yaw_plot])
            MultiScope.refreshAll()
        else:
            try:
                self.steeringScope.graphicsLayoutWidget.close()
            except AttributeError:
                pass


def main():
    rclpy.init()
    node = PathFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()