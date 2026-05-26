#! /usr/bin/env python3

from hal.products.mats import SDCSRoadMap
from pal.utilities.math import wrap_to_pi

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
        self.declare_parameter('kd_steering', 0.20)
        self.kp_steering = float(self.get_parameter('kp_steering').value)
        self.kd_steering = float(self.get_parameter('kd_steering').value)

        # Lane Stanley is a bounded visual bias on top of the existing
        # pure-pursuit + gyro-damped steering. This node remains the only
        # /cmd_vel_nav publisher.
        self.declare_parameter('lane_bias_gain', 0.70)
        self.declare_parameter('lane_bias_max', 0.10)
        self.declare_parameter('lane_bias_turn_start', 0.18)
        self.declare_parameter('lane_bias_turn_end', 0.38)
        self.declare_parameter('stanley_trust_min', 0.35)
        self.declare_parameter('stanley_timeout_sec', 0.40)
        self.lane_bias_gain = float(self.get_parameter('lane_bias_gain').value)
        self.lane_bias_max = float(self.get_parameter('lane_bias_max').value)
        self.lane_bias_turn_start = float(self.get_parameter('lane_bias_turn_start').value)
        self.lane_bias_turn_end = float(self.get_parameter('lane_bias_turn_end').value)
        self.stanley_trust_min = float(self.get_parameter('stanley_trust_min').value)
        self.stanley_timeout_sec = float(self.get_parameter('stanley_timeout_sec').value)

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
        self.wp = SDCSRoadMap().generate_path(self.waypoints) * self.scale
        # -------------ADDED N_1
        self.auto_align_start = True
        self.auto_aligned = False
        #----------
        self.N = len(self.wp[0, :])
        self.wpi = 0
        self._path_start_initialized = False
        self.start_gate_radius = 0.30
        self._last_auto_align_wait_log = 0.0
        self._last_start_wait_log = 0.0
        self.wp_prior = []
        self.current_steering = 0
        self._wp_in_ros_frame = False

        self.stanley_delta = 0.0
        self.stanley_trust = 0.0
        self.stanley_last_t = 0.0
        self.pp_delta_raw  = 0.0
        self.lane_bias_last = 0.0
        self.lane_bias_weight = 0.0

        self.publisher = self.create_publisher(Twist, '/cmd_vel_nav', 1)
        self.cyclic = False
        # ±0.52 rad = ±30°, max steering angle per QCar 2 hardware spec (Table 11).
        self.max_steering_angle = 0.52

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
        self.create_subscription(Float32, '/lane_keeping/delta', self._stanley_delta_cb, 5)
        self.create_subscription(Float32, '/lane_keeping/trust', self._stanley_trust_cb, 5)

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
            f'PathFollower ready | PP+PD with bounded lane bias | '
            f'lane_bias_gain={self.lane_bias_gain} '
            f'lane_bias_max={self.lane_bias_max} '
            f'trust_min={self.stanley_trust_min}')

    def _cmd_waypoints_cb(self, msg: Path):
        if not msg.poses:
            self.get_logger().warn('/cmd_waypoints empty — ignoring')
            return
        xs = [p.pose.position.x for p in msg.poses]
        ys = [p.pose.position.y for p in msg.poses]
        self.wp              = np.array([xs, ys])
        self.N               = self.wp.shape[1]
        self.wpi             = 0
        self._path_start_initialized = False
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
        self.stanley_last_t = time.time()

    def _lane_turn_scale(self, pp_delta: float) -> float:
        start = max(0.0, self.lane_bias_turn_start)
        end = max(start + 1e-3, self.lane_bias_turn_end)
        abs_pp = abs(pp_delta)
        if abs_pp <= start:
            return 1.0
        if abs_pp >= end:
            return 0.0
        return float(1.0 - (abs_pp - start) / (end - start))

    def _apply_lane_bias(self, pp_delta: float) -> float:
        self.pp_delta_raw = pp_delta

        stanley_is_fresh = (time.time() - self.stanley_last_t) <= self.stanley_timeout_sec
        if not stanley_is_fresh or self.stanley_trust < self.stanley_trust_min:
            weight = 0.0
        else:
            trust_range = max(1e-6, 1.0 - self.stanley_trust_min)
            trust_scale = np.clip((self.stanley_trust - self.stanley_trust_min) / trust_range, 0.0, 1.0)
            turn_scale = self._lane_turn_scale(pp_delta)
            weight = float(np.clip(self.lane_bias_gain * trust_scale * turn_scale, 0.0, 1.0))

        lane_bias = float(np.clip(self.stanley_delta, -self.lane_bias_max, self.lane_bias_max))
        applied_bias = weight * lane_bias
        steering = float(np.clip(
            pp_delta + applied_bias,
            -self.max_steering_angle,
            self.max_steering_angle))

        self.lane_bias_last = applied_bias
        self.lane_bias_weight = weight

        def f32(v):
            m = Float32(); m.data = float(v); return m

        self.pub_pp_delta.publish(f32(pp_delta))
        self.pub_stanley_delta.publish(f32(self.stanley_delta))
        self.pub_blended_delta.publish(f32(steering))
        self.pub_blend_alpha.publish(f32(weight))

        return steering

    def parameter_update_callback(self, params):
        for param in params:
            if param.name == 'node_values' and param.type_ == param.Type.INTEGER_ARRAY:
                self.waypoints = list(param.value)
                self.wp = SDCSRoadMap().generate_path(self.waypoints) * 0.975
                self.N = len(self.wp[0, :])
                self.wpi = 0
                self._path_start_initialized = False
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
            elif param.name == 'lane_bias_gain' and param.type_ == param.Type.DOUBLE:
                self.lane_bias_gain = float(np.clip(param.value, 0.0, 1.0))
                self.get_logger().info(f'lane_bias_gain -> {self.lane_bias_gain:.3f}')
            elif param.name == 'lane_bias_max' and param.type_ == param.Type.DOUBLE:
                self.lane_bias_max = float(np.clip(param.value, 0.0, self.max_steering_angle))
                self.get_logger().info(f'lane_bias_max -> {self.lane_bias_max:.3f}')
            elif param.name == 'lane_bias_turn_start' and param.type_ == param.Type.DOUBLE:
                self.lane_bias_turn_start = float(np.clip(param.value, 0.0, self.max_steering_angle))
                self.get_logger().info(f'lane_bias_turn_start -> {self.lane_bias_turn_start:.3f}')
            elif param.name == 'lane_bias_turn_end' and param.type_ == param.Type.DOUBLE:
                self.lane_bias_turn_end = float(np.clip(param.value, 0.0, self.max_steering_angle))
                self.get_logger().info(f'lane_bias_turn_end -> {self.lane_bias_turn_end:.3f}')
            elif param.name == 'stanley_trust_min' and param.type_ == param.Type.DOUBLE:
                self.stanley_trust_min = float(np.clip(param.value, 0.0, 1.0))
                self.get_logger().info(f'stanley_trust_min -> {self.stanley_trust_min:.3f}')
            elif param.name == 'stanley_timeout_sec' and param.type_ == param.Type.DOUBLE:
                self.stanley_timeout_sec = float(np.clip(param.value, 0.05, 2.0))
                self.get_logger().info(f'stanley_timeout_sec -> {self.stanley_timeout_sec:.3f}')
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
        self.qcar2_measurred_speed = (msg.velocity[0] / (720.0 * 4.0)) * ((13.0 * 19.0) / (70.0 * 30.0)) * (2.0 * np.pi) * 0.033

    def imu_callback(self, msg):
        self.gyroscope = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]


    def _roadmap_to_map_rotation(self):
        angle_offset = self.rotation_offset[0]
        return np.array([
            [np.cos(-angle_offset * np.pi / 180), -np.sin(-angle_offset * np.pi / 180)],
            [np.sin(-angle_offset * np.pi / 180),  np.cos(-angle_offset * np.pi / 180)]
        ])

    def _waypoints_in_map_frame(self):
        if self._wp_in_ros_frame:
            return np.array(self.wp)

        t = np.array([self.translation_offset[0], self.translation_offset[1]])
        return ((self.wp.T + t) @ self._roadmap_to_map_rotation()).T

    def _warn_once_per_second(self, attr_name, message):
        now = time.time()
        if now - getattr(self, attr_name, 0.0) >= 1.0:
            self.get_logger().warn(message)
            setattr(self, attr_name, now)

    def _initialize_path_start_from_current_pose(self):
        if self._path_start_initialized:
            return True

        if self.fused_pose_x is None or self.fused_pose_y is None:
            self._warn_once_per_second(
                '_last_start_wait_log',
                'Waiting for /qcar2_pose_fused before starting path index')
            return False

        if self.N <= 0:
            return False

        current_xy = np.array([self.fused_pose_x, self.fused_pose_y])
        wp_map = self._waypoints_in_map_frame()
        deltas = wp_map.T - current_xy
        dists = np.linalg.norm(deltas, axis=1)
        nearest_i = int(np.argmin(dists))
        nearest_dist = float(dists[nearest_i])

        self.wpi = int(np.clip(nearest_i, 0, self.N - 1))
        self._path_start_initialized = True

        if nearest_dist > self.start_gate_radius:
            self.get_logger().warn(
                f'Starting path from nearest waypoint {self.wpi}, '
                f'but EKF pose is {nearest_dist:.2f}m away')
        else:
            self.get_logger().info(
                f'Starting path from waypoint {self.wpi} '
                f'(EKF distance {nearest_dist:.2f}m)')

        return True


#----------------Change N2 auto_align_roadmap_to_current_pose----------------
    def auto_align_roadmap_to_current_pose(self):
        """
        Align SDCSRoadMap coordinates to the current Cartographer map frame.

        It assumes:
        - The QCar is physically placed at the first waypoint of the current route.
        - /qcar2_pose_fused is already available.
        - self.wp has shape (2, N), where row 0 = x and row 1 = y.
        """

        try:
            if self.fused_pose_x is None or self.fused_pose_y is None or self.fused_pose_yaw is None:
                self._warn_once_per_second(
                    '_last_auto_align_wait_log',
                    'Roadmap auto-align waiting for /qcar2_pose_fused')
                return False

            current_xy = np.array([
                self.fused_pose_x,
                self.fused_pose_y
            ])
            current_yaw = self.fused_pose_yaw

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
            self.get_logger().warn(f"Roadmap auto-align failed using /qcar2_pose_fused: {e}")
            return False
    #----------------End of auto_align_roadmap_to_current_pose----------------N2

    def path_publisher(self):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        for i in range(self.wpi):
            if i >= self.N:
                i = self.N - 1
            pose = PoseStamped()
            R_QLabs_ROS = self._roadmap_to_map_rotation()
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
        if self.auto_align_start and not self.auto_aligned and not self._wp_in_ros_frame:
            self.auto_aligned = self.auto_align_roadmap_to_current_pose()
            if not self.auto_aligned:
                return

        #--END OF CHANGE N3---------------------

        if not self._initialize_path_start_from_current_pose():
            return

        if round(self.t_plot) % 2 == 0:
            self.path_publisher()

        try:
            if not self.path_complete:
                wp_1 = np.array(self.wp[:, self.wpi])

                R_QLabs_ROS = self._roadmap_to_map_rotation()
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

                WaypointDist = max(np.linalg.norm(v_car), 0.05)
                psi = np.arctan2(v_car[1], v_car[0])

                pp_delta = np.arctan2(2 * L * np.sin(psi), WaypointDist)

                dist = np.linalg.norm([p[0] - wp_1_mod[0], p[1] - wp_1_mod[1]])

                v_eff = max(self.qcar2_measurred_speed, 0.05)
                lookahead_dist = max(v_eff * 1.7, 0.30)

                if dist < lookahead_dist:
                    if self.wpi < self.N - 1:
                        self.wpi += skip_index

                self.wpi = np.clip(self.wpi, 0, self.N - 1)

                if self.wpi > self.N - 100:
                    speed_command = min(speed_command, 0.2)

                wp_final = np.array(self.wp[:, -1])
                wp_final_mod = wp_final if self._wp_in_ros_frame else (wp_final + t) @ R_QLabs_ROS
                dist_to_final = np.linalg.norm([p[0] - wp_final_mod[0],
                                                p[1] - wp_final_mod[1]])
                if dist_to_final < 0.25:
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

                steering = self._apply_lane_bias(float(pp_delta_damped))
                self.current_steering = steering

                # ----- Publish controller diagnostics for Foxglove -----
                self._publish_controller_diagnostics(
                    dist=dist,
                    dist_to_final=dist_to_final,
                    psi=psi,
                    steering_cmd=float(steering),
                    speed_cmd=float(speed_command),
                )

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
        QCarCommands.linear.x = enable * np.clip(
            speed_command * np.power(np.cos(self.current_steering), 2), 0.0, 0.7)
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
        elif (
            self.lane_bias_weight > 1e-3
        ):
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
