#! /usr/bin/env python3

from pal.utilities.math import wrap_to_pi

import time
import numpy as np
import scipy.signal as signal
from scipy.spatial.transform import Rotation as R
from pal.utilities.scope import MultiScope

from rclpy.duration import Duration
import rclpy
from geometry_msgs.msg import Point, PoseStamped
from rclpy.node import Node
from nav_msgs.msg import Path
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, JointState
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Bool, Float32
from visualization_msgs.msg import Marker

class QcarEKF:

    def __init__(self, x0, P0, Q, R):
        self.L = 0.257
        self.I = np.eye(3)
        self.xHat = x0
        self.P = P0
        self.Q = Q
        self.R = R
        self.C = np.eye(3)

    def f(self, X, u, dt):
        return X + dt * u[0] * np.array([
            [np.cos(X[2, 0])],
            [np.sin(X[2, 0])],
            [np.tan(u[1]) / self.L]
        ])

    def Jf(self, X, u, dt):
        return np.array([
            [1, 0, -dt * u[0] * np.sin(X[2, 0])],
            [0, 1,  dt * u[0] * np.cos(X[2, 0])],
            [0, 0, 1]
        ])

    def prediction(self, dt, u):
        F = self.Jf(self.xHat, u, dt)
        self.P = F @ self.P @ F.T + self.Q
        self.xHat = self.f(self.xHat, u, dt)
        self.xHat[2] = wrap_to_pi(self.xHat[2])

    def correction(self, y):
        H = self.C
        PH = self.P @ H.T
        S = H @ PH + self.R
        K = PH @ np.linalg.inv(S)
        z = y - H @ self.xHat
        if len(y) > 1:
            z[2] = wrap_to_pi(z[2])
        else:
            z = wrap_to_pi(z)
        self.xHat += K @ z
        self.xHat[2] = wrap_to_pi(self.xHat[2])
        self.P = (self.I - K @ H) @ self.P


class GyroKF:

    def __init__(self, x0, P0, Q, R):
        self.I = np.eye(2)
        self.xHat = x0
        self.P = P0
        self.Q = Q
        self.R = R
        self.A = np.array([[0, -1], [0, 0]])
        self.B = np.array([[1], [0]])
        self.C = np.array([[1, 0]])

    def prediction(self, dt, u):
        Ad = self.I + self.A * dt
        self.xHat = Ad @ self.xHat + dt * self.B * u
        self.P = Ad @ self.P @ Ad.T + self.Q

    def correction(self, y):
        PC = self.P @ self.C.T
        S = self.C @ PC + self.R
        K = PC @ np.linalg.inv(S)
        z = wrap_to_pi(y - self.C @ self.xHat)
        self.xHat += K @ z
        self.xHat[0] = wrap_to_pi(self.xHat[0])
        self.P = (self.I - K @ self.C) @ self.P


class PathFollower(Node):

    def __init__(self):
        super().__init__('path_follower')

        self.declare_parameter('node_values', [0, 8, 10])
        self.waypoints = list(self.get_parameter('node_values').get_parameter_value().integer_array_value)
        self.declare_parameter('route_frame', 'map')
        self.route_frame = self.get_parameter(
            'route_frame').get_parameter_value().string_value
        self.declare_parameter('assume_start_at_origin', False)
        self.assume_start_at_origin = bool(
            self.get_parameter('assume_start_at_origin').get_parameter_value().bool_value)
        self.declare_parameter('use_recorded_map', True)
        self.use_recorded_map = self.get_parameter(
            'use_recorded_map').get_parameter_value().bool_value
        self.declare_parameter('recorded_min_node_spacing_m', 0.10)
        self.recorded_min_node_spacing_m = float(
            self.get_parameter('recorded_min_node_spacing_m').get_parameter_value().double_value)
        self.declare_parameter('recorded_min_yaw_change_rad', 0.20)
        self.recorded_min_yaw_change_rad = float(
            self.get_parameter('recorded_min_yaw_change_rad').get_parameter_value().double_value)
        self.declare_parameter('generated_waypoint_spacing_m', 0.03)
        self.generated_waypoint_spacing_m = float(
            self.get_parameter('generated_waypoint_spacing_m').get_parameter_value().double_value)

        self.declare_parameter('desired_speed', [0.4])
        self.desired_speed = list(self.get_parameter('desired_speed').get_parameter_value().double_array_value)
        self.declare_parameter('steering_sign', 1.0)
        self.steering_sign = float(
            self.get_parameter('steering_sign').get_parameter_value().double_value)

        self.declare_parameter('visualize_pose', [True])
        self.pose_visualize_flag = True

        self.scale = 1.0

        self.declare_parameter('rotation_offset', [0.0])
        self.rotation_offset = list(self.get_parameter('rotation_offset').get_parameter_value().double_array_value)

        self.declare_parameter('translation_offset', [0.0, 0.0])
        self.translation_offset = list(self.get_parameter('translation_offset').get_parameter_value().double_array_value)

        self.declare_parameter('start_path', [True])
        self.start_path_requested = list(
            self.get_parameter('start_path').get_parameter_value().bool_array_value)[0]
        self.path_execute_flag = False

        self.declare_parameter('mission_pickup_xy',  [0.0, 0.0])
        self.declare_parameter('mission_dropoff_xy', [0.0, 0.0])
        self.declare_parameter('mission_enable',     [False])

        # Stanley blend is kept at 0 — pure pursuit only
        self.stanley_blend     = 0.0
        self.stanley_trust_min = 999.0

        self.declare_parameter('min_lookahead_m', 0.25)
        self.min_lookahead_m = float(
            self.get_parameter('min_lookahead_m').get_parameter_value().double_value)
        self.declare_parameter('lookahead_gain', 1.7)
        self.lookahead_gain = float(
            self.get_parameter('lookahead_gain').get_parameter_value().double_value)
        self.declare_parameter('progress_search_max_step', 30)
        self.progress_search_max_step = int(
            self.get_parameter('progress_search_max_step').get_parameter_value().integer_value)
        self.declare_parameter('finish_progress_window', 80)
        self.finish_progress_window = int(
            self.get_parameter('finish_progress_window').get_parameter_value().integer_value)
        self._stall_count = 0

        self.declare_parameter('curve_lookahead_min_m', 0.18)
        self.curve_lookahead_min_m = float(
            self.get_parameter('curve_lookahead_min_m').get_parameter_value().double_value)
        self.declare_parameter('curve_lookahead_max_m', 0.70)
        self.curve_lookahead_max_m = float(
            self.get_parameter('curve_lookahead_max_m').get_parameter_value().double_value)
        self.declare_parameter('curvature_lookahead_gain', 1.0)
        self.curvature_lookahead_gain = float(
            self.get_parameter('curvature_lookahead_gain').get_parameter_value().double_value)
        self.declare_parameter('min_curve_speed', 0.16)
        self.min_curve_speed = float(
            self.get_parameter('min_curve_speed').get_parameter_value().double_value)
        self.declare_parameter('curvature_speed_gain', 1.5)
        self.curvature_speed_gain = float(
            self.get_parameter('curvature_speed_gain').get_parameter_value().double_value)
        self.declare_parameter('lateral_error_slowdown_threshold_m', 0.12)
        self.lateral_error_slowdown_threshold_m = float(
            self.get_parameter('lateral_error_slowdown_threshold_m').get_parameter_value().double_value)

        self.add_on_set_parameters_callback(self.parameter_update_callback)

        self.target_frame = self.declare_parameter(
            'target_frame', 'base_link').get_parameter_value().string_value
        self.declare_parameter('control_hz', 30.0)
        control_hz = float(
            self.get_parameter('control_hz').get_parameter_value().double_value)
        control_hz = float(np.clip(control_hz, 5.0, 80.0))

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.dt = 1.0 / control_hz

        x0 = np.zeros((3, 1))
        P0 = np.eye(3)
        R_combined = np.diagflat([0.1, 0.1, 0.01])

        self.qcar2_ekf = QcarEKF(x0=x0, P0=P0,
                                  Q=np.diagflat([0.0001, 0.0001, 0.001]),
                                  R=R_combined)
        self.pose_ekf = np.zeros((3, 1))

        self.gyro_kf = GyroKF(x0=np.zeros((2, 1)), P0=np.eye(2),
                               Q=np.diagflat([0.01, 0.01]),
                               R=np.diagflat([.1]))

        self.yaw = 0
        self.cutoff_frequency_filter = min(15.0, 0.45 * control_hz)
        self.a1, self.b1 = self.filter_coefficients(self.cutoff_frequency_filter, self.dt)

        self.path_control_timer = self.create_timer(self.dt, self.path_planner)
        self.timer = self.create_timer(self.dt, self.tf_timer)
        self.translation = [0, 0, 0]
        self.rotation = [0, 0, 0]
        self._origin_xy = None
        self._wp_in_ros_frame = False
        self.wp = np.zeros((2, 0), dtype=float)
        self.N = 0
        self.wpi = 0
        self.wp_prior = []
        self.current_steering = 0

        self.stanley_delta = 0.0
        self.stanley_trust = 0.0
        self.pp_delta_raw  = 0.0

        self.publisher = self.create_publisher(Twist, '/cmd_vel_nav', 1)
        self.cyclic = False
        self.max_steering_angle = 0.6

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

        self._override_cmd   = None
        self._override_stamp = -999.0
        self.create_subscription(Twist, '/nav/override_cmd', self._override_cb, 1)

        self.path_publisher_topic  = self.create_publisher(Path, '/planned_path', 1)
        self.path_marker_publisher = self.create_publisher(Marker, '/planned_path_marker', 1)
        self.path_nodes_marker_publisher = self.create_publisher(Marker, '/planned_path_nodes', 1)
        self.target_waypoint_marker_publisher = self.create_publisher(Marker, '/planned_target_waypoint', 1)
        self.path_status_publisher = self.create_publisher(Bool, '/path_status', 1)
        self.robot_pose_publisher  = self.create_publisher(PoseStamped, '/robot_pose', 10)

        self.create_subscription(Path, '/cmd_waypoints', self._cmd_waypoints_cb, 1)

        self.pub_pp_delta      = self.create_publisher(Float32, '/nav/pp_delta',      2)
        self.pub_stanley_delta = self.create_publisher(Float32, '/nav/stanley_delta', 2)
        self.pub_blended_delta = self.create_publisher(Float32, '/nav/blended_delta', 2)
        self.pub_blend_alpha   = self.create_publisher(Float32, '/nav/blend_alpha',   2)

        self.t0 = time.time()
        self.t_plot = 0
        self.plot_visualized = False
        self.scopeTimer = self.create_timer(0.1, self.scopeDataTimer)

        self.get_logger().info(
            f'PathFollower ready | PURE PURSUIT BASELINE | '
            f'route_frame={self.route_frame} | '
            f'stanley_blend={self.stanley_blend} trust_min={self.stanley_trust_min}')

    def _cmd_waypoints_cb(self, msg: Path):
        if not msg.poses:
            self.get_logger().warn('/cmd_waypoints empty — ignoring')
            return
        xs = [p.pose.position.x for p in msg.poses]
        ys = [p.pose.position.y for p in msg.poses]
        self.wp = np.array([xs, ys], dtype=float)
        self.N               = self.wp.shape[1]
        self.wpi             = 0
        self.path_complete   = False
        self.path_execute_flag = self.start_path_requested
        self._wp_in_ros_frame = True
        self._stall_count = 0
        self.get_logger().warn(
            f'/cmd_waypoints overwrote self.wp raw={len(xs)} kept={self.N} '
            f'first=({xs[0]:.2f},{ys[0]:.2f}) last=({xs[-1]:.2f},{ys[-1]:.2f})')

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
                self.get_logger().warn(
                    'node_values updated, but path_follower is planner-driven and will wait for /cmd_waypoints')
            elif param.name == 'desired_speed' and param.type_ == param.Type.DOUBLE_ARRAY:
                self.desired_speed = list(param.value)
                self.get_logger().info('new desired speed...')
            elif param.name == 'steering_sign' and param.type_ in (param.Type.DOUBLE, param.Type.INTEGER):
                self.steering_sign = float(param.value)
                self.get_logger().warn(f'steering_sign updated: {self.steering_sign:.1f}')
            elif param.name == 'rotation_offset' and param.type_ == param.Type.DOUBLE_ARRAY:
                self.rotation_offset = list(param.value)
            elif param.name == 'translation_offset' and param.type_ == param.Type.DOUBLE_ARRAY:
                self.translation_offset = list(param.value)
            elif param.name == 'assume_start_at_origin' and param.type_ == param.Type.BOOL:
                self.assume_start_at_origin = bool(param.value)
                self._origin_xy = None
                self.get_logger().warn(
                    f'assume_start_at_origin={self.assume_start_at_origin}; origin will reset on next TF')
            elif param.name == 'recorded_min_node_spacing_m' and param.type_ == param.Type.DOUBLE:
                self.recorded_min_node_spacing_m = float(param.value)
            elif param.name == 'recorded_min_yaw_change_rad' and param.type_ == param.Type.DOUBLE:
                self.recorded_min_yaw_change_rad = float(param.value)
            elif param.name == 'generated_waypoint_spacing_m' and param.type_ == param.Type.DOUBLE:
                self.generated_waypoint_spacing_m = float(param.value)
                self.get_logger().info(
                    f'generated_waypoint_spacing_m updated: {self.generated_waypoint_spacing_m:.2f}m')
            elif param.name == 'start_path' and param.type_ == param.Type.BOOL_ARRAY:
                self.start_path_requested = list(param.value)[0]
                self.path_execute_flag = self.start_path_requested and self.N > 0 and not self.path_complete
                self.get_logger().info('path status changed!')
            elif param.name == 'min_lookahead_m' and param.type_ == param.Type.DOUBLE:
                self.min_lookahead_m = float(param.value)
                self.get_logger().info(f'min_lookahead_m updated: {self.min_lookahead_m:.3f}m')
            elif param.name == 'lookahead_gain' and param.type_ == param.Type.DOUBLE:
                self.lookahead_gain = float(param.value)
                self.get_logger().info(f'lookahead_gain updated: {self.lookahead_gain:.2f}')
            elif param.name == 'progress_search_max_step' and param.type_ == param.Type.INTEGER:
                self.progress_search_max_step = int(param.value)
                self.get_logger().info(
                    f'progress_search_max_step updated: {self.progress_search_max_step}')
            elif param.name == 'finish_progress_window' and param.type_ == param.Type.INTEGER:
                self.finish_progress_window = int(param.value)
                self.get_logger().info(f'finish_progress_window updated: {self.finish_progress_window}')
            elif param.name == 'curve_lookahead_min_m' and param.type_ == param.Type.DOUBLE:
                self.curve_lookahead_min_m = float(param.value)
                self.get_logger().info(f'curve_lookahead_min_m updated: {self.curve_lookahead_min_m:.3f}m')
            elif param.name == 'curve_lookahead_max_m' and param.type_ == param.Type.DOUBLE:
                self.curve_lookahead_max_m = float(param.value)
                self.get_logger().info(f'curve_lookahead_max_m updated: {self.curve_lookahead_max_m:.3f}m')
            elif param.name == 'curvature_lookahead_gain' and param.type_ == param.Type.DOUBLE:
                self.curvature_lookahead_gain = float(param.value)
                self.get_logger().info(f'curvature_lookahead_gain updated: {self.curvature_lookahead_gain:.2f}')
            elif param.name == 'min_curve_speed' and param.type_ == param.Type.DOUBLE:
                self.min_curve_speed = float(param.value)
                self.get_logger().info(f'min_curve_speed updated: {self.min_curve_speed:.3f}')
            elif param.name == 'curvature_speed_gain' and param.type_ == param.Type.DOUBLE:
                self.curvature_speed_gain = float(param.value)
                self.get_logger().info(f'curvature_speed_gain updated: {self.curvature_speed_gain:.2f}')
            elif param.name == 'lateral_error_slowdown_threshold_m' and param.type_ == param.Type.DOUBLE:
                self.lateral_error_slowdown_threshold_m = float(param.value)
                self.get_logger().info(f'lateral_error_slowdown_threshold_m updated: {self.lateral_error_slowdown_threshold_m:.3f}m')
            elif param.name == 'stanley_blend' or param.name == 'stanley_trust_min':
                self.get_logger().warn(f'{param.name} is hardcoded — ignoring')
            elif param.name == 'mission_pickup_xy' and param.type_ == param.Type.DOUBLE_ARRAY:
                self.get_logger().info(f'mission_pickup_xy updated: {list(param.value)}')
            elif param.name == 'mission_dropoff_xy' and param.type_ == param.Type.DOUBLE_ARRAY:
                self.get_logger().info(f'mission_dropoff_xy updated: {list(param.value)}')
            elif param.name == 'mission_enable' and param.type_ == param.Type.BOOL_ARRAY:
                self.get_logger().info(f'mission_enable updated: {list(param.value)}')
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

    def _override_cb(self, msg: Twist):
        self._override_cmd   = msg
        self._override_stamp = time.time()

    def _waypoint_in_route_frame(self, wp_xy):
        if self._wp_in_ros_frame:
            return np.array(wp_xy, dtype=float)

        angle_offset = self.rotation_offset[0]
        r_qlabs_ros = np.array([
            [np.cos(-angle_offset * np.pi / 180), -np.sin(-angle_offset * np.pi / 180)],
            [np.sin(-angle_offset * np.pi / 180),  np.cos(-angle_offset * np.pi / 180)]
        ])
        t = np.array([self.translation_offset[0], self.translation_offset[1]])
        return (np.array(wp_xy, dtype=float) + t) @ r_qlabs_ros

    def path_publisher(self):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.route_frame
        marker_msg = Marker()
        marker_msg.header = path_msg.header
        marker_msg.ns = 'planned_path'
        marker_msg.id = 0
        marker_msg.type = Marker.LINE_STRIP
        marker_msg.action = Marker.ADD
        marker_msg.scale.x = 0.12
        marker_msg.color.a = 1.0
        marker_msg.color.r = 1.0
        marker_msg.color.g = 0.1
        marker_msg.color.b = 1.0
        marker_msg.pose.orientation.w = 1.0
        nodes_marker = Marker()
        nodes_marker.header = path_msg.header
        nodes_marker.ns = 'planned_path_nodes'
        nodes_marker.id = 1
        nodes_marker.type = Marker.SPHERE_LIST
        nodes_marker.action = Marker.ADD
        nodes_marker.scale.x = 0.08
        nodes_marker.scale.y = 0.08
        nodes_marker.scale.z = 0.08
        nodes_marker.color.a = 1.0
        nodes_marker.color.r = 0.0
        nodes_marker.color.g = 0.9
        nodes_marker.color.b = 1.0
        nodes_marker.pose.orientation.w = 1.0
        if self.N <= 0:
            marker_msg.action = Marker.DELETE
            nodes_marker.action = Marker.DELETE
            self.path_publisher_topic.publish(path_msg)
            self.path_marker_publisher.publish(marker_msg)
            self.path_nodes_marker_publisher.publish(nodes_marker)
            return
        for i in range(self.N):
            pose = PoseStamped()
            wp_1_mod = self._waypoint_in_route_frame(self.wp[:, i])
            pose.header = path_msg.header
            pose.pose.position.x = wp_1_mod[0]
            pose.pose.position.y = wp_1_mod[1]
            path_msg.poses.append(pose)
            point = Point()
            point.x = float(wp_1_mod[0])
            point.y = float(wp_1_mod[1])
            point.z = 0.0
            marker_msg.points.append(point)
            nodes_marker.points.append(point)
        self.path_publisher_topic.publish(path_msg)
        self.path_marker_publisher.publish(marker_msg)
        self.path_nodes_marker_publisher.publish(nodes_marker)

    def _publish_target_waypoint_marker(self, wp_xy):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.route_frame
        marker.ns = 'planned_target_waypoint'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.16
        marker.scale.y = 0.16
        marker.scale.z = 0.16
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.85
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = float(wp_xy[0])
        marker.pose.position.y = float(wp_xy[1])
        marker.pose.position.z = 0.0
        self.target_waypoint_marker_publisher.publish(marker)

    def _clear_target_waypoint_marker(self):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.route_frame
        marker.ns = 'planned_target_waypoint'
        marker.id = 0
        marker.action = Marker.DELETE
        self.target_waypoint_marker_publisher.publish(marker)

    def _update_progress_index(self, robot_xy, max_step=None):
        if self.N <= 1:
            return

        robot_xy = np.array(robot_xy, dtype=float)
        start_idx = int(np.clip(self.wpi, 0, self.N - 1))
        if max_step is None:
            max_step = self.progress_search_max_step
        stop_idx = min(self.N, start_idx + int(max_step) + 1)

        best_idx = start_idx
        best_dist = float('inf')
        for idx in range(start_idx, stop_idx):
            wp_route = self._waypoint_in_route_frame(self.wp[:, idx])
            dist = float(np.linalg.norm(wp_route - robot_xy))
            if dist < best_dist:
                best_idx = idx
                best_dist = dist

        if best_idx > self.wpi:
            self.wpi = best_idx

    def _select_lookahead_index(self, robot_xy, lookahead_dist):
        if self.N <= 0:
            return 0

        lookahead_dist = max(float(lookahead_dist), 0.05)

        target_idx = int(np.clip(self.wpi, 0, self.N - 1))
        traveled = 0.0
        prev_wp = self._waypoint_in_route_frame(self.wp[:, target_idx])
        while target_idx < self.N - 1 and traveled < lookahead_dist:
            target_idx += 1
            next_wp = self._waypoint_in_route_frame(self.wp[:, target_idx])
            traveled += float(np.linalg.norm(next_wp - prev_wp))
            prev_wp = next_wp

        return target_idx

    def _estimate_local_curvature(self, progress_idx, look_pts=20):
        """Heading-change-over-arc-length curvature estimate at progress_idx. Returns 1/m."""
        if self.N < 3:
            return 0.0
        i0 = int(np.clip(progress_idx,              0, self.N - 1))
        i1 = int(np.clip(progress_idx + look_pts // 2, 0, self.N - 1))
        i2 = int(np.clip(progress_idx + look_pts,    0, self.N - 1))
        if i0 == i1 or i1 == i2:
            return 0.0
        p0 = self._waypoint_in_route_frame(self.wp[:, i0])
        p1 = self._waypoint_in_route_frame(self.wp[:, i1])
        p2 = self._waypoint_in_route_frame(self.wp[:, i2])
        h01 = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
        h12 = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        arc_len = float(np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1))
        if arc_len < 0.01:
            return 0.0
        return abs(float(wrap_to_pi(h12 - h01))) / arc_len

    def _compute_cross_track_error(self, robot_xy, progress_idx):
        """Signed cross-track error to the path segment at progress_idx. Positive = left of path."""
        if self.N < 2:
            return 0.0
        i0 = int(np.clip(progress_idx, 0, self.N - 2))
        i1 = i0 + 1
        p0 = self._waypoint_in_route_frame(self.wp[:, i0])
        p1 = self._waypoint_in_route_frame(self.wp[:, i1])
        seg = p1 - p0
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-6:
            return 0.0
        to_robot = robot_xy - p0
        return float(seg[0] * to_robot[1] - seg[1] * to_robot[0]) / seg_len

    def path_planner(self):
        max_speed = 1.5
        enable = 1
        speed_command = self.desired_speed[0]

        self.t_plot = time.time() - self.t0
        self.ekf_filter_timer()

        self.path_publisher()

        # External override: trip_planner reverse mode takes full control
        if self._override_cmd is not None and time.time() - self._override_stamp < 0.3:
            self.publisher.publish(self._override_cmd)
            self.path_status()
            return

        if self.N <= 0:
            self.current_steering = 0.0
            self._clear_target_waypoint_marker()
            self.nav_command(0.0, 0.0)
            self.path_status()
            return

        try:
            if not self.path_complete:
                L = 0.256

                th = self.qcar2_ekf.xHat[2, 0]
                p = [self.qcar2_ekf.xHat[0, 0], self.qcar2_ekf.xHat[1, 0]]

                try:
                    p = [self.translation.x, self.translation.y]
                    th = self.yaw
                except AttributeError:
                    p = [0, 0]
                    th = 0

                robot_xy = np.array(p, dtype=float)

                v_eff = max(self.qcar2_measurred_speed, 0.05)

                prev_wpi = self.wpi
                self._update_progress_index(robot_xy)

                # Adaptive lookahead: velocity-based but clamped by local path curvature
                curvature = self._estimate_local_curvature(self.wpi)
                v_based = v_eff * self.lookahead_gain
                curv_limit = self.curve_lookahead_max_m / (1.0 + self.curvature_lookahead_gain * curvature)
                lookahead_dist = float(np.clip(
                    min(v_based, curv_limit),
                    self.curve_lookahead_min_m,
                    self.curve_lookahead_max_m))

                # Curvature-based speed: slow down into tight corners
                curv_speed = speed_command / (1.0 + self.curvature_speed_gain * curvature)
                speed_command = float(np.clip(curv_speed, self.min_curve_speed, speed_command))

                # Cross-track error: reduce lookahead and speed when off-path
                cte = self._compute_cross_track_error(robot_xy, self.wpi)
                abs_cte = abs(cte)
                if abs_cte > self.lateral_error_slowdown_threshold_m:
                    excess = abs_cte - self.lateral_error_slowdown_threshold_m
                    cte_factor = max(0.6, 1.0 - excess * 2.0)
                    speed_command = max(self.min_curve_speed, speed_command * cte_factor)
                    lookahead_dist = max(self.curve_lookahead_min_m, lookahead_dist * cte_factor)

                target_idx = self._select_lookahead_index(robot_xy, lookahead_dist)
                Rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                for _ in range(40):
                    wp_1 = np.array(self.wp[:, target_idx])
                    wp_1_mod = self._waypoint_in_route_frame(wp_1)
                    v = [wp_1_mod[0] - p[0], wp_1_mod[1] - p[1]]
                    v_car = v @ Rot
                    if v_car[0] >= 0.02 or target_idx >= self.N - 1:
                        break
                    target_idx += 1
                self._publish_target_waypoint_marker(wp_1_mod)

                WaypointDist = max(np.linalg.norm(v_car), 0.05)
                psi = np.arctan2(v_car[1], v_car[0])

                pp_delta = np.arctan2(2 * L * np.sin(psi), WaypointDist)

                dist = np.linalg.norm([p[0] - wp_1_mod[0], p[1] - wp_1_mod[1]])

                if target_idx > self.N - 100:
                    speed_command = min(speed_command, 0.2)

                wp_final_mod = self._waypoint_in_route_frame(self.wp[:, -1])
                dist_to_final = float(np.linalg.norm(np.array(p) - wp_final_mod))
                finish_window = max(1, min(self.finish_progress_window, self.N - 1))
                near_route_end = self.wpi >= self.N - finish_window

                if near_route_end and dist_to_final < 0.25:
                    speed_command        = 0.0
                    pp_delta             = 0.0
                    self.current_steering = 0.0
                    self.wp_prior        = self.wp
                    self.path_complete   = True

                Kp_steering = 1.1
                kd_steering = 5
                gyro_filtered = self.apply_filter('gyro', self.gyroscope[2], self.a1, self.b1)
                pp_delta_damped = np.clip(
                    Kp_steering * pp_delta - gyro_filtered * np.pi / 180 * kd_steering,
                    -self.max_steering_angle,
                    self.max_steering_angle)

                self.pp_delta_raw  = float(pp_delta_damped)
                self.stanley_delta = 0.0
                self.stanley_trust = 0.0

                steering = float(np.clip(pp_delta_damped,
                                         -self.max_steering_angle,
                                         self.max_steering_angle))
                self.current_steering = steering

                if abs(pp_delta_damped) >= self.max_steering_angle - 0.01 and self.wpi == prev_wpi:
                    self._stall_count += 1
                    if self._stall_count % 80 == 0:
                        self.get_logger().warn(
                            f'Steering saturated ({pp_delta_damped:.2f} rad) '
                            f'progress_idx stuck at {self.wpi} '
                            f'lookahead={lookahead_dist:.2f}m target={target_idx}')
                else:
                    self._stall_count = 0

                def f32(v):
                    m = Float32(); m.data = float(v); return m

                self.pub_pp_delta.publish(f32(self.pp_delta_raw))
                self.pub_stanley_delta.publish(f32(0.0))
                self.pub_blended_delta.publish(f32(self.current_steering))
                self.pub_blend_alpha.publish(f32(0.0))

                if int(self.t_plot * 5) != int((self.t_plot - self.dt) * 5):
                    self.get_logger().info(
                        f"progress_idx={self.wpi} target_idx={target_idx} "
                        f"gap={target_idx - self.wpi} "
                        f"lookahead={lookahead_dist:.3f}m "
                        f"curv={curvature:.2f}/m "
                        f"cte={cte:.3f}m "
                        f"v={speed_command:.2f} "
                        f"steering={steering:.3f}")

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
        QCarCommands = Twist()
        QCarCommands.linear.x = enable * np.clip(
            speed_command * np.power(np.cos(self.current_steering), 2), 0.0, 0.7)
        QCarCommands.angular.z = enable * self.steering_sign * self.current_steering
        self.publisher.publish(QCarCommands)

    def path_status(self):
        msg = Bool()
        msg.data = self.path_complete
        self.path_status_publisher.publish(msg)

    def tf_timer(self):
        from_frame_rel = self.route_frame
        to_frame_rel = self.target_frame
        try:
            t = self.tf_buffer.lookup_transform(from_frame_rel, to_frame_rel, rclpy.time.Time())
            self.translation = t.transform.translation
            if self.assume_start_at_origin:
                raw_xy = np.array([float(self.translation.x), float(self.translation.y)], dtype=float)
                if self._origin_xy is None:
                    self._origin_xy = raw_xy
                    self.get_logger().warn(
                        f'Origin locked to first TF pose: raw=({raw_xy[0]:.3f},{raw_xy[1]:.3f}) '
                        f'-> local node 0 at (0.000,0.000)')
                local_xy = raw_xy - self._origin_xy
                self.translation.x = float(local_xy[0])
                self.translation.y = float(local_xy[1])
            rotation = [t.transform.rotation.x, t.transform.rotation.y,
                        t.transform.rotation.z, t.transform.rotation.w]
            roll, pitch, self.yaw = R.from_quat(rotation).as_euler('xyz')
            self.gyro_kf.correction(self.yaw)
            y = np.array([[self.translation.x], [self.translation.y], [self.gyro_kf.xHat[0, 0]]])
            self.qcar2_ekf.correction(y)

            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = self.route_frame
            pose_msg.pose.position.x = float(self.translation.x)
            pose_msg.pose.position.y = float(self.translation.y)
            pose_msg.pose.orientation.x = t.transform.rotation.x
            pose_msg.pose.orientation.y = t.transform.rotation.y
            pose_msg.pose.orientation.z = t.transform.rotation.z
            pose_msg.pose.orientation.w = t.transform.rotation.w
            self.robot_pose_publisher.publish(pose_msg)
        except TransformException as ex:
            self.get_logger().info(f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')

    def ekf_filter_timer(self):
        speed = self.qcar2_measurred_speed
        delta = self.current_steering
        self.qcar2_ekf.prediction(self.dt, [speed, delta])
        try:
            th_gyro = self.gyroscope[2]
        except AttributeError:
            th_gyro = 0
        self.gyro_kf.prediction(self.dt, th_gyro)

    def scopeDataTimer(self):
        if not getattr(self, 'pose_visualize_flag', False):
            return
        if not hasattr(self, 'steeringScope') or self.steeringScope is None:
            return
        if self.pose_visualize_flag:
            p = [self.qcar2_ekf.xHat[0, 0], self.qcar2_ekf.xHat[1, 0], self.qcar2_ekf.xHat[2, 0]]
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
            self.steeringScope.axes[3].sample(self.t_plot, [self.yaw, self.qcar2_ekf.xHat[2, 0]])
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
