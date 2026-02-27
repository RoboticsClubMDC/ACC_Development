#! /usr/bin/env python3
"""
nav_to_pose — Stanley path-tracking controller for QCar2
=========================================================

v3: Replaces pure-pursuit with a **Stanley controller** which naturally damps
heading oscillation.  Adds **curvature feedforward** so the car leads into
corners, and **adaptive speed** that slows in tight curves.

Algorithms / filters / math used
---------------------------------
1. **Stanley controller** (Stanford, Thrun 2006)
       δ = ψ_error + arctan(k_e · e_cte / (v + k_soft))
   Heading-error term provides natural damping; cross-track term provides
   lateral convergence.

2. **Path curvature feedforward**
       δ_ff = arctan(L · κ)
   Pre-steers into upcoming curves using the discrete curvature of the path.

3. **Extended Kalman Filter (EKF)** for fused pose estimation
   (bicycle motion model + SLAM/cartographer correction).

4. **Gyroscope Kalman Filter** for heading estimation
   (IMU angular velocity + SLAM heading correction).

5. **2nd-order Butterworth low-pass filter** (15 Hz cut-off) on gyroscope.

6. **Steering rate-limiter** (actuator constraint, 3.0 rad/s max slew).

7. **Curvature-adaptive speed control**
       v_cmd = v_desired · (1 − α · |κ|) , clamped above v_min.
"""

# Quanser specific packages
from hal.products.mats import SDCSRoadMap
from pal.utilities.math import wrap_to_pi

# Generic python packages
import time
import numpy as np
import scipy.signal as signal
from scipy.spatial.transform import Rotation as R
from pal.utilities.scope import MultiScope

# ROS specific packages
from rclpy.duration import Duration
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from nav_msgs.msg import Path
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Imu, JointState
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Bool


# ===================================================================
#  State estimation (unchanged from original)
# ===================================================================

class QcarEKF:
    """Extended Kalman Filter for QCar2 pose [x, y, θ]."""

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
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        z = y - H @ self.xHat
        if len(y) > 1:
            z[2] = wrap_to_pi(z[2])
        else:
            z = wrap_to_pi(z)
        self.xHat += K @ z
        self.xHat[2] = wrap_to_pi(self.xHat[2])
        self.P = (self.I - K @ H) @ self.P


class GyroKF:
    """Kalman Filter fusing gyroscope heading with SLAM heading."""

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
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)
        z = wrap_to_pi(y - self.C @ self.xHat)
        self.xHat += K @ z
        self.xHat[0] = wrap_to_pi(self.xHat[0])
        self.P = (self.I - K @ self.C) @ self.P


# ===================================================================
#  Path follower using Stanley controller
# ===================================================================

class PathFollower(Node):

    def __init__(self):
        super().__init__('path_follower')

        # --------------- ROS parameters ---------------
        self.declare_parameter('node_values', [0, 8, 10, 14, 20, 10])
        self.waypoints = list(self.get_parameter("node_values").get_parameter_value().integer_array_value)

        self.declare_parameter('desired_speed', [0.4])
        self.desired_speed = list(self.get_parameter("desired_speed").get_parameter_value().double_array_value)

        self.declare_parameter('visualize_pose', [False])
        self.pose_visualize_flag = list(self.get_parameter("visualize_pose").get_parameter_value().bool_array_value)[0]

        self.scale = 1.0

        self.declare_parameter('rotation_offset', [90.0])
        self.rotation_offset = list(self.get_parameter("rotation_offset").get_parameter_value().double_array_value)

        self.declare_parameter('translation_offset', [0.0, -0.125])
        self.translation_offset = list(self.get_parameter("translation_offset").get_parameter_value().double_array_value)

        self.declare_parameter('start_path', [False])
        self.path_execute_flag = list(self.get_parameter("start_path").get_parameter_value().bool_array_value)[0]

        self.add_on_set_parameters_callback(self.parameter_update_callback)

        self.target_frame = self.declare_parameter(
            'target_frame', 'base_link').get_parameter_value().string_value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --------------- Control parameters ---------------
        self.dt = 1 / 80
        self.L = 0.256                  # wheelbase [m]
        self.max_steering_angle = 0.6   # [rad]
        self.max_steer_rate = 3.0       # [rad/s] — actuator slew limit

        # Stanley gains
        self.k_e = 2.5                  # cross-track gain
        self.k_soft = 0.1              # softening constant [m/s]
        self.k_heading = 1.0           # heading error gain (≤1 to soften)

        # Curvature feedforward
        self.k_ff = 1.0                 # feedforward gain

        # Curvature-adaptive speed:  v = v_des * (1 - k_curv_speed * |κ|)
        self.k_curv_speed = 0.5
        self.v_min_curve = 0.15         # minimum speed in tight curves [m/s]

        # --------------- EKF / KF ---------------
        x0 = np.zeros((3, 1))
        P0 = np.eye(3)

        self.qcar2_ekf = QcarEKF(
            x0=x0, P0=P0,
            Q=np.diagflat([0.0001, 0.0001, 0.001]),
            R=np.diagflat([0.1, 0.1, 0.01]))
        self.pose_ekf = np.zeros((3, 1))

        self.gyro_kf = GyroKF(
            x0=np.zeros((2, 1)), P0=np.eye(2),
            Q=np.diagflat([0.01, 0.01]),
            R=np.diagflat([0.1]))

        self.yaw = 0
        self.cutoff_frequency_filter = 15.0
        self.a1, self.b1 = self.filter_coefficients(self.cutoff_frequency_filter, self.dt)

        # --------------- Timers ---------------
        self.path_control_timer = self.create_timer(self.dt, self.path_planner)
        self.timer = self.create_timer(self.dt, self.tf_timer)

        # --------------- Path ---------------
        self.translation = [0, 0, 0]
        self.rotation = [0, 0, 0]
        self.wp = SDCSRoadMap().generate_path(self.waypoints) * self.scale
        self.N = len(self.wp[0, :])
        self.wpi = 0
        self.wp_prior = []
        self.current_steering = 0.0
        self.previous_steering = 0.0

        # Precompute path geometry (headings, curvatures, transformed coords)
        self._rebuild_path()

        # --------------- Publishers / Subscribers ---------------
        self.publisher = self.create_publisher(Twist, '/cmd_vel_nav', 1)
        self.cyclic = False

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

        self.path_publisher_topic = self.create_publisher(Path, '/planned_path', 1)
        self.path_status_publisher = self.create_publisher(Bool, '/path_status', 1)

        # Multiscope
        self.t0 = time.time()
        self.t_plot = 0
        self.plot_visualized = False
        self.scopeTimer = self.create_timer(0.1, self.scopeDataTimer)

    # =============================================================
    #  Path geometry precomputation
    # =============================================================
    def _rebuild_path(self):
        """Precompute transformed waypoints, path headings, and curvatures."""
        self.wp_t = self._transform_all_waypoints()   # (N, 2)
        self.N = self.wp_t.shape[0]
        self.path_headings = self._compute_path_headings()
        self.path_curvatures = self._compute_path_curvatures()
        self.wpi = 0

    def _transform_all_waypoints(self):
        angle_offset = self.rotation_offset[0]
        c = np.cos(-angle_offset * np.pi / 180)
        s = np.sin(-angle_offset * np.pi / 180)
        R_rot = np.array([[c, -s], [s, c]])
        t = np.array([self.translation_offset[0], self.translation_offset[1]])
        wp_shifted = self.wp[:2, :].T + t
        return wp_shifted @ R_rot

    def _compute_path_headings(self):
        """Heading at each waypoint from the tangent vector."""
        headings = np.zeros(self.N)
        for i in range(self.N - 1):
            dx = self.wp_t[i + 1, 0] - self.wp_t[i, 0]
            dy = self.wp_t[i + 1, 1] - self.wp_t[i, 1]
            headings[i] = np.arctan2(dy, dx)
        headings[-1] = headings[-2]  # copy last
        return headings

    def _compute_path_curvatures(self):
        """Discrete curvature: κ = Δθ / Δs  at each waypoint."""
        curvatures = np.zeros(self.N)
        for i in range(1, self.N - 1):
            dtheta = wrap_to_pi(self.path_headings[i] - self.path_headings[i - 1])
            ds = np.linalg.norm(self.wp_t[i] - self.wp_t[i - 1])
            if ds > 1e-6:
                curvatures[i] = dtheta / ds
        # Smooth curvatures with a small running average to reduce noise
        kernel = 15
        if self.N > kernel:
            pad = kernel // 2
            c_padded = np.pad(curvatures, pad, mode='edge')
            curvatures = np.convolve(c_padded, np.ones(kernel) / kernel, mode='valid')[:self.N]
        return curvatures

    # =============================================================
    #  Stanley controller helpers
    # =============================================================
    def _find_closest_segment(self, p):
        """Find the path segment closest to position p.

        Returns (idx, projection_frac) where the closest point on the path
        is between wp_t[idx] and wp_t[idx+1], at fraction `projection_frac`
        along that segment.

        Only searches forward from self.wpi to avoid going backward.
        """
        p = np.array(p)
        best_dist = np.inf
        best_idx = self.wpi
        best_frac = 0.0

        search_start = max(0, self.wpi - 5)
        search_end = min(self.N - 1, self.wpi + 300)

        for i in range(search_start, search_end):
            a = self.wp_t[i]
            b = self.wp_t[i + 1]
            ab = b - a
            ab_len_sq = np.dot(ab, ab)
            if ab_len_sq < 1e-12:
                continue
            t_frac = np.clip(np.dot(p - a, ab) / ab_len_sq, 0.0, 1.0)
            proj = a + t_frac * ab
            d = np.linalg.norm(p - proj)
            if d < best_dist:
                best_dist = d
                best_idx = i
                best_frac = t_frac

        return best_idx, best_frac

    def _get_path_heading_at(self, idx, frac):
        """Interpolated path heading between idx and idx+1."""
        if idx >= self.N - 1:
            return self.path_headings[self.N - 1]
        h0 = self.path_headings[idx]
        h1 = self.path_headings[min(idx + 1, self.N - 1)]
        # Unwrap for interpolation
        dh = wrap_to_pi(h1 - h0)
        return wrap_to_pi(h0 + frac * dh)

    def _get_curvature_at(self, idx, frac):
        """Interpolated curvature between idx and idx+1."""
        if idx >= self.N - 1:
            return 0.0
        k0 = self.path_curvatures[idx]
        k1 = self.path_curvatures[min(idx + 1, self.N - 1)]
        return k0 + frac * (k1 - k0)

    def _compute_cross_track_error(self, p, idx, frac):
        """Signed cross-track error.  Positive = car is to the RIGHT of the path."""
        p = np.array(p)
        a = self.wp_t[idx]
        b = self.wp_t[min(idx + 1, self.N - 1)]
        proj = a + frac * (b - a)
        error_vec = p - proj

        # Path tangent and right-normal
        tangent = b - a
        t_len = np.linalg.norm(tangent)
        if t_len < 1e-6:
            return 0.0
        tangent = tangent / t_len
        # Right normal: rotate tangent 90° clockwise
        normal_right = np.array([tangent[1], -tangent[0]])

        return float(np.dot(error_vec, normal_right))

    # =============================================================
    #  Parameter callback
    # =============================================================
    def parameter_update_callback(self, params):
        for param in params:
            if param.name == 'node_values' and param.type_ == param.Type.INTEGER_ARRAY:
                self.waypoints = list(param.value)
                self.wp = SDCSRoadMap().generate_path(self.waypoints) * 0.975
                self.N = len(self.wp[0, :])
                self.previous_steering = 0.0
                self.current_steering = 0.0
                self.path_complete = False
                self._rebuild_path()
                self.get_logger().info('nodes updated!')
                print(self.waypoints)

            elif param.name == 'desired_speed' and param.type_ == param.Type.DOUBLE_ARRAY:
                self.desired_speed = list(param.value)
                self.get_logger().info('new desired speed...')

            elif param.name == 'rotation_offset' and param.type_ == param.Type.DOUBLE_ARRAY:
                self.rotation_offset = list(param.value)
                self._rebuild_path()

            elif param.name == 'translation_offset' and param.type_ == param.Type.DOUBLE_ARRAY:
                self.translation_offset = list(param.value)
                self._rebuild_path()

            elif param.name == 'start_path' and param.type_ == param.Type.BOOL_ARRAY:
                self.path_execute_flag = list(param.value)[0]
                self.get_logger().info('path status changed!')

            elif param.name == 'visualize_pose' and param.type_ == param.Type.BOOL_ARRAY:
                self.pose_visualize_flag = list(param.value)[0]
                if self.pose_visualize_flag and not self.plot_visualized:
                    self.get_logger().info('Pose visualizing...')
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
                    self.steeringScope.axes[2].attachSignal(name='delta')
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

    # =============================================================
    #  Signal processing helpers
    # =============================================================
    def filter_coefficients(self, freq, dt):
        nyq = 0.5 / dt
        b, a = signal.butter(2, freq / nyq)
        self.hist = {'gyro': {'in': [0.0] * 3, 'out': [0.0] * 3}}
        return a, b

    def apply_filter(self, key, x, a, b):
        h = self.hist[key]
        h['in'] = [x] + h['in'][:2]
        y = (b[0]*h['in'][0] + b[1]*h['in'][1] + b[2]*h['in'][2]
             - a[1]*h['out'][0] - a[2]*h['out'][1])
        h['out'] = [y] + h['out'][:2]
        return y

    # =============================================================
    #  ROS callbacks
    # =============================================================
    def object_detector_callback(self, msg):
        self.motion_flag = msg.data

    def joint_state_callback(self, msg):
        self.qcar2_measurred_speed = (
            (msg.velocity[0] / (720.0 * 4.0))
            * ((13.0 * 19.0) / (70.0 * 30.0))
            * (2.0 * np.pi) * 0.033)

    def imu_callback(self, msg):
        self.gyroscope = [msg.angular_velocity.x,
                          msg.angular_velocity.y,
                          msg.angular_velocity.z]

    # =============================================================
    #  Path publisher (RViz visualisation)
    # =============================================================
    def path_publisher(self):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        for i in range(self.wpi, self.N):
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = "map"
            pose.pose.position.x = float(self.wp_t[i, 0])
            pose.pose.position.y = float(self.wp_t[i, 1])
            path_msg.poses.append(pose)
        self.path_publisher_topic.publish(path_msg)

    # =============================================================
    #  MAIN CONTROL LOOP — Stanley controller
    # =============================================================
    def path_planner(self):
        enable = 1
        speed_des = self.desired_speed[0]
        speed_command = speed_des          # default; overwritten below
        self.t_plot = time.time() - self.t0

        # EKF predict
        self.ekf_filter_timer()

        # Publish path for visualisation every 2 s
        if round(self.t_plot) % 2 == 0:
            self.path_publisher()

        try:
            if not self.path_complete:

                # --- Car pose (prefer TF, fall back to EKF) ---
                try:
                    p = [self.translation.x, self.translation.y]
                    th = self.yaw
                except AttributeError:
                    p = [self.qcar2_ekf.xHat[0, 0], self.qcar2_ekf.xHat[1, 0]]
                    th = self.qcar2_ekf.xHat[2, 0]

                # Front-axle position (Stanley is defined at front axle)
                p_front = [p[0] + self.L * np.cos(th),
                           p[1] + self.L * np.sin(th)]

                # ==================================================
                # 1.  Find closest point on the path
                # ==================================================
                seg_idx, seg_frac = self._find_closest_segment(p_front)
                self.wpi = max(self.wpi, seg_idx)

                # ==================================================
                # 2.  Heading error   ψ_e = ψ_path − ψ_car
                # ==================================================
                path_heading = self._get_path_heading_at(seg_idx, seg_frac)
                heading_error = wrap_to_pi(path_heading - th)

                # ==================================================
                # 3.  Cross-track error  (+ = car right of path)
                # ==================================================
                cte = self._compute_cross_track_error(p_front, seg_idx, seg_frac)

                # ==================================================
                # 4.  Curvature feedforward
                # ==================================================
                kappa = self._get_curvature_at(seg_idx, seg_frac)
                delta_ff = np.arctan(self.L * kappa)

                # ==================================================
                # 5.  Stanley law
                #     δ = k_h · ψ_e + arctan(k_e · e / (v + k_s)) + δ_ff
                # ==================================================
                v = max(abs(self.qcar2_measurred_speed), 0.05)

                delta_heading  = self.k_heading * heading_error
                delta_cte      = np.arctan2(self.k_e * cte, v + self.k_soft)
                delta_ff_term  = self.k_ff * delta_ff

                steering_raw = float(delta_heading + delta_cte + delta_ff_term)

                # ==================================================
                # 6.  Curvature-adaptive speed
                # ==================================================
                abs_kappa = min(abs(kappa), 10.0)
                speed_scale = max(1.0 - self.k_curv_speed * abs_kappa, 0.0)
                speed_command = max(speed_des * speed_scale, self.v_min_curve)

                # ==================================================
                # 7.  End-of-path handling
                # ==================================================
                dist_to_end = np.linalg.norm(
                    np.array(p) - self.wp_t[self.N - 1])

                if self.wpi >= self.N - 5 and dist_to_end < 0.4:
                    speed_command = 0.0
                    steering_raw = 0.0
                    self.wp_prior = self.wp
                    self.path_complete = True

                if self.wpi > self.N - 100:
                    speed_command = min(speed_command, 0.2)

                # ==================================================
                # 8.  Steering rate limiter (actuator constraint)
                # ==================================================
                max_step = self.max_steer_rate * self.dt
                diff = steering_raw - self.previous_steering
                diff = float(np.clip(diff, -max_step, max_step))
                steering = self.previous_steering + diff
                steering = float(np.clip(steering,
                                         -self.max_steering_angle,
                                          self.max_steering_angle))

                self.previous_steering = steering
                self.current_steering = steering

        except KeyboardInterrupt:
            speed_command = 0.0

        # Enable / disable
        if self.path_execute_flag and self.motion_flag:
            enable = 1.0
        if not self.path_execute_flag or not self.motion_flag or self.path_complete:
            enable = 0.0

        self.nav_command(enable, speed_command)
        self.path_status()

    # =============================================================
    def nav_command(self, enable, speed_command):
        cmd = Twist()
        cmd.linear.x = float(enable * np.clip(
            speed_command * np.cos(self.current_steering) ** 2,
            0.05, 0.7))
        cmd.angular.z = float(enable * self.current_steering)
        self.publisher.publish(cmd)

    def path_status(self):
        msg = Bool()
        msg.data = self.path_complete
        self.path_status_publisher.publish(msg)

    # =============================================================
    #  TF / EKF
    # =============================================================
    def tf_timer(self):
        try:
            t = self.tf_buffer.lookup_transform("map", self.target_frame,
                                                 rclpy.time.Time())
            self.translation = t.transform.translation
            rot = [t.transform.rotation.x, t.transform.rotation.y,
                   t.transform.rotation.z, t.transform.rotation.w]
            _, _, self.yaw = R.from_quat(rot).as_euler('xyz')

            self.gyro_kf.correction(self.yaw)
            y = np.array([[self.translation.x],
                          [self.translation.y],
                          [self.gyro_kf.xHat[0, 0]]])
            self.qcar2_ekf.correction(y)

        except TransformException as ex:
            self.get_logger().info(f'TF error: {ex}')

    def ekf_filter_timer(self):
        self.qcar2_ekf.prediction(self.dt,
                                   [self.qcar2_measurred_speed,
                                    self.current_steering])
        try:
            th_gyro = self.gyroscope[2]
        except AttributeError:
            th_gyro = 0
        self.gyro_kf.prediction(self.dt, th_gyro)

    # =============================================================
    #  Scope / debug
    # =============================================================
    def scopeDataTimer(self):
        if self.pose_visualize_flag:
            p = [self.qcar2_ekf.xHat[0, 0],
                 self.qcar2_ekf.xHat[1, 0],
                 self.qcar2_ekf.xHat[2, 0]]
            if self.t_plot > 200:
                self.t0 = time.time()
                for ax in self.steeringScope.axes:
                    ax.clean()
                MultiScope.refreshAll()
            try:
                x_ref = self.translation.x
                y_ref = self.translation.y
            except AttributeError:
                x_ref = y_ref = 0
            self.steeringScope.axes[0].sample(self.t_plot, [x_ref, p[0]])
            self.steeringScope.axes[1].sample(self.t_plot, [y_ref, p[1]])
            self.steeringScope.axes[2].sample(self.t_plot, [self.current_steering])
            self.steeringScope.axes[3].sample(self.t_plot,
                                               [self.yaw, self.qcar2_ekf.xHat[2, 0]])
            MultiScope.refreshAll()
        else:
            try:
                self.steeringScope.graphicsLayoutWidget.close()
                self.get_logger().info('previous scope closed...')
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