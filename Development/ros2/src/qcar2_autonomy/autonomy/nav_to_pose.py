#! /usr/bin/env python3

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
from std_msgs.msg import Bool, Float32


# region: EKF / KF helpers (unchanged)
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

# endregion


class PathFollower(Node):

    def __init__(self):
        super().__init__('path_follower')

        # ---- Existing parameters (unchanged) ----
        self.declare_parameter('node_values', [0, 8, 10])
        self.waypoints = list(self.get_parameter('node_values').get_parameter_value().integer_array_value)

        self.declare_parameter('desired_speed', [0.6])
        self.desired_speed = list(self.get_parameter('desired_speed').get_parameter_value().double_array_value)

        self.declare_parameter('visualize_pose', [False])
        self.pose_visualize_flag = list(self.get_parameter('visualize_pose').get_parameter_value().bool_array_value)[0]

        self.scale = 1.0

        self.declare_parameter('rotation_offset', [90.0])
        self.rotation_offset = list(self.get_parameter('rotation_offset').get_parameter_value().double_array_value)

        self.declare_parameter('translation_offset', [0.0, 0.0])
        self.translation_offset = list(self.get_parameter('translation_offset').get_parameter_value().double_array_value)

        self.declare_parameter('start_path', [False])
        self.path_execute_flag = list(self.get_parameter('start_path').get_parameter_value().bool_array_value)[0]

        # ---- Stanley blending parameters (NEW) ----
        self.declare_parameter('stanley_blend',     0.3)   # 0=pure pursuit only, 1=stanley only
        self.declare_parameter('stanley_trust_min', 0.8)   # trust threshold to engage stanley
        self.stanley_blend     = self.get_parameter('stanley_blend').value
        self.stanley_trust_min = self.get_parameter('stanley_trust_min').value

        self.add_on_set_parameters_callback(self.parameter_update_callback)

        self.target_frame = self.declare_parameter(
            'target_frame', 'base_link').get_parameter_value().string_value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.dt = 1 / 80

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
        self.cutoff_frequency_filter = 15.0
        self.a1, self.b1 = self.filter_coefficients(self.cutoff_frequency_filter, self.dt)

        self.path_control_timer = self.create_timer(self.dt, self.path_planner)
        self.timer = self.create_timer(self.dt, self.tf_timer)
        self.translation = [0, 0, 0]
        self.rotation = [0, 0, 0]
        self.wp = SDCSRoadMap().generate_path(self.waypoints) * self.scale
        self.N = len(self.wp[0, :])
        self.wpi = 0
        self.wp_prior = []
        self.current_steering = 0

        # ---- Stanley state (NEW) ----
        self.stanley_delta = 0.0
        self.stanley_trust = 0.0
        self.pp_delta_raw  = 0.0   # pure pursuit delta before blending (for plot)

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

        self.path_publisher_topic = self.create_publisher(Path, '/planned_path', 1)
        self.path_status_publisher = self.create_publisher(Bool, '/path_status', 1)

        # ---- Stanley subscribers (NEW) ----
        self.create_subscription(Float32, '/lane_stanley/delta', self._stanley_delta_cb, 2)
        self.create_subscription(Float32, '/lane_stanley/trust', self._stanley_trust_cb, 2)

        # ---- Debug publishers for plot (NEW) ----
        self.pub_pp_delta      = self.create_publisher(Float32, '/nav/pp_delta',      2)
        self.pub_stanley_delta = self.create_publisher(Float32, '/nav/stanley_delta', 2)
        self.pub_blended_delta = self.create_publisher(Float32, '/nav/blended_delta', 2)
        self.pub_blend_alpha   = self.create_publisher(Float32, '/nav/blend_alpha',   2)

        self.t0 = time.time()
        self.t_plot = 0
        self.plot_visualized = False
        self.scopeTimer = self.create_timer(0.1, self.scopeDataTimer)

        self.get_logger().info(
            f'PathFollower ready | stanley_blend={self.stanley_blend} '
            f'trust_min={self.stanley_trust_min}')

    # ------------------------------------------------------------------
    # Stanley callbacks (NEW)
    # ------------------------------------------------------------------
    def _stanley_delta_cb(self, msg: Float32):
        self.stanley_delta = float(msg.data)

    def _stanley_trust_cb(self, msg: Float32):
        self.stanley_trust = float(msg.data)

    # ------------------------------------------------------------------
    # Steering blend (NEW)
    # ------------------------------------------------------------------
    def _blend_steering(self, pp_delta: float) -> float:
        """
        Blend pure pursuit and Stanley:
          - If Stanley trust < threshold: use pure pursuit only
          - Otherwise: weighted blend
          alpha = stanley_blend * stanley_trust
        """
        self.pp_delta_raw = pp_delta

        if self.stanley_trust < self.stanley_trust_min:
            alpha = 0.0
        else:
            alpha = np.clip(self.stanley_blend * self.stanley_trust, 0.0, 1.0)

        blended = (1.0 - alpha) * pp_delta + alpha * self.stanley_delta
        blended = float(np.clip(blended, -self.max_steering_angle, self.max_steering_angle))

        # Publish debug signals
        def f32(v):
            m = Float32(); m.data = float(v); return m

        self.pub_pp_delta.publish(f32(pp_delta))
        self.pub_stanley_delta.publish(f32(self.stanley_delta))
        self.pub_blended_delta.publish(f32(blended))
        self.pub_blend_alpha.publish(f32(alpha))

        return blended

    # ------------------------------------------------------------------
    # All original methods below — only path_planner has one line changed
    # ------------------------------------------------------------------

    def parameter_update_callback(self, params):
        for param in params:
            if param.name == 'node_values' and param.type_ == param.Type.INTEGER_ARRAY:
                self.waypoints = list(param.value)
                self.wp = SDCSRoadMap().generate_path(self.waypoints) * 0.975
                self.N = len(self.wp[0, :])
                self.wpi = 0
                self.previous_steering_value = 0
                self.path_complete = False
                self.get_logger().info('nodes updated!')
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
            elif param.name == 'stanley_blend' and param.type_ == param.Type.DOUBLE:
                self.stanley_blend = float(param.value)
                self.get_logger().info(f'stanley_blend updated to {self.stanley_blend}')
            elif param.name == 'stanley_trust_min' and param.type_ == param.Type.DOUBLE:
                self.stanley_trust_min = float(param.value)
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

    def path_publisher(self):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        for i in range(self.wpi):
            if i >= self.N:
                i = self.N - 1
            pose = PoseStamped()
            angle_offset = self.rotation_offset[0]
            R_QLabs_ROS = np.array([
                [np.cos(-angle_offset * np.pi / 180), -np.sin(-angle_offset * np.pi / 180)],
                [np.sin(-angle_offset * np.pi / 180),  np.cos(-angle_offset * np.pi / 180)]
            ])
            t = np.array([self.translation_offset[0], self.translation_offset[1]])
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
        self.ekf_filter_timer()

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
                wp_1_mod = (wp_1 + t) @ R_QLabs_ROS

                L = 0.256

                th = self.qcar2_ekf.xHat[2, 0]
                p = [self.qcar2_ekf.xHat[0, 0], self.qcar2_ekf.xHat[1, 0]]

                try:
                    p = [self.translation.x, self.translation.y]
                    th = self.yaw
                except AttributeError:
                    p = [0, 0]
                    th = 0

                v = [wp_1_mod[0] - p[0], wp_1_mod[1] - p[1]]
                Rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                v_car = v @ Rot

                WaypointDist = max(np.linalg.norm(v_car), 0.05)
                psi = np.arctan2(v_car[1], v_car[0])

                # Pure pursuit delta
                pp_delta = np.arctan2(2 * L * np.sin(psi), WaypointDist)

                dist = np.linalg.norm([p[0] - wp_1_mod[0], p[1] - wp_1_mod[1]])

                v_eff = max(self.qcar2_measurred_speed, 0.05)
                lookahead_dist = max(v_eff * 1.7, 0.30)

                if dist < lookahead_dist:
                    if self.wpi < self.N - 2:
                        self.wpi += skip_index

                self.wpi = np.clip(self.wpi, 0, self.N - 5)

                if self.wpi >= self.N - 5:
                    if dist < 0.4:
                        speed_command = 0.0
                        pp_delta = 0.0
                        self.wp_prior = self.wp
                        self.path_complete = True

                if self.wpi > self.N - 100:
                    speed_command = 0.2

                Kp_steering = 1.1
                kd_steering = 5
                gyro_filtered = self.apply_filter('gyro', self.gyroscope[2], self.a1, self.b1)

                # ---- Pure pursuit with gyro damping ----
                pp_delta_damped = np.clip(
                    Kp_steering * pp_delta - gyro_filtered * np.pi / 180 * kd_steering,
                    -self.max_steering_angle,
                    self.max_steering_angle)

                # ---- Blend with Stanley (NEW) ----
                steering = self._blend_steering(pp_delta_damped)

                self.current_steering = steering

                if int(self.t_plot * 5) != int((self.t_plot - self.dt) * 5):
                    self.get_logger().info(
                        f"wpi={self.wpi} dist={dist:.2f} "
                        f"pp={pp_delta_damped:.3f} "
                        f"st={self.stanley_delta:.3f} "
                        f"trust={self.stanley_trust:.2f} "
                        f"blended={steering:.3f} "
                        f"v={speed_command:.2f}")

        except KeyboardInterrupt:
            speed_command = 0.0

        if self.path_execute_flag and self.motion_flag:
            enable = 1.0
        if not self.path_execute_flag or not self.motion_flag or self.path_complete:
            enable = 0.0

        self.nav_command(enable, speed_command)
        self.path_status()

    def nav_command(self, enable, speed_command):
        QCarCommands = Twist()
        QCarCommands.linear.x = enable * np.clip(
            speed_command * np.power(np.cos(self.current_steering), 2), 0.05, 0.7)
        QCarCommands.angular.z = enable * self.current_steering
        self.publisher.publish(QCarCommands)

    def path_status(self):
        msg = Bool()
        msg.data = self.path_complete
        self.path_status_publisher.publish(msg)

    def tf_timer(self):
        from_frame_rel = 'map'
        to_frame_rel = self.target_frame
        try:
            t = self.tf_buffer.lookup_transform(from_frame_rel, to_frame_rel, rclpy.time.Time())
            self.translation = t.transform.translation
            rotation = [t.transform.rotation.x, t.transform.rotation.y,
                        t.transform.rotation.z, t.transform.rotation.w]
            roll, pitch, self.yaw = R.from_quat(rotation).as_euler('xyz')
            self.gyro_kf.correction(self.yaw)
            y = np.array([[self.translation.x], [self.translation.y], [self.gyro_kf.xHat[0, 0]]])
            self.qcar2_ekf.correction(y)
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
            self.steeringScope.axes[2].sample(self.t_plot, [self.current_steering])
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