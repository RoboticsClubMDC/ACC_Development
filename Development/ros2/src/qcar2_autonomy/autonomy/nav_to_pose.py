#!/usr/bin/env python3

from pal.utilities.math import wrap_to_pi
import time
import numpy as np
import scipy.signal as signal
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool
from tf2_ros import TransformException, Buffer, TransformListener
from rcl_interfaces.msg import SetParametersResult


# region: state estimation filters

class QcarEKF:

    def __init__(self, x0, P0, Q, R):
        self.L = 0.257
        self.I = np.eye(3)
        self.xHat = x0
        self.P = P0
        self.Q = Q
        self.R = R
        self.C = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

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
        z[2] = wrap_to_pi(z[2])
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


class NavToPose(Node):

    def __init__(self):
        super().__init__('nav_to_pose')

        self.declare_parameter('desired_speed', [0.6])
        self.desired_speed = list(self.get_parameter('desired_speed').get_parameter_value().double_array_value)

        self.declare_parameter('start_path', [True])
        self.path_execute_flag = list(self.get_parameter('start_path').get_parameter_value().bool_array_value)[0]

        self.declare_parameter('visualize_pose', [False])
        self.pose_visualize_flag = list(self.get_parameter('visualize_pose').get_parameter_value().bool_array_value)[0]

        self.declare_parameter('target_frame', 'base_link')
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value

        self.add_on_set_parameters_callback(self.parameter_update_callback)

        self.dt = 1 / 80

        # EKF setup
        self.qcar2_ekf = QcarEKF(
            x0=np.zeros((3, 1)),
            P0=np.eye(3),
            Q=np.diagflat([0.0001, 0.0001, 0.001]),
            R=np.diagflat([0.1, 0.1, 0.01])
        )
        self.gyro_kf = GyroKF(
            x0=np.zeros((2, 1)),
            P0=np.eye(2),
            Q=np.diagflat([0.01, 0.01]),
            R=np.diagflat([0.1])
        )

        self.cutoff_frequency_filter = 15.0
        self.a1, self.b1 = self._filter_coefficients(self.cutoff_frequency_filter, self.dt)

        self.yaw = 0
        self.translation = None
        self.current_steering = 0
        self.qcar2_measurred_speed = 0
        self.gyroscope = [0, 0, 0]
        self.motion_flag = True
        self.max_steering_angle = 0.6

        # waypoints in map frame, 2xN
        self.wp = np.zeros((2, 1))
        self.N = 1
        self.wpi = 0
        self.path_complete = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel_nav', 1)
        self.path_status_pub = self.create_publisher(Bool, '/path_status', 1)
        self.path_viz_pub = self.create_publisher(Path, '/planned_path', 1)
        self.robot_pose_pub = self.create_publisher(PoseStamped, '/robot_pose', 10)

        # subscribers
        self.create_subscription(Path, '/cmd_waypoints', self.waypoints_callback, 1)
        self.create_subscription(JointState, '/qcar2_joint', self.joint_state_callback, 1)
        self.create_subscription(Bool, '/motion_enable', self.motion_enable_callback, 1)
        self.create_subscription(Imu, '/qcar2_imu', self.imu_callback, 10)

        self.create_timer(self.dt, self.path_planner)
        self.create_timer(self.dt, self.tf_timer)

        self.t0 = time.time()
        self.t_plot = 0

    # ---- param callback ----

    def parameter_update_callback(self, params):
        for param in params:
            if param.name == 'desired_speed' and param.type_ == param.Type.DOUBLE_ARRAY:
                self.desired_speed = list(param.value)
            elif param.name == 'start_path' and param.type_ == param.Type.BOOL_ARRAY:
                self.path_execute_flag = list(param.value)[0]
        return SetParametersResult(successful=True)

    # ---- subscriber callbacks ----

    def waypoints_callback(self, msg: Path):
        if not msg.poses:
            return
        xs = [p.pose.position.x for p in msg.poses]
        ys = [p.pose.position.y for p in msg.poses]
        self.wp = np.array([xs, ys])
        self.N = self.wp.shape[1]
        self.wpi = 0
        self.path_complete = False

    def motion_enable_callback(self, msg):
        self.motion_flag = msg.data

    def joint_state_callback(self, msg):
        self.qcar2_measurred_speed = (
            (msg.velocity[0] / (720.0 * 4.0))
            * ((13.0 * 19.0) / (70.0 * 30.0))
            * (2.0 * np.pi) * 0.033
        )

    def imu_callback(self, msg):
        self.gyroscope = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]

    # ---- filters ----

    def _filter_coefficients(self, freq, dt):
        from scipy.signal import butter
        norm_cut = freq / (0.5 * (1 / dt))
        b, a = butter(2, norm_cut)
        self.hist = {'gyro': {'in': [0.0] * 3, 'out': [0.0] * 3}}
        return a, b

    def _apply_filter(self, key, new_input, a, b):
        h = self.hist[key]
        h['in'] = [new_input] + h['in'][:2]
        y = (b[0]*h['in'][0] + b[1]*h['in'][1] + b[2]*h['in'][2]
             - a[1]*h['out'][0] - a[2]*h['out'][1])
        h['out'] = [y] + h['out'][:2]
        return y

    # ---- TF + EKF ----

    def tf_timer(self):
        try:
            t = self.tf_buffer.lookup_transform('map', self.target_frame, rclpy.time.Time())
            self.translation = t.transform.translation
            rot = [t.transform.rotation.x, t.transform.rotation.y,
                   t.transform.rotation.z, t.transform.rotation.w]
            _, _, self.yaw = R.from_quat(rot).as_euler('xyz')

            self.gyro_kf.correction(self.yaw)
            y = np.array([[self.translation.x], [self.translation.y], [self.gyro_kf.xHat[0, 0]]])
            self.qcar2_ekf.correction(y)

            self._publish_robot_pose()
        except TransformException as ex:
            self.get_logger().info(f'TF error: {ex}')

    def _ekf_predict(self):
        self.qcar2_ekf.prediction(self.dt, [self.qcar2_measurred_speed, self.current_steering])
        self.gyro_kf.prediction(self.dt, self.gyroscope[2])

    def _publish_robot_pose(self):
        if self.translation is None:
            return
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = self.translation.x
        msg.pose.position.y = self.translation.y
        q = R.from_euler('z', self.yaw).as_quat()
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        self.robot_pose_pub.publish(msg)

    # ---- path visualization ----

    def _publish_path_viz(self):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        if self.wp is None or self.wp.shape[1] < 1:
            self.path_viz_pub.publish(path_msg)
            return
        for i in range(min(self.wpi, self.wp.shape[1] - 1)):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(self.wp[0, i])
            pose.pose.position.y = float(self.wp[1, i])
            path_msg.poses.append(pose)
        self.path_viz_pub.publish(path_msg)

    # ---- main control loop ----

    def path_planner(self):
        self.t_plot = time.time() - self.t0
        self._ekf_predict()

        if round(self.t_plot) % 2 == 0:
            self._publish_path_viz()

        speed_command = self.desired_speed[0]

        try:
            if self.translation is None or self.wp is None or self.wp.shape[1] < 2:
                speed_command = 0.0
                self.current_steering = 0.0
                self.path_complete = True
            else:
                self.N = self.wp.shape[1]
                self.wpi = int(np.clip(self.wpi, 0, max(self.N - 2, 0)))

                wp_1 = self.wp[:, self.wpi]
                p_map = np.array([float(self.translation.x), float(self.translation.y)])
                th = float(self.yaw)

                v_map = (wp_1 - p_map).reshape(2, 1)
                Rot = np.array([[np.cos(th), -np.sin(th)],
                                [np.sin(th),  np.cos(th)]])
                v_car = (Rot.T @ v_map).reshape(2,)

                WaypointDist = max(float(np.linalg.norm(v_car)), 0.05)
                psi = float(np.arctan2(v_car[1], v_car[0]))
                L = 0.256
                delta = float(np.arctan2(2 * L * np.sin(psi), WaypointDist))

                wp_final = self.wp[:, -1]
                dist_to_target = float(np.linalg.norm(p_map - wp_1))
                dist_to_final = float(np.linalg.norm(p_map - wp_final))

                v_eff = max(self.qcar2_measurred_speed, 0.05)
                lookahead_dist = max(0.30, v_eff * 1.7)

                if dist_to_target < lookahead_dist and self.wpi < self.N - 2:
                    self.wpi += 1

                if dist_to_final < 0.50 or self.wpi >= self.N - 2:
                    speed_command = 0.0
                    self.current_steering = 0.0
                    self.path_complete = True
                else:
                    gyro_filtered = self._apply_filter('gyro', self.gyroscope[2], self.a1, self.b1)
                    steering = np.clip(
                        1.1 * delta - gyro_filtered * np.pi / 180 * 7,
                        -self.max_steering_angle,
                        self.max_steering_angle
                    )
                    self.current_steering = steering

                    if int(self.t_plot * 5) != int((self.t_plot - self.dt) * 5):
                        self.get_logger().info(
                            f'wpi={self.wpi}/{self.N} Ld={lookahead_dist:.2f} '
                            f'distT={dist_to_target:.2f} distF={dist_to_final:.2f} '
                            f'delta={delta:.2f} steer={steering:.3f} v={speed_command:.2f}'
                        )

        except KeyboardInterrupt:
            speed_command = 0.0
            self.current_steering = 0.0

        enable = 1.0 if (self.path_execute_flag and self.motion_flag and not self.path_complete) else 0.0

        cmd = Twist()
        cmd.linear.x = enable * np.clip(speed_command * np.power(np.cos(self.current_steering), 2), 0.05, 6.0)
        cmd.angular.z = enable * self.current_steering
        self.cmd_vel_pub.publish(cmd)

        status = Bool()
        status.data = self.path_complete
        self.path_status_pub.publish(status)


def main():
    rclpy.init()
    node = NavToPose()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()