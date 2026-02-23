#! /usr/bin/env python3

# Quanser specific packages
from hal.products.mats import SDCSRoadMap
from pal.utilities.math import wrap_to_pi

# Generic python packages
import time  # Time library
import numpy as np
import scipy.signal as signal
from scipy.spatial.transform import Rotation as R
from pal.utilities.scope import MultiScope

# ROS specific packages
from rclpy.duration import Duration  # Handles time for ROS 2
import rclpy  # Python client library for ROS 2
from geometry_msgs.msg import PoseStamped  # Pose with ref frame and timestamp
from rclpy.node import Node
from nav_msgs.msg import Path
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Imu, JointState
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Bool

# ---- Stanley controller library (your uploaded control.py) ----
from hal.utilities.control import StanleyController


'''
Description:

Navigates a robot from an initial pose to a goal pose described by a series of
given nodes based on Quanser's SDCSRoadMap class
'''

# region: Helper classes for state estimation
class QcarEKF:

    def __init__(self, x0, P0, Q, R):
        self.L = 0.257

        self.I = np.eye(3)
        self.xHat = x0
        self.P = P0
        self.Q = Q
        self.R = R

        self.C = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

    # ==============  SECTION A -  Motion Model ====================
    def f(self, X, u, dt):
        # Kinematic Bicycle Model:
        # - X = [x, y, theta]
        # - u[0] = v (speed in [m/s])
        # - u[1] = delta (steering Angle in [rad])
        # - dt: change in time since last update

        return X + dt * u[0] * np.array([
            [np.cos(X[2,0])],
            [np.sin(X[2,0])],
            [np.tan(u[1]) / self.L]
        ])

    # ==============  SECTION B -  Motion Model Jacobian ====================
    def Jf(self, X, u, dt):
        return np.array([
                [1, 0, -dt*u[0]*np.sin(X[2,0])],
                [0, 1, dt*u[0]*np.cos(X[2,0])],
                [0, 0, 1]
        ])

    # ==============  SECTION C -  Motion Model Prediction ====================
    def prediction(self, dt, u):
        F = self.Jf(self.xHat, u, dt)
        self.P = F@self.P@np.transpose(F) + self.Q

        self.xHat = self.f(self.xHat, u, dt)
        self.xHat[2] = wrap_to_pi(self.xHat[2])
        return

    # ==============  SECTION D -  Measurement correction ====================
    def correction(self, y):
        H = self.C
        P_times_HTransposed = self.P @ np.transpose(H)

        S = H @ P_times_HTransposed + self.R
        K = P_times_HTransposed @ np.linalg.inv(S)

        z = (y - H@self.xHat)
        if len(y) > 1:
            z[2] = wrap_to_pi(z[2])
        else:
            z = wrap_to_pi(z)

        self.xHat += K @ z
        self.xHat[2] = wrap_to_pi(self.xHat[2])

        self.P = (self.I - K@H) @ self.P
        return


class GyroKF:

    def __init__(self, x0, P0, Q, R):
        self.I = np.eye(2)
        self.xHat = x0
        self.P = P0
        self.Q = Q
        self.R = R

        # State Space Representation Matrices
        self.A = np.array([
            [0, -1],
            [0, 0]
        ])
        self.B = np.array([
            [1],
            [0]
        ])
        self.C = np.array([
            [1, 0]
        ])

    # ==========  SECTION F -  Gyro Heading Prediction  ================
    def prediction(self, dt, u):
        Ad = self.I + self.A*dt
        self.xHat = Ad@self.xHat + dt*self.B*u
        self.P = Ad@self.P@np.transpose(Ad) + self.Q

    # ==========  SECTION G -  GPS Heading Correction  ================
    def correction(self, y):
        P_times_CTransposed = self.P @ np.transpose(self.C)

        S = self.C @ P_times_CTransposed + self.R
        K = P_times_CTransposed @ np.linalg.inv(S)

        z = y - self.C@self.xHat
        z = wrap_to_pi(z)

        self.xHat += K @ z
        self.xHat[0] = wrap_to_pi(self.xHat[0])

        self.P = (self.I - K@self.C) @ self.P
        return

# endregion


class PathFollower(Node):

    def __init__(self):
        super().__init__('path_follower')

        # define new parameters for node to use
        self.declare_parameter('node_values', [0, 8, 10])
        self.waypoints = list(self.get_parameter("node_values").get_parameter_value().integer_array_value)

        self.declare_parameter('desired_speed', [0.3])
        self.desired_speed = list(self.get_parameter("desired_speed").get_parameter_value().double_array_value)

        self.declare_parameter('visualize_pose', [False])
        self.pose_visualize_flag = list(self.get_parameter("visualize_pose").get_parameter_value().bool_array_value)[0]

        self.scale = 1.0

        self.declare_parameter('rotation_offset', [90.0])
        self.rotation_offset = list(self.get_parameter("rotation_offset").get_parameter_value().double_array_value)

        self.declare_parameter('translation_offset', [0.0, 0.0])
        self.translation_offset = list(self.get_parameter("translation_offset").get_parameter_value().double_array_value)

        self.declare_parameter('start_path', [True])
        self.path_execute_flag = list(self.get_parameter("start_path").get_parameter_value().bool_array_value)[0]

        self.add_on_set_parameters_callback(self.parameter_update_callback)

        # Declare and acquire `target_frame` parameter
        self.target_frame = self.declare_parameter(
            'target_frame', 'base_link').get_parameter_value().string_value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # parameters common to all methods
        self.dt = 1 / 200

        # Initial estimates for QCar state and covariance matrix (P)
        x0 = np.zeros((3, 1))
        P0 = np.eye(3)

        # For pose estimation
        R_combined = np.diagflat([0.0001, 0.0001, 0.0001])

        self.qcar2_ekf = QcarEKF(
            x0=x0,
            P0=P0,
            Q=np.diagflat([0.01, 0.01, 0.01]),
            R=R_combined
        )
        self.pose_ekf = np.zeros((3, 1))

        # for updating heading based on pose and gyro
        self.gyro_kf = GyroKF(
            x0=np.zeros((2, 1)),
            P0=np.eye(2),
            Q=np.diagflat([0.01, 0.01]),
            R=np.diagflat([.1])
        )

        # timer for both pose and gyro EKF
        self.yaw = 0

        # butterworth filter parameters (still used for optional diagnostics; not used in steering anymore)
        self.cutoff_frequency_filter = 15.0
        self.a1, self.b1 = self.filter_coefficients(self.cutoff_frequency_filter, self.dt)

        # Main timers
        self.path_control_timer = self.create_timer(self.dt, self.path_planner)
        self.timer = self.create_timer(self.dt, self.tf_timer)

        # Waypoint and path specific settings
        self.translation = [0, 0, 0]
        self.rotation = [0, 0, 0]

        self.wp = SDCSRoadMap().generate_path(self.waypoints) * self.scale
        self.N = len(self.wp[0, :])

        # -------------------- Stanley Controller setup --------------------
        # Stanley gain taken from vehicle_control.py lab values (K_stanley = 1)
        self.K_stanley = 0.75
        self.wp_mod = self.transform_path_to_map(self.wp)
        self.stanley = StanleyController(waypoints=self.wp_mod, k=self.K_stanley, cyclic=False)
        # -----------------------------------------------------------------

        self.current_steering = 0.0
        self.max_steering_angle = 1.0

        self.publisher = self.create_publisher(Twist, '/cmd_vel_nav', 1)
        self.cyclic = False

        self.joint_state_subscriber = self.create_subscription(JointState, '/qcar2_joint',
                                                              self.joint_state_callback, 1)
        self.qcar2_measurred_speed = 0.0

        self.object_detection_flag = self.create_subscription(Bool, '/motion_enable',
                                                              self.object_detector_callback, 1)
        self.motion_flag = True
        self.path_complete = False

        self.imu_subscrition = self.create_subscription(Imu, '/qcar2_imu', self.imu_callback, 10)
        self.gyroscope = [0, 0, 0]

        self.path_publisher_topic = self.create_publisher(Path, '/planned_path', 1)
        self.path_status_publisher = self.create_publisher(Bool, '/path_status', 1)

        # Multiscope info
        self.t0 = time.time()
        self.t_plot = 0
        self.plot_visualized = False
        self.scopeTimer = self.create_timer(0.1, self.scopeDataTimer)

    # ---------------- Stanley utilities ----------------
    def transform_path_to_map(self, wp_xy_2xN):
        """
        Applies your rotation_offset + translation_offset to the entire waypoint path
        so the controller works in the same frame as your TF pose (map frame).
        """
        angle_offset = self.rotation_offset[0]
        R_QLabs_ROS = np.array([
            [np.cos(-angle_offset * np.pi / 180), -np.sin(-angle_offset * np.pi / 180)],
            [np.sin(-angle_offset * np.pi / 180),  np.cos(-angle_offset * np.pi / 180)]
        ])
        t = np.array([self.translation_offset[0], self.translation_offset[1]])

        wp_mod = ((wp_xy_2xN.T + t) @ R_QLabs_ROS).T  # back to (2, N)
        return wp_mod
    # ---------------------------------------------------

    def parameter_update_callback(self, params):
        for param in params:

            if param.name == 'node_values' and param.type_ == param.Type.INTEGER_ARRAY:
                # regenerate + transform the path
                self.waypoints = list(param.value)
                self.wp = SDCSRoadMap().generate_path(self.waypoints) * 0.975
                self.N = len(self.wp[0, :])

                self.wp_mod = self.transform_path_to_map(self.wp)
                self.stanley.updatePath(self.wp_mod, cyclic=False)

                self.current_steering = 0.0
                self.path_complete = False
                self.get_logger().info('nodes updated!')
                print(self.waypoints)

            elif param.name == 'desired_speed' and param.type_ == param.Type.DOUBLE_ARRAY:
                self.desired_speed = list(param.value)
                self.get_logger().info('new desired speed...')
                print(self.desired_speed)

            elif param.name == 'rotation_offset' and param.type_ == param.Type.DOUBLE_ARRAY:
                self.rotation_offset = list(param.value)
                self.wp_mod = self.transform_path_to_map(self.wp)
                self.stanley.updatePath(self.wp_mod, cyclic=False)

            elif param.name == 'translation_offset' and param.type_ == param.Type.DOUBLE_ARRAY:
                self.translation_offset = list(param.value)
                self.wp_mod = self.transform_path_to_map(self.wp)
                self.stanley.updatePath(self.wp_mod, cyclic=False)

            elif param.name == 'start_path' and param.type_ == param.Type.BOOL_ARRAY:
                self.path_execute_flag = list(param.value)[0]
                self.get_logger().info('path status changed!')

            elif param.name == 'visualize_pose' and param.type_ == param.Type.BOOL_ARRAY:
                self.pose_visualize_flag = list(param.value)[0]
                if self.pose_visualize_flag and not self.plot_visualized:
                    self.get_logger().info(
                        'Pose performance to be displayed.. Note: visualizing pose will impact driving performance...'
                    )

                    tf = 200

                    self.steeringScope = MultiScope(
                        rows=4,
                        cols=1,
                        title='Vehicle Steering Control',
                        fps=10
                    )

                    self.steeringScope.addAxis(
                        row=0,
                        col=0,
                        timeWindow=tf,
                        yLabel='x Position [m]',
                        yLim=(-2.5, 2.5)
                    )
                    self.steeringScope.axes[0].attachSignal(name='x_meas')
                    self.steeringScope.axes[0].attachSignal(name='x_ekf')

                    self.steeringScope.addAxis(
                        row=1,
                        col=0,
                        timeWindow=tf,
                        yLabel='y Position [m]',
                        yLim=(-1, 6)
                    )
                    self.steeringScope.axes[1].attachSignal(name='y_meas')
                    self.steeringScope.axes[1].attachSignal(name='y_ekf')

                    self.steeringScope.addAxis(
                        row=2,
                        col=0,
                        timeWindow=tf,
                        yLabel='steering cmd [rad]',
                        yLim=(-0.6, 0.6)
                    )
                    self.steeringScope.axes[2].attachSignal(name='delta')

                    self.steeringScope.addAxis(
                        row=3,
                        col=0,
                        timeWindow=tf,
                        yLabel='heading',
                        yLim=(-np.pi, np.pi)
                    )
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
        self.hist = {
            'gyro': {'in': [0.0] * 3, 'out': [0.0] * 3},
        }

        return a, b

    def apply_filter(self, key, new_input, a, b):
        h = self.hist[key]
        h['in'] = [new_input] + h['in'][:2]
        y = (
            b[0] * h['in'][0] +
            b[1] * h['in'][1] +
            b[2] * h['in'][2] -
            a[1] * h['out'][0] -
            a[2] * h['out'][1]
        )
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
        path_msg.header.frame_id = "map"

        # publish up to the controller's current index (if exposed), else publish the full path
        # NOTE: if StanleyController exposes "currentWaypoint", we can use it; otherwise just publish full path.
        current_i = getattr(self.stanley, "currentWaypoint", self.N)

        for i in range(int(np.clip(current_i, 0, self.N))):
            if i >= self.N:
                i = self.N - 1
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = "map"
            pose.pose.position.x = float(self.wp_mod[0, i])
            pose.pose.position.y = float(self.wp_mod[1, i])
            path_msg.poses.append(pose)

        self.path_publisher_topic.publish(path_msg)

    def path_planner(self):
        max_speed = 1.5
        enable = 1.0
        speed_command = float(self.desired_speed[0])

        self.t_plot = time.time() - self.t0

        # update ekf_filter
        self.ekf_filter_timer()

        # publish latest path value based on current waypoint every 2 seconds
        if round(self.t_plot) % 2 == 0:
            self.path_publisher()

        try:
            if not self.path_complete:
                # Pose used by controller (prefer TF pose + yaw if available)
                th = float(self.qcar2_ekf.xHat[2, 0])
                p = np.array([float(self.qcar2_ekf.xHat[0, 0]), float(self.qcar2_ekf.xHat[1, 0])])

                try:
                    p = np.array([float(self.translation.x), float(self.translation.y)])
                    th = float(self.yaw)
                except AttributeError:
                    p = np.array([0.0, 0.0])
                    th = 0.0

                # Stanley uses speed in the cross-track term; keep small minimum to avoid division blow-up
                speed_for_stanley = max(float(self.qcar2_measurred_speed), 0.05)

                # ---- Stanley steering command ----
                steering_cmd = float(self.stanley.update(p, th, speed_for_stanley))
                self.current_steering = np.clip(steering_cmd, -self.max_steering_angle, self.max_steering_angle)

                # ---- Completion condition ----
                if getattr(self.stanley, "pathComplete", False):
                    speed_command = 0.0
                    self.current_steering = 0.0
                    self.path_complete = True

                # Optional slowdown near end if controller exposes remaining distance / index
                # If your Stanley implementation exposes something like currentWaypoint, you can slow down here.
                current_i = getattr(self.stanley, "currentWaypoint", None)
                if current_i is not None and current_i > self.N - 100:
                    speed_command = min(speed_command, 0.2)

        except KeyboardInterrupt:
            speed_command = 0.0
            self.current_steering = 0.0

        # enable gating
        if self.path_execute_flag is True:
            if self.motion_flag is True:
                enable = 1.0
        if self.path_execute_flag is False or self.motion_flag is False or self.path_complete:
            enable = 0.0

        # publishing commands
        self.nav_command(enable, speed_command)
        self.path_status()

    def nav_command(self, enable, speed_command):
        QCarCommands = Twist()
        QCarCommands.linear.x = enable * np.clip(speed_command * np.power(np.cos(self.current_steering), 1), 0.05, 0.7)
        QCarCommands.angular.z = enable * self.current_steering
        self.publisher.publish(QCarCommands)

    def path_status(self):
        msg = Bool()
        msg.data = self.path_complete
        self.path_status_publisher.publish(msg)

    def tf_timer(self):
        from_frame_rel = "map"
        to_frame_rel = self.target_frame

        try:
            t = self.tf_buffer.lookup_transform(
                from_frame_rel,
                to_frame_rel,
                rclpy.time.Time()
            )
            self.translation = t.transform.translation
            rotation = [
                t.transform.rotation.x,
                t.transform.rotation.y,
                t.transform.rotation.z,
                t.transform.rotation.w
            ]
            roll, pitch, self.yaw = R.from_quat(rotation).as_euler('xyz')

            # correct the GPS when we get new data from cartographer package
            self.gyro_kf.correction(self.yaw)
            y = np.array([
                [self.translation.x],
                [self.translation.y],
                [self.gyro_kf.xHat[0, 0]]
            ])

            self.qcar2_ekf.correction(y)

        except TransformException as ex:
            self.get_logger().info(f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
            return

    def ekf_filter_timer(self):
        # always predict based on the steering and measured speed
        speed = float(self.qcar2_measurred_speed)
        delta = float(self.current_steering)
        self.qcar2_ekf.prediction(self.dt, [speed, delta])

        try:
            th_gyro = float(self.gyroscope[2])
        except AttributeError:
            th_gyro = 0.0

        self.gyro_kf.prediction(self.dt, th_gyro)

    def scopeDataTimer(self):
        if self.pose_visualize_flag:
            p = [self.qcar2_ekf.xHat[0, 0], self.qcar2_ekf.xHat[1, 0], self.qcar2_ekf.xHat[2, 0]]

            if self.t_plot > 200:
                self.t0 = time.time()
                self.steeringScope.axes[0].clean()
                self.steeringScope.axes[1].clean()
                self.steeringScope.axes[2].clean()
                self.steeringScope.axes[3].clean()
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