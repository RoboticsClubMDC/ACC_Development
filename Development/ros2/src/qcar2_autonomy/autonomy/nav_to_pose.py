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


from enum import Enum

class IntersectionState(Enum):
    DRIVE   = 0   # normal path following
    STOPPED = 1   # hard-stopped at node, waiting for YOLO rule
    WAITING = 2   # timed wait (stop sign / yield)
    DONE    = 3   # wait complete, advance past node

# Intersection node IDs — car will hard-stop here and check for sign/light
INTERSECTION_NODES = {10, 1, 4, 6, 14, 20, 19, 11, 12, 9}

class PathFollower(Node):

    def __init__(self):
        super().__init__('path_follower')

        # ---- Existing parameters ----
        self.declare_parameter('node_values', [0, 8, 10])
        self.waypoints = list(self.get_parameter('node_values').get_parameter_value().integer_array_value)

        self.declare_parameter('desired_speed', [0.3])
        self.desired_speed = list(self.get_parameter('desired_speed').get_parameter_value().double_array_value)

        self.declare_parameter('visualize_pose', [True])
        self.pose_visualize_flag = True  # hardcoded on

        self.scale = 1.0

        self.declare_parameter('rotation_offset', [82.0])
        self.rotation_offset = list(self.get_parameter('rotation_offset').get_parameter_value().double_array_value)

        self.declare_parameter('translation_offset', [0.0, 0.0])
        self.translation_offset = list(self.get_parameter('translation_offset').get_parameter_value().double_array_value)

        self.declare_parameter('start_path', [True])
        self.path_execute_flag = list(self.get_parameter('start_path').get_parameter_value().bool_array_value)[0]

        self.declare_parameter('mission_pickup_xy',  [0.0, 0.0])
        self.declare_parameter('mission_dropoff_xy', [0.0, 0.0])
        self.declare_parameter('mission_enable',     [False])

        # ---- Stanley blending — HARDCODED, not runtime-tunable ----
        self.stanley_blend     = 0.3   # 5% max stanley contribution
        self.stanley_trust_min = 0.50   # needs 80% lane confidence to activate

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
        self._wp_in_ros_frame = False  # True when path came from TripPlanner (already transformed)

        # ── Intersection node XY table (QLabs frame) ─────────────────────────
        _rm = SDCSRoadMap()
        self._node_xy = {}   # node_id -> np.array([x, y])
        for nid in INTERSECTION_NODES:
            try:
                if hasattr(_rm, 'get_node_pose'):
                    pose = np.array(_rm.get_node_pose(nid)).reshape(-1)
                else:
                    pose = np.array(_rm.nodes[nid].pose).reshape(-1)
                self._node_xy[nid] = np.array([float(pose[0]), float(pose[1])])
            except Exception as e:
                self.get_logger().warn(f'Could not load intersection node {nid}: {e}')

        # Intersection state machine
        self._isec_state       = IntersectionState.DRIVE
        self._isec_wpi         = -1    # which wpi triggered the current intersection
        self._isec_wait_until  = 0.0   # absolute time.time() to release timed waits
        self._isec_nodes_done  = set() # node ids already handled on this path
        self._intersection_wpi = {}    # wpi -> node_id for current path

        # ── Intersection topics ───────────────────────────────────────────────
        # ---- Stanley state ----
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

        self.path_publisher_topic  = self.create_publisher(Path, '/planned_path', 1)
        self.path_status_publisher = self.create_publisher(Bool, '/path_status', 1)
        self.at_intersection_pub   = self.create_publisher(Bool, '/at_intersection', 1)

        # /intersection_rule — yolo_detector tells us what it sees at this node
        # Values: "NONE" | "STOP" | "YIELD" | "RED" | "GREEN" | "YELLOW"
        self._intersection_rule = "NONE"
        from std_msgs.msg import String as StringMsg
        self.create_subscription(StringMsg, '/intersection_rule',
                                 lambda m: setattr(self, '_intersection_rule', m.data), 1)

        # /robot_pose — TripPlanner needs this to plan each leg
        self.robot_pose_publisher = self.create_publisher(PoseStamped, '/robot_pose', 10)

        # /cmd_waypoints — TripPlanner sends pre-transformed paths here
        self.create_subscription(Path, '/cmd_waypoints', self._cmd_waypoints_cb, 1)

        # ---- Stanley subscribers ----
        self.create_subscription(Float32, '/lane_stanley/delta', self._stanley_delta_cb, 2)
        self.create_subscription(Float32, '/lane_stanley/trust', self._stanley_trust_cb, 2)

        # ---- Debug publishers for live plot ----
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
    # /cmd_waypoints — TripPlanner sends path already in ROS/map frame
    # ------------------------------------------------------------------
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
        self.path_execute_flag = True
        self._wp_in_ros_frame = True
        # Reset intersection state for new path
        self._isec_state      = IntersectionState.DRIVE
        self._isec_wpi        = -1
        self._isec_nodes_done = set()
        self._intersection_rule = "NONE"
        # Find which wpi indices correspond to intersection nodes
        self._intersection_wpi = self._mark_intersection_waypoints()
        self.get_logger().info(
            f'/cmd_waypoints received N={self.N} '
            f'intersection_wpi={list(self._intersection_wpi.keys())}')

    # ------------------------------------------------------------------
    # Intersection waypoint marking
    # ------------------------------------------------------------------
    def _mark_intersection_waypoints(self):
        """
        For each intersection node, find the single closest waypoint index
        in self.wp (already in ROS frame). Returns dict: wpi -> node_id.
        """
        mapping = {}
        if not self._node_xy or self.N == 0:
            return mapping

        angle_offset = self.rotation_offset[0]
        R = np.array([
            [np.cos(-angle_offset * np.pi / 180), -np.sin(-angle_offset * np.pi / 180)],
            [np.sin(-angle_offset * np.pi / 180),  np.cos(-angle_offset * np.pi / 180)]
        ])
        t = np.array(self.translation_offset)

        for nid, xy_q in self._node_xy.items():
            # Transform node from QLabs → ROS frame (same as path waypoints)
            xy_ros = (xy_q + t) @ R
            # Find nearest waypoint
            dists = np.linalg.norm(
                self.wp.T - xy_ros, axis=1)   # shape (N,)
            best_wpi = int(np.argmin(dists))
            # Only mark if close enough to actually be on this path
            if dists[best_wpi] < 0.8:
                # Don't double-mark same wpi for two nodes
                if best_wpi not in mapping:
                    mapping[best_wpi] = nid
                    self.get_logger().info(
                        f'  node {nid} → wpi {best_wpi} dist={dists[best_wpi]:.3f}m')

        return mapping

    # ------------------------------------------------------------------
    # Stanley callbacks
    # ------------------------------------------------------------------
    def _stanley_delta_cb(self, msg: Float32):
        self.stanley_delta = float(msg.data)

    def _stanley_trust_cb(self, msg: Float32):
        self.stanley_trust = float(msg.data)

    # ------------------------------------------------------------------
    # Steering blend
    # ------------------------------------------------------------------
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
                if self._wp_in_ros_frame:
                    wp_1_mod = wp_1  # already in ROS/map frame — TripPlanner did the transform
                else:
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

                pp_delta = np.arctan2(2 * L * np.sin(psi), WaypointDist)

                dist = np.linalg.norm([p[0] - wp_1_mod[0], p[1] - wp_1_mod[1]])

                v_eff = max(self.qcar2_measurred_speed, 0.05)
                lookahead_dist = max(v_eff * 1.7, 0.30)

                if dist < lookahead_dist:
                    if self.wpi < self.N - 2:
                        self.wpi += skip_index

                self.wpi = np.clip(self.wpi, 0, self.N - 1)

                # ── End of path stop ──────────────────────────────────────────
                if self.wpi > self.N - 100:
                    speed_command = min(speed_command, 0.2)
                if self.wpi >= self.N - 1 and dist < 0.4:
                    speed_command = 0.0
                    pp_delta = 0.0
                    self.wp_prior = self.wp
                    self.path_complete = True

                # ── Intersection node state machine ───────────────────────────
                now_isec = time.time()
                node_id  = self._intersection_wpi.get(self.wpi)

                if (self._isec_state == IntersectionState.DRIVE
                        and node_id is not None
                        and node_id not in self._isec_nodes_done):
                    # Arrived at intersection node — hard stop
                    self._isec_state      = IntersectionState.STOPPED
                    self._isec_wpi        = self.wpi
                    self._intersection_rule = "NONE"
                    self.get_logger().info(
                        f'INTERSECTION node {node_id} @ wpi={self.wpi} — HARD STOP')
                    ai = Bool(); ai.data = True
                    self.at_intersection_pub.publish(ai)

                if self._isec_state == IntersectionState.STOPPED:
                    speed_command = 0.0   # hard stop
                    rule = self._intersection_rule.upper().strip()

                    if rule == "STOP":
                        self._isec_state     = IntersectionState.WAITING
                        self._isec_wait_until = now_isec + 3.0
                        self.get_logger().info('Intersection rule: STOP SIGN — waiting 3s')

                    elif rule == "YIELD":
                        self._isec_state     = IntersectionState.WAITING
                        self._isec_wait_until = now_isec + 1.5
                        self.get_logger().info('Intersection rule: YIELD — waiting 1.5s')

                    elif rule in ("GREEN", "YELLOW"):
                        self._isec_state = IntersectionState.DONE
                        self.get_logger().info(f'Intersection rule: {rule} — go')

                    elif rule == "NONE":
                        # Nothing detected after brief grace period — treat as yield
                        if not hasattr(self, '_isec_stopped_at'):
                            self._isec_stopped_at = now_isec
                        elif now_isec - self._isec_stopped_at > 1.0:
                            self._isec_state     = IntersectionState.WAITING
                            self._isec_wait_until = now_isec + 1.5
                            self.get_logger().info('Intersection: nothing detected — default yield 1.5s')
                            del self._isec_stopped_at

                if self._isec_state == IntersectionState.WAITING:
                    speed_command = 0.0
                    if now_isec >= self._isec_wait_until:
                        self._isec_state = IntersectionState.DONE

                if self._isec_state == IntersectionState.DONE:
                    nid = self._intersection_wpi.get(self._isec_wpi, -1)
                    self._isec_nodes_done.add(nid)
                    self._isec_state = IntersectionState.DRIVE
                    self._isec_wpi   = -1
                    ai = Bool(); ai.data = False
                    self.at_intersection_pub.publish(ai)
                    self.get_logger().info(f'Intersection node {nid} cleared — DRIVE')
                # ─────────────────────────────────────────────────────────────

                Kp_steering = 1.1
                kd_steering = 5
                gyro_filtered = self.apply_filter('gyro', self.gyroscope[2], self.a1, self.b1)

                pp_delta_damped = np.clip(
                    Kp_steering * pp_delta - gyro_filtered * np.pi / 180 * kd_steering,
                    -self.max_steering_angle,
                    self.max_steering_angle)

                # Blend pure pursuit with Stanley
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

            # Publish /robot_pose so TripPlanner can plan next leg
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
                self.stanley_delta,
                self.current_steering,
                float(np.clip(self.stanley_blend * self.stanley_trust, 0.0, 1.0))
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