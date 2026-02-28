# =========================
# OVERALL: path_follower.py  (FULL FILE)
# ✅ Adds /path_follower/qcar_state publisher so your tripPlanner subscription works
# =========================
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
from enum import Enum

# ROS specific packages
from rclpy.duration import Duration # Handles time for ROS 2
import rclpy # Python client library for ROS 2
from geometry_msgs.msg import PoseStamped # Pose with ref frame and timestamp
from rclpy.node import Node
from nav_msgs.msg import Path
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Imu, JointState
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Bool, UInt8   # ✅ ADDED UInt8


'''
Description:

Navigates a robot from an initial pose to a goal pose described by a series of
given nodes based on Quanser's SDCSRoadMap class.

UPGRADE:
- You can provide pickup/dropoff as XY coordinates in meters (QLabs map meters)
- It will snap current pose -> closest node (FIXED: ROS->QLabs conversion)
- Snap goal XY -> closest node
- A* shortest path between nodes
- Append exact goal XY as final waypoint so you actually end at that coordinate

MISSION:
- Provide TWO coordinates: pickup_xy and dropoff_xy
- Go to pickup, STOP 3s
- Go to dropoff, STOP 3s
- Go back to Taxi Hub (FIXED to node 10), DONE
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

    def f(self, X, u, dt):
        return X + dt * u[0] * np.array([
            [np.cos(X[2,0])],
            [np.sin(X[2,0])],
            [np.tan(u[1]) / self.L]
        ])

    def Jf(self, X, u, dt):
        return np.array([
            [1, 0, -dt*u[0]*np.sin(X[2,0])],
            [0, 1, dt*u[0]*np.cos(X[2,0])],
            [0, 0, 1]
        ])

    def prediction(self, dt, u):
        F = self.Jf(self.xHat, u, dt)
        self.P = F@self.P@np.transpose(F) + self.Q

        self.xHat = self.f(self.xHat, u, dt)
        self.xHat[2] = wrap_to_pi(self.xHat[2])
        return

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

    def prediction(self, dt, u):
        Ad = self.I + self.A*dt
        self.xHat = Ad@self.xHat + dt*self.B*u
        self.P = Ad@self.P@np.transpose(Ad) + self.Q

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


class MissionStage(Enum):
    IDLE = 0
    TO_PICKUP = 1
    WAIT_AT_PICKUP = 2
    TO_DROPOFF = 3
    WAIT_AT_DROPOFF = 4
    TO_HUB = 5
    DONE = 6


class PathFollower(Node):

    def __init__(self):
        super().__init__('path_follower')

        # Roadmap (we use it for snapping pose->node and A*)
        self.roadmap = SDCSRoadMap()


        self.qcar_state_pub = self.create_publisher(UInt8, '/path_follower/qcar_state', 10)
        self._last_qcar_state = None
        self._publish_qcar_state(5)   # MAGENTA at hub/ready at startup

        # ---------------- Existing params ----------------
        self.declare_parameter('node_values', [0, 8, 10])
        self.waypoints = list(self.get_parameter("node_values").get_parameter_value().integer_array_value)

        self.declare_parameter('desired_speed', [0.6])
        self.desired_speed = list(self.get_parameter("desired_speed").get_parameter_value().double_array_value)

        self.declare_parameter('visualize_pose', [False])
        self.pose_visualize_flag = list(self.get_parameter("visualize_pose").get_parameter_value().bool_array_value)[0]

        self.scale = 1.0

        self.declare_parameter('rotation_offset', [82.0])
        self.rotation_offset = list(self.get_parameter("rotation_offset").get_parameter_value().double_array_value)

        self.declare_parameter('translation_offset', [0.0, 0.0])
        self.translation_offset = list(self.get_parameter("translation_offset").get_parameter_value().double_array_value)

        self.declare_parameter('start_path', [True])
        self.path_execute_flag = list(self.get_parameter("start_path").get_parameter_value().bool_array_value)[0]

        # ---------------- NEW mission params ----------------
        self.declare_parameter('mission_enable', [False])
        self.mission_enable = list(self.get_parameter("mission_enable").get_parameter_value().bool_array_value)[0]

        self.declare_parameter('mission_use_xy', [True])
        self.mission_use_xy = list(self.get_parameter("mission_use_xy").get_parameter_value().bool_array_value)[0]

        self.declare_parameter('mission_pickup_xy', [0.0, 0.0])
        self.pickup_xy = list(self.get_parameter("mission_pickup_xy").get_parameter_value().double_array_value)

        self.declare_parameter('mission_dropoff_xy', [0.0, 0.0])
        self.dropoff_xy = list(self.get_parameter("mission_dropoff_xy").get_parameter_value().double_array_value)

        self.declare_parameter('mission_stop_seconds', [3.0])
        self.stop_seconds = float(list(self.get_parameter("mission_stop_seconds").get_parameter_value().double_array_value)[0])

        # mission state
        self.mission_stage = MissionStage.IDLE
        self.mission_initialized = False
        self.current_goal_desc = "none"
        self.mission_pause_until = 0.0

        self.add_on_set_parameters_callback(self.parameter_update_callback)

        # Declare and acquire `target_frame` parameter
        self.target_frame = self.declare_parameter(
            'target_frame', 'base_link').get_parameter_value().string_value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # parameters common to all methods
        self.dt = 1/80

        # EKF init
        x0 = np.zeros((3,1))
        P0 = np.eye(3)
        R_combined = np.diagflat([0.1, 0.1, 0.01])

        self.qcar2_ekf = QcarEKF(
            x0=x0,
            P0=P0,
            Q=np.diagflat([0.0001, 0.0001, 0.001]),
            R=R_combined)

        self.pose_ekf = np.zeros((3,1))

        self.gyro_kf = GyroKF(
            x0=np.zeros((2,1)),
            P0=np.eye(2),
            Q=np.diagflat([0.01, 0.01]),
            R=np.diagflat([.1])
        )

        # filter parameters
        self.yaw = 0
        self.cutoff_frequency_filter = 15.0
        self.a1, self.b1 = self.filter_coefficients(self.cutoff_frequency_filter, self.dt)

        # timers
        self.path_control_timer = self.create_timer(self.dt, self.path_planner)
        self.timer = self.create_timer(self.dt, self.tf_timer)

        # TF pose placeholders
        self.translation = [0,0,0]
        self.rotation = [0,0,0]

        # start with manual node_values path until mission replans
        self.wp = self.roadmap.generate_path(self.waypoints) * self.scale
        if self.wp is None:
            self.wp = np.zeros((2,1))
        self.N = self.wp.shape[1]
        self.wpi = 0
        self.wp_prior = []
        self.current_steering = 0

        self.publisher = self.create_publisher(Twist,'/cmd_vel_nav', 1)
        self.max_steering_angle = 0.6

        self.joint_state_subscriber = self.create_subscription(JointState, '/qcar2_joint', self.joint_state_callback, 1)
        self.qcar2_measurred_speed = 0

        self.object_detection_flag = self.create_subscription(Bool, '/motion_enable', self.object_detector_callback, 1)
        self.motion_flag = True
        self.path_complete = False

        self.imu_subscrition = self.create_subscription(Imu, '/qcar2_imu', self.imu_callback, 10)
        self.gyroscope = [0,0,0]

        self.path_publisher_topic = self.create_publisher(Path, '/planned_path', 1)
        self.path_status_publisher = self.create_publisher(Bool, '/path_status', 1)

        # ---------------- TAXI HUB FIX ----------------
        self.hub_xy = list(self._get_node_xy(10))
        self.get_logger().info(f"Taxi hub locked to node 10 => hub_xy={self.hub_xy}")
        # ------------------------------------------------

        # ---------------- ✅ NEW: publish qcar_state for LEDs ----------------
        # 2 = BLUE pickup
        # 4 = GREEN driving
        # 5 = MAGENTA hub/ready
        # 6 = ORANGE dropoff
        # --------------------------------------------------------------------

        # Multiscope info
        self.t0 = time.time()
        self.t_plot = 0
        self.plot_visualized = False
        self.scopeTimer = self.create_timer(0.1, self.scopeDataTimer)

    # ---------------- ✅ NEW: qcar_state publisher helpers ----------------
    def _publish_qcar_state(self, s: int):
        s = int(s)
        if self._last_qcar_state == s:
            return
        self._last_qcar_state = s
        m = UInt8()
        m.data = s
        self.qcar_state_pub.publish(m)

    def _update_led_state_from_context(self):
        """
        Keeps LEDs correct even if we miss a single transition.
        """
        if self.mission_enable:
            if self.mission_stage == MissionStage.WAIT_AT_PICKUP:
                self._publish_qcar_state(2)   # BLUE
            elif self.mission_stage == MissionStage.WAIT_AT_DROPOFF:
                self._publish_qcar_state(6)   # ORANGE
            elif self.mission_stage == MissionStage.DONE:
                self._publish_qcar_state(5)   # MAGENTA
            else:
                # driving stages: TO_PICKUP / TO_DROPOFF / TO_HUB
                self._publish_qcar_state(4)   # GREEN
        else:
            # manual path: green while moving, magenta when complete
            if self.path_execute_flag and self.motion_flag and not self.path_complete:
                self._publish_qcar_state(4)   # GREEN
            else:
                self._publish_qcar_state(5)   # MAGENTA
    # --------------------------------------------------------------------

    # --------------------- FRAME FIX HELPERS ---------------------
    def _R_qlabs_to_ros_2d(self):
        angle_offset = float(self.rotation_offset[0])
        return np.array([
            [np.cos(-angle_offset*np.pi/180.0), -np.sin(-angle_offset*np.pi/180.0)],
            [np.sin(-angle_offset*np.pi/180.0),  np.cos(-angle_offset*np.pi/180.0)]
        ])

    def _ros_to_qlabs_xy(self, x_ros, y_ros):
        Rq = self._R_qlabs_to_ros_2d()
        t = np.array([float(self.translation_offset[0]), float(self.translation_offset[1])])
        p_ros = np.array([float(x_ros), float(y_ros)])
        p_qlabs = (p_ros @ Rq.T) - t
        return float(p_qlabs[0]), float(p_qlabs[1])

    # --------------------- Roadmap helpers ---------------------
    def _roadmap_node_count(self):
        if hasattr(self.roadmap, "nodes"):
            try:
                return len(self.roadmap.nodes)
            except Exception:
                pass
        if hasattr(self.roadmap, "get_node_pose"):
            i = 0
            while True:
                try:
                    _ = self.roadmap.get_node_pose(i)
                    i += 1
                    if i > 1000:
                        return i
                except Exception:
                    return i
        return 0

    def _get_node_xy(self, node_id):
        if hasattr(self.roadmap, "get_node_pose"):
            pose = np.array(self.roadmap.get_node_pose(node_id)).reshape(-1)
            return float(pose[0]), float(pose[1])
        if hasattr(self.roadmap, "nodes"):
            pose = np.array(self.roadmap.nodes[node_id].pose).reshape(-1)
            return float(pose[0]), float(pose[1])
        raise RuntimeError("SDCSRoadMap does not expose node poses in a known way.")

    def _closest_node_to_xy(self, x, y):
        n = self._roadmap_node_count()
        if n <= 0:
            self.get_logger().error("Could not determine roadmap node count.")
            return 0

        best_i = 0
        best_d = float("inf")
        for i in range(n):
            try:
                nx, ny = self._get_node_xy(i)
                d = (nx - x)**2 + (ny - y)**2
                if d < best_d:
                    best_d = d
                    best_i = i
            except Exception:
                continue
        return best_i

    def _plan_shortest_path_nodes(self, start_node, goal_node):
        if hasattr(self.roadmap, "find_shortest_path"):
            return self.roadmap.find_shortest_path(start_node, goal_node)
        return self.roadmap.generate_path([start_node, goal_node])

    def _set_new_waypoint_path(self, path_2xn, info=""):
        if path_2xn is None or np.size(path_2xn) == 0:
            self.get_logger().error(f"Planner returned empty path. {info}")
            self.wp = np.zeros((2,1))
            self.N = 1
            self.wpi= 0
            self.path_complete = True
            return

        wp = np.array(path_2xn)
        if wp.ndim != 2 or wp.shape[0] != 2:
            self.get_logger().error(f"Planned path has wrong shape: {wp.shape}. {info}")
            self.wp = np.zeros((2,1))
            self.N = 1
            self.wpi = 0
            self.path_complete = True
            return

        self.wp = wp * self.scale
        self.N = self.wp.shape[1]
        self.wpi = 0
        self.path_complete = False
        self.wp_prior = self.wp
        self.get_logger().info(f"New path set. N={self.N}. {info}")

    def _append_goal_xy(self, path_2xn, goal_xy):
        wp = np.array(path_2xn)
        goal_xy = np.array(goal_xy).reshape(2,)
        if wp.ndim != 2 or wp.shape[0] != 2:
            return wp
        goal_col = goal_xy.reshape(2,1)
        return np.hstack([wp, goal_col])

    def _plan_leg_to_xy(self, goal_xy, stage_name=""):
        try:
            px_ros = float(self.translation.x)
            py_ros = float(self.translation.y)
        except Exception:
            self.get_logger().warn("No TF pose yet; cannot plan leg.")
            return False

        px_qlabs, py_qlabs = self._ros_to_qlabs_xy(px_ros, py_ros)

        start_node = self._closest_node_to_xy(px_qlabs, py_qlabs)
        goal_node = self._closest_node_to_xy(float(goal_xy[0]), float(goal_xy[1]))

        base_path = self._plan_shortest_path_nodes(start_node, goal_node)
        if base_path is None or np.size(base_path) == 0:
            self.get_logger().error(f"A* returned empty path. [{stage_name}]")
            return False

        full_path = self._append_goal_xy(base_path, goal_xy)

        self.current_goal_desc = (
            f"{stage_name} goal_xy=({goal_xy[0]:.2f},{goal_xy[1]:.2f}) "
            f"start_node={start_node} goal_node={goal_node} "
            f"start_pose_qlabs=({px_qlabs:.2f},{py_qlabs:.2f})"
        )
        self._set_new_waypoint_path(full_path, info=self.current_goal_desc)
        return True

    # --------------------- Parameter callback ---------------------
    def parameter_update_callback(self, params):
        for param in params:

            if param.name == 'node_values' and param.type_ == param.Type.INTEGER_ARRAY:
                self.waypoints = list(param.value)
                wp_new = self.roadmap.generate_path(self.waypoints) * self.scale
                self._set_new_waypoint_path(wp_new, info="[node_values updated]")
                self.get_logger().info('nodes updated!')
                print(self.waypoints)

            elif param.name == 'desired_speed' and param.type_ == param.Type.DOUBLE_ARRAY:
                self.desired_speed = list(param.value)
                self.get_logger().info('new desired speed...')
                print(self.desired_speed)

            elif param.name == 'rotation_offset' and param.type_ == param.Type.DOUBLE_ARRAY:
                self.rotation_offset = list(param.value)

            elif param.name == 'translation_offset' and param.type_ == param.Type.DOUBLE_ARRAY:
                self.translation_offset = list(param.value)

            elif param.name == 'start_path' and param.type_ == param.Type.BOOL_ARRAY:
                self.path_execute_flag = list(param.value)[0]
                self.get_logger().info('path status changed!')

            # --- mission params ---
            elif param.name == 'mission_enable' and param.type_ == param.Type.BOOL_ARRAY:
                self.mission_enable = list(param.value)[0]
                self.get_logger().info(f"mission_enable set to {self.mission_enable}")
                self.mission_initialized = False
                self.mission_stage = MissionStage.IDLE
                self.path_complete = False
                self.mission_pause_until = 0.0

            elif param.name == 'mission_use_xy' and param.type_ == param.Type.BOOL_ARRAY:
                self.mission_use_xy = list(param.value)[0]
                self.get_logger().info(f"mission_use_xy set to {self.mission_use_xy}")
                self.mission_initialized = False
                self.mission_stage = MissionStage.IDLE
                self.path_complete = False
                self.mission_pause_until = 0.0

            elif param.name == 'mission_pickup_xy' and param.type_ == param.Type.DOUBLE_ARRAY:
                self.pickup_xy = list(param.value)
                self.get_logger().info(f"pickup_xy set to {self.pickup_xy}")
                self.mission_initialized = False
                self.mission_stage = MissionStage.IDLE

            elif param.name == 'mission_dropoff_xy' and param.type_ == param.Type.DOUBLE_ARRAY:
                self.dropoff_xy = list(param.value)
                self.get_logger().info(f"dropoff_xy set to {self.dropoff_xy}")
                self.mission_initialized = False
                self.mission_stage = MissionStage.IDLE

            elif param.name == 'mission_stop_seconds' and param.type_ == param.Type.DOUBLE_ARRAY:
                self.stop_seconds = float(list(param.value)[0])
                self.get_logger().info(f"mission_stop_seconds set to {self.stop_seconds}")

            return SetParametersResult(successful=True)

        return SetParametersResult(successful=True)

    # --------------------- Filters / callbacks ---------------------
    def filter_coefficients(self, freq, dt):
        nyq_freq = 0.5*(1/dt)
        norm_cut = freq/nyq_freq
        b, a = signal.butter(2, norm_cut)
        self.hist = {'gyro': {'in': [0.0]*3, 'out': [0.0]*3}}
        return a, b

    def apply_filter(self, key, new_input, a, b):
        h = self.hist[key]
        h['in'] = [new_input] + h['in'][:2]
        y = (
            b[0]*h['in'][0] +
            b[1]*h['in'][1] +
            b[2]*h['in'][2] -
            a[1]*h['out'][0] -
            a[2]*h['out'][1]
        )
        h['out'] = [y] + h['out'][:2]
        return y

    def object_detector_callback(self, msg):
        self.motion_flag = msg.data

    def joint_state_callback(self, msg):
        self.qcar2_measurred_speed = (msg.velocity[0]/(720.0*4.0))*((13.0*19.0)/(70.0*30.0))*(2.0*np.pi)*0.033

    def imu_callback(self, msg):
        self.gyroscope = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]

    # --------------------- Path publishing ---------------------
    def path_publisher(self):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"

        if self.wp is None or self.wp.shape[1] < 1:
            self.path_publisher_topic.publish(path_msg)
            return

        upto = min(self.wpi, self.wp.shape[1]-1)
        for i in range(upto):
            pose = PoseStamped()

            angle_offset = self.rotation_offset[0]
            R_QLabs_ROS = np.array([[np.cos(-angle_offset*np.pi/180), -np.sin(-angle_offset*np.pi/180)],
                                    [np.sin(-angle_offset*np.pi/180),  np.cos(-angle_offset*np.pi/180)]])
            t = np.array([self.translation_offset[0], self.translation_offset[1]])

            wp_i_mod = (np.array([self.wp[0,i], self.wp[1,i]]) + t) @ R_QLabs_ROS

            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = "map"
            pose.pose.position.x = wp_i_mod[0]
            pose.pose.position.y = wp_i_mod[1]
            path_msg.poses.append(pose)

        self.path_publisher_topic.publish(path_msg)

    # --------------------- Mission helpers ---------------------
    def _mission_in_pause(self):
        return time.time() < self.mission_pause_until

    def _start_pause(self, seconds, reason=""):
        self.mission_pause_until = time.time() + float(seconds)
        if reason:
            self.get_logger().info(f"MISSION PAUSE: {reason} for {seconds:.1f}s")

    # --------------------- Main control loop ---------------------
    def path_planner(self):

        enable = 1.0
        speed_command = self.desired_speed[0]
        self.t_plot = time.time() - self.t0

        # update ekf filters
        self.ekf_filter_timer()

        # publish path every 2 seconds
        if round(self.t_plot) % 2 == 0:
            self.path_publisher()

        # =======================
        #   Mission logic (XY)
        # =======================
        if self.mission_enable:

            if self._mission_in_pause():
                speed_command = 0.0
                self.current_steering = 0.0
                self.path_complete = True

            
            else:
                if not self.mission_initialized:
                    ok = self._plan_leg_to_xy(self.pickup_xy, stage_name="TO_PICKUP") if self.mission_use_xy else False
                    if ok:
                        self.mission_initialized = True
                        self.mission_stage = MissionStage.TO_PICKUP
                        self._publish_qcar_state(4)  # ✅ driving => GREEN

                if self.path_complete and self.mission_initialized:

                    if self.mission_stage == MissionStage.TO_PICKUP:
                        self.get_logger().info("Arrived at PICKUP. Stopping 3s...")
                        self.mission_stage = MissionStage.WAIT_AT_PICKUP
                        self._publish_qcar_state(2)  # ✅ pickup => BLUE
                        self._start_pause(self.stop_seconds, reason="at PICKUP")

                    elif self.mission_stage == MissionStage.WAIT_AT_PICKUP:
                        self.get_logger().info("Leaving PICKUP. Planning to DROPOFF...")
                        ok = self._plan_leg_to_xy(self.dropoff_xy, stage_name="TO_DROPOFF") if self.mission_use_xy else False
                        if ok:
                            self.mission_stage = MissionStage.TO_DROPOFF
                            self._publish_qcar_state(4)  # ✅ driving => GREEN
                            self.path_complete = False

                    elif self.mission_stage == MissionStage.TO_DROPOFF:
                        self.get_logger().info("Arrived at DROPOFF. Stopping 3s...")
                        self.mission_stage = MissionStage.WAIT_AT_DROPOFF
                        self._publish_qcar_state(6)  # ✅ dropoff => ORANGE
                        self._start_pause(self.stop_seconds, reason="at DROPOFF")

                    elif self.mission_stage == MissionStage.WAIT_AT_DROPOFF:
                        self.get_logger().info(f"Leaving DROPOFF. Planning back to HUB (node 10) hub_xy={self.hub_xy}...")
                        ok = self._plan_leg_to_xy(self.hub_xy, stage_name="TO_HUB")
                        if ok:
                            self.mission_stage = MissionStage.TO_HUB
                            self._publish_qcar_state(4)  # ✅ driving => GREEN
                            self.path_complete = False

                    elif self.mission_stage == MissionStage.TO_HUB:
                        self.get_logger().info("Arrived at HUB. Mission DONE.")
                        self.mission_stage = MissionStage.DONE
                        self._publish_qcar_state(5)  # ✅ hub/ready => MAGENTA
                        self.path_complete = True

        # =======================
        #   Track current path
        # =======================
        try:
            if self.mission_enable and self.mission_stage == MissionStage.DONE:
                speed_command = 0.0
                self.current_steering = 0.0
                self.path_complete = True

            if self.mission_enable and self._mission_in_pause():
                speed_command = 0.0
                self.current_steering = 0.0
                self.path_complete = True

            else:
                if self.wp is None or self.wp.shape[1] < 2:
                    speed_command = 0.0
                    self.current_steering = 0.0
                    self.path_complete = True
                else:
                    self.N = self.wp.shape[1]
                    self.wpi = int(np.clip(self.wpi, 0, max(self.N - 2, 0)))

                    wp_1 = np.array(self.wp[:, self.wpi])

                    angle_offset = self.rotation_offset[0]
                    R_QLabs_ROS = np.array([[np.cos(-angle_offset*np.pi/180), -np.sin(-angle_offset*np.pi/180)],
                                            [np.sin(-angle_offset*np.pi/180),  np.cos(-angle_offset*np.pi/180)]])
                    t_off = np.array([self.translation_offset[0], self.translation_offset[1]])
                    wp_1_mod = (wp_1 + t_off) @ R_QLabs_ROS

                    wp_next = np.array(self.wp[:, min(self.wpi+1, self.N-1)])
                    wp_next_mod = (wp_next + t_off) @ R_QLabs_ROS

                    dx = wp_next_mod[0] - wp_1_mod[0]
                    dy = wp_next_mod[1] - wp_1_mod[1]
                    path_heading = np.arctan2(dy, dx)

                    self.get_logger().info(f"Path heading (deg): {np.degrees(path_heading):.2f}")

                    cte = self._cte_to_waypoint(wp_1_mod)

                    L = 0.256

                    try:
                        p_map = np.array([float(self.translation.x), float(self.translation.y)])
                        th = float(self.yaw)
                    except Exception:
                        p_map = np.array([0.0, 0.0])
                        th = 0.0

                    v_map = (wp_1_mod - p_map).reshape(2, 1)

                    Rot = np.array([[np.cos(th), -np.sin(th)],
                                    [np.sin(th),  np.cos(th)]])
                    v_car = (Rot.T @ v_map).reshape(2,)

                    WaypointDist = float(np.linalg.norm(v_car))
                    WaypointDist = max(WaypointDist, 0.05)
                    psi = float(np.arctan2(v_car[1], v_car[0]))

                    delta = float(np.arctan2(2 * L * np.sin(psi), WaypointDist))

                    dist_to_target = float(np.linalg.norm(p_map - wp_1_mod))

                    wp_final = np.array(self.wp[:, -1])
                    wp_final_mod = (wp_final + t_off) @ R_QLabs_ROS
                    dist_to_final = float(np.linalg.norm(p_map - wp_final_mod))

                    v_eff = max(self.qcar2_measurred_speed, 0.05)
                    lookahead_dist = max(0.30, v_eff * 1.7)

                    # heading_err = self._heading_error_to_path(self.wpi)

                    if dist_to_target < lookahead_dist and self.wpi < self.N - 2:
                        self.wpi += 1

                    near_final = (dist_to_final < 0.50)          # looser tolerance
                    at_end = (self.wpi >= self.N - 2)            # reached end of waypoint list

                    if near_final or at_end:
                        speed_command = 0.0
                        self.current_steering = 0.0
                        self.path_complete = True

                    Kp_steering = 1.1
                    kd_steering = 7

                    gyro_filtered = self.apply_filter('gyro', self.gyroscope[2], self.a1, self.b1)

                    steering = np.clip(
                        Kp_steering * delta - gyro_filtered * np.pi/180 * kd_steering,
                        -self.max_steering_angle,
                        self.max_steering_angle)

                    self.current_steering = steering

                    if int(self.t_plot * 5) != int((self.t_plot - self.dt) * 5):
                        stage = self.mission_stage.name if self.mission_enable else "MANUAL"
                        self.get_logger().info(
                            f"stage={stage} wpi={self.wpi}/{self.N} "
                            f"Ld={lookahead_dist:.2f} distT={dist_to_target:.2f} distF={dist_to_final:.2f} "
                            f"delta={delta:.2f} gyro={gyro_filtered:.3f} steer={steering:.3f} v={speed_command:.2f} "
                            f"{self.current_goal_desc}"
                        )

        except KeyboardInterrupt:
            speed_command = 0.0
            self.current_steering = 0.0

        if self.path_execute_flag and self.motion_flag and not self.path_complete:
            enable = 1.0
        else:
            enable = 0.0

        # ✅ ALWAYS push LED state based on current stage/context
        # self._update_led_state_from_context()

        self.nav_command(enable, speed_command)
        self.path_status()

    def nav_command(self, enable, speed_command):
        QCarCommands = Twist()
        QCarCommands.linear.x = enable*np.clip(speed_command*np.power(np.cos(self.current_steering), 2), 0.05, 0.7)
        QCarCommands.angular.z = enable*self.current_steering
        self.publisher.publish(QCarCommands)

    def path_status(self):
        msg = Bool()
        msg.data = self.path_complete
        self.path_status_publisher.publish(msg)

    # --------------------- TF + EKF ---------------------
    def tf_timer(self):
        from_frame_rel= "map"
        to_frame_rel = self.target_frame

        try:
            t = self.tf_buffer.lookup_transform(from_frame_rel, to_frame_rel, rclpy.time.Time())
            self.translation = t.transform.translation

            rotation = [t.transform.rotation.x,
                        t.transform.rotation.y,
                        t.transform.rotation.z,
                        t.transform.rotation.w]
            roll, pitch, self.yaw = R.from_quat(rotation).as_euler('xyz')

            self.gyro_kf.correction(self.yaw)

            y = np.array([
                [self.translation.x],
                [self.translation.y],
                [self.gyro_kf.xHat[0,0]]
            ])
            self.qcar2_ekf.correction(y)

        except TransformException as ex:
            self.get_logger().info(f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
            return

    def ekf_filter_timer(self):
        speed = self.qcar2_measurred_speed
        delta = self.current_steering
        self.qcar2_ekf.prediction(self.dt, [speed, delta])

        try:
            th_gyro = self.gyroscope[2]
        except AttributeError:
            th_gyro = 0
        self.gyro_kf.prediction(self.dt, th_gyro)

    # --------------------- Visualization ---------------------
    def scopeDataTimer(self):
        if self.pose_visualize_flag:
            p = [self.qcar2_ekf.xHat[0,0], self.qcar2_ekf.xHat[1,0], self.qcar2_ekf.xHat[2,0]]

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
            self.steeringScope.axes[3].sample(self.t_plot, [self.yaw, self.qcar2_ekf.xHat[2,0]])

            MultiScope.refreshAll()
        else:
            try:
                self.steeringScope.graphicsLayoutWidget.close()
            except AttributeError:
                pass

    def _path_heading_at_index(self, i: int):
        if self.wp is None or self.wp.shape[1] < 2:
            return 0.0

        N = self.wp.shape[1]
        i = int(np.clip(i, 0, N - 2))

        angle_offset = float(self.rotation_offset[0])
        R_QLabs_ROS = np.array([
            [np.cos(-angle_offset*np.pi/180.0), -np.sin(-angle_offset*np.pi/180.0)],
            [np.sin(-angle_offset*np.pi/180.0),  np.cos(-angle_offset*np.pi/180.0)]
        ])
        t_off = np.array([float(self.translation_offset[0]), float(self.translation_offset[1])])

        p0 = (np.array([self.wp[0, i],   self.wp[1, i]])   + t_off) @ R_QLabs_ROS
        p1 = (np.array([self.wp[0, i+1], self.wp[1, i+1]]) + t_off) @ R_QLabs_ROS

        dx = float(p1[0] - p0[0])
        dy = float(p1[1] - p0[1])
        return float(np.arctan2(dy, dx))

    def _heading_error_to_path(self, idx_for_heading: int):
        path_h = self._path_heading_at_index(idx_for_heading)
        try:
            robot_yaw = float(self.yaw)
        except Exception:
            robot_yaw = 0.0
        return float(wrap_to_pi(path_h - robot_yaw))

    def _cte_to_waypoint(self, wp_map_xy):
        try:
            rx = float(self.translation.x)
            ry = float(self.translation.y)
            yaw = float(self.yaw)
        except Exception:
            return 0.0

        dx = float(wp_map_xy[0] - rx)
        dy = float(wp_map_xy[1] - ry)

        c = np.cos(yaw)
        s = np.sin(yaw)
        y_r = -s*dx + c*dy
        return float(y_r)


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