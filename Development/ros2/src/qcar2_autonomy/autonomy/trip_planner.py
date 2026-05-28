#!/usr/bin/env python3

import time
import numpy as np
from enum import Enum

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Bool
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, SetParametersResult
from rclpy.parameter import ParameterType

from hal.products.mats import SDCSRoadMap


class MissionStage(Enum):
    IDLE           = 0
    TO_PICKUP      = 1
    WAIT_AT_PICKUP = 2
    TO_DROPOFF     = 3
    WAIT_AT_DROPOFF = 4
    TO_HUB         = 5
    TO_INTERMEDIATE = 6
    WAIT_AT_INTERMEDIATE = 7


class TripPlanner(Node):

    def __init__(self):
        super().__init__('trip_planner')

        self.roadmap = SDCSRoadMap()

        self.qcar_hardware_client = self.create_client(SetParameters, '/qcar2_hardware/set_parameters')
        if not self.qcar_hardware_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('qcar2_hardware LED service not available — LEDs disabled')
            self.qcar_hardware_client = None
        else:
            self.get_logger().info('connected to qcar2_hardware.')

        self.declare_parameter('taxi_node',          [10])
        self.declare_parameter('trip_nodes',         [-1])
        self.declare_parameter('pickup_node',        [-1])
        self.declare_parameter('dropoff_node',       [-1])
        self.declare_parameter('stop_seconds',       [3.0])
        self.declare_parameter('rotation_offset',    [90.0])
        self.declare_parameter('translation_offset', [0.0, 0.0])

        self.taxi_node          = int(list(self.get_parameter('taxi_node').get_parameter_value().integer_array_value)[0])
        self.trip_nodes         = self._normalize_node_list(
            self.get_parameter('trip_nodes').get_parameter_value().integer_array_value)
        self.pickup_node        = int(list(self.get_parameter('pickup_node').get_parameter_value().integer_array_value)[0])
        self.dropoff_node       = int(list(self.get_parameter('dropoff_node').get_parameter_value().integer_array_value)[0])
        self.stop_seconds       = float(list(self.get_parameter('stop_seconds').get_parameter_value().double_array_value)[0])
        self.rotation_offset    = list(self.get_parameter('rotation_offset').get_parameter_value().double_array_value)
        self.translation_offset = list(self.get_parameter('translation_offset').get_parameter_value().double_array_value)

        self.add_on_set_parameters_callback(self.parameter_update_callback)

        self.hub_xy     = list(self._get_node_xy(self.taxi_node))
        if len(self.trip_nodes) >= 2:
            self.pickup_node = self.trip_nodes[0]
            self.dropoff_node = self.trip_nodes[-1]
        self.pickup_xy  = list(self._get_node_xy(self.pickup_node))  if self.pickup_node  >= 0 else None
        self.dropoff_xy = list(self._get_node_xy(self.dropoff_node)) if self.dropoff_node >= 0 else None
        self.get_logger().info(
            f'Hub=node{self.taxi_node}{self.hub_xy}. '
            f'waiting for trip_nodes, e.g. `ros2 param set /trip_planner trip_nodes "[1, 8]"`.')

        self.LED_RED     = 0
        self.LED_GREEN   = 1
        self.LED_BLUE    = 2
        self.LED_MAGENTA = 5
        self.LED_ORANGE  = 6

        self.robot_pose             = None
        self.path_status            = False
        self._path_completed_event  = False
        self.motion_enabled         = True

        self._auto_aligned         = False
        self.startup_done          = False
        self._startup_path_sent    = False
        self.ready_for_rides       = False
        self.new_ride_requested    = False

        self.mission_stage = MissionStage.IDLE
        self.pause_until   = 0.0
        self.picked_up     = False
        self.dropped_off   = False
        self._last_led     = None
        self.active_trip_nodes = []
        self.active_goal_nodes = []
        self.active_goal_index = 0

        self.waypoints_pub = self.create_publisher(Path, '/cmd_waypoints', 1)

        self.create_subscription(Bool,        '/path_status',  self.path_status_callback,  10)
        self.create_subscription(PoseStamped, '/robot_pose',   self.robot_pose_callback,   10)
        self.create_subscription(Bool,        '/motion_enable', self._motion_enable_callback, 10)

        self._set_led(self.LED_MAGENTA)
        self.create_timer(0.1, self.loop)

    def path_status_callback(self, msg):
        prev = self.path_status
        self.path_status = bool(msg.data)
        if not prev and self.path_status:
            if ((not self.startup_done and self._startup_path_sent) or
                    self.mission_stage in (MissionStage.TO_PICKUP,
                                           MissionStage.TO_INTERMEDIATE,
                                           MissionStage.TO_DROPOFF,
                                           MissionStage.TO_HUB)):
                self._path_completed_event = True

    def robot_pose_callback(self, msg: PoseStamped):
        self.robot_pose = msg

    def _motion_enable_callback(self, msg):
        self.motion_enabled = bool(msg.data)

    def parameter_update_callback(self, params):
        ride_param_updated = False
        ride_param_names = {'trip_nodes', 'pickup_node', 'dropoff_node'}

        if any(p.name in ride_param_names for p in params):
            if not (self.ready_for_rides and self.mission_stage == MissionStage.IDLE):
                reason = 'trip_planner is not parked at the hub and ready for a new ride yet'
                self.get_logger().warn(reason)
                return SetParametersResult(successful=False, reason=reason)

        for p in params:
            if p.name == 'trip_nodes' and p.type_ == p.Type.INTEGER_ARRAY:
                nodes = self._normalize_node_list(p.value)
                invalid = self._invalid_nodes(nodes)
                if invalid:
                    reason = f'Invalid trip_nodes {invalid}; valid node IDs are 0..{self._node_count() - 1}'
                    self.get_logger().error(reason)
                    return SetParametersResult(successful=False, reason=reason)
                self.trip_nodes = nodes
                if len(nodes) >= 2:
                    self.pickup_node = nodes[0]
                    self.dropoff_node = nodes[-1]
                    self.pickup_xy = list(self._get_node_xy(self.pickup_node))
                    self.dropoff_xy = list(self._get_node_xy(self.dropoff_node))
                    self.get_logger().info(
                        f'trip_nodes updated: {self.trip_nodes}; '
                        f'node {self.taxi_node} will be appended automatically after the dropoff.')
                else:
                    self.pickup_node = -1
                    self.dropoff_node = -1
                    self.pickup_xy = None
                    self.dropoff_xy = None
                    self.get_logger().info('trip_nodes cleared; waiting for at least two ride nodes.')
                ride_param_updated = True
            elif p.name == 'pickup_node' and p.type_ == p.Type.INTEGER_ARRAY:
                values = list(p.value)
                pickup_node = int(values[0]) if values else -1
                if pickup_node >= 0 and not self._node_exists(pickup_node):
                    reason = f'Invalid pickup_node {pickup_node}; valid node IDs are 0..{self._node_count() - 1}'
                    self.get_logger().error(reason)
                    return SetParametersResult(successful=False, reason=reason)
                self.pickup_node = pickup_node
                self.trip_nodes = []
                self.pickup_xy = list(self._get_node_xy(self.pickup_node)) if self.pickup_node >= 0 else None
                self.get_logger().info(f'pickup_node updated: {self.pickup_node} -> {self.pickup_xy}')
                ride_param_updated = True
            elif p.name == 'dropoff_node' and p.type_ == p.Type.INTEGER_ARRAY:
                values = list(p.value)
                dropoff_node = int(values[0]) if values else -1
                if dropoff_node >= 0 and not self._node_exists(dropoff_node):
                    reason = f'Invalid dropoff_node {dropoff_node}; valid node IDs are 0..{self._node_count() - 1}'
                    self.get_logger().error(reason)
                    return SetParametersResult(successful=False, reason=reason)
                self.dropoff_node = dropoff_node
                self.trip_nodes = []
                self.dropoff_xy = list(self._get_node_xy(self.dropoff_node)) if self.dropoff_node >= 0 else None
                self.get_logger().info(f'dropoff_node updated: {self.dropoff_node} -> {self.dropoff_xy}')
                ride_param_updated = True
            elif p.name == 'stop_seconds' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.stop_seconds = float(list(p.value)[0])
            elif p.name == 'rotation_offset' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.rotation_offset = list(p.value)
            elif p.name == 'translation_offset' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.translation_offset = list(p.value)

        if ride_param_updated and self.ready_for_rides and self.mission_stage == MissionStage.IDLE:
            if self._pending_ride_nodes():
                self.new_ride_requested = True
            else:
                self.get_logger().info(
                    'Ride order incomplete; set trip_nodes "[pickup, dropoff]" '
                    'or set both pickup_node and dropoff_node.')

        return SetParametersResult(successful=True)

    def _normalize_node_list(self, values):
        return [int(v) for v in list(values) if int(v) >= 0]

    def _invalid_nodes(self, nodes):
        return [node for node in nodes if not self._node_exists(node)]

    def _node_exists(self, node_id):
        try:
            self._get_node_xy(int(node_id))
            return True
        except Exception:
            return False

    def _pending_ride_nodes(self):
        if len(self.trip_nodes) >= 2:
            return list(self.trip_nodes)
        if self.pickup_node >= 0 and self.dropoff_node >= 0:
            return [int(self.pickup_node), int(self.dropoff_node)]
        return []

    def _clear_pending_ride(self):
        self.trip_nodes = []
        self.pickup_node = -1
        self.dropoff_node = -1
        self.pickup_xy = None
        self.dropoff_xy = None

    def _rot_matrix(self):
        angle = float(self.rotation_offset[0]) * np.pi / 180.0
        return np.array([
            [np.cos(-angle), -np.sin(-angle)],
            [np.sin(-angle),  np.cos(-angle)]
        ])

    def _ros_to_qlabs(self, x, y):
        R_mat = self._rot_matrix()
        t     = np.array(self.translation_offset)
        return tuple((np.array([float(x), float(y)]) @ R_mat.T) - t)

    def _qlabs_path_to_ros(self, wp_2xn):
        R_mat   = self._rot_matrix()
        t_off   = np.array(self.translation_offset)
        path_msg = Path()
        path_msg.header.stamp    = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        for i in range(wp_2xn.shape[1]):
            pt   = (wp_2xn[:, i] + t_off) @ R_mat
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(pt[0])
            pose.pose.position.y = float(pt[1])
            path_msg.poses.append(pose)
        return path_msg

    def _get_node_xy(self, node_id):
        if hasattr(self.roadmap, 'get_node_pose'):
            pose = np.array(self.roadmap.get_node_pose(node_id)).reshape(-1)
        else:
            pose = np.array(self.roadmap.nodes[node_id].pose).reshape(-1)
        return float(pose[0]), float(pose[1])

    def _get_node_theta(self, node_id):
        if hasattr(self.roadmap, 'get_node_pose'):
            pose = np.array(self.roadmap.get_node_pose(node_id)).reshape(-1)
        else:
            pose = np.array(self.roadmap.nodes[node_id].pose).reshape(-1)
        return float(pose[2]) if pose.size >= 3 else 0.0

    def _auto_align(self):
        """One-shot calibration of rotation_offset/translation_offset.
        Assumes the car is parked at the hub node (taxi_node). Derives the
        QLabs->map transform from the hub node's known QLabs pose and the car's
        measured pose in the Cartographer map frame, so node coords convert to
        the live map frame without manual tuning."""
        if self.robot_pose is None:
            return False

        rx = float(self.robot_pose.pose.position.x)
        ry = float(self.robot_pose.pose.position.y)
        o  = self.robot_pose.pose.orientation
        map_yaw = float(np.arctan2(2.0 * (o.w * o.z + o.x * o.y),
                                   1.0 - 2.0 * (o.y * o.y + o.z * o.z)))

        hx, hy  = self._get_node_xy(self.taxi_node)
        h_theta = self._get_node_theta(self.taxi_node)

        # _qlabs_path_to_ros rotates QLabs vectors CCW by rotation_offset, so
        # map_yaw = qlabs_theta + rotation_offset.
        rot_rad = float(np.arctan2(np.sin(map_yaw - h_theta),
                                   np.cos(map_yaw - h_theta)))
        self.rotation_offset = [float(np.degrees(rot_rad))]

        R_mat = self._rot_matrix()
        t_off = np.array([rx, ry]) @ R_mat.T - np.array([hx, hy])
        self.translation_offset = [float(t_off[0]), float(t_off[1])]

        self.get_logger().info(
            f'Auto-aligned at hub node {self.taxi_node}: '
            f'rotation_offset={self.rotation_offset[0]:.2f} deg, '
            f'translation_offset=[{self.translation_offset[0]:.3f}, {self.translation_offset[1]:.3f}]')
        return True

    def _node_count(self):
        if hasattr(self.roadmap, 'nodes'):
            try:
                return len(self.roadmap.nodes)
            except Exception:
                pass
        if hasattr(self.roadmap, 'get_node_pose'):
            i = 0
            while True:
                try:
                    self.roadmap.get_node_pose(i)
                    i += 1
                    if i > 1000:
                        return i
                except Exception:
                    return i
        return 0

    def _closest_node(self, x, y):
        best_i, best_d = 0, float('inf')
        for i in range(self._node_count()):
            try:
                nx, ny = self._get_node_xy(i)
                d = (nx - x) ** 2 + (ny - y) ** 2
                if d < best_d:
                    best_d = d
                    best_i = i
            except Exception:
                continue
        return best_i

    def _plan_to_xy(self, goal_xy):
        if self.robot_pose is None:
            return None

        rx, ry   = float(self.robot_pose.pose.position.x), float(self.robot_pose.pose.position.y)
        px_q, py_q = self._ros_to_qlabs(rx, ry)

        start_node = self._closest_node(px_q, py_q)
        goal_node  = self._closest_node(float(goal_xy[0]), float(goal_xy[1]))
        goal_col = np.array(goal_xy).reshape(2, 1)

        if start_node == goal_node:
            cur_col = np.array([px_q, py_q]).reshape(2, 1)
            if float(np.linalg.norm(cur_col - goal_col)) < 0.01:
                return goal_col
            return np.hstack([cur_col, goal_col])

        if hasattr(self.roadmap, 'find_shortest_path'):
            base = self.roadmap.find_shortest_path(start_node, goal_node)
        else:
            base = self.roadmap.generate_path([start_node, goal_node])

        if base is None or np.size(base) == 0:
            self.get_logger().error(f'No path from node {start_node} to {goal_node}')
            return None

        return np.hstack([np.array(base), goal_col])

    def _send_path_to(self, goal_xy, label=''):
        wp = self._plan_to_xy(goal_xy)
        if wp is None:
            return False
        self.waypoints_pub.publish(self._qlabs_path_to_ros(wp))
        self.get_logger().info(f'Path published -> {label} ({goal_xy[0]:.2f}, {goal_xy[1]:.2f})')
        return True

    def _snap_to_exact(self, goal_xy, label=''):
        if self.robot_pose is None:
            return self._send_path_to(goal_xy, label)

        rx, ry = float(self.robot_pose.pose.position.x), float(self.robot_pose.pose.position.y)
        R_mat  = self._rot_matrix()
        t      = np.array(self.translation_offset)
        cur_q  = np.array([rx, ry]) @ R_mat.T - t
        goal_q = np.array([float(goal_xy[0]), float(goal_xy[1])])

        dist = float(np.linalg.norm(cur_q - goal_q))
        if dist < 0.10:
            self.get_logger().info(f'Already within 0.10m of {label}, no snap needed')
            return True

        wp = np.stack([cur_q, goal_q], axis=1)
        self.waypoints_pub.publish(self._qlabs_path_to_ros(wp))
        self.get_logger().info(f'Snap path -> {label} ({goal_xy[0]:.2f}, {goal_xy[1]:.2f}) dist={dist:.3f}m')
        return True

    def _set_led(self, led_id: int):
        if self.qcar_hardware_client is None:
            return
        led_id = int(led_id)
        if self._last_led == led_id:
            return
        self._last_led = led_id
        param = Parameter()
        param.name                  = 'led_color_id'
        param.value.type            = ParameterType.PARAMETER_INTEGER
        param.value.integer_value   = led_id
        req                         = SetParameters.Request()
        req.parameters              = [param]
        self.qcar_hardware_client.call_async(req)

    def _distance_to_node(self, node_id):
        if self.robot_pose is None:
            return float('inf')
        rx, ry = float(self.robot_pose.pose.position.x), float(self.robot_pose.pose.position.y)
        cur_q = np.array(self._ros_to_qlabs(rx, ry))
        goal_q = np.array(self._get_node_xy(node_id))
        return float(np.linalg.norm(cur_q - goal_q))

    def _active_goal_label(self, index=None):
        idx = self.active_goal_index if index is None else int(index)
        trip_len = len(self.active_trip_nodes)
        if idx == 0:
            return 'PICKUP'
        if idx < trip_len - 1:
            return f'STOP {idx}'
        if idx < trip_len:
            return 'DROPOFF'
        return 'HUB'

    def _stage_for_goal_index(self, index):
        idx = int(index)
        trip_len = len(self.active_trip_nodes)
        if idx == 0:
            return MissionStage.TO_PICKUP
        if idx < trip_len - 1:
            return MissionStage.TO_INTERMEDIATE
        if idx < trip_len:
            return MissionStage.TO_DROPOFF
        return MissionStage.TO_HUB

    def _set_drive_led_for_stage(self, stage):
        if stage == MissionStage.TO_PICKUP:
            self._set_led(self.LED_GREEN)
        elif stage == MissionStage.TO_HUB:
            self._set_led(self.LED_ORANGE)
        else:
            self._set_led(self.LED_BLUE)

    def _send_active_goal(self):
        if self.active_goal_index >= len(self.active_goal_nodes):
            self._complete_mission()
            return True

        goal_node = self.active_goal_nodes[self.active_goal_index]
        goal_xy = list(self._get_node_xy(goal_node))
        label = self._active_goal_label()
        stage = self._stage_for_goal_index(self.active_goal_index)

        self.mission_stage = stage
        self._path_completed_event = False

        if self._distance_to_node(goal_node) < 0.30:
            self.get_logger().info(
                f'Already at {label} node {goal_node}; handling arrival without a new path.')
            self._handle_active_arrival(time.time())
            return True

        ok = self._send_path_to(goal_xy, label=f'{label} node {goal_node}')
        if ok:
            self._set_drive_led_for_stage(stage)
        return ok

    def _advance_to_next_goal(self):
        self.active_goal_index += 1
        if not self._send_active_goal():
            self._abort_active_ride('Could not publish next trip leg; ride aborted.')

    def _handle_active_arrival(self, now):
        if self.active_goal_index >= len(self.active_goal_nodes):
            self._complete_mission()
            return

        goal_node = self.active_goal_nodes[self.active_goal_index]
        goal_xy = list(self._get_node_xy(goal_node))
        label = self._active_goal_label()

        if self.mission_stage == MissionStage.TO_HUB:
            self._complete_mission()
            return

        self._snap_to_exact(goal_xy, label=f'{label}-snap')

        if self.active_goal_index == 0:
            self.mission_stage = MissionStage.WAIT_AT_PICKUP
            self.picked_up = True
            self._set_led(self.LED_BLUE)
            self.pause_until = now + self.stop_seconds
            self.get_logger().info(
                f'Arrived at PICKUP node {goal_node}. Waiting {self.stop_seconds:.1f}s.')
        elif self.active_goal_index < len(self.active_trip_nodes) - 1:
            self.mission_stage = MissionStage.WAIT_AT_INTERMEDIATE
            self._set_led(self.LED_BLUE)
            self.pause_until = now + self.stop_seconds
            self.get_logger().info(
                f'Arrived at intermediate node {goal_node}. Waiting {self.stop_seconds:.1f}s.')
        else:
            self.mission_stage = MissionStage.WAIT_AT_DROPOFF
            self.dropped_off = True
            self._set_led(self.LED_ORANGE)
            self.pause_until = now + self.stop_seconds
            self.get_logger().info(
                f'Arrived at DROPOFF node {goal_node}. Waiting {self.stop_seconds:.1f}s.')

    def _complete_mission(self):
        self.active_trip_nodes = []
        self.active_goal_nodes = []
        self.active_goal_index = 0
        self.mission_stage = MissionStage.IDLE
        self.picked_up = False
        self.dropped_off = False
        self.ready_for_rides = True
        self._set_led(self.LED_MAGENTA)
        self.get_logger().info(f'Mission complete. Returned to HUB node {self.taxi_node}. Ready for next ride.')

    def _abort_active_ride(self, reason):
        self.active_trip_nodes = []
        self.active_goal_nodes = []
        self.active_goal_index = 0
        self.mission_stage = MissionStage.IDLE
        self.ready_for_rides = True
        self._set_led(self.LED_MAGENTA)
        self.get_logger().error(reason)

    def loop(self):
        now = time.time()

        if self.robot_pose is None:
            return

        if not self._auto_aligned:
            self._auto_aligned = self._auto_align()
            if not self._auto_aligned:
                return

        if not self.startup_done:
            if not self._startup_path_sent:
                rx, ry = float(self.robot_pose.pose.position.x), float(self.robot_pose.pose.position.y)
                cur_q  = np.array(self._ros_to_qlabs(rx, ry))
                dist_to_hub = float(np.linalg.norm(cur_q - np.array(self.hub_xy)))
                if dist_to_hub < 0.30:
                    self._startup_path_sent = True
                    self.startup_done    = True
                    self.ready_for_rides = True
                    self._set_led(self.LED_MAGENTA)
                    self.get_logger().info(
                        f'Already at HUB (dist={dist_to_hub:.2f}m). Skipping startup drive. Ready for rides.')
                    return
                ok = self._send_path_to(self.hub_xy, label='HUB (startup)')
                if ok:
                    self._startup_path_sent = True
                return
            if self._path_completed_event:
                self._path_completed_event = False
                self.startup_done    = True
                self.ready_for_rides = True
                self._set_led(self.LED_MAGENTA)
                self.get_logger().info('Startup complete. Ready for rides.')
            return

        # External-stop LED override: if the object detector halts the car mid-leg
        # (stop sign / traffic light on /motion_enable), show RED. Restore the
        # leg's drive colour when motion resumes. Legitimate stops are the WAIT_*
        # stages (pickup/dropoff/hub) and keep their own colours.
        if self.mission_stage in (MissionStage.TO_PICKUP,
                                  MissionStage.TO_INTERMEDIATE,
                                  MissionStage.TO_DROPOFF,
                                  MissionStage.TO_HUB):
            if not self.motion_enabled:
                self._set_led(self.LED_RED)
            else:
                self._set_drive_led_for_stage(self.mission_stage)

        if self.ready_for_rides and not self.new_ride_requested:
            return

        if self.ready_for_rides and self.new_ride_requested:
            self.new_ride_requested = False
            ride_nodes = self._pending_ride_nodes()
            if len(ride_nodes) < 2:
                self.get_logger().warn(
                    'Ride requested but no complete node list is set — '
                    'use `ros2 param set /trip_planner trip_nodes "[1, 8]"`.')
                return
            invalid = self._invalid_nodes(ride_nodes)
            if invalid:
                self.get_logger().error(f'Ride requested with invalid nodes {invalid}; ignoring.')
                return

            self.ready_for_rides    = False
            self.picked_up          = False
            self.dropped_off        = False
            self.active_trip_nodes  = list(ride_nodes)
            self.active_goal_nodes  = list(ride_nodes)
            if self.active_goal_nodes[-1] != self.taxi_node:
                self.active_goal_nodes.append(self.taxi_node)
            self.active_goal_index = 0
            self._clear_pending_ride()
            self.get_logger().info(
                f'Ride dispatched: requested nodes {self.active_trip_nodes}; '
                f'executing goals {self.active_goal_nodes}.')
            if not self._send_active_goal():
                self._abort_active_ride('Could not publish first trip leg; ride aborted.')
            return

        if self.mission_stage in (MissionStage.WAIT_AT_PICKUP,
                                  MissionStage.WAIT_AT_INTERMEDIATE,
                                  MissionStage.WAIT_AT_DROPOFF) and now >= self.pause_until:
            self._advance_to_next_goal()
            return

        if now < self.pause_until:
            return

        if not self._path_completed_event:
            return
        self._path_completed_event = False

        if self.mission_stage in (MissionStage.TO_PICKUP,
                                  MissionStage.TO_INTERMEDIATE,
                                  MissionStage.TO_DROPOFF,
                                  MissionStage.TO_HUB):
            self._handle_active_arrival(now)


def main():
    rclpy.init()
    node = TripPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()
