#!/usr/bin/env python3
"""
trip_planner — taxi-mission state machine for the ACC 2026 ride loop.

State machine (per official ACC 2026 rules):
    IDLE (MAGENTA, parked at HUB, ready_for_rides=True)
      └─ new ride dispatched (ride_waypoints param set or new_ride_requested true)
         ▼
    TO_PICKUP (GREEN; RED while /motion_enable=False)
      └─ pose-arrival at pickup_node → halt + snap /initialpose → BLUE
         ▼
    WAIT_AT_PICKUP (BLUE) — stop_seconds pause
      └─ resume → GREEN → TO_DROPOFF
         ▼
    TO_DROPOFF (GREEN; RED while /motion_enable=False)
      └─ pose-arrival at dropoff_node → halt + snap → ORANGE
         ▼
    WAIT_AT_DROPOFF (ORANGE) — stop_seconds pause
      └─ resume → GREEN → TO_HUB
         ▼
    TO_HUB (GREEN; RED while /motion_enable=False)
      └─ pose-arrival at HUB node → halt + snap → MAGENTA → IDLE

Key behaviors (added 2026-05-27):
    - **Extended path**: path goes from current_node THROUGH goal_node TO one
      downstream node, so pure-pursuit always has a forward target. PP no
      longer degrades in the final 0.5 m approach.
    - **Pose-arrival**: instead of waiting for path_follower to report
      path_complete=True, we monitor /robot_pose against the known node's
      XY coords. Arrival fires when ‖pose - node_xy‖ < arrival_tolerance_m
      (default 0.10 m).
    - **Halt-and-snap**: on arrival, set path_follower control_mode=idle
      (stops the car instantly), then publish /initialpose with the node's
      exact pose. AMCL re-localizes, drift bounded for the next leg.
    - **/motion_enable → LED RED**: orthogonal to FSM. When perception
      stops the car (stop sign, traffic light), LED flips GREEN → RED
      during navigation legs. Returns to GREEN when motion_enabled True.

Ride representation (param):
    ride_waypoints: int array of SDCSRoadMap node IDs
        [pickup_node, dropoff_node]
    Intermediate stops (stop signs, traffic lights) are handled by
    /motion_enable, NOT by adding nodes to ride_waypoints.

Legacy XY params (pickup_xy, dropoff_xy) still work for back-compat: if
ride_waypoints is empty, we find the nearest node to each XY at dispatch.
"""

import math
import time
import numpy as np
from enum import Enum

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from std_msgs.msg import Bool
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterDescriptor, SetParametersResult
from rcl_interfaces.msg import ParameterType as ParameterTypeMsg
from rclpy.parameter import ParameterType

from hal.products.mats import SDCSRoadMap


class MissionStage(Enum):
    IDLE            = 0
    TO_PICKUP       = 1
    WAIT_AT_PICKUP  = 2
    TO_DROPOFF      = 3
    WAIT_AT_DROPOFF = 4
    TO_HUB          = 5


# LED IDs from qcar2_hardware.cpp:333-346
LED_RED     = 0
LED_GREEN   = 1
LED_BLUE    = 2
LED_YELLOW  = 3
LED_CYAN    = 4
LED_MAGENTA = 5
LED_ORANGE  = 6


def _stage_to_led(stage: MissionStage) -> int:
    """Default LED color for a given mission stage (no /motion_enable override)."""
    if stage in (MissionStage.TO_PICKUP, MissionStage.TO_DROPOFF, MissionStage.TO_HUB):
        return LED_GREEN
    if stage == MissionStage.WAIT_AT_PICKUP:
        return LED_BLUE
    if stage == MissionStage.WAIT_AT_DROPOFF:
        return LED_ORANGE
    return LED_MAGENTA  # IDLE


def _is_navigating(stage: MissionStage) -> bool:
    return stage in (MissionStage.TO_PICKUP, MissionStage.TO_DROPOFF, MissionStage.TO_HUB)


class TripPlanner(Node):

    def __init__(self):
        super().__init__('trip_planner')

        self.roadmap = SDCSRoadMap()

        # ────── Service clients ──────
        # 2026-05-27: clients are NEVER set to None — we create them once and
        # poll service_is_ready() at call time. This survives any startup order
        # (trip_planner can launch before or after qcar2_hardware/path_follower).
        # We log once when each service becomes ready so the user knows when
        # LEDs / halt-on-arrival start working.
        self.qcar_hardware_client = self.create_client(
            SetParameters, '/qcar2_hardware/set_parameters')
        self.path_follower_client = self.create_client(
            SetParameters, '/path_follower/set_parameters')
        self._qcar_hw_ready_logged = False
        self._pf_ready_logged = False
        self.get_logger().info(
            'Service clients created (qcar2_hardware, path_follower). '
            'Will use them as soon as they are advertised — no blocking wait.')

        # ────── Params ──────
        self.declare_parameter('taxi_node',           [10])
        self.declare_parameter('pickup_xy',           [0.0, 0.0])
        self.declare_parameter('dropoff_xy',          [0.0, 0.0])
        self.declare_parameter('stop_seconds',        [3.0])
        self.declare_parameter('rotation_offset',     [82.0])
        self.declare_parameter('translation_offset',  [0.0, 0.0])

        # New (2026-05-27): node-id-based ride API.
        # Empty default + explicit INTEGER_ARRAY descriptor — otherwise ROS
        # infers BYTE_ARRAY from `[]` and rejects later "[14,22]" sets with
        # "expected Type.BYTE_ARRAY got Type.INTEGER_ARRAY".
        self.declare_parameter(
            'ride_waypoints', [],
            ParameterDescriptor(type=ParameterTypeMsg.PARAMETER_INTEGER_ARRAY),
        )
        # Tight tolerance for pickup/dropoff — judges measure these.
        self.declare_parameter('arrival_tolerance_m',     0.07)
        # Looser tolerance for HUB — it's a wide parking area, the node
        # there is somewhat ambiguous, and the snap re-anchors anyway.
        self.declare_parameter('hub_arrival_tolerance_m', 0.30)
        self.declare_parameter('extend_past_goal',        True)

        self.taxi_node          = int(list(self.get_parameter('taxi_node').get_parameter_value().integer_array_value)[0])
        self.pickup_xy          = list(self.get_parameter('pickup_xy').get_parameter_value().double_array_value)
        self.dropoff_xy         = list(self.get_parameter('dropoff_xy').get_parameter_value().double_array_value)
        self.stop_seconds       = float(list(self.get_parameter('stop_seconds').get_parameter_value().double_array_value)[0])
        self.rotation_offset    = list(self.get_parameter('rotation_offset').get_parameter_value().double_array_value)
        self.translation_offset = list(self.get_parameter('translation_offset').get_parameter_value().double_array_value)
        self.ride_waypoints           = list(self.get_parameter('ride_waypoints').get_parameter_value().integer_array_value)
        self.arrival_tolerance_m      = float(self.get_parameter('arrival_tolerance_m').get_parameter_value().double_value)
        self.hub_arrival_tolerance_m  = float(self.get_parameter('hub_arrival_tolerance_m').get_parameter_value().double_value)
        self.extend_past_goal         = bool(self.get_parameter('extend_past_goal').get_parameter_value().bool_value)

        self.add_on_set_parameters_callback(self.parameter_update_callback)

        # ────── HUB ──────
        self.hub_node = self.taxi_node
        self.hub_xy_ros = list(self._node_xy_ros(self.hub_node))
        self.get_logger().info(
            f'HUB at node {self.hub_node}: ROS=({self.hub_xy_ros[0]:.3f}, {self.hub_xy_ros[1]:.3f})')

        # ────── State ──────
        self.robot_pose            = None
        self.path_status           = False
        self._path_completed_event = False
        self.motion_enabled        = True

        self.startup_done       = False
        self._startup_path_sent = False
        self.ready_for_rides    = False
        self.new_ride_requested = False

        self.mission_stage = MissionStage.IDLE
        self.pause_until   = 0.0
        self._last_led     = None
        self._halt_active  = False    # are we currently in halt-on-arrival?

        # Current-ride node refs (resolved from ride_waypoints or pickup_xy/dropoff_xy)
        self.current_pickup_node  = None
        self.current_dropoff_node = None

        # ────── Topics ──────
        self.waypoints_pub   = self.create_publisher(Path, '/cmd_waypoints', 1)
        self.initialpose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 1)

        self.create_subscription(Bool,        '/path_status',   self._path_status_cb,   10)
        self.create_subscription(PoseStamped, '/robot_pose',    self._robot_pose_cb,    10)
        self.create_subscription(Bool,        '/motion_enable', self._motion_enable_cb, 10)

        self._set_led(LED_MAGENTA)
        self.create_timer(0.1, self.loop)

    # ───────────────────────── Callbacks ─────────────────────────

    def _path_status_cb(self, msg: Bool):
        prev = self.path_status
        self.path_status = bool(msg.data)
        if not prev and self.path_status:
            self._path_completed_event = True

    def _robot_pose_cb(self, msg: PoseStamped):
        self.robot_pose = msg

    def _motion_enable_cb(self, msg: Bool):
        was = self.motion_enabled
        self.motion_enabled = bool(msg.data)
        if was == self.motion_enabled:
            return
        # Only override LED during NAVIGATING stages — don't stomp BLUE/ORANGE
        # at pickup/dropoff or MAGENTA at HUB.
        if _is_navigating(self.mission_stage):
            if not self.motion_enabled:
                self._set_led(LED_RED)
            else:
                self._set_led(_stage_to_led(self.mission_stage))

    def parameter_update_callback(self, params):
        for p in params:
            if p.name == 'pickup_xy' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.pickup_xy = list(p.value)
            elif p.name == 'dropoff_xy' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.dropoff_xy = list(p.value)
            elif p.name == 'stop_seconds' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.stop_seconds = float(list(p.value)[0])
            elif p.name == 'rotation_offset' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.rotation_offset = list(p.value)
            elif p.name == 'translation_offset' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.translation_offset = list(p.value)
            elif p.name == 'ride_waypoints' and p.type_ == p.Type.INTEGER_ARRAY:
                self.ride_waypoints = list(p.value)
                self.get_logger().info(f'ride_waypoints updated: {self.ride_waypoints}')
            elif p.name == 'arrival_tolerance_m' and p.type_ == p.Type.DOUBLE:
                self.arrival_tolerance_m = float(p.value)
            elif p.name == 'hub_arrival_tolerance_m' and p.type_ == p.Type.DOUBLE:
                self.hub_arrival_tolerance_m = float(p.value)
            elif p.name == 'extend_past_goal' and p.type_ == p.Type.BOOL:
                self.extend_past_goal = bool(p.value)

        # Setting any ride-relevant param while parked = dispatch a new ride.
        if self.ready_for_rides and self.mission_stage == MissionStage.IDLE:
            self.new_ride_requested = True

        return SetParametersResult(successful=True)

    # ───────────────────────── Frame helpers ─────────────────────────
    # The roadmap stores nodes in QLabs frame. ROS map frame differs by
    # rotation_offset (deg) and translation_offset (m). The transform
    # below mirrors what path_follower's auto-align computed at startup.

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

    # ───────────────────────── Per-node helpers ─────────────────────────

    def _get_node_xy_qlabs(self, node_id):
        """Raw node XY in QLabs frame (as SDCSRoadMap stores it)."""
        if hasattr(self.roadmap, 'get_node_pose'):
            pose = np.array(self.roadmap.get_node_pose(node_id)).reshape(-1)
        else:
            pose = np.array(self.roadmap.nodes[node_id].pose).reshape(-1)
        return float(pose[0]), float(pose[1])

    def _get_node_theta_qlabs(self, node_id):
        """Node yaw (radians) in QLabs frame, or 0.0 if not defined."""
        if hasattr(self.roadmap, 'get_node_pose'):
            pose = np.array(self.roadmap.get_node_pose(node_id)).reshape(-1)
        else:
            pose = np.array(self.roadmap.nodes[node_id].pose).reshape(-1)
        return float(pose[2]) if pose.size >= 3 else 0.0

    def _node_xy_ros(self, node_id):
        """Node XY transformed to ROS map frame."""
        qx, qy = self._get_node_xy_qlabs(node_id)
        R_mat = self._rot_matrix()
        t_off = np.array(self.translation_offset)
        p_ros = (np.array([qx, qy]) + t_off) @ R_mat
        return float(p_ros[0]), float(p_ros[1])

    def _node_theta_ros(self, node_id):
        """Node yaw transformed to ROS frame. ROS_yaw = QLabs_yaw - rotation_offset."""
        theta_q = self._get_node_theta_qlabs(node_id)
        angle_rad = float(self.rotation_offset[0]) * np.pi / 180.0
        return theta_q - angle_rad

    def _node_count(self):
        if hasattr(self.roadmap, 'nodes'):
            try:
                return len(self.roadmap.nodes)
            except Exception:
                pass
        return 0

    def _closest_node(self, x_qlabs, y_qlabs):
        """Nearest node ID for a QLabs-frame point."""
        best_i, best_d = 0, float('inf')
        for i in range(self._node_count()):
            try:
                nx, ny = self._get_node_xy_qlabs(i)
                d = (nx - x_qlabs) ** 2 + (ny - y_qlabs) ** 2
                if d < best_d:
                    best_d = d
                    best_i = i
            except Exception:
                continue
        return best_i

    def _downstream_node(self, node_id):
        """Pick a legal next-node along outEdges for path extension. None if dead-end.

        `edge.toNode` in hal.utilities.path_planning.RoadMapEdge is the RoadMapNode
        OBJECT, not its integer index. SDCSRoadMap.generate_path expects integers,
        so we resolve the object back to its index by identity match in
        self.roadmap.nodes.
        """
        try:
            node = self.roadmap.nodes[node_id]
            edges = getattr(node, 'outEdges', None) or []
            if not edges:
                return None
            to_obj = edges[0].toNode
            # Resolve object → integer index. Identity comparison is correct here
            # because there's exactly one RoadMapNode instance per index.
            for i, n in enumerate(self.roadmap.nodes):
                if n is to_obj:
                    return int(i)
            # Fallback: if it's already an integer (some implementations differ)
            if isinstance(to_obj, int):
                return int(to_obj)
        except Exception:
            pass
        return None

    def _current_node_id(self):
        """Closest node to the robot's current pose."""
        if self.robot_pose is None:
            return None
        rx = float(self.robot_pose.pose.position.x)
        ry = float(self.robot_pose.pose.position.y)
        qx, qy = self._ros_to_qlabs(rx, ry)
        return self._closest_node(qx, qy)

    # ───────────────────────── Path planning ─────────────────────────

    def _plan_node_seq(self, sequence):
        """Generate path through a sequence of node IDs. Returns 2xN numpy or None."""
        if not sequence or len(sequence) < 2:
            return None
        try:
            if hasattr(self.roadmap, 'generate_path'):
                base = self.roadmap.generate_path(nodeSequence=sequence)
            else:
                base = self.roadmap.find_shortest_path(sequence[0], sequence[-1])
        except Exception as e:
            self.get_logger().error(f'generate_path failed for {sequence}: {e}')
            return None
        if base is None or np.size(base) == 0:
            self.get_logger().error(f'No path through node sequence {sequence}')
            return None
        return np.array(base)

    def _send_extended_path_to_node(self, goal_node, label=''):
        """Send a path: current_node → goal_node → (one downstream node).
        The extra node ensures PP always has a forward target near the goal.
        """
        start_node = self._current_node_id()
        if start_node is None:
            return False
        sequence = [start_node, goal_node]
        if self.extend_past_goal:
            ds = self._downstream_node(goal_node)
            if ds is not None and ds != start_node:
                sequence.append(ds)
        wp = self._plan_node_seq(sequence)
        if wp is None:
            return False
        self.waypoints_pub.publish(self._qlabs_path_to_ros(wp))
        self.get_logger().info(
            f'Path published -> {label} (nodes={sequence}, '
            f'waypoints={wp.shape[1]})')
        return True

    # ───────────────────────── Arrival detection ─────────────────────────

    def _dist_to_node(self, node_id):
        """Distance (m) from robot pose to node's ROS-frame XY."""
        if self.robot_pose is None or node_id is None:
            return float('inf')
        rx = float(self.robot_pose.pose.position.x)
        ry = float(self.robot_pose.pose.position.y)
        nx, ny = self._node_xy_ros(node_id)
        return math.hypot(rx - nx, ry - ny)

    def _goal_node_for_stage(self):
        if self.mission_stage == MissionStage.TO_PICKUP:
            return self.current_pickup_node
        if self.mission_stage == MissionStage.TO_DROPOFF:
            return self.current_dropoff_node
        if self.mission_stage == MissionStage.TO_HUB:
            return self.hub_node
        return None

    def _check_pose_arrival(self):
        """Returns True iff pose is within tolerance of the current goal node.

        HUB uses a looser tolerance (hub_arrival_tolerance_m, default 0.30 m)
        because the parking area is wide and the snap re-anchors the EKF
        regardless of exact stop position.

        PICKUP/DROPOFF use arrival_tolerance_m (default 0.07 m) — judges
        measure these.
        """
        goal = self._goal_node_for_stage()
        if goal is None:
            return False
        tolerance = (self.hub_arrival_tolerance_m
                     if self.mission_stage == MissionStage.TO_HUB
                     else self.arrival_tolerance_m)
        return self._dist_to_node(goal) < tolerance

    # ───────────────────────── Halt + Snap ─────────────────────────

    def _set_pf_control_mode(self, mode: str):
        """Send /path_follower/set_parameters control_mode = idle | autonomous | manual."""
        if not self.path_follower_client.service_is_ready():
            # Don't spam logs — fallback (no halt) is documented at startup.
            return
        if not self._pf_ready_logged:
            self.get_logger().info('path_follower service now ready — halt-on-arrival enabled.')
            self._pf_ready_logged = True
        param = Parameter()
        param.name                = 'control_mode'
        param.value.type          = ParameterType.PARAMETER_STRING
        param.value.string_value  = mode
        req                       = SetParameters.Request()
        req.parameters            = [param]
        self.path_follower_client.call_async(req)

    def _halt_path_follower(self):
        """Stop the car immediately by switching path_follower to idle."""
        self._set_pf_control_mode('idle')
        self._halt_active = True

    def _resume_path_follower(self):
        """Resume autonomous driving after a halt."""
        self._set_pf_control_mode('autonomous')
        self._halt_active = False

    def _snap_initialpose_to_node(self, node_id, label=''):
        """Publish /initialpose at the node's exact ROS-frame pose with tight covariance.
        AMCL hard-snaps its belief, killing any drift accumulated this leg.
        """
        nx, ny = self._node_xy_ros(node_id)
        ntheta = self._node_theta_ros(node_id)

        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = float(nx)
        msg.pose.pose.position.y = float(ny)
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.z = float(math.sin(ntheta / 2.0))
        msg.pose.pose.orientation.w = float(math.cos(ntheta / 2.0))
        # Tight covariance — we trust this hard.
        msg.pose.covariance = [
            0.01, 0,    0, 0, 0, 0,
            0,    0.01, 0, 0, 0, 0,
            0,    0,    0, 0, 0, 0,
            0,    0,    0, 0, 0, 0,
            0,    0,    0, 0, 0, 0,
            0,    0,    0, 0, 0, 0.05,
        ]
        # Publish a few times so AMCL definitely catches one.
        for _ in range(3):
            self.initialpose_pub.publish(msg)
        self.get_logger().info(
            f'Snapped /initialpose to node {node_id} ({label}): '
            f'xy=({nx:.3f}, {ny:.3f}) yaw={math.degrees(ntheta):.1f}°')

    # ───────────────────────── LEDs ─────────────────────────

    def _set_led(self, led_id: int):
        if not self.qcar_hardware_client.service_is_ready():
            # Cache desired LED so we'll send it once the service comes up.
            self._last_led = None  # force re-send when ready
            return
        if not self._qcar_hw_ready_logged:
            self.get_logger().info('qcar2_hardware service now ready — LEDs enabled.')
            self._qcar_hw_ready_logged = True
        led_id = int(led_id)
        if self._last_led == led_id:
            return
        self._last_led = led_id
        param = Parameter()
        param.name                = 'led_color_id'
        param.value.type          = ParameterType.PARAMETER_INTEGER
        param.value.integer_value = led_id
        req                       = SetParameters.Request()
        req.parameters            = [param]
        self.qcar_hardware_client.call_async(req)

    def _set_navigation_led(self):
        """Set GREEN if motion_enabled else RED. Call this on entry to a TO_* stage."""
        self._set_led(LED_RED if not self.motion_enabled else LED_GREEN)

    # ───────────────────────── Ride dispatch ─────────────────────────

    def _resolve_ride_nodes(self):
        """Resolve current pickup/dropoff node IDs from either ride_waypoints
        (new API) or pickup_xy/dropoff_xy (legacy)."""
        if len(self.ride_waypoints) >= 2:
            self.current_pickup_node  = int(self.ride_waypoints[0])
            self.current_dropoff_node = int(self.ride_waypoints[-1])
            self.get_logger().info(
                f'Ride from node {self.current_pickup_node} → node '
                f'{self.current_dropoff_node} (from ride_waypoints)')
            return True
        if self.pickup_xy and self.dropoff_xy:
            pxq, pyq = self._ros_to_qlabs(self.pickup_xy[0], self.pickup_xy[1])
            dxq, dyq = self._ros_to_qlabs(self.dropoff_xy[0], self.dropoff_xy[1])
            self.current_pickup_node  = self._closest_node(pxq, pyq)
            self.current_dropoff_node = self._closest_node(dxq, dyq)
            self.get_logger().info(
                f'Ride from node {self.current_pickup_node} → node '
                f'{self.current_dropoff_node} (resolved from XY)')
            return True
        return False

    # ───────────────────────── Main loop ─────────────────────────

    def loop(self):
        now = time.time()

        if self.robot_pose is None:
            return

        # ── Startup: drive to HUB once, snap, declare ready ──
        if not self.startup_done:
            if not self._startup_path_sent:
                # Halt may be left over from a previous run; resume so PP drives.
                self._resume_path_follower()
                ok = self._send_extended_path_to_node(self.hub_node, label='HUB (startup)')
                if ok:
                    self._startup_path_sent = True
                    self._set_navigation_led()
                return
            if self._check_pose_arrival_to(self.hub_node):
                self._halt_path_follower()
                self._snap_initialpose_to_node(self.hub_node, label='HUB-startup')
                self.startup_done    = True
                self.ready_for_rides = True
                self.mission_stage   = MissionStage.IDLE
                self._set_led(LED_MAGENTA)
                self.get_logger().info('Startup complete. Parked at HUB. Ready for rides.')
            return

        # ── Idle, parked at HUB, waiting for dispatch ──
        if self.ready_for_rides and not self.new_ride_requested:
            return

        # ── Dispatch new ride ──
        if self.ready_for_rides and self.new_ride_requested:
            self.new_ride_requested = False
            if not self._resolve_ride_nodes():
                self.get_logger().error(
                    'No ride waypoints — set ride_waypoints OR pickup_xy/dropoff_xy first.')
                return
            self.ready_for_rides = False
            self._resume_path_follower()
            ok = self._send_extended_path_to_node(
                self.current_pickup_node, label='PICKUP')
            if ok:
                self.mission_stage = MissionStage.TO_PICKUP
                self._set_navigation_led()
            return

        # ── Pause windows (BLUE / ORANGE) — wait, then advance ──
        if self.mission_stage == MissionStage.WAIT_AT_PICKUP and now >= self.pause_until:
            self._resume_path_follower()
            ok = self._send_extended_path_to_node(
                self.current_dropoff_node, label='DROPOFF')
            if ok:
                self.mission_stage = MissionStage.TO_DROPOFF
                self._set_navigation_led()
            return

        if self.mission_stage == MissionStage.WAIT_AT_DROPOFF and now >= self.pause_until:
            self._resume_path_follower()
            ok = self._send_extended_path_to_node(self.hub_node, label='HUB')
            if ok:
                self.mission_stage = MissionStage.TO_HUB
                self._set_navigation_led()
            return

        if now < self.pause_until:
            return

        # ── Pose-arrival check during TO_* stages ──
        if _is_navigating(self.mission_stage) and self._check_pose_arrival():
            arrived_node = self._goal_node_for_stage()
            self._halt_path_follower()
            self._snap_initialpose_to_node(arrived_node, label=self.mission_stage.name)

            if self.mission_stage == MissionStage.TO_PICKUP:
                self.mission_stage = MissionStage.WAIT_AT_PICKUP
                self._set_led(LED_BLUE)
                self.pause_until = now + self.stop_seconds
                self.get_logger().info(
                    f'Arrived at PICKUP (node {arrived_node}). Waiting {self.stop_seconds:.1f}s.')

            elif self.mission_stage == MissionStage.TO_DROPOFF:
                self.mission_stage = MissionStage.WAIT_AT_DROPOFF
                self._set_led(LED_ORANGE)
                self.pause_until = now + self.stop_seconds
                self.get_logger().info(
                    f'Arrived at DROPOFF (node {arrived_node}). Waiting {self.stop_seconds:.1f}s.')

            elif self.mission_stage == MissionStage.TO_HUB:
                self.mission_stage = MissionStage.IDLE
                self.ready_for_rides = True
                self._set_led(LED_MAGENTA)
                self.get_logger().info(
                    f'Mission complete. Parked at HUB (node {arrived_node}). '
                    f'Ready for next ride.')

    def _check_pose_arrival_to(self, node_id):
        """Standalone pose-distance check (currently used for startup HUB arrival).
        Uses the looser hub_arrival_tolerance_m because startup target IS HUB."""
        return self._dist_to_node(node_id) < self.hub_arrival_tolerance_m


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
