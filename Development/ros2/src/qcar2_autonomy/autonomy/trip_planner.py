#!/usr/bin/env python3

import math
import time
import numpy as np
from enum import Enum

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseStamped, Twist
from nav_msgs.msg import Path
from std_msgs.msg import Bool
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, SetParametersResult
from rclpy.parameter import ParameterType
from visualization_msgs.msg import Marker

from autonomy.recorded_map_utils import (
    build_waypoint_path_from_nodes,
    compute_path_headings,
    find_latest_recording_map,
    load_recording_map,
)
from autonomy.recorded_graph_planner import RecordedMapGraphPlanner
from autonomy.sdcs_graph_planner import SDCSGraphPlanner


# SDCS big-map, right-hand traffic node 10 (taxi hub) expressed in PNG/reference coordinates.
DEFAULT_TAXI_HUB_XY = [-1.28205, -0.45991]

# Fitted from manually sampled ROS map coordinates for SDCS nodes 0, 2, 4, 6, 8, 10.
# Converts SDCS PNG/reference coordinates into the live ROS map frame.
DEFAULT_SDCS_TO_MAP_TRANSLATION = [-0.08658752363019608, -0.053780886731217004]
DEFAULT_SDCS_TO_MAP_ROTATION_DEG = -79.63094749827502
DEFAULT_SDCS_TO_MAP_SCALE = 1.0216993081930361


class MissionStage(Enum):
    IDLE            = 0
    TO_PICKUP       = 1
    WAIT_AT_PICKUP  = 2
    TO_DROPOFF      = 3
    WAIT_AT_DROPOFF = 4
    TO_HUB          = 5
    REVERSING       = 6


class TripPlanner(Node):

    def __init__(self):
        super().__init__('trip_planner')

        self.qcar_hardware_client = self.create_client(SetParameters, '/qcar2_hardware/set_parameters')
        if not self.qcar_hardware_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('qcar2_hardware LED service not available — LEDs disabled')
            self.qcar_hardware_client = None
        else:
            self.get_logger().info('connected to qcar2_hardware.')

        self.declare_parameter('route_frame',        'map')
        self.declare_parameter('hub_xy',             DEFAULT_TAXI_HUB_XY)
        self.declare_parameter('pickup_xy',          [0.125, 4.395])
        self.declare_parameter('dropoff_xy',         [-0.905, 0.800])
        self.declare_parameter('stop_seconds',       [3.0])
        self.declare_parameter('recorded_min_node_spacing_m', 0.10)
        self.declare_parameter('recorded_min_yaw_change_rad', 0.20)
        self.declare_parameter('generated_waypoint_spacing_m', 0.03)
        self.declare_parameter('goal_on_route_tolerance_m', 0.12)
        self.declare_parameter('goals_are_sdcs_frame', True)
        self.declare_parameter('sdcs_to_map_translation', DEFAULT_SDCS_TO_MAP_TRANSLATION)
        self.declare_parameter('sdcs_to_map_rotation_deg', DEFAULT_SDCS_TO_MAP_ROTATION_DEG)
        self.declare_parameter('sdcs_to_map_scale', DEFAULT_SDCS_TO_MAP_SCALE)
        self.declare_parameter('one_way_route', True)
        self.declare_parameter('allow_route_wrap', True)
        self.declare_parameter('route_wrap_gap_tolerance_m', 1.0)
        self.declare_parameter('heading_aware_start', True)
        self.declare_parameter('max_start_heading_error_rad', 1.5708)
        self.declare_parameter('heading_score_weight_m_per_rad', 0.30)
        self.declare_parameter('reverse_threshold_m', 1.5)
        self.declare_parameter('reverse_speed', 0.15)
        self.declare_parameter('dispatch_ride', False)

        self.route_frame        = self.get_parameter('route_frame').get_parameter_value().string_value
        self.hub_xy             = list(self.get_parameter('hub_xy').get_parameter_value().double_array_value)
        self.pickup_xy          = list(self.get_parameter('pickup_xy').get_parameter_value().double_array_value)
        self.dropoff_xy         = list(self.get_parameter('dropoff_xy').get_parameter_value().double_array_value)
        self.stop_seconds       = float(list(self.get_parameter('stop_seconds').get_parameter_value().double_array_value)[0])
        self.recorded_min_node_spacing_m = float(
            self.get_parameter('recorded_min_node_spacing_m').get_parameter_value().double_value)
        self.recorded_min_yaw_change_rad = float(
            self.get_parameter('recorded_min_yaw_change_rad').get_parameter_value().double_value)
        self.generated_waypoint_spacing_m = float(
            self.get_parameter('generated_waypoint_spacing_m').get_parameter_value().double_value)
        self.goal_on_route_tolerance_m = float(
            self.get_parameter('goal_on_route_tolerance_m').get_parameter_value().double_value)
        self.goals_are_sdcs_frame = bool(
            self.get_parameter('goals_are_sdcs_frame').get_parameter_value().bool_value)
        self.sdcs_to_map_translation = list(
            self.get_parameter('sdcs_to_map_translation').get_parameter_value().double_array_value)
        self.sdcs_to_map_rotation_deg = float(
            self.get_parameter('sdcs_to_map_rotation_deg').get_parameter_value().double_value)
        self.sdcs_to_map_scale = float(
            self.get_parameter('sdcs_to_map_scale').get_parameter_value().double_value)
        self.one_way_route = bool(
            self.get_parameter('one_way_route').get_parameter_value().bool_value)
        self.allow_route_wrap = bool(
            self.get_parameter('allow_route_wrap').get_parameter_value().bool_value)
        self.route_wrap_gap_tolerance_m = float(
            self.get_parameter('route_wrap_gap_tolerance_m').get_parameter_value().double_value)
        self.heading_aware_start = bool(
            self.get_parameter('heading_aware_start').get_parameter_value().bool_value)
        self.max_start_heading_error_rad = float(
            self.get_parameter('max_start_heading_error_rad').get_parameter_value().double_value)
        self.heading_score_weight_m_per_rad = float(
            self.get_parameter('heading_score_weight_m_per_rad').get_parameter_value().double_value)
        self.reverse_threshold_m = float(
            self.get_parameter('reverse_threshold_m').get_parameter_value().double_value)
        self.reverse_speed = float(
            self.get_parameter('reverse_speed').get_parameter_value().double_value)

        # Node-ID goals: -1 means "snap to nearest node from xy coords"
        self.declare_parameter('hub_node_id',     10)
        self.declare_parameter('pickup_node_id',  -1)
        self.declare_parameter('dropoff_node_id', -1)
        self.hub_node_id     = int(self.get_parameter('hub_node_id').get_parameter_value().integer_value)
        self.pickup_node_id  = int(self.get_parameter('pickup_node_id').get_parameter_value().integer_value)
        self.dropoff_node_id = int(self.get_parameter('dropoff_node_id').get_parameter_value().integer_value)

        self.add_on_set_parameters_callback(self.parameter_update_callback)

        # Pre-convert SDCS goal coords → map frame once at startup
        self.hub_xy_map     = self._goal_to_route_frame(self.hub_xy)
        self.pickup_xy_map  = self._goal_to_route_frame(self.pickup_xy)
        self.dropoff_xy_map = self._goal_to_route_frame(self.dropoff_xy)

        self.get_logger().info(f'Route frame: {self.route_frame}')
        self.get_logger().info(
            f'Goal input frame: {"SDCS/PNG" if self.goals_are_sdcs_frame else self.route_frame}')
        self.get_logger().info(
            f'SDCS->map: scale={self.sdcs_to_map_scale:.4f} '
            f'rot={self.sdcs_to_map_rotation_deg:.2f}° '
            f't={self.sdcs_to_map_translation}')
        self.get_logger().info(
            f'Goals (map): hub=({self.hub_xy_map[0]:.3f},{self.hub_xy_map[1]:.3f}) '
            f'hub_node={self.hub_node_id} | '
            f'pickup=({self.pickup_xy_map[0]:.3f},{self.pickup_xy_map[1]:.3f}) '
            f'pickup_node={self.pickup_node_id} | '
            f'dropoff=({self.dropoff_xy_map[0]:.3f},{self.dropoff_xy_map[1]:.3f}) '
            f'dropoff_node={self.dropoff_node_id}')
        self.recorded_points = np.zeros((2, 0), dtype=float)
        self.recorded_waypoints = np.zeros((2, 0), dtype=float)
        self.recorded_waypoint_yaws = np.array([], dtype=float)
        self.recorded_nodes = []
        self.recorded_map_path = None
        self.recorded_loop = False
        self.recorded_loop_gap = float('inf')
        self.graph_planner = None
        self.LED_GREEN   = 1
        self.LED_BLUE    = 2
        self.LED_RED     = 3
        self.LED_MAGENTA = 5
        self.LED_ORANGE  = 6

        self.robot_pose             = None
        self.path_status            = False
        self._path_completed_event  = False

        self.startup_done          = False
        self._startup_path_sent    = False
        self.ready_for_rides       = False
        self.new_ride_requested    = False
        self._pending_dispatch     = False

        self.mission_stage       = MissionStage.IDLE
        self.pause_until         = 0.0
        self.picked_up           = False
        self.dropped_off         = False
        self._last_led           = None
        self._reverse_target     = None
        self._reverse_next_stage = MissionStage.IDLE

        self.waypoints_pub    = self.create_publisher(Path, '/cmd_waypoints', 1)
        self._override_pub    = self.create_publisher(Twist, '/nav/override_cmd', 1)
        self.raw_nodes_marker_pub = self.create_publisher(Marker, '/planner/raw_recorded_nodes', 1)
        self.control_nodes_marker_pub = self.create_publisher(Marker, '/planner/control_nodes', 1)
        self.dense_route_marker_pub = self.create_publisher(Marker, '/planner/dense_recorded_route', 1)
        self.active_route_marker_pub = self.create_publisher(Marker, '/planner/active_route_nodes', 1)

        self._load_recorded_map()
        self._build_graph_planner()

        self.create_subscription(Bool,        '/path_status',  self.path_status_callback,  10)
        self.create_subscription(PoseStamped, '/robot_pose',   self.robot_pose_callback,   10)

        # Startup begins by driving to the hub, so do not show the idle/ready color yet.
        self._set_led(self.LED_ORANGE)
        self.create_timer(0.1, self.loop)

    def _make_points_marker(self, ns, marker_id, rgb, scale=0.08):
        marker = Marker()
        marker.header.frame_id = self.route_frame
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.a = 1.0
        marker.color.r = float(rgb[0])
        marker.color.g = float(rgb[1])
        marker.color.b = float(rgb[2])
        marker.pose.orientation.w = 1.0
        return marker

    def _make_line_marker(self, ns, marker_id, rgb, width=0.04):
        marker = Marker()
        marker.header.frame_id = self.route_frame
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = width
        marker.color.a = 1.0
        marker.color.r = float(rgb[0])
        marker.color.g = float(rgb[1])
        marker.color.b = float(rgb[2])
        marker.pose.orientation.w = 1.0
        return marker

    def _marker_points_from_xy(self, points_2xn):
        pts = []
        if points_2xn.size == 0:
            return pts
        for i in range(points_2xn.shape[1]):
            p = Point()
            p.x = float(points_2xn[0, i])
            p.y = float(points_2xn[1, i])
            p.z = 0.0
            pts.append(p)
        return pts

    def _publish_debug_markers(self, raw_nodes=None, control_points=None, dense_waypoints=None, active_route=None):
        stamp = self.get_clock().now().to_msg()

        if raw_nodes is not None:
            marker = self._make_points_marker('planner_raw_recorded_nodes', 0, (1.0, 1.0, 0.0), scale=0.09)
            marker.header.stamp = stamp
            marker.points = self._marker_points_from_xy(raw_nodes)
            self.raw_nodes_marker_pub.publish(marker)

        if control_points is not None:
            marker = self._make_points_marker('planner_control_nodes', 1, (0.0, 1.0, 1.0), scale=0.10)
            marker.header.stamp = stamp
            marker.points = self._marker_points_from_xy(control_points)
            self.control_nodes_marker_pub.publish(marker)

        if dense_waypoints is not None:
            marker = self._make_line_marker('planner_dense_recorded_route', 2, (0.1, 1.0, 0.1), width=0.03)
            marker.header.stamp = stamp
            marker.points = self._marker_points_from_xy(dense_waypoints)
            self.dense_route_marker_pub.publish(marker)

        if active_route is not None:
            marker = self._make_points_marker('planner_active_route_nodes', 3, (0.1, 0.4, 1.0), scale=0.11)
            marker.header.stamp = stamp
            marker.points = self._marker_points_from_xy(active_route)
            self.active_route_marker_pub.publish(marker)

    def path_status_callback(self, msg):
        prev = self.path_status
        self.path_status = bool(msg.data)
        if not prev and self.path_status:
            self._path_completed_event = True

    def robot_pose_callback(self, msg: PoseStamped):
        self.robot_pose = msg

    def _recompute_map_goals(self):
        self.hub_xy_map     = self._goal_to_route_frame(self.hub_xy)
        self.pickup_xy_map  = self._goal_to_route_frame(self.pickup_xy)
        self.dropoff_xy_map = self._goal_to_route_frame(self.dropoff_xy)

    def _build_graph_planner(self):
        if self.recorded_waypoints.shape[1] < 2:
            self.graph_planner = None
            self.get_logger().error('Cannot build recorded graph: no recorded waypoints loaded')
            return False

        rot_rad = math.radians(self.sdcs_to_map_rotation_deg)
        rough_graph = SDCSGraphPlanner(
            waypoint_spacing=self.generated_waypoint_spacing_m,
            scale=self.sdcs_to_map_scale,
            rot_rad=rot_rad,
            translation=self.sdcs_to_map_translation,
        )
        self.graph_planner = RecordedMapGraphPlanner(
            self.recorded_waypoints,
            rough_graph,
        )
        edge_count = sum(len(edges) for edges in self.graph_planner._adj)
        missing_count = len(getattr(self.graph_planner, '_missing_edges', []))
        self.get_logger().info(
            f'RecordedMapGraphPlanner ready: source={self.recorded_map_path} '
            f'nodes={len(self.graph_planner._adj)} edges={edge_count} '
            f'missing_edges={missing_count} route_points={self.recorded_waypoints.shape[1]}')
        if edge_count == 0:
            self.get_logger().error('Recorded graph has no valid edges')
            return False
        return True

    def parameter_update_callback(self, params):
        goal_param_changed = False
        for p in params:
            if p.name == 'pickup_xy' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.pickup_xy = list(p.value)
                self.pickup_xy_map = self._goal_to_route_frame(self.pickup_xy)
                self.get_logger().info(
                    f'pickup_xy updated: {self.pickup_xy} '
                    f'map=({self.pickup_xy_map[0]:.3f},{self.pickup_xy_map[1]:.3f})')
                goal_param_changed = True
            elif p.name == 'dropoff_xy' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.dropoff_xy = list(p.value)
                self.dropoff_xy_map = self._goal_to_route_frame(self.dropoff_xy)
                self.get_logger().info(
                    f'dropoff_xy updated: {self.dropoff_xy} '
                    f'map=({self.dropoff_xy_map[0]:.3f},{self.dropoff_xy_map[1]:.3f})')
                goal_param_changed = True
            elif p.name == 'hub_xy' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.hub_xy = list(p.value)
                self.hub_xy_map = self._goal_to_route_frame(self.hub_xy)
                self.get_logger().info(
                    f'hub_xy updated: {self.hub_xy} '
                    f'map=({self.hub_xy_map[0]:.3f},{self.hub_xy_map[1]:.3f})')
                goal_param_changed = True
            elif p.name == 'hub_node_id' and p.type_ == p.Type.INTEGER:
                self.hub_node_id = int(p.value)
                self.get_logger().info(f'hub_node_id updated: {self.hub_node_id}')
                goal_param_changed = True
            elif p.name == 'pickup_node_id' and p.type_ == p.Type.INTEGER:
                self.pickup_node_id = int(p.value)
                self.get_logger().info(f'pickup_node_id updated: {self.pickup_node_id}')
                goal_param_changed = True
            elif p.name == 'dropoff_node_id' and p.type_ == p.Type.INTEGER:
                self.dropoff_node_id = int(p.value)
                self.get_logger().info(f'dropoff_node_id updated: {self.dropoff_node_id}')
                goal_param_changed = True
            elif p.name == 'stop_seconds' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.stop_seconds = float(list(p.value)[0])
            elif p.name == 'recorded_min_node_spacing_m' and p.type_ == p.Type.DOUBLE:
                self.recorded_min_node_spacing_m = float(p.value)
                self._load_recorded_map()
                self._build_graph_planner()
            elif p.name == 'recorded_min_yaw_change_rad' and p.type_ == p.Type.DOUBLE:
                self.recorded_min_yaw_change_rad = float(p.value)
                self._load_recorded_map()
                self._build_graph_planner()
            elif p.name == 'generated_waypoint_spacing_m' and p.type_ == p.Type.DOUBLE:
                self.generated_waypoint_spacing_m = float(p.value)
                self._load_recorded_map()
                self._build_graph_planner()
            elif p.name == 'goal_on_route_tolerance_m' and p.type_ == p.Type.DOUBLE:
                self.goal_on_route_tolerance_m = float(p.value)
            elif p.name == 'goals_are_sdcs_frame' and p.type_ == p.Type.BOOL:
                self.goals_are_sdcs_frame = bool(p.value)
                self.get_logger().info(
                    f'goals_are_sdcs_frame updated: {self.goals_are_sdcs_frame}')
                goal_param_changed = True
            elif p.name == 'sdcs_to_map_translation' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.sdcs_to_map_translation = list(p.value)
                self.get_logger().info(
                    f'sdcs_to_map_translation updated: {self.sdcs_to_map_translation}')
                self._recompute_map_goals()
                self._build_graph_planner()
                goal_param_changed = True
            elif p.name == 'sdcs_to_map_rotation_deg' and p.type_ == p.Type.DOUBLE:
                self.sdcs_to_map_rotation_deg = float(p.value)
                self.get_logger().info(
                    f'sdcs_to_map_rotation_deg updated: {self.sdcs_to_map_rotation_deg:.2f}')
                self._recompute_map_goals()
                self._build_graph_planner()
                goal_param_changed = True
            elif p.name == 'sdcs_to_map_scale' and p.type_ == p.Type.DOUBLE:
                self.sdcs_to_map_scale = float(p.value)
                self.get_logger().info(
                    f'sdcs_to_map_scale updated: {self.sdcs_to_map_scale:.4f}')
                self._recompute_map_goals()
                self._build_graph_planner()
                goal_param_changed = True
            elif p.name == 'one_way_route' and p.type_ == p.Type.BOOL:
                self.one_way_route = bool(p.value)
                self.get_logger().info(f'one_way_route updated: {self.one_way_route}')
                goal_param_changed = True
            elif p.name == 'allow_route_wrap' and p.type_ == p.Type.BOOL:
                self.allow_route_wrap = bool(p.value)
                self.get_logger().info(f'allow_route_wrap updated: {self.allow_route_wrap}')
                goal_param_changed = True
            elif p.name == 'route_wrap_gap_tolerance_m' and p.type_ == p.Type.DOUBLE:
                self.route_wrap_gap_tolerance_m = float(p.value)
                self.get_logger().info(
                    f'route_wrap_gap_tolerance_m updated: {self.route_wrap_gap_tolerance_m:.2f}')
                goal_param_changed = True
            elif p.name == 'heading_aware_start' and p.type_ == p.Type.BOOL:
                self.heading_aware_start = bool(p.value)
                self.get_logger().info(f'heading_aware_start updated: {self.heading_aware_start}')
                goal_param_changed = True
            elif p.name == 'max_start_heading_error_rad' and p.type_ == p.Type.DOUBLE:
                self.max_start_heading_error_rad = float(p.value)
                self.get_logger().info(
                    f'max_start_heading_error_rad updated: {self.max_start_heading_error_rad:.2f}')
                goal_param_changed = True
            elif p.name == 'heading_score_weight_m_per_rad' and p.type_ == p.Type.DOUBLE:
                self.heading_score_weight_m_per_rad = float(p.value)
                self.get_logger().info(
                    f'heading_score_weight_m_per_rad updated: {self.heading_score_weight_m_per_rad:.2f}')
                goal_param_changed = True
            elif p.name == 'reverse_threshold_m' and p.type_ == p.Type.DOUBLE:
                self.reverse_threshold_m = float(p.value)
                self.get_logger().info(f'reverse_threshold_m updated: {self.reverse_threshold_m:.2f}m')
            elif p.name == 'reverse_speed' and p.type_ == p.Type.DOUBLE:
                self.reverse_speed = float(p.value)
                self.get_logger().info(f'reverse_speed updated: {self.reverse_speed:.3f}m/s')
            elif p.name == 'dispatch_ride' and p.type_ == p.Type.BOOL:
                if bool(p.value):
                    if self.startup_done and self.mission_stage == MissionStage.IDLE:
                        self.new_ride_requested = True
                        self.get_logger().info('dispatch_ride triggered: dispatching ride now')
                    elif not self.startup_done:
                        self._pending_dispatch = True
                        self.get_logger().info(
                            'dispatch_ride triggered: queued (startup not done yet)')
                    else:
                        self.get_logger().warn(
                            f'dispatch_ride triggered but mission in progress: '
                            f'stage={self.mission_stage.name}')

        if self.ready_for_rides and self.mission_stage == MissionStage.IDLE:
            self.new_ride_requested = True
        elif goal_param_changed:
            if not self.startup_done:
                # Params changed before startup finished — remember to dispatch when ready
                self._pending_dispatch = True
                self.get_logger().info(
                    'Goal params updated before startup — ride will dispatch after hub arrival')
            else:
                self._replan_active_goal()

        return SetParametersResult(successful=True)

    def _replan_active_goal(self):
        if self.robot_pose is None:
            return

        if not self.startup_done:
            ok = self._send_path_to(self.hub_xy_map, label='HUB (updated)',
                                    goal_node_id=self.hub_node_id)
            if ok:
                self._startup_path_sent = True
            return

        if self.mission_stage == MissionStage.TO_PICKUP:
            self._send_path_to(self.pickup_xy_map, label='PICKUP (updated)',
                               goal_node_id=self.pickup_node_id)
        elif self.mission_stage == MissionStage.TO_DROPOFF:
            self._send_path_to(self.dropoff_xy_map, label='DROPOFF (updated)',
                               goal_node_id=self.dropoff_node_id)
        elif self.mission_stage == MissionStage.TO_HUB:
            self._send_path_to(self.hub_xy_map, label='HUB (updated)',
                               goal_node_id=self.hub_node_id)

    def _load_recorded_map(self):
        latest_map = find_latest_recording_map(frame_id=self.route_frame)
        if latest_map is None:
            self.get_logger().error(
                f'No recorded map found for trip planner in frame_id={self.route_frame}')
            return False

        data = load_recording_map(latest_map)
        frame_id = data.get('frame_id', '')
        if frame_id != self.route_frame:
            self.get_logger().error(
                f'Recorded map frame mismatch: file={frame_id} '
                f'expected={self.route_frame}')
            return False
        nodes = data.get('nodes', [])
        filtered_nodes, control_points, dense_waypoints = build_waypoint_path_from_nodes(
            nodes,
            min_node_spacing=self.recorded_min_node_spacing_m,
            min_yaw_change=self.recorded_min_yaw_change_rad,
            waypoint_spacing=self.generated_waypoint_spacing_m,
        )
        raw_points = np.zeros((2, 0), dtype=float)
        if nodes:
            raw_points = np.array(
                [[float(node['x']) for node in nodes],
                 [float(node['y']) for node in nodes]],
                dtype=float,
            )
        self.recorded_nodes = filtered_nodes
        self.recorded_points = control_points
        self.recorded_waypoints = dense_waypoints
        self.recorded_waypoint_yaws = compute_path_headings(self.recorded_waypoints)
        self.recorded_map_path = str(latest_map)
        if self.recorded_points.shape[1] < 2 or self.recorded_waypoints.shape[1] < 2:
            self.get_logger().error('Recorded map has too few points for planning')
            return False

        first = self.recorded_waypoints[:, 0]
        last = self.recorded_waypoints[:, -1]
        self.recorded_loop_gap = float(np.linalg.norm(first - last))
        self.recorded_loop = self.recorded_loop_gap < 0.75
        self.get_logger().info(
            f'Loaded recorded map for planner: {latest_map.name} '
            f'raw_nodes={len(nodes)} control_nodes={self.recorded_points.shape[1]} '
            f'dense_waypoints={self.recorded_waypoints.shape[1]} '
            f'loop={self.recorded_loop} loop_gap={self.recorded_loop_gap:.2f}m')
        self._publish_debug_markers(
            raw_nodes=raw_points,
            control_points=self.recorded_points,
            dense_waypoints=self.recorded_waypoints,
        )
        return True

    def _map_path_to_ros(self, wp_2xn):
        path_msg = Path()
        path_msg.header.stamp    = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.route_frame
        for i in range(wp_2xn.shape[1]):
            pt   = wp_2xn[:, i]
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(pt[0])
            pose.pose.position.y = float(pt[1])
            path_msg.poses.append(pose)
        return path_msg

    def _attach_exact_endpoint(self, route, point_xy, prepend=False, replace_thresh=0.05):
        point_xy = np.array(point_xy, dtype=float).reshape(2)
        point_col = point_xy.reshape(2, 1)
        if route.size == 0:
            return point_col

        endpoint = route[:, 0] if prepend else route[:, -1]
        if float(np.linalg.norm(endpoint - point_xy)) <= float(replace_thresh):
            if prepend:
                route[:, 0] = point_xy
            else:
                route[:, -1] = point_xy
            return route

        if prepend:
            return np.hstack([point_col, route])
        return np.hstack([route, point_col])

    def _goal_to_route_frame(self, goal_xy):
        goal = np.array([float(goal_xy[0]), float(goal_xy[1])], dtype=float)
        if not self.goals_are_sdcs_frame:
            return goal

        angle = np.deg2rad(self.sdcs_to_map_rotation_deg)
        rot = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)],
        ])
        trans = np.array(self.sdcs_to_map_translation, dtype=float)
        return self.sdcs_to_map_scale * (goal @ rot) + trans

    def _pose_yaw(self):
        if self.robot_pose is None:
            return None

        q = self.robot_pose.pose.orientation
        return float(np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)))

    def _route_can_wrap(self):
        if not self.allow_route_wrap:
            return False
        if self.recorded_loop:
            return True
        return self.recorded_loop_gap <= self.route_wrap_gap_tolerance_m

    def _plan_to_xy(self, goal_xy_map, goal_node_id: int = -1):
        """
        Plan a route from current robot pose to goal_xy_map (map frame).
        If goal_node_id >= 0, that node is used directly — no snapping needed.
        Everything works in map frame; no SDCS transform at query time.
        """
        if self.robot_pose is None:
            return None
        if self.graph_planner is None and not self._build_graph_planner():
            return None

        rx        = float(self.robot_pose.pose.position.x)
        ry        = float(self.robot_pose.pose.position.y)
        robot_yaw = self._pose_yaw()
        goal      = np.asarray(goal_xy_map, dtype=float).ravel()

        # Goal node: direct ID or nearest-position snap
        if goal_node_id >= 0:
            goal_node = goal_node_id
            goal_col  = self.graph_planner.node_position(goal_node_id).reshape(2, 1)
        else:
            goal_node, goal_dist = self.graph_planner.find_nearest_node_position_only(goal)
            goal_col = goal.reshape(2, 1)
            self.get_logger().debug(
                f'Goal snapped to node {goal_node} (dist={goal_dist:.2f}m)')

        # Edge-based start snap (heading-aware)
        edge_result = None
        if self.heading_aware_start and robot_yaw is not None:
            edge_result = self.graph_planner.find_start_on_edge(
                (rx, ry), robot_yaw,
                heading_weight=self.heading_score_weight_m_per_rad,
                max_heading_error=self.max_start_heading_error_rad,
            )

        if edge_result is not None:
            fi, ti, proj_pt, remaining_wp, proj_dist, heading_err = edge_result
            self.get_logger().info(
                f'Edge snap: edge=({fi}→{ti}) proj_dist={proj_dist:.3f}m '
                f'heading_err={math.degrees(heading_err):.1f}° '
                f'goal_node={goal_node} robot=({rx:.2f},{ry:.2f})')

            node_seq = self.graph_planner.astar(ti, goal_node)
            if node_seq is None:
                self.get_logger().error(
                    f'A* found no path from node {ti} to node {goal_node}')
                return None

            route_len = self.graph_planner.route_length_m(node_seq)
            self.get_logger().info(
                f'A* path: nodes={node_seq} graph_length={route_len:.2f}m')

            astar_path = self.graph_planner.build_path(node_seq)
            cols = [proj_pt]
            if remaining_wp.shape[1] > 0:
                cols.append(remaining_wp)
            if astar_path.shape[1] > 0:
                cols.append(astar_path)
            cols.append(goal_col)
            path = np.hstack(cols)

            first5 = [(float(path[0, i]), float(path[1, i]))
                      for i in range(min(5, path.shape[1]))]
            self.get_logger().info(
                f'Path ready: {path.shape[1]} waypoints | first5={first5}')

        else:
            # Fallback: node snap when all edges fail heading filter
            if self.heading_aware_start and robot_yaw is not None:
                start_node, start_score = self.graph_planner.find_nearest_node(
                    (rx, ry), robot_yaw,
                    heading_weight=self.heading_score_weight_m_per_rad,
                    max_heading_error=self.max_start_heading_error_rad,
                )
            else:
                start_node, start_score = self.graph_planner.find_nearest_node_position_only(
                    (rx, ry))

            self.get_logger().warn(
                f'Edge snap failed — fallback to node {start_node} (score={start_score:.2f}m) '
                f'goal_node={goal_node} robot=({rx:.2f},{ry:.2f})')

            node_seq = self.graph_planner.astar(start_node, goal_node)
            if node_seq is None:
                self.get_logger().error(
                    f'A* found no path from node {start_node} to node {goal_node}')
                return None

            route_len = self.graph_planner.route_length_m(node_seq)
            self.get_logger().info(
                f'A* path: nodes={node_seq} graph_length={route_len:.2f}m')

            path = self.graph_planner.build_path(node_seq)
            if path.shape[1] == 0:
                self.get_logger().error('build_path returned empty path')
                return None

            robot_col = np.array([[rx], [ry]])
            path = np.hstack([robot_col, path, goal_col])
            self.get_logger().info(f'Path ready: {path.shape[1]} waypoints')

        self._publish_debug_markers(active_route=path)
        return path

    def _send_path_to(self, goal_xy_map, label='', goal_node_id: int = -1):
        """goal_xy_map must already be in map frame."""
        wp = self._plan_to_xy(goal_xy_map, goal_node_id)
        if wp is None:
            return False
        self.waypoints_pub.publish(self._map_path_to_ros(wp))
        self._publish_debug_markers(active_route=wp)
        self.get_logger().info(
            f'Path published -> {label} '
            f'goal_map=({float(goal_xy_map[0]):.3f},{float(goal_xy_map[1]):.3f})'
            + (f' node_id={goal_node_id}' if goal_node_id >= 0 else ''))
        return True

    def _snap_to_exact(self, goal_xy_map, label=''):
        """Publish a straight two-point path to snap to an exact map-frame goal."""
        if self.robot_pose is None:
            return self._send_path_to(goal_xy_map, label)

        rx, ry  = float(self.robot_pose.pose.position.x), float(self.robot_pose.pose.position.y)
        cur_q   = np.array([rx, ry])
        goal_q  = np.asarray(goal_xy_map, dtype=float).ravel()
        dist    = float(np.linalg.norm(cur_q - goal_q))
        if dist < 0.10:
            self.get_logger().info(f'Already within 0.10m of {label}, no snap needed')
            return True

        wp = np.stack([cur_q, goal_q], axis=1)
        self.waypoints_pub.publish(self._map_path_to_ros(wp))
        self.get_logger().info(
            f'Snap path -> {label} ({goal_q[0]:.2f},{goal_q[1]:.2f}) dist={dist:.3f}m')
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

    def _should_reverse(self, goal_xy_map):
        """True if goal is close and behind the robot — better to reverse than loop around."""
        if self.robot_pose is None:
            return False
        rx   = float(self.robot_pose.pose.position.x)
        ry   = float(self.robot_pose.pose.position.y)
        goal = np.array(goal_xy_map, dtype=float)
        dist = float(np.linalg.norm(goal - np.array([rx, ry])))
        if dist > self.reverse_threshold_m:
            return False
        yaw = self._pose_yaw()
        if yaw is None:
            return False
        forward  = np.array([np.cos(yaw), np.sin(yaw)])
        to_goal  = goal - np.array([rx, ry])
        return float(np.dot(forward, to_goal)) < 0.0

    def _start_reverse(self, goal_xy_map, next_stage: MissionStage):
        self._reverse_target     = list(goal_xy_map)
        self._reverse_next_stage = next_stage
        self.mission_stage       = MissionStage.REVERSING
        self._set_led(self.LED_RED)
        self.get_logger().info(
            f'Reversing to ({float(goal_xy_map[0]):.3f},{float(goal_xy_map[1]):.3f}) '
            f'→ {next_stage.name} at {self.reverse_speed:.2f}m/s')

    def _handle_reversing(self):
        """Called every loop tick while REVERSING. Publishes override cmd; transitions when done."""
        if self.robot_pose is None or self._reverse_target is None:
            return
        rx   = float(self.robot_pose.pose.position.x)
        ry   = float(self.robot_pose.pose.position.y)
        dist = float(np.linalg.norm(
            np.array([rx, ry]) - np.array(self._reverse_target)))

        if dist < 0.20:
            # Arrived — send stop then transition
            self._override_pub.publish(Twist())
            self._reverse_target = None
            ns = self._reverse_next_stage
            self.get_logger().info(
                f'Reverse complete ({dist:.3f}m from target) → {ns.name}')
            if ns == MissionStage.WAIT_AT_PICKUP:
                self._on_arrived_pickup()
            elif ns == MissionStage.WAIT_AT_DROPOFF:
                self._on_arrived_dropoff()
            else:
                self._on_arrived_hub()
        else:
            cmd = Twist()
            cmd.linear.x  = -self.reverse_speed
            cmd.angular.z = 0.0
            self._override_pub.publish(cmd)

    def _on_arrived_pickup(self):
        self.mission_stage = MissionStage.WAIT_AT_PICKUP
        self.picked_up     = True
        self._set_led(self.LED_BLUE)
        self.pause_until   = time.time() + self.stop_seconds
        self.get_logger().info('Arrived at PICKUP.')

    def _on_arrived_dropoff(self):
        self.mission_stage = MissionStage.WAIT_AT_DROPOFF
        self.dropped_off   = True
        self._set_led(self.LED_ORANGE)
        self.pause_until   = time.time() + self.stop_seconds
        self.get_logger().info('Arrived at DROPOFF.')

    def _on_arrived_hub(self):
        self.mission_stage   = MissionStage.IDLE
        self.picked_up       = False
        self.dropped_off     = False
        self._set_led(self.LED_MAGENTA)
        self.ready_for_rides = True
        if self._pending_dispatch:
            self._pending_dispatch  = False
            self.new_ride_requested = True
            self.get_logger().info('Pending ride dispatching now.')
        else:
            self.get_logger().info('Mission complete. Ready for next ride.')

    def loop(self):
        now = time.time()

        if self.robot_pose is None:
            return

        # --- Startup: drive to hub once before accepting rides ---
        if not self.startup_done:
            if not self._startup_path_sent:
                ok = self._send_path_to(self.hub_xy_map, label='HUB (startup)',
                                        goal_node_id=self.hub_node_id)
                if ok:
                    self._startup_path_sent = True
                    self._set_led(self.LED_ORANGE)
                return
            if self._path_completed_event:
                self._path_completed_event = False
                self.startup_done    = True
                self.ready_for_rides = True
                self._set_led(self.LED_MAGENTA)
                self.get_logger().info('Startup complete. Ready for rides.')
                if self._pending_dispatch:
                    self._pending_dispatch  = False
                    self.new_ride_requested = True
                    self.get_logger().info('Pending ride dispatching now.')
            return

        # --- Reverse mode: trip_planner drives directly ---
        if self.mission_stage == MissionStage.REVERSING:
            self._handle_reversing()
            return

        if self.ready_for_rides and not self.new_ride_requested:
            return

        # --- Dispatch to pickup ---
        if self.ready_for_rides and self.new_ride_requested:
            self.new_ride_requested = False
            self.ready_for_rides    = False
            self.picked_up          = False
            self.dropped_off        = False
            if self._should_reverse(self.pickup_xy_map):
                self._start_reverse(self.pickup_xy_map, MissionStage.WAIT_AT_PICKUP)
            else:
                ok = self._send_path_to(self.pickup_xy_map, label='PICKUP',
                                        goal_node_id=self.pickup_node_id)
                if ok:
                    self.mission_stage = MissionStage.TO_PICKUP
                    self._set_led(self.LED_GREEN)
            return

        # --- Leave pickup stop ---
        if self.mission_stage == MissionStage.WAIT_AT_PICKUP and now >= self.pause_until:
            if self._should_reverse(self.dropoff_xy_map):
                self._start_reverse(self.dropoff_xy_map, MissionStage.WAIT_AT_DROPOFF)
            else:
                ok = self._send_path_to(self.dropoff_xy_map, label='DROPOFF',
                                        goal_node_id=self.dropoff_node_id)
                if ok:
                    self.mission_stage = MissionStage.TO_DROPOFF
                    self._set_led(self.LED_BLUE)
            return

        # --- Leave dropoff stop ---
        if self.mission_stage == MissionStage.WAIT_AT_DROPOFF and now >= self.pause_until:
            if self._should_reverse(self.hub_xy_map):
                self._start_reverse(self.hub_xy_map, MissionStage.IDLE)
            else:
                ok = self._send_path_to(self.hub_xy_map, label='HUB',
                                        goal_node_id=self.hub_node_id)
                if ok:
                    self.mission_stage = MissionStage.TO_HUB
                    self._set_led(self.LED_ORANGE)
            return

        if now < self.pause_until:
            return

        if not self._path_completed_event:
            return
        self._path_completed_event = False

        # --- Path-following arrivals ---
        if self.mission_stage == MissionStage.TO_PICKUP:
            self._snap_to_exact(self.pickup_xy_map, label='PICKUP-snap')
            self._on_arrived_pickup()

        elif self.mission_stage == MissionStage.TO_DROPOFF:
            self._snap_to_exact(self.dropoff_xy_map, label='DROPOFF-snap')
            self._on_arrived_dropoff()

        elif self.mission_stage == MissionStage.TO_HUB:
            self._on_arrived_hub()


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
