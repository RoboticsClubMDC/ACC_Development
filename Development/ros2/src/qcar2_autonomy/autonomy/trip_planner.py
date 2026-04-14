#!/usr/bin/env python3

import time
import numpy as np
from enum import Enum

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Bool
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, SetParametersResult
from rclpy.parameter import ParameterType
from visualization_msgs.msg import Marker

from autonomy.recorded_map_utils import (
    build_dense_recorded_segment,
    build_waypoint_path_from_nodes,
    closest_recorded_node_index,
    find_latest_recording_map,
    load_recording_map,
)


# SDCS big-map, right-hand traffic node 10 (taxi hub) expressed in map coordinates.
DEFAULT_TAXI_HUB_XY = [-1.28205, -0.45991]


class MissionStage(Enum):
    IDLE           = 0
    TO_PICKUP      = 1
    WAIT_AT_PICKUP = 2
    TO_DROPOFF     = 3
    WAIT_AT_DROPOFF = 4
    TO_HUB         = 5


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

        self.add_on_set_parameters_callback(self.parameter_update_callback)

        self.get_logger().info(f'Route frame: {self.route_frame}')
        self.get_logger().info(f'Hub in {self.route_frame} frame: {self.hub_xy}')
        self.recorded_points = np.zeros((2, 0), dtype=float)
        self.recorded_waypoints = np.zeros((2, 0), dtype=float)
        self.recorded_nodes = []
        self.recorded_map_path = None
        self.recorded_loop = False

        self.LED_GREEN   = 1
        self.LED_BLUE    = 2
        self.LED_MAGENTA = 5
        self.LED_ORANGE  = 6

        self.robot_pose             = None
        self.path_status            = False
        self._path_completed_event  = False

        self.startup_done          = False
        self._startup_path_sent    = False
        self.ready_for_rides       = False
        self.new_ride_requested    = False

        self.mission_stage = MissionStage.IDLE
        self.pause_until   = 0.0
        self.picked_up     = False
        self.dropped_off   = False
        self._last_led     = None

        self.waypoints_pub = self.create_publisher(Path, '/cmd_waypoints', 1)
        self.raw_nodes_marker_pub = self.create_publisher(Marker, '/planner/raw_recorded_nodes', 1)
        self.control_nodes_marker_pub = self.create_publisher(Marker, '/planner/control_nodes', 1)
        self.dense_route_marker_pub = self.create_publisher(Marker, '/planner/dense_recorded_route', 1)
        self.active_route_marker_pub = self.create_publisher(Marker, '/planner/active_route_nodes', 1)

        self._load_recorded_map()

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

    def parameter_update_callback(self, params):
        goal_param_changed = False
        for p in params:
            if p.name == 'pickup_xy' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.pickup_xy = list(p.value)
                self.get_logger().info(f'pickup_xy updated: {self.pickup_xy}')
                goal_param_changed = True
            elif p.name == 'dropoff_xy' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.dropoff_xy = list(p.value)
                self.get_logger().info(f'dropoff_xy updated: {self.dropoff_xy}')
                goal_param_changed = True
            elif p.name == 'hub_xy' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.hub_xy = list(p.value)
                self.get_logger().info(f'hub_xy updated: {self.hub_xy}')
                goal_param_changed = True
            elif p.name == 'stop_seconds' and p.type_ == p.Type.DOUBLE_ARRAY:
                self.stop_seconds = float(list(p.value)[0])
            elif p.name == 'recorded_min_node_spacing_m' and p.type_ == p.Type.DOUBLE:
                self.recorded_min_node_spacing_m = float(p.value)
                self._load_recorded_map()
            elif p.name == 'recorded_min_yaw_change_rad' and p.type_ == p.Type.DOUBLE:
                self.recorded_min_yaw_change_rad = float(p.value)
                self._load_recorded_map()
            elif p.name == 'generated_waypoint_spacing_m' and p.type_ == p.Type.DOUBLE:
                self.generated_waypoint_spacing_m = float(p.value)
                self._load_recorded_map()
            elif p.name == 'goal_on_route_tolerance_m' and p.type_ == p.Type.DOUBLE:
                self.goal_on_route_tolerance_m = float(p.value)

        if self.ready_for_rides and self.mission_stage == MissionStage.IDLE:
            self.new_ride_requested = True
        elif goal_param_changed:
            self._replan_active_goal()

        return SetParametersResult(successful=True)

    def _replan_active_goal(self):
        if self.robot_pose is None:
            return

        if not self.startup_done:
            ok = self._send_path_to(self.hub_xy, label='HUB (updated)')
            if ok:
                self._startup_path_sent = True
            return

        if self.mission_stage == MissionStage.TO_PICKUP:
            self._send_path_to(self.pickup_xy, label='PICKUP (updated)')
        elif self.mission_stage == MissionStage.TO_DROPOFF:
            self._send_path_to(self.dropoff_xy, label='DROPOFF (updated)')
        elif self.mission_stage == MissionStage.TO_HUB:
            self._send_path_to(self.hub_xy, label='HUB (updated)')

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
        self.recorded_map_path = str(latest_map)
        if self.recorded_points.shape[1] < 2 or self.recorded_waypoints.shape[1] < 2:
            self.get_logger().error('Recorded map has too few points for planning')
            return False

        first = self.recorded_waypoints[:, 0]
        last = self.recorded_waypoints[:, -1]
        self.recorded_loop = float(np.linalg.norm(first - last)) < 0.75
        self.get_logger().info(
            f'Loaded recorded map for planner: {latest_map.name} '
            f'raw_nodes={len(nodes)} control_nodes={self.recorded_points.shape[1]} '
            f'dense_waypoints={self.recorded_waypoints.shape[1]} '
            f'loop={self.recorded_loop}')
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

    def _plan_to_xy(self, goal_xy):
        if self.robot_pose is None:
            return None
        if self.recorded_waypoints.shape[1] < 2 and not self._load_recorded_map():
            return None

        rx = float(self.robot_pose.pose.position.x)
        ry = float(self.robot_pose.pose.position.y)
        cur = np.array([rx, ry], dtype=float)
        goal = np.array([float(goal_xy[0]), float(goal_xy[1])], dtype=float)

        start_idx = closest_recorded_node_index(self.recorded_waypoints, cur)
        goal_idx = closest_recorded_node_index(self.recorded_waypoints, goal)
        if start_idx is None or goal_idx is None:
            self.get_logger().error('Failed to find nearest recorded nodes')
            return None

        start_nearest = self.recorded_waypoints[:, start_idx]
        goal_nearest = self.recorded_waypoints[:, goal_idx]
        start_route_dist = float(np.linalg.norm(start_nearest - cur))
        goal_route_dist = float(np.linalg.norm(goal_nearest - goal))

        route = build_dense_recorded_segment(
            self.recorded_waypoints,
            start_idx,
            goal_idx,
            spacing=self.generated_waypoint_spacing_m,
            closed_loop=self.recorded_loop,
        )
        if route.shape[1] == 0:
            return None

        # The recorded map is only a guide corridor. The actual current pose and
        # requested goal remain the true endpoints of the planned path.
        if goal_route_dist > self.goal_on_route_tolerance_m:
            self.get_logger().warn(
                f'Goal ({goal[0]:.2f}, {goal[1]:.2f}) is {goal_route_dist:.2f}m off the recorded route; '
                'keeping the recorded route as a guide, then appending the exact goal coordinate')

        if start_route_dist > 0.25:
            self.get_logger().warn(
                f'Robot start pose is {start_route_dist:.2f}m from the recorded route; '
                'prepending the exact current pose before joining the recorded route')

        route = self._attach_exact_endpoint(route, cur, prepend=True)
        route = self._attach_exact_endpoint(route, goal, prepend=False)

        self.get_logger().info(
            f'Planner route start_idx={start_idx} goal_idx={goal_idx} '
            f'pts={route.shape[1]} goal=({goal[0]:.2f},{goal[1]:.2f}) '
            f'start_route_dist={start_route_dist:.2f} goal_route_dist={goal_route_dist:.2f} '
            f'using_dense_route={self.recorded_waypoints.shape[1]}')
        return route

    def _send_path_to(self, goal_xy, label=''):
        wp = self._plan_to_xy(goal_xy)
        if wp is None:
            return False
        self.waypoints_pub.publish(self._map_path_to_ros(wp))
        self._publish_debug_markers(active_route=wp)
        self.get_logger().info(f'Path published -> {label} ({goal_xy[0]:.2f}, {goal_xy[1]:.2f})')
        return True

    def _snap_to_exact(self, goal_xy, label=''):
        if self.robot_pose is None:
            return self._send_path_to(goal_xy, label)

        rx, ry = float(self.robot_pose.pose.position.x), float(self.robot_pose.pose.position.y)
        cur_q  = np.array([rx, ry])
        goal_q = np.array([float(goal_xy[0]), float(goal_xy[1])])

        dist = float(np.linalg.norm(cur_q - goal_q))
        if dist < 0.10:
            self.get_logger().info(f'Already within 0.10m of {label}, no snap needed')
            return True

        wp = np.stack([cur_q, goal_q], axis=1)
        self.waypoints_pub.publish(self._map_path_to_ros(wp))
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

    def loop(self):
        now = time.time()

        if self.robot_pose is None:
            return

        if not self.startup_done:
            if not self._startup_path_sent:
                ok = self._send_path_to(self.hub_xy, label='HUB (startup)')
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
            return

        if self.ready_for_rides and not self.new_ride_requested:
            return

        if self.ready_for_rides and self.new_ride_requested:
            self.new_ride_requested = False
            self.ready_for_rides    = False
            self.picked_up          = False
            self.dropped_off        = False
            ok = self._send_path_to(self.pickup_xy, label='PICKUP')
            if ok:
                self.mission_stage = MissionStage.TO_PICKUP
                self._set_led(self.LED_GREEN)
            return

        if self.mission_stage == MissionStage.WAIT_AT_PICKUP and now >= self.pause_until:
            ok = self._send_path_to(self.dropoff_xy, label='DROPOFF')
            if ok:
                self.mission_stage = MissionStage.TO_DROPOFF
                self._set_led(self.LED_BLUE)
            return

        if self.mission_stage == MissionStage.WAIT_AT_DROPOFF and now >= self.pause_until:
            ok = self._send_path_to(self.hub_xy, label='HUB')
            if ok:
                self.mission_stage = MissionStage.TO_HUB
                self._set_led(self.LED_ORANGE)
            return

        if now < self.pause_until:
            return

        if not self._path_completed_event:
            return
        self._path_completed_event = False

        if self.mission_stage == MissionStage.TO_PICKUP:
            self._snap_to_exact(self.pickup_xy, label='PICKUP-snap')
            self.mission_stage = MissionStage.WAIT_AT_PICKUP
            self.picked_up     = True
            self._set_led(self.LED_BLUE)
            self.pause_until   = now + self.stop_seconds
            self.get_logger().info('Arrived at PICKUP.')

        elif self.mission_stage == MissionStage.TO_DROPOFF:
            self._snap_to_exact(self.dropoff_xy, label='DROPOFF-snap')
            self.mission_stage = MissionStage.WAIT_AT_DROPOFF
            self.dropped_off   = True
            self._set_led(self.LED_ORANGE)
            self.pause_until   = now + self.stop_seconds
            self.get_logger().info('Arrived at DROPOFF.')

        elif self.mission_stage == MissionStage.TO_HUB:
            self.mission_stage = MissionStage.IDLE
            self.picked_up     = False
            self.dropped_off   = False
            self._set_led(self.LED_MAGENTA)
            self.ready_for_rides = True
            self.get_logger().info('Mission complete. Ready for next ride.')


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
