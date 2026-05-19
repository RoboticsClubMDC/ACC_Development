#!/usr/bin/env python3
"""
auto_teach_mapper.py
Generates a full right-hand-traffic SDCS route and publishes it to /cmd_waypoints
so that path_follower drives the car autonomously while path_teacher records the map.

Run via auto_teach_launch.py — do not use standalone.
"""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from autonomy.recorded_map_utils import get_recording_maps_dir
from autonomy.sdcs_graph_planner import SDCS_EDGES, SDCS_NODES, SDCSGraphPlanner

# Right-hand traffic coverage route.
# Ordered to defer the tight 14→16 roundabout (189°, 0.678 m radius) until arc 5,
# after the car is well-established on the path.  Goes 14→20 on the first outer
# pass to avoid the hardest segment while the car is still near the start.
FULL_MAP_ROUTE_NODE_SEQ = [
    # Arc 1: outer right loop — avoids roundabout, ends at taxi hub
    0, 2, 4, 14, 20, 22, 10,
    # Arc 2: hub → inner upper section → outer lower left → hub
    1, 13, 19, 17, 15, 6, 8, 10,
    # Arc 3: hub → inner kink loops → back to origin
    1, 7, 5, 3, 1, 8, 23, 21, 16, 18, 11, 12, 0,
    # Arc 4: outer lower loop
    2, 4, 6, 0,
    # Arc 5: outer right with roundabout (14→16) — car is now settled
    2, 4, 14, 16, 17, 20, 22, 9, 0,
    # Arc 6: outer lower → hub (final stop)
    2, 4, 6, 8, 10,
]

# Defaults for your current QLabs spawn.  The mapper prefers live TF yaw when
# available because QLabs and ROS can disagree on yaw sign conventions.
DEFAULT_SPAWN_XY = [0.0, 0.0]
DEFAULT_SPAWN_YAW_DEG = -33.0
DEFAULT_MAP_SCALE = 1.0216993081930361
DEFAULT_MAP_TRANSLATION = [0.0, 0.0]


class AutoTeachMapper(Node):
    def __init__(self):
        super().__init__('auto_teach_mapper')

        self.declare_parameter('route_frame',       'map')
        self.declare_parameter('robot_frame',       'base_link')
        self.declare_parameter('waypoint_spacing',  0.03)
        self.declare_parameter('publish_delay_sec', 3.0)
        self.declare_parameter('route_node_sequence', FULL_MAP_ROUTE_NODE_SEQ)
        self.declare_parameter('spawn_xy', DEFAULT_SPAWN_XY)
        self.declare_parameter('spawn_yaw_deg', DEFAULT_SPAWN_YAW_DEG)
        self.declare_parameter('align_start_yaw_to_robot', True)
        self.declare_parameter('map_scale', DEFAULT_MAP_SCALE)
        self.declare_parameter('map_translation', DEFAULT_MAP_TRANSLATION)
        self.declare_parameter('anchor_start_to_spawn', True)
        self.declare_parameter('align_start_yaw_to_spawn', True)

        self.route_frame      = self.get_parameter('route_frame').get_parameter_value().string_value
        self.robot_frame      = self.get_parameter('robot_frame').get_parameter_value().string_value
        waypoint_spacing      = float(self.get_parameter('waypoint_spacing').get_parameter_value().double_value)
        self.publish_delay    = float(self.get_parameter('publish_delay_sec').get_parameter_value().double_value)
        self.route_node_seq   = list(
            self.get_parameter('route_node_sequence').get_parameter_value().integer_array_value)
        self.spawn_xy         = list(self.get_parameter('spawn_xy').get_parameter_value().double_array_value)
        self.spawn_yaw_deg    = float(self.get_parameter('spawn_yaw_deg').get_parameter_value().double_value)
        self.align_start_yaw_to_robot = bool(
            self.get_parameter('align_start_yaw_to_robot').get_parameter_value().bool_value)
        self.map_scale        = float(self.get_parameter('map_scale').get_parameter_value().double_value)
        self.map_translation  = list(
            self.get_parameter('map_translation').get_parameter_value().double_array_value)
        self.anchor_start_to_spawn = bool(
            self.get_parameter('anchor_start_to_spawn').get_parameter_value().bool_value)
        self.align_start_yaw_to_spawn = bool(
            self.get_parameter('align_start_yaw_to_spawn').get_parameter_value().bool_value)

        if len(self.spawn_xy) < 2:
            raise ValueError('spawn_xy must contain [x, y]')
        if len(self.map_translation) < 2:
            raise ValueError('map_translation must contain [x, y]')
        if len(self.route_node_seq) < 2:
            raise ValueError('route_node_sequence must contain at least two nodes')

        self._route_published = False
        self._route_complete  = False

        qos_latched = QoSProfile(depth=1)
        qos_latched.reliability = ReliabilityPolicy.RELIABLE
        qos_latched.durability  = DurabilityPolicy.TRANSIENT_LOCAL

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._path_pub = self.create_publisher(Path, '/cmd_waypoints', qos_latched)
        self.create_subscription(Bool, '/path_status', self._path_status_cb, 10)

        self._waypoint_spacing = waypoint_spacing
        self._path_msg = None
        self._route_len_m = 0.0
        self._path_msg_waypoint_count = 0
        self._covered_edge_count = 0
        self._known_edge_count = 0
        self._missing_route_edges = []
        self._map_rotation_deg = 0.0

        maps_dir = get_recording_maps_dir()
        self.get_logger().info(
            f'AutoTeachMapper ready | '
            f'start_node={self.route_node_seq[0]} goal_node={self.route_node_seq[-1]} | '
            f'map will be saved to {maps_dir}/')
        self.get_logger().info(
            f'SDCS->map for auto-teach: scale={self.map_scale:.4f} '
            f't={self.map_translation[:2]} '
            f'anchor_start={self.anchor_start_to_spawn} '
            f'spawn_xy={self.spawn_xy[:2]} fallback_spawn_yaw={self.spawn_yaw_deg:.2f}deg '
            f'align_to_robot_yaw={self.align_start_yaw_to_robot}')
        self.get_logger().info(
            f'Publishing route to /cmd_waypoints in {self.publish_delay:.1f}s '
            f'— path_teacher will begin recording as the car moves')

        self._publish_timer = self.create_timer(self.publish_delay, self._publish_once)

    def _build_path_msg(self, waypoint_spacing: float, start_yaw_deg: float) -> Path:
        self._map_rotation_deg = self._compute_map_rotation_deg(start_yaw_deg)
        planner = SDCSGraphPlanner(
            waypoint_spacing=waypoint_spacing,
            scale=self.map_scale,
            rot_rad=math.radians(self._map_rotation_deg),
            translation=self.map_translation[:2],
        )

        self._validate_route_sequence()

        waypoints_2xn = planner.build_path(self.route_node_seq)   # 2×N in map frame
        if self.anchor_start_to_spawn and waypoints_2xn.shape[1] > 0:
            spawn = np.array(self.spawn_xy[:2], dtype=float).reshape(2, 1)
            waypoints_2xn = waypoints_2xn + (spawn - waypoints_2xn[:, [0]])

        self._route_len_m = planner.route_length_m(self.route_node_seq)
        self._path_msg_waypoint_count = waypoints_2xn.shape[1]

        path_msg = Path()
        path_msg.header.frame_id = self.route_frame

        for i in range(self._path_msg_waypoint_count):
            pose = PoseStamped()
            pose.header.frame_id = self.route_frame
            pose.pose.position.x = float(waypoints_2xn[0, i])
            pose.pose.position.y = float(waypoints_2xn[1, i])
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        return path_msg

    def _compute_map_rotation_deg(self, start_yaw_deg: float) -> float:
        if not self.align_start_yaw_to_spawn:
            return 0.0

        start_node = int(self.route_node_seq[0])
        sdcs_start_yaw_deg = math.degrees(SDCS_NODES[start_node][2])
        return sdcs_start_yaw_deg - start_yaw_deg

    def _lookup_robot_yaw_deg(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.route_frame,
                self.robot_frame,
                rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().warn(
                f'Waiting for {self.route_frame}->{self.robot_frame} before publishing route: {ex}')
            return None

        qx = float(transform.transform.rotation.x)
        qy = float(transform.transform.rotation.y)
        qz = float(transform.transform.rotation.z)
        qw = float(transform.transform.rotation.w)
        yaw_rad = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz))
        return math.degrees(yaw_rad)

    def _validate_route_sequence(self):
        graph_edges = {(int(fi), int(ti)) for fi, ti, _radius in SDCS_EDGES}
        route_edges = set()
        missing_segments = []

        for fi, ti in zip(self.route_node_seq[:-1], self.route_node_seq[1:]):
            edge = (int(fi), int(ti))
            route_edges.add(edge)
            if edge not in graph_edges:
                missing_segments.append(edge)

        if missing_segments:
            raise ValueError(
                f'route_node_sequence contains invalid directed edges: {missing_segments}')

        self._known_edge_count = len(graph_edges)
        self._covered_edge_count = len(route_edges)
        self._missing_route_edges = sorted(graph_edges - route_edges)

    def _publish_once(self):
        if self._route_published:
            return

        start_yaw_deg = self.spawn_yaw_deg
        if self.align_start_yaw_to_robot:
            robot_yaw_deg = self._lookup_robot_yaw_deg()
            if robot_yaw_deg is None:
                return
            start_yaw_deg = robot_yaw_deg
            self.get_logger().warn(
                f'Auto-teach using live start yaw from TF: {start_yaw_deg:.2f}deg '
                f'(fallback spawn_yaw_deg={self.spawn_yaw_deg:.2f}deg)')

        self._path_msg = self._build_path_msg(self._waypoint_spacing, start_yaw_deg)
        self._publish_timer.cancel()
        self._route_published = True

        self._path_msg.header.stamp = self.get_clock().now().to_msg()
        self._path_pub.publish(self._path_msg)

        self.get_logger().warn(
            f'Route published to /cmd_waypoints | '
            f'{self._path_msg_waypoint_count} waypoints | '
            f'{self._route_len_m:.2f}m | '
            f'rot={self._map_rotation_deg:.2f}deg | '
            f'covered_edges={self._covered_edge_count}/{self._known_edge_count} | '
            f'Mapping started — path_teacher is recording nodes')
        if self._missing_route_edges:
            self.get_logger().info(
                f'Teach route intentionally skips unused directed edges: {self._missing_route_edges}')

    def _path_status_cb(self, msg: Bool):
        if msg.data and self._route_published and not self._route_complete:
            self._route_complete = True
            maps_dir = get_recording_maps_dir()
            self.get_logger().warn(
                f'Route complete — car stopped | '
                f'Taught map saved in {maps_dir}/ | '
                f'Ctrl+C to shut down all nodes')


def main(args=None):
    rclpy.init(args=args)
    node = AutoTeachMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
