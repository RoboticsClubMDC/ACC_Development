#!/usr/bin/env python3

from pathlib import Path

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path as PathMsg
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker

from autonomy.recorded_map_utils import find_latest_recording_map, load_recording_map


class RecordedMapVisualizer(Node):
    def __init__(self):
        super().__init__('recorded_map_visualizer')

        self.declare_parameter('route_frame', 'map')
        self.declare_parameter('map_path', '')
        self.declare_parameter('publish_hz', 1.0)
        self.declare_parameter('line_width', 0.025)
        self.declare_parameter('node_size', 0.08)

        self.route_frame = self.get_parameter('route_frame').get_parameter_value().string_value
        self.map_path = self.get_parameter('map_path').get_parameter_value().string_value
        self.publish_hz = float(self.get_parameter('publish_hz').get_parameter_value().double_value)
        self.line_width = float(self.get_parameter('line_width').get_parameter_value().double_value)
        self.node_size = float(self.get_parameter('node_size').get_parameter_value().double_value)

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.path_pub = self.create_publisher(PathMsg, '/recorded_map_path', qos)
        self.line_pub = self.create_publisher(Marker, '/recorded_map_line_marker', qos)
        self.nodes_pub = self.create_publisher(Marker, '/recorded_map_nodes_marker', qos)

        self._loaded_path = None
        self._loaded_mtime = None
        self._path_msg = None
        self._line_marker = None
        self._nodes_marker = None

        period = 1.0 / max(self.publish_hz, 0.1)
        self.timer = self.create_timer(period, self._timer_cb)
        self._load_latest_if_needed(force=True)

    def _resolve_map_path(self):
        if self.map_path:
            path = Path(self.map_path).expanduser()
            return path if path.exists() else None
        return find_latest_recording_map(frame_id=self.route_frame)

    def _load_latest_if_needed(self, force=False):
        path = self._resolve_map_path()
        if path is None:
            self.get_logger().warn('No recorded map JSON found to visualize')
            return False

        path = Path(path)
        mtime = path.stat().st_mtime
        if not force and self._loaded_path == path and self._loaded_mtime == mtime:
            return True

        payload = load_recording_map(path)
        nodes = payload.get('nodes', [])
        frame_id = payload.get('frame_id') or self.route_frame
        if not nodes:
            self.get_logger().warn(f'Recorded map has no nodes: {path}')
            return False

        self._path_msg = self._make_path_msg(nodes, frame_id)
        self._line_marker = self._make_line_marker(nodes, frame_id)
        self._nodes_marker = self._make_nodes_marker(nodes, frame_id)
        self._loaded_path = path
        self._loaded_mtime = mtime

        self.get_logger().warn(
            f'Visualizing recorded map: {path.name} | nodes={len(nodes)} | frame={frame_id}')
        return True

    def _make_path_msg(self, nodes, frame_id):
        msg = PathMsg()
        msg.header.frame_id = frame_id
        for node in nodes:
            pose = PoseStamped()
            pose.header.frame_id = frame_id
            pose.pose.position.x = float(node['x'])
            pose.pose.position.y = float(node['y'])
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        return msg

    def _make_line_marker(self, nodes, frame_id):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.ns = 'recorded_map_line'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = self.line_width
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        for node in nodes:
            point = Point()
            point.x = float(node['x'])
            point.y = float(node['y'])
            point.z = 0.01
            marker.points.append(point)
        return marker

    def _make_nodes_marker(self, nodes, frame_id):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.ns = 'recorded_map_nodes'
        marker.id = 1
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = self.node_size
        marker.scale.y = self.node_size
        marker.scale.z = self.node_size
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        for node in nodes:
            point = Point()
            point.x = float(node['x'])
            point.y = float(node['y'])
            point.z = 0.02
            marker.points.append(point)
        return marker

    def _timer_cb(self):
        if not self._load_latest_if_needed():
            return

        stamp = self.get_clock().now().to_msg()
        self._path_msg.header.stamp = stamp
        self._line_marker.header.stamp = stamp
        self._nodes_marker.header.stamp = stamp
        for pose in self._path_msg.poses:
            pose.header.stamp = stamp

        self.path_pub.publish(self._path_msg)
        self.line_pub.publish(self._line_marker)
        self.nodes_pub.publish(self._nodes_marker)


def main(args=None):
    rclpy.init(args=args)
    node = RecordedMapVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
