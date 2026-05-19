#!/usr/bin/env python3

import json
import math
from datetime import datetime

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Empty
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import Marker

from autonomy.recorded_map_utils import get_recording_maps_dir


class PathTeacher(Node):
    def __init__(self):
        super().__init__('path_teacher')

        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('assume_start_at_origin', False)
        self.declare_parameter('sample_period', 0.10)
        self.declare_parameter('node_interval_sec', 1.0)
        self.declare_parameter('min_node_spacing_m', 0.10)
        self.declare_parameter('min_node_yaw_change_rad', 0.20)
        self.declare_parameter('path_topic', '/taught_path')
        self.declare_parameter('line_marker_topic', '/taught_path_marker')
        self.declare_parameter('node_marker_topic', '/taught_nodes_marker')
        self.declare_parameter('line_marker_width', 0.08)
        self.declare_parameter('node_marker_size', 0.14)
        self.declare_parameter('map_name', 'taught_map')
        self.declare_parameter('mapping_enabled', True)
        self.declare_parameter('node_delete_wait', 30.0)

        self.global_frame = self.get_parameter(
            'global_frame').get_parameter_value().string_value
        self.robot_frame = self.get_parameter(
            'robot_frame').get_parameter_value().string_value
        self.assume_start_at_origin = bool(
            self.get_parameter('assume_start_at_origin').get_parameter_value().bool_value)
        self.sample_period = float(self.get_parameter(
            'sample_period').get_parameter_value().double_value)
        self.node_interval_sec = float(self.get_parameter(
            'node_interval_sec').get_parameter_value().double_value)
        self.min_node_spacing_m = float(self.get_parameter(
            'min_node_spacing_m').get_parameter_value().double_value)
        self.min_node_yaw_change_rad = float(self.get_parameter(
            'min_node_yaw_change_rad').get_parameter_value().double_value)
        self.path_topic = self.get_parameter(
            'path_topic').get_parameter_value().string_value
        self.line_marker_topic = self.get_parameter(
            'line_marker_topic').get_parameter_value().string_value
        self.node_marker_topic = self.get_parameter(
            'node_marker_topic').get_parameter_value().string_value
        self.line_marker_width = float(self.get_parameter(
            'line_marker_width').get_parameter_value().double_value)
        self.node_marker_size = float(self.get_parameter(
            'node_marker_size').get_parameter_value().double_value)
        self.map_name = self.get_parameter(
            'map_name').get_parameter_value().string_value
        self.mapping_enabled = self.get_parameter(
            'mapping_enabled').get_parameter_value().bool_value
        self.node_delete_wait = float(self.get_parameter(
            'node_delete_wait').get_parameter_value().double_value)

        self.save_directory = get_recording_maps_dir()
        self.save_directory.mkdir(parents=True, exist_ok=True)
        self.save_path = self._make_versioned_save_path()

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.path_pub = self.create_publisher(Path, self.path_topic, qos)
        self.line_marker_pub = self.create_publisher(Marker, self.line_marker_topic, qos)
        self.node_marker_pub = self.create_publisher(Marker, self.node_marker_topic, qos)
        self.create_subscription(Bool, '/mapping_enable', self._mapping_enable_cb, 2)
        self.create_subscription(Empty, '/undo_last_node', self._undo_last_node_cb, 2)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.path_msg = Path()
        self.path_msg.header.frame_id = self.global_frame
        self.line_marker_msg = self._make_line_marker()
        self.node_marker_msg = self._make_node_marker()

        self.saved_nodes = []
        self.last_record_time = None
        self._undo_cooldown_until = 0.0
        self._origin_xy = None

        self.timer = self.create_timer(self.sample_period, self._timer_cb)
        self.get_logger().info(
            'PathTeacher ready. '
            f'Saving nodes every {self.node_interval_sec:.1f}s to {self.save_path} | '
            f'frame={self.global_frame} | '
            f'assume_start_at_origin={self.assume_start_at_origin} | '
            f'min_spacing={self.min_node_spacing_m:.2f}m '
            f'min_yaw={self.min_node_yaw_change_rad:.2f}rad')
        self.get_logger().info(
            'Pause/resume recording with /mapping_enable '
            '(false=pause, true=resume)')

    def _undo_last_node_cb(self, _msg: Empty):
        if not self.saved_nodes:
            self.get_logger().warn('Undo requested but no nodes to remove')
            return
        removed = self.saved_nodes.pop()
        self.path_msg.poses.pop()
        self.line_marker_msg.points.pop()
        self.node_marker_msg.points.pop()
        self.last_record_time = self.saved_nodes[-1]['stamp_sec'] if self.saved_nodes else None
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        self._undo_cooldown_until = now_sec + self.node_delete_wait
        stamp = self.get_clock().now().to_msg()
        self.path_msg.header.stamp = stamp
        self.line_marker_msg.header.stamp = stamp
        self.node_marker_msg.header.stamp = stamp
        self.path_pub.publish(self.path_msg)
        self.line_marker_pub.publish(self.line_marker_msg)
        self.node_marker_pub.publish(self.node_marker_msg)
        self._write_map_file()
        self.get_logger().warn(
            f'Undid node {removed["index"]}: x={removed["x"]:.2f}, y={removed["y"]:.2f} '
            f'— {len(self.saved_nodes)} nodes remaining | '
            f'recording paused {self.node_delete_wait:.0f}s')

    def _mapping_enable_cb(self, msg: Bool):
        enabled = bool(msg.data)
        if enabled == self.mapping_enabled:
            return
        self.mapping_enabled = enabled
        self.last_record_time = None if enabled else self.last_record_time
        state = 'resumed' if enabled else 'paused'
        self.get_logger().warn(f'Mapping {state}')

    def _make_versioned_save_path(self):
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{self.map_name}_{stamp}.json'
        return self.save_directory / filename

    def _make_line_marker(self):
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.ns = 'path_teacher_line'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = self.line_marker_width
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        return marker

    def _make_node_marker(self):
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.ns = 'path_teacher_nodes'
        marker.id = 1
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = self.node_marker_size
        marker.scale.y = self.node_marker_size
        marker.scale.z = self.node_marker_size
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        return marker

    def _timer_cb(self):
        if not self.mapping_enabled:
            return
        transform = self._lookup_robot_transform()
        if transform is None:
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if now_sec < self._undo_cooldown_until:
            return

        if self.last_record_time is None:
            self._record_node(transform, now_sec)
            return

        if now_sec - self.last_record_time >= self.node_interval_sec:
            if not self._should_record_node(transform):
                self.get_logger().debug(
                    'Skipped node save because movement/yaw change was too small')
                return
            self._record_node(transform, now_sec)

    def _lookup_robot_transform(self):
        try:
            return self.tf_buffer.lookup_transform(
                self.global_frame,
                self.robot_frame,
                rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().debug(
                f'Could not transform {self.robot_frame} into {self.global_frame}: {ex}')
            return None

    def _transform_to_xy_yaw(self, transform):
        x = float(transform.transform.translation.x)
        y = float(transform.transform.translation.y)

        if self.assume_start_at_origin:
            if self._origin_xy is None:
                self._origin_xy = (x, y)
                self.get_logger().warn(
                    f'Recording origin locked to first TF pose: raw=({x:.3f},{y:.3f}) '
                    f'-> node 0 at (0.000,0.000)')
            x -= self._origin_xy[0]
            y -= self._origin_xy[1]

        qx = float(transform.transform.rotation.x)
        qy = float(transform.transform.rotation.y)
        qz = float(transform.transform.rotation.z)
        qw = float(transform.transform.rotation.w)
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz))

        return x, y, yaw

    def _should_record_node(self, transform):
        if not self.saved_nodes:
            return True

        x, y, yaw = self._transform_to_xy_yaw(transform)
        last = self.saved_nodes[-1]
        dx = x - float(last['x'])
        dy = y - float(last['y'])
        dist = math.hypot(dx, dy)
        yaw_delta = abs(math.atan2(
            math.sin(yaw - float(last['yaw'])),
            math.cos(yaw - float(last['yaw']))))
        return (
            dist >= self.min_node_spacing_m or
            yaw_delta >= self.min_node_yaw_change_rad
        )

    def _record_node(self, transform, now_sec):
        x, y, yaw = self._transform_to_xy_yaw(transform)

        stamp = self.get_clock().now().to_msg()

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.global_frame
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation = transform.transform.rotation
        self.path_msg.header.stamp = stamp
        self.path_msg.poses.append(pose)

        point = Point()
        point.x = x
        point.y = y
        self.line_marker_msg.header.stamp = stamp
        self.line_marker_msg.points.append(point)
        self.node_marker_msg.header.stamp = stamp
        self.node_marker_msg.points.append(point)

        node_index = len(self.saved_nodes)
        node_data = {
            'index': node_index,
            'node_id': node_index,
            'x': x,
            'y': y,
            'yaw': yaw,
            'stamp_sec': now_sec,
        }
        self.saved_nodes.append(node_data)
        self.last_record_time = now_sec

        self.path_pub.publish(self.path_msg)
        self.line_marker_pub.publish(self.line_marker_msg)
        self.node_marker_pub.publish(self.node_marker_msg)
        self._write_map_file()

        self.get_logger().info(
            f'saved node {node_index}: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f} '
            f'file={self.save_path.name}')

    def _write_map_file(self):
        payload = {
            'map_name': self.map_name,
            'frame_id': self.global_frame,
            'node_interval_sec': self.node_interval_sec,
            'saved_at': datetime.now().isoformat(),
            'node_count': len(self.saved_nodes),
            'nodes': self.saved_nodes,
        }
        self.save_path.write_text(json.dumps(payload, indent=2))


def main(args=None):
    rclpy.init(args=args)
    node = PathTeacher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
