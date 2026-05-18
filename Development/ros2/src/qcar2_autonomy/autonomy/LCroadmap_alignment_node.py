#!/usr/bin/env python3

import math
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformListener, TransformBroadcaster


class RoadmapAlignmentNode(Node):
    """
    Publishes a TF transform between Cartographer's live map frame and
    the official SDCS roadmap frame.

    It assumes the QCar is physically placed at the start_node when this
    node first aligns.

    TF published:
        map -> roadmap

    Meaning:
        Official SDCS roadmap coordinates can be transformed into the
        Cartographer map frame.
    """

    def __init__(self):
        super().__init__('roadmap_alignment_node')

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('roadmap_frame', 'roadmap')
        self.declare_parameter('start_node', 10)

        # Extra corrections for physical QCar later
        self.declare_parameter('extra_x_offset', 0.0)
        self.declare_parameter('extra_y_offset', 0.0)
        self.declare_parameter('extra_yaw_offset_deg', 0.0)

        # If true, compute once and keep broadcasting that calibration
        self.declare_parameter('align_once', True)

        self.map_frame = self.get_parameter('map_frame').value
        self.robot_frame = self.get_parameter('robot_frame').value
        self.roadmap_frame = self.get_parameter('roadmap_frame').value
        self.start_node = int(self.get_parameter('start_node').value)

        self.extra_x_offset = float(self.get_parameter('extra_x_offset').value)
        self.extra_y_offset = float(self.get_parameter('extra_y_offset').value)
        self.extra_yaw_offset_deg = float(self.get_parameter('extra_yaw_offset_deg').value)
        self.align_once = bool(self.get_parameter('align_once').value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.aligned = False
        self.transform_msg = None

        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info(
            f"Roadmap alignment node started. start_node={self.start_node}, "
            f"{self.map_frame}->{self.roadmap_frame}, robot_frame={self.robot_frame}"
        )

    def get_sdcs_node_pose(self, node_id: int):
        """
        Returns node pose [x, y, theta] in SDCS roadmap metric coordinates.

        This mirrors the right-side traffic node definitions from mats.py.
        """

        scale = 0.002035
        x_offset = 1134
        y_offset = 2363

        pi = math.pi
        half_pi = pi / 2.0

        # Right-side traffic nodePoses from mats.py
        node_poses_px = [
            [1134, 2299, -half_pi],
            [1266, 2323, half_pi],
            [1688, 2896, 0.0],
            [1688, 2763, pi],
            [2242, 2323, half_pi],
            [2109, 2323, -half_pi],
            [1632, 1822, pi],
            [1741, 1955, 0.0],
            [766, 1822, pi],
            [766, 1955, 0.0],
            [504, 2589, -42.0 * pi / 180.0],
            [1134, 1300, -half_pi],
            [1134, 1454, -half_pi],
            [1266, 1454, half_pi],
            [2242, 905, half_pi],
            [2109, 1454, -half_pi],
            [1580, 540, -80.6 * pi / 180.0],
            [1854.4, 814.5, -9.4 * pi / 180.0],
            [1440, 856, -138.0 * pi / 180.0],
            [1523, 958, 42.0 * pi / 180.0],
            [1134, 153, pi],
            [1134, 286, 0.0],
            [159, 905, -half_pi],
            [291, 905, half_pi],
        ]

        if node_id < 0 or node_id >= len(node_poses_px):
            raise ValueError(f"Invalid SDCS node id: {node_id}")

        x_px, y_px, theta = node_poses_px[node_id]

        x = scale * (x_px - x_offset)
        y = scale * (y_offset - y_px)

        return np.array([x, y]), float(theta)

    @staticmethod
    def yaw_from_quaternion(q):
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    @staticmethod
    def quaternion_from_yaw(yaw):
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        return 0.0, 0.0, qz, qw

    def compute_alignment(self):
        tf_msg = self.tf_buffer.lookup_transform(
            self.map_frame,
            self.robot_frame,
            rclpy.time.Time()
        )

        p_map = np.array([
            tf_msg.transform.translation.x,
            tf_msg.transform.translation.y
        ])

        theta_map = self.yaw_from_quaternion(tf_msg.transform.rotation)

        p_node, theta_node = self.get_sdcs_node_pose(self.start_node)

        extra_yaw = math.radians(self.extra_yaw_offset_deg)

        # Rotation from roadmap frame into Cartographer map frame
        yaw_map_from_roadmap = theta_map - theta_node + extra_yaw

        c = math.cos(yaw_map_from_roadmap)
        s = math.sin(yaw_map_from_roadmap)
        R = np.array([
            [c, -s],
            [s,  c]
        ])

        # p_map = R @ p_roadmap + t
        t = p_map - R @ p_node

        # Physical/manual correction in map frame
        t += np.array([self.extra_x_offset, self.extra_y_offset])

        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.child_frame_id = self.roadmap_frame

        msg.transform.translation.x = float(t[0])
        msg.transform.translation.y = float(t[1])
        msg.transform.translation.z = 0.0

        qx, qy, qz, qw = self.quaternion_from_yaw(yaw_map_from_roadmap)
        msg.transform.rotation.x = qx
        msg.transform.rotation.y = qy
        msg.transform.rotation.z = qz
        msg.transform.rotation.w = qw

        self.transform_msg = msg
        self.aligned = True

        self.get_logger().info(
            f"Aligned roadmap using node {self.start_node}: "
            f"node_xy={p_node}, node_yaw_deg={math.degrees(theta_node):.2f}, "
            f"map_xy={p_map}, map_yaw_deg={math.degrees(theta_map):.2f}, "
            f"tf_yaw_deg={math.degrees(yaw_map_from_roadmap):.2f}, "
            f"tf_translation=[{t[0]:.3f}, {t[1]:.3f}]"
        )

    def timer_callback(self):
        if not self.aligned or not self.align_once:
            try:
                self.compute_alignment()
            except Exception as e:
                self.get_logger().warn(
                    f"Waiting for TF {self.map_frame}->{self.robot_frame}: {e}"
                )
                return

        if self.transform_msg is not None:
            self.transform_msg.header.stamp = self.get_clock().now().to_msg()
            self.tf_broadcaster.sendTransform(self.transform_msg)


def main(args=None):
    rclpy.init(args=args)
    node = RoadmapAlignmentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()