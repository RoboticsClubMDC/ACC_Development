#! /usr/bin/env python3

import os

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image


_DEFAULT_PNG = (
    "/workspaces/isaac_ros-dev/backup/Quanser_Academic_Resources/"
    "6_teaching/4_Autonomous_Systems/SDCS/skills_activities/"
    "04-vehicle_control/cityscape.png"
)
_FALLBACK_PNGS = (
    # In-container repo mounts
    "/workspaces/isaac_ros-dev/Development/python_resources/roadmap/SDCS_MapLayout.png",
    "/workspaces/isaac_ros-dev/python_resources/roadmap/SDCS_MapLayout.png",
    "/workspaces/isaac_ros-dev/backup/Quanser_Academic_Resources/"
    "5_research/sdcs/roadmap/SDCS_MapLayout.png",
    # Host paths (visible if the container bind-mounts /home)
    "/home/admin/Downloads/SDCS_MapLayout.png",
    "/home/admin/Downloads/cityscape.png",
    # Relative paths from where launch is invoked
    "./python_resources/roadmap/SDCS_MapLayout.png",
    "./Development/python_resources/roadmap/SDCS_MapLayout.png",
    "./backup/Quanser_Academic_Resources/"
    "6_teaching/4_Autonomous_Systems/SDCS/skills_activities/"
    "04-vehicle_control/cityscape.png",
)


class SdcsMapPublisher(Node):
    def __init__(self):
        super().__init__('sdcs_map_publisher')

        self.declare_parameter('image_path', _DEFAULT_PNG)
        self.declare_parameter('topic', '/sdcs_map_image')
        self.declare_parameter('publish_rate_hz', 1.0)

        path = self.get_parameter('image_path').value
        topic = self.get_parameter('topic').value
        rate = float(self.get_parameter('publish_rate_hz').value)

        resolved = self._resolve_path(path)
        if resolved is None:
            self.get_logger().error(
                f'SDCS map image not found. Tried: {path}, and fallbacks {_FALLBACK_PNGS}'
            )
            raise FileNotFoundError(path)

        bgr = cv2.imread(resolved, cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().error(f'cv2.imread returned None for {resolved}')
            raise RuntimeError(f'failed to load {resolved}')

        self.get_logger().info(
            f'Loaded SDCS map: {resolved} ({bgr.shape[1]}x{bgr.shape[0]})'
        )

        self.bridge = CvBridge()
        self._msg = self.bridge.cv2_to_imgmsg(bgr, 'bgr8')
        self._msg.header.frame_id = 'map'

        qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub = self.create_publisher(Image, topic, qos)

        # Publish once immediately (transient local makes late subscribers get it)
        # and then on a slow timer in case any subscriber wants a refresh.
        self._publish_once()
        self.timer = self.create_timer(1.0 / max(rate, 0.1), self._publish_once)

    def _resolve_path(self, path):
        candidates = [path] + list(_FALLBACK_PNGS)
        for p in candidates:
            if p and os.path.isfile(p):
                return p
        return None

    def _publish_once(self):
        self._msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self._msg)


def main():
    rclpy.init()
    try:
        node = SdcsMapPublisher()
    except (FileNotFoundError, RuntimeError):
        rclpy.shutdown()
        return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
