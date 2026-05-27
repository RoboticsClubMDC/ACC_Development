#! /usr/bin/env python3
"""Publish the SDCS course PNG as a nav_msgs/OccupancyGrid so RViz can
render it anchored in the `map` frame, on top of cartographer's slam grid.

All geometry knobs (resolution, origin x/y, yaw) are exposed as ROS
parameters so you can nudge alignment from the launch file or from the
command line via `--ros-args -p`.
"""

import math
import os

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import OccupancyGrid, MapMetaData
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy


_FALLBACK_PNGS = (
    "/workspaces/isaac_ros-dev/backup/Quanser_Academic_Resources/"
    "6_teaching/4_Autonomous_Systems/SDCS/skills_activities/"
    "04-vehicle_control/cityscape.png",
    "/workspaces/isaac_ros-dev/Development/python_resources/roadmap/SDCS_MapLayout.png",
    "/workspaces/isaac_ros-dev/python_resources/roadmap/SDCS_MapLayout.png",
    "/workspaces/isaac_ros-dev/backup/Quanser_Academic_Resources/"
    "5_research/sdcs/roadmap/SDCS_MapLayout.png",
    "/home/arturo-noble/Downloads/cityscape.png",
    "/home/admin/Downloads/SDCS_MapLayout.png",
    "/home/admin/Downloads/cityscape.png",
)

_ROADMAP_SOURCE_SCALE_M_PER_PX = 0.002035
_REFERENCE_CITYSCAPE_WIDTH_PX = 2400.0
_REFERENCE_CITYSCAPE_HEIGHT_PX = 3049.0


def _yaw_to_quat(yaw):
    q = Quaternion()
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


class SdcsMapPublisher(Node):
    def __init__(self):
        super().__init__('sdcs_map_publisher')

        # Path + frame
        self.declare_parameter('image_path', '')
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('topic', '/sdcs_map_grid')

        # Geometry — defaults derived from the Quanser SDCSRoadMap scale.
        # origin_x/origin_y is where the BOTTOM-LEFT pixel of the image
        # lands in the map frame. Tune live with:
        #   ros2 param set /sdcs_map_publisher origin_x  -1.5
        # 0.0 means auto-scale from the known SDCS source image. The roadmap
        # itself is in meters using 0.002035 m/source-pixel.
        self.declare_parameter('resolution', 0.0)
        self.declare_parameter('origin_x', -2.308)
        self.declare_parameter('origin_y', -2.500)
        self.declare_parameter('origin_yaw', 0.0)
        self.declare_parameter('invert', False)
        self.declare_parameter('publish_rate_hz', 1.0)

        # Resolve image path
        path = self.get_parameter('image_path').value
        candidates = [path] if path else []
        candidates += list(_FALLBACK_PNGS)
        resolved = None
        for p in candidates:
            if p and os.path.isfile(p):
                resolved = p
                break
        if resolved is None:
            self.get_logger().error(
                'SDCS map PNG not found. Tried:\n  ' + '\n  '.join(p for p in candidates if p)
            )
            raise FileNotFoundError(path or '(none)')

        raw = cv2.imread(resolved, cv2.IMREAD_UNCHANGED)
        if raw is None:
            self.get_logger().error(f'cv2.imread returned None for {resolved}')
            raise RuntimeError(resolved)

        if raw.ndim == 3 and raw.shape[2] == 4:
            bgr = raw[:, :, :3].astype(np.float32)
            alpha = raw[:, :, 3:4].astype(np.float32) / 255.0
            white = np.full_like(bgr, 255.0)
            bgr = alpha * bgr + (1.0 - alpha) * white
            gray = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        elif raw.ndim == 3:
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        else:
            gray = raw

        invert = bool(self.get_parameter('invert').value)
        if invert:
            gray = 255 - gray

        # OccupancyGrid: bright PNG pixels -> free (0), dark -> occupied (100).
        # Use full 0..100 range so RViz "raw" color scheme shows shading.
        occ = (255 - gray.astype(np.int16)) * 100 // 255
        occ = np.clip(occ, 0, 100).astype(np.int8)

        # OccupancyGrid data is row-major from BOTTOM row up; OpenCV gives
        # top-down rows, so flip vertically.
        occ = np.flipud(occ)

        h, w = occ.shape
        self.get_logger().info(
            f'Loaded SDCS PNG: {resolved} ({w}x{h}) -> OccupancyGrid'
        )

        msg = OccupancyGrid()
        msg.header.frame_id = self.get_parameter('frame_id').value
        meta = MapMetaData()
        resolution = float(self.get_parameter('resolution').value)
        if resolution <= 0.0:
            resolution_x = (
                _REFERENCE_CITYSCAPE_WIDTH_PX * _ROADMAP_SOURCE_SCALE_M_PER_PX / w
            )
            resolution_y = (
                _REFERENCE_CITYSCAPE_HEIGHT_PX * _ROADMAP_SOURCE_SCALE_M_PER_PX / h
            )
            resolution = 0.5 * (resolution_x + resolution_y)
            self.get_logger().info(
                f'Auto SDCS map resolution={resolution:.6f} m/px '
                f'(x={resolution_x:.6f}, y={resolution_y:.6f})'
            )

        meta.resolution = resolution
        meta.width = w
        meta.height = h
        meta.origin.position.x = float(self.get_parameter('origin_x').value)
        meta.origin.position.y = float(self.get_parameter('origin_y').value)
        meta.origin.position.z = 0.0
        meta.origin.orientation = _yaw_to_quat(
            float(self.get_parameter('origin_yaw').value)
        )
        msg.info = meta
        msg.data = occ.flatten().tolist()
        self._msg = msg

        qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        topic = self.get_parameter('topic').value
        self.pub = self.create_publisher(OccupancyGrid, topic, qos)

        self._publish_once()
        rate = float(self.get_parameter('publish_rate_hz').value)
        self.timer = self.create_timer(1.0 / max(rate, 0.1), self._publish_once)

        self.add_on_set_parameters_callback(self._on_params)

        self.get_logger().info(
            f"Publishing {w}x{h} grid on '{topic}' frame='{msg.header.frame_id}' "
            f"resolution={meta.resolution} origin=({meta.origin.position.x:.3f},"
            f"{meta.origin.position.y:.3f}) — tune via "
            f"`ros2 param set {self.get_name()} origin_x|origin_y|origin_yaw|resolution ...`"
        )

    def _publish_once(self):
        self._msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self._msg)

    def _on_params(self, params):
        # Allow live tweaking of geometry without restarting the node.
        changed = False
        for p in params:
            if p.name == 'resolution':
                self._msg.info.resolution = float(p.value)
                changed = True
            elif p.name == 'origin_x':
                self._msg.info.origin.position.x = float(p.value)
                changed = True
            elif p.name == 'origin_y':
                self._msg.info.origin.position.y = float(p.value)
                changed = True
            elif p.name == 'origin_yaw':
                self._msg.info.origin.orientation = _yaw_to_quat(float(p.value))
                changed = True
        if changed:
            self._publish_once()
        from rcl_interfaces.msg import SetParametersResult
        return SetParametersResult(successful=True)


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
