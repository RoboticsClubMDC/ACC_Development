#!/usr/bin/env python3

import os
import sys
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

if os.environ.get("QCAR2_FORCE_CPU", "").strip() in ("1", "true", "True"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.getlogin = lambda: os.environ.get("USER", "admin")


def add_mdc_paths():
    candidates = [
        "/workspaces/isaac_ros-dev/MDC_libraries/python",
        str(Path.home() / "Documents/ACC_Development/Development/MDC_libraries/python"),
        "/home/nvidia/Documents/ACC_Development_luigi/Development/MDC_libraries/python",
        "/home/nvidia/Documents/ACC_Development/Development/MDC_libraries/python",
    ]

    env_path = os.getenv("MDC_PYTHON_PATH", "").strip()
    if env_path:
        candidates = env_path.split(":") + candidates

    for path in candidates:
        if path and Path(path).exists() and path not in sys.path:
            sys.path.insert(0, path)


add_mdc_paths()

from pit.YOLO.utils import QCar2DepthAligned



#Node Creation to have an stable source of aligned RGB and Depth images from the D435 camera. 
#This node will be used by other perception nodes to get the aligned images and camera info.
class D435AlignedSource(Node):
    def __init__(self):
        super().__init__("d435_aligned_source")

        self.bridge = CvBridge()

        self.declare_parameter("is_physical", False)
        self.declare_parameter("distance_scale", 0.1)
        self.declare_parameter("publish_rate", 10.0)
        self.declare_parameter("frame_id", "aligned_camera_optical_frame")

        self.declare_parameter("fx", 455.20)
        self.declare_parameter("fy", 459.43)
        self.declare_parameter("cx", 308.53)
        self.declare_parameter("cy", 213.56)

        self.is_physical = bool(self.get_parameter("is_physical").value)
        self.distance_scale = float(self.get_parameter("distance_scale").value)
        self.publish_rate = float(self.get_parameter("publish_rate").value)
        self.frame_id = str(self.get_parameter("frame_id").value)

        self.fx = float(self.get_parameter("fx").value)
        self.fy = float(self.get_parameter("fy").value)
        self.cx = float(self.get_parameter("cx").value)
        self.cy = float(self.get_parameter("cy").value)

        self.rgb_pub = self.create_publisher(
            Image, "/perception/d435/rgb/image_raw", 10
        )
        self.depth_pub = self.create_publisher(
            Image, "/perception/d435/depth/image_rect", 10
        )
        self.info_pub = self.create_publisher(
            CameraInfo, "/perception/d435/camera_info", 10
        )

        self.get_logger().info("Initializing QCar2DepthAligned...")
        self.get_logger().info(
            f"is_physical={self.is_physical}, distance_scale={self.distance_scale}"
        )

        self.camera = QCar2DepthAligned(isPhyscial=self.is_physical)

        if not self.is_physical and hasattr(self.camera, "camera"):
            try:
                self.camera.camera.readMode = 0
            except Exception:
                pass

        period = 1.0 / max(self.publish_rate, 0.5)
        self.timer = self.create_timer(period, self.on_timer)

        self.frame_count = 0
        self.get_logger().info("D435 aligned source started.")


#intrinsic information helper based on D435 specifications, this will be used to publish the camera info message along with the aligned RGB and depth images.
    def make_camera_info(self, stamp, width, height):
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.width = int(width)
        msg.height = int(height)

        msg.k = [
            self.fx, 0.0, self.cx,
            0.0, self.fy, self.cy,
            0.0, 0.0, 1.0,
        ]

        msg.p = [
            self.fx, 0.0, self.cx, 0.0,
            0.0, self.fy, self.cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]

        msg.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]

        msg.distortion_model = "plumb_bob"
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        return msg
    
#timer for camera counting, reading and publishing the aligned RGB and depth images along with the camera info. This will be called at the rate specified by the publish_rate parameter.
    def on_timer(self):
        self.frame_count += 1

        try:
            new_frame = self.camera.read()
        except Exception as exc:
            self.get_logger().error(f"QCar2DepthAligned.read() failed: {exc}")
            return

        rgb = self.camera.rgb
        depth = self.camera.depth

        if rgb is None or depth is None:
            self.get_logger().warn("RGB or depth is None.", throttle_duration_sec=2.0)
            return

        rgb = np.asarray(rgb)
        depth = np.asarray(depth)

        if depth.ndim == 3 and depth.shape[2] == 1:
            depth = depth[:, :, 0]

        depth = depth.astype(np.float32, copy=False)

        if np.isfinite(depth).any() and np.nanmedian(depth) > 20.0:
            depth = depth / 1000.0

        depth = depth * self.distance_scale

        stamp = self.get_clock().now().to_msg()

        rgb_msg = self.bridge.cv2_to_imgmsg(rgb, encoding="bgr8")
        rgb_msg.header.stamp = stamp
        rgb_msg.header.frame_id = self.frame_id

        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = self.frame_id

        h, w = depth.shape[:2]
        info_msg = self.make_camera_info(stamp, w, h)

        self.rgb_pub.publish(rgb_msg)
        self.depth_pub.publish(depth_msg)
        self.info_pub.publish(info_msg)

        if self.frame_count % 30 == 1:
            self.get_logger().info(
                f"Published D435 aligned frames rgb={rgb.shape} depth={depth.shape} "
                f"depth_med={float(np.nanmedian(depth)):.3f} new_frame={new_frame}"
            )
    def destroy_node(self):
        try:
            self.get_logger().info("Terminating QCar2DepthAligned.")
            self.camera.terminate()
        except Exception as exc:
            self.get_logger().warn(f"Error terminating QCar2DepthAligned: {exc}")

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = D435AlignedSource()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
