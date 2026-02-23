#!/usr/bin/env python3
"""
dataset_collector.py

Subscribes to a ROS2 Image topic (default: /camera/csi_image) and saves frames
to /workspaces/isaac_ros-dev/ros2/src/qcar2_autonomy/autonomy/datasets.

Saves every N seconds (default: 1.0). Also optionally shows a debug window.
"""

import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class DatasetCollector(Node):
    def __init__(self):
        super().__init__("dataset_collector")

        # ---------- Params ----------
        self.declare_parameter("image_topic", "/camera/csi_image")
        self.declare_parameter("save_period_s", 0.1)
        self.declare_parameter(
            "dataset_dir",
            "/workspaces/isaac_ros-dev/ros2/src/qcar2_autonomy/autonomy/datasets",
        )
        self.declare_parameter("prefix", "frame")
        self.declare_parameter("image_ext", "jpg")  # jpg or png
        self.declare_parameter("jpeg_quality", 95)  # only used for jpg
        self.declare_parameter("show_window", False)
        self.declare_parameter("window_name", "dataset_collector")

        self.image_topic = self.get_parameter("image_topic").value
        self.save_period_s = float(self.get_parameter("save_period_s").value)
        self.dataset_dir = Path(self.get_parameter("dataset_dir").value)
        self.prefix = str(self.get_parameter("prefix").value)
        self.image_ext = str(self.get_parameter("image_ext").value).lower().strip(".")
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.show_window = bool(self.get_parameter("show_window").value)
        self.window_name = str(self.get_parameter("window_name").value)

        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        self.bridge = CvBridge()
        self.last_save_t = 0.0
        self.frame_idx = 0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(Image, self.image_topic, self.image_cb, qos)

        self.get_logger().info("Dataset Collector Initialized.")
        self.get_logger().info(f'Subscribed to: {self.image_topic}')
        self.get_logger().info(f"Saving to: {self.dataset_dir}")
        self.get_logger().info(f"Period: {self.save_period_s:.3f}s, ext: .{self.image_ext}")

        if self.show_window:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def image_cb(self, msg: Image):
        # Convert ROS Image -> OpenCV BGR
        try:
            # Your CSI topic is typically bgr8 already
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge convert failed: {e}")
            return

        if self.show_window:
            cv2.imshow(self.window_name, frame)
            cv2.waitKey(1)

        now = time.time()
        if (now - self.last_save_t) < self.save_period_s:
            return

        self.last_save_t = now

        # Timestamped filename
        stamp = msg.header.stamp
        # Works even if stamp is zero (still unique via time.time)
        ts = f"{stamp.sec:010d}_{stamp.nanosec:09d}"
        fname = f"{self.prefix}_{ts}_{self.frame_idx:06d}.{self.image_ext}"
        out_path = self.dataset_dir / fname

        try:
            if self.image_ext in ("jpg", "jpeg"):
                cv2.imwrite(
                    str(out_path),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
                )
            elif self.image_ext == "png":
                cv2.imwrite(str(out_path), frame)
            else:
                # Default fallback
                cv2.imwrite(str(out_path.with_suffix(".png")), frame)

            self.frame_idx += 1
        except Exception as e:
            self.get_logger().error(f"Failed to save image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = DatasetCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if bool(node.get_parameter("show_window").value):
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
