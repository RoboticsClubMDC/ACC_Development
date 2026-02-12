#!/usr/bin/env python3

import os
import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.exceptions import ParameterUninitializedException

from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge

from ultralytics import YOLO


class YoloDetector(Node):
    """
    Ultralytics YOLO detector for QCar2 (virtual-friendly, autonomy-only changes).

    Defaults (virtual):
      RGB:   /camera/color_image (bgr8)
      Depth: /camera/depth_image (mono16)

    Publishes:
      /qcar_camera/rgb_yolo  (bgr8 annotated for RViz2)
      /qcar_camera/rgb       (bgr8 passthrough)
      /qcar_camera/depth     (32FC1 meters)
      /motion_enable         (Bool)
    """

    def __init__(self):
        super().__init__("yolo_detector")

        # Make Ultralytics config/cache writable even in containers
        os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
        os.environ.setdefault("TORCH_HOME", "/tmp/torch")

        # Params
        self.declare_parameter("rgb_topic", "/camera/color_image")
        self.declare_parameter("depth_topic", "/camera/depth_image")

        self.declare_parameter("weights", "yolov8n.pt")
        self.declare_parameter("conf", 0.30)
        self.declare_parameter("max_rate_hz", 12.0)

        # Filtering: list of class IDs, or empty => no filter
        # Default is COCO: car=2, traffic light=9, stop sign=11
        self.declare_parameter("classes", [2, 9, 11])

        # Depth scale for mono16: usually millimeters -> meters
        self.declare_parameter("depth_scale", 0.001)

        # Motion gating (optional)
        self.declare_parameter("stop_conf", 0.90)
        self.declare_parameter("stop_distance_m", 0.60)
        self.declare_parameter("hold_stop_s", 3.0)
        self.declare_parameter("cooldown_s", 10.0)

        # QoS: your camera publisher is RELIABLE in your setup
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.bridge = CvBridge()

        # Publishers
        self.pub_ann = self.create_publisher(Image, "/qcar_camera/rgb_yolo", 10)
        self.pub_rgb = self.create_publisher(Image, "/qcar_camera/rgb", 10)
        self.pub_depth = self.create_publisher(Image, "/qcar_camera/depth", 10)
        self.pub_motion = self.create_publisher(Bool, "/motion_enable", 1)

        # Subscriptions
        rgb_topic = self.get_parameter("rgb_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        self.sub_rgb = self.create_subscription(Image, rgb_topic, self.on_rgb, qos)
        self.sub_depth = self.create_subscription(Image, depth_topic, self.on_depth, qos)

        # Model
        weights = self.get_parameter("weights").value
        self.get_logger().info(f"Loading Ultralytics model: {weights}")
        self.model = YOLO(weights)

        # State
        self.depth_m = None
        self.last_infer_t = 0.0
        self.motion_enable = True
        self.disable_until_t = 0.0
        self.last_stop_t = 0.0

        self._empty_rgb = 0
        self._empty_depth = 0
        self._infer_count = 0

        # Publish /motion_enable regularly
        self.motion_timer = self.create_timer(1.0 / 50.0, self.publish_motion)

        self.get_logger().info(f"Subscribed RGB:   {rgb_topic}")
        self.get_logger().info(f"Subscribed Depth: {depth_topic}")
        self.get_logger().info("Annotated topic for RViz2: /qcar_camera/rgb_yolo")

    def publish_motion(self):
        msg = Bool()
        msg.data = bool(self.motion_enable)
        self.pub_motion.publish(msg)

    # ---------- Decoders (no cv_bridge for incoming messages) ----------

    def _decode_bgr8(self, msg: Image):
        # Guard against empty/partial frames
        if msg.height == 0 or msg.width == 0 or msg.step == 0:
            self._empty_rgb += 1
            if self._empty_rgb % 30 == 0:
                self.get_logger().warn(f"Empty RGB frames so far: {self._empty_rgb} (waiting for camera...)")
            return None

        if msg.encoding.lower() != "bgr8":
            self.get_logger().warn(f"Unexpected RGB encoding: {msg.encoding} (expected bgr8)")
            return None

        if msg.step < msg.width * 3:
            self.get_logger().warn(f"RGB step too small ({msg.step} < {msg.width*3})")
            return None

        buf = np.frombuffer(msg.data, dtype=np.uint8)
        expected = msg.height * msg.step
        if buf.size < expected:
            self.get_logger().warn(f"RGB buffer too small: {buf.size} < {expected}")
            return None

        # Reshape using step (row stride), then crop to width
        img = buf[:expected].reshape((msg.height, msg.step // 3, 3))[:, :msg.width, :]
        return np.ascontiguousarray(img)

    def _decode_depth_mono16_to_meters(self, msg: Image):
        if msg.height == 0 or msg.width == 0 or msg.step == 0:
            self._empty_depth += 1
            if self._empty_depth % 30 == 0:
                self.get_logger().warn(f"Empty DEPTH frames so far: {self._empty_depth} (waiting for camera...)")
            return None

        if msg.encoding.lower() != "mono16":
            self.get_logger().warn(f"Unexpected depth encoding: {msg.encoding} (expected mono16)")
            return None

        if msg.step < msg.width * 2:
            self.get_logger().warn(f"Depth step too small ({msg.step} < {msg.width*2})")
            return None

        buf = np.frombuffer(msg.data, dtype=np.uint16)
        expected = msg.height * (msg.step // 2)
        if buf.size < expected:
            self.get_logger().warn(f"Depth buffer too small: {buf.size} < {expected}")
            return None

        depth16 = buf[:expected].reshape((msg.height, msg.step // 2))[:, :msg.width]
        scale = float(self.get_parameter("depth_scale").value)
        depth_m = depth16.astype(np.float32) * scale
        return np.ascontiguousarray(depth_m)

    # ---------- Callbacks ----------

    def on_depth(self, msg: Image):
        depth_m = self._decode_depth_mono16_to_meters(msg)
        if depth_m is None:
            return

        self.depth_m = depth_m

        # Republish standardized depth (meters) as 32FC1
        out = self.bridge.cv2_to_imgmsg(depth_m, encoding="32FC1")
        out.header = msg.header
        self.pub_depth.publish(out)

    def on_rgb(self, msg: Image):
        bgr = self._decode_bgr8(msg)
        if bgr is None:
            return

        # Republish raw RGB
        self.pub_rgb.publish(self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8"))

        # Rate limit inference
        max_rate = float(self.get_parameter("max_rate_hz").value)
        now = time.time()
        if max_rate > 0 and (now - self.last_infer_t) < (1.0 / max_rate):
            return
        self.last_infer_t = now

        conf = float(self.get_parameter("conf").value)

        # Handle "classes" parameter safely (classes:=[] can appear uninitialized)
        try:
            classes = self.get_parameter("classes").value
        except ParameterUninitializedException:
            classes = []

        # Ultralytics expects None for "no filter"
        cls_arg = classes if isinstance(classes, (list, tuple)) and len(classes) > 0 else None

        results = self.model.predict(source=bgr, conf=conf, classes=cls_arg, verbose=False)
        r0 = results[0]

        annotated = r0.plot()  # BGR uint8
        annotated = np.ascontiguousarray(annotated)

        # Make it obvious the pipeline is working
        cv2.putText(
            annotated, "YOLO RUNNING",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA
        )

        # Log detection count occasionally
        n = 0 if r0.boxes is None else len(r0.boxes)
        self._infer_count += 1
        if self._infer_count % 10 == 0:
            self.get_logger().info(f"YOLO detections: {n}")

        # (Optional) motion gating on stop sign near camera using depth
        # Note: this works only if COCO "stop sign" is detected (class 11) and depth is valid.
        stop_conf = float(self.get_parameter("stop_conf").value)
        stop_dist = float(self.get_parameter("stop_distance_m").value)
        hold_s = float(self.get_parameter("hold_stop_s").value)
        cooldown_s = float(self.get_parameter("cooldown_s").value)

        detected_stop = False
        best_d = None

        if r0.boxes is not None and len(r0.boxes) > 0 and self.depth_m is not None:
            h, w = self.depth_m.shape[:2]

            for b in r0.boxes:
                cls = int(b.cls[0].item())
                score = float(b.conf[0].item())
                name = self.model.names.get(cls, str(cls))

                if name != "stop sign" or score < stop_conf:
                    continue

                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                cx = int((x1 + x2) * 0.5)
                cy = int((y1 + y2) * 0.5)
                cx = max(0, min(w - 1, cx))
                cy = max(0, min(h - 1, cy))

                d = float(self.depth_m[cy, cx])
                if not np.isfinite(d) or d <= 0.0:
                    continue

                if d <= stop_dist:
                    detected_stop = True
                    best_d = d if best_d is None else min(best_d, d)

        if detected_stop and (now - self.last_stop_t) >= cooldown_s and now >= self.disable_until_t:
            self.last_stop_t = now
            self.disable_until_t = now + hold_s
            self.motion_enable = False
            self.get_logger().info(f"STOP SIGN detected at ~{best_d:.2f}m, disabling motion for {hold_s:.1f}s.")

        if now >= self.disable_until_t:
            self.motion_enable = True

        # Publish annotated image for RViz2
        ann_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        ann_msg.header = msg.header
        self.pub_ann.publish(ann_msg)


def main():
    rclpy.init()
    node = YoloDetector()
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


if __name__ == "__main__":
    main()
