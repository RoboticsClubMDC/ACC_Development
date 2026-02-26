#!/usr/bin/env python3
import os
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from ultralytics import YOLO


# Class IDs (must match your training YAML)
NO_GO      = 2

# Overlay colors (BGR)
COLORS_BGR = {
    NO_GO:      (0, 0, 255),     # red
}

ALPHA = 0.45
NO_GO_MARGIN_PX = 10


def overlay_mask(img_bgr: np.ndarray, mask_bool: np.ndarray, color_bgr, alpha: float) -> np.ndarray:
    """Overlay a boolean mask onto an image with alpha blending."""
    if mask_bool is None or mask_bool.sum() == 0:
        return img_bgr
    overlay = img_bgr.copy()
    overlay[mask_bool] = color_bgr
    return cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)


class SidewalkDetectionNode(Node):
    def __init__(self):
        super().__init__("sidewalk_detection")

        # ROS params
        self.declare_parameter("image_topic", "/camera/color_image")
        self.declare_parameter("model_path", "ros2/src/qcar2_autonomy/models/sidewalk_seg_yolo.pt")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("device", 0)

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        model_path_param = self.get_parameter("model_path").get_parameter_value().string_value
        imgsz = int(self.get_parameter("imgsz").get_parameter_value().integer_value)
        device = int(self.get_parameter("device").get_parameter_value().integer_value)

        model_path = self._resolve_model_path(model_path_param)

        self.get_logger().info(f"Subscribing to: {image_topic}")
        self.get_logger().info(f"Using model: {model_path}")
        self.get_logger().info(f"imgsz: {imgsz}, device: {device}")

        self.bridge = CvBridge()
        self.model = YOLO(model_path)

        # Publishers
        self.pub_overlay   = self.create_publisher(Image,  "/sidewalk_detection/overlay", qos_profile_sensor_data)
        self.pub_no_go_mgn = self.create_publisher(Image,  "/sidewalk_detection/no_go_margin", qos_profile_sensor_data)
        self.pub_debug     = self.create_publisher(String, "/sidewalk_detection/debug_detections", 10)

        self.imgsz = imgsz
        self.device = device

        # Precompute dilation kernel
        k = 2 * NO_GO_MARGIN_PX + 1
        self.no_go_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        # Subscriber
        self.sub = self.create_subscription(Image, image_topic, self.image_cb, qos_profile_sensor_data)

    def _resolve_model_path(self, model_path_param: str) -> str:
        """Resolve model path for both source-tree and installed layouts."""
        if os.path.isabs(model_path_param):
            return model_path_param

        # Try install space: <prefix>/share/<relative>
        prefix_paths = os.environ.get("AMENT_PREFIX_PATH", "").split(":")
        for p in prefix_paths:
            cand = os.path.join(p, "share", model_path_param)
            if os.path.exists(cand):
                return cand

        # Fallback: relative to CWD
        return os.path.abspath(model_path_param)

    def image_cb(self, msg: Image):
        # ROS Image -> OpenCV BGR
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge failed: encoding={msg.encoding} error={e}")
            return

        if img_bgr is None:
            return

        h, w = img_bgr.shape[:2]

        # Run YOLO segmentation
        res = self.model.predict(
            img_bgr,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )[0]

        # Union masks per class
        union = {NO_GO: np.zeros((h, w), dtype=bool)}

        dbg_lines = []

        # Fill union masks from YOLO output
        if res.masks is not None and res.boxes is not None and len(res.boxes) == len(res.masks.data):
            masks = res.masks.data.cpu().numpy()            # (N, mh, mw)
            clss  = res.boxes.cls.cpu().numpy().astype(int) # (N,)
            confs = res.boxes.conf.cpu().numpy()            # (N,)

            for i, (m, c) in enumerate(zip(masks, clss)):
                if c not in union:
                    continue

                m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                union[c] |= m_resized

                area_px = int(m_resized.sum())
                conf = float(confs[i])
                dbg_lines.append(f"i={i} class={c} conf={conf:.3f} area_px={area_px}")

        if not dbg_lines:
            dbg_lines = ["no detections"]

        dbg = String()
        dbg.data = " | ".join(dbg_lines)
        self.pub_debug.publish(dbg)

        # Overlay (draw sidewalk last)
        out = overlay_mask(img_bgr.copy(), union[NO_GO], COLORS_BGR[NO_GO], ALPHA)

        overlay_msg = self.bridge.cv2_to_imgmsg(out, encoding="bgr8")
        overlay_msg.header = msg.header
        self.pub_overlay.publish(overlay_msg)

        # Publish sidewalk mask + margin (mono8)
        no_go_u8 = (union[NO_GO].astype(np.uint8) * 255)
        no_go_margin_u8 = cv2.dilate(no_go_u8, self.no_go_kernel, iterations=1)
        margin_msg = self.bridge.cv2_to_imgmsg(no_go_margin_u8, encoding="mono8")
        margin_msg.header = msg.header
        self.pub_no_go_mgn.publish(margin_msg)

def main():
    rclpy.init()
    node = SidewalkDetectionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()