#!/usr/bin/env python3
import os
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ultralytics import YOLO


# Class IDs (must match your training YAML)
MUST_DRIVE = 0
CAN_DRIVE  = 1
NO_GO      = 2

# BGR colors for overlay
COLORS_BGR = {
    MUST_DRIVE: (0, 255, 0),     # green
    CAN_DRIVE:  (0, 255, 255),   # yellow (unused for now)
    NO_GO:      (0, 0, 255),     # red
}

ALPHA = 0.45  # overlay transparency

NO_GO_MARGIN_PX = 10  # safety buffer in pixels

def overlay_mask(img_bgr: np.ndarray, mask_bool: np.ndarray, color_bgr, alpha: float) -> np.ndarray:
    """Overlay a boolean mask onto an image with alpha blending."""
    if mask_bool is None or mask_bool.sum() == 0:
        return img_bgr
    overlay = img_bgr.copy()
    overlay[mask_bool] = color_bgr
    return cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)


class LaneSegNode(Node):
    def __init__(self):
        super().__init__("lane_seg_node")

        # Parameters (can be set from launch)
        self.declare_parameter("image_topic", "/camera/color_image")
        self.declare_parameter("model_path", "ros2/src/qcar2_autonomy/models/lane_seg_yolo.pt")
        self.declare_parameter("imgsz", 640)

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        model_path_param = self.get_parameter("model_path").get_parameter_value().string_value
        self.imgsz = int(self.get_parameter("imgsz").get_parameter_value().integer_value)

        # Resolve model path:
        # - if absolute, use it
        # - else resolve relative to package install share directory via AMENT_PREFIX_PATH
        model_path = model_path_param
        if not os.path.isabs(model_path):
            # Try to find installed package share folder
            # Typical install: <prefix>/share/qcar2_autonomy/models/...
            prefix_paths = os.environ.get("AMENT_PREFIX_PATH", "").split(":")
            found = None
            for p in prefix_paths:
                cand = os.path.join(p, "share", model_path)
                if os.path.exists(cand):
                    found = cand
                    break
            if found:
                model_path = found
            else:
                # fallback: relative to current working dir
                model_path = os.path.abspath(model_path)

        self.get_logger().info(f"Subscribing to: {image_topic}")
        self.get_logger().info(f"Using model: {model_path}")
        self.get_logger().info(f"imgsz: {self.imgsz}")

        self.bridge = CvBridge()
        self.model = YOLO(model_path)

        self.pub_overlay = self.create_publisher(Image, "/lane_seg/overlay", qos_profile_sensor_data)
        self.pub_no_go   = self.create_publisher(Image, "/lane_seg/no_go_mask", qos_profile_sensor_data)
        self.pub_no_go_mgn = self.create_publisher(Image, "/lane_seg/no_go_margin", qos_profile_sensor_data)  # NEW
        
        self.sub = self.create_subscription(
            Image,
            image_topic,
            self.image_cb,
            qos_profile_sensor_data
        )

    def image_cb(self, msg: Image):
        # ROS Image -> OpenCV BGR
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge failed: encoding={msg.encoding} error={e}")
            return

        if img_bgr is None:
            self.get_logger().warn(f"Received empty image (encoding={msg.encoding}), skipping frame")
            return

        h, w = img_bgr.shape[:2]

        # Run segmentation (single frame)
        res = self.model.predict(img_bgr, imgsz=self.imgsz, device=0, verbose=False)[0]

        # Build union masks per class in ORIGINAL image size
        union = {
            MUST_DRIVE: np.zeros((h, w), dtype=bool),
            CAN_DRIVE:  np.zeros((h, w), dtype=bool),
            NO_GO:      np.zeros((h, w), dtype=bool),
        }

        if res.masks is not None and res.boxes is not None and len(res.boxes) == len(res.masks.data):
            masks = res.masks.data.cpu().numpy()            # (N, mh, mw)
            clss  = res.boxes.cls.cpu().numpy().astype(int) # (N,)
            for m, c in zip(masks, clss):
                if c not in union:
                    continue
                # Resize inference mask -> original frame size
                m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                union[c] |= m_resized

        # Create overlay (no_go drawn last so it's always visible)
        out = img_bgr.copy()
        for c in [CAN_DRIVE, MUST_DRIVE, NO_GO]:
            out = overlay_mask(out, union[c], COLORS_BGR[c], ALPHA)

        # Publish overlay
        overlay_msg = self.bridge.cv2_to_imgmsg(out, encoding="bgr8")
        overlay_msg.header = msg.header
        self.pub_overlay.publish(overlay_msg)

        # Publish no_go mask (mono8 0/255)
        no_go_u8 = (union[NO_GO].astype(np.uint8) * 255)
        mask_msg = self.bridge.cv2_to_imgmsg(no_go_u8, encoding="mono8")
        mask_msg.header = msg.header
        self.pub_no_go.publish(mask_msg)

        # NEW: publish dilated "no_go margin" mask (mono8 0/255)
        # (10 px means a kernel of size (2*10+1) = 21, centered; ellipse is usually the cleanest buffer.)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * NO_GO_MARGIN_PX + 1, 2 * NO_GO_MARGIN_PX + 1)
        )
        no_go_margin_u8 = cv2.dilate(no_go_u8, kernel, iterations=1)

        margin_msg = self.bridge.cv2_to_imgmsg(no_go_margin_u8, encoding="mono8")
        margin_msg.header = msg.header
        self.pub_no_go_mgn.publish(margin_msg)


def main():
    rclpy.init()
    node = LaneSegNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()