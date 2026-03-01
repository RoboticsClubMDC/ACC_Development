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
LANE = 0

# Overlay colors (BGR)
COLORS_BGR = {
    LANE: (0, 255, 0),     # green
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


class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__("lane_detection")

        # ROS params
        self.declare_parameter("image_topic", "/camera/color_image")
        self.declare_parameter("model_path", "ros2/src/qcar2_autonomy/models/lane_seg_yolo.pt")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("device", 0)
        
        self.declare_parameter("lane_seed_x_frac", 0.55)
        self.last_lane_cx = None
        self.last_lane_seen = 0
        self.declare_parameter("lane_lost_frames", 10)  # how long we keep last choice if it vanishes

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
        self.pub_overlay       = self.create_publisher(Image,  "/lane_detection/overlay",       qos_profile_sensor_data)
        self.pub_lane_selected = self.create_publisher(Image,  "/lane_detection/lane_selected", qos_profile_sensor_data)
        self.pub_debug         = self.create_publisher(String, "/lane_detection/debug_detections", 10)

        self.imgsz = imgsz
        self.device = device

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
        union = {
            LANE: np.zeros((h, w), dtype=bool)}

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

        # -------- Lane selection with hysteresis (choose one lane and stick to it) --------
        lane_bool = union[LANE]
        lane_sel = np.zeros((h, w), dtype=bool)

        if lane_bool.any():
            lane_u8 = lane_bool.astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(lane_u8)

            comps = []
            for lab in range(1, num_labels):
                ys, xs = np.where(labels == lab)
                if xs.size < 200:  # ignore tiny blobs
                    continue
                cx = float(xs.mean())
                area = float(xs.size)
                comps.append((lab, cx, area))

            if comps:
                seed_x = float(self.get_parameter("lane_seed_x_frac").value) * w
                target_cx = self.last_lane_cx if self.last_lane_cx is not None else seed_x

                # closest to last/seed, tie-break by larger area
                comps.sort(key=lambda t: (abs(t[1] - target_cx), -t[2]))
                best_lab, best_cx, _ = comps[0]

                lane_sel = (labels == best_lab)
                self.last_lane_cx = best_cx
                self.last_lane_seen = 0
            else:
                self.last_lane_seen += 1
        else:
            self.last_lane_seen += 1

        # Unlock if lane missing too long
        if self.last_lane_seen > int(self.get_parameter("lane_lost_frames").value):
            self.last_lane_cx = None
            self.last_lane_seen = 0
        # -------------------------------------------------------------------------------

        # Overlay (lane_detection shows ONLY the selected lane)
        out = overlay_mask(img_bgr.copy(), lane_sel, COLORS_BGR[LANE], ALPHA)

        overlay_msg = self.bridge.cv2_to_imgmsg(out, encoding="bgr8")
        overlay_msg.header = msg.header
        self.pub_overlay.publish(overlay_msg)

        # Publish selected lane mask (mono8)
        lane_sel_u8 = (lane_sel.astype(np.uint8) * 255)
        lane_sel_msg = self.bridge.cv2_to_imgmsg(lane_sel_u8, encoding="mono8")
        lane_sel_msg.header = msg.header
        self.pub_lane_selected.publish(lane_sel_msg)

def main():
    rclpy.init()
    node = LaneDetectionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()