#!/usr/bin/env python3

import os
import cv2
import numpy as np

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from ultralytics import YOLO

from autonomy.cuda_utils import (
    clear_cuda_cache,
    is_cuda_runtime_error,
    select_yolo_device,
)


LANE = 0
COLORS_BGR = {LANE: (0, 255, 0)}
ALPHA = 0.45


def overlay_mask(img_bgr, mask_bool, color_bgr, alpha):
    if mask_bool is None or mask_bool.sum() == 0:
        return img_bgr
    overlay = img_bgr.copy()
    overlay[mask_bool] = color_bgr
    return cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)


class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__("lane_detection")

        self.declare_parameter("image_topic",      "/camera/csi_image")
        self.declare_parameter("model_path",       "ros2/src/qcar2_autonomy/models/lane_seg_yolo.pt")
        self.declare_parameter("imgsz",            640)
        self.declare_parameter("device",           0)
        self.declare_parameter("use_cuda",         True)
        self.declare_parameter("allow_cpu_fallback", False)
        self.declare_parameter("lane_seed_x_frac", 0.55)
        self.declare_parameter("lane_lost_frames", 10)

        image_topic      = self.get_parameter("image_topic").get_parameter_value().string_value
        model_path_param = self.get_parameter("model_path").get_parameter_value().string_value
        imgsz            = int(self.get_parameter("imgsz").get_parameter_value().integer_value)
        device           = int(self.get_parameter("device").get_parameter_value().integer_value)
        use_cuda         = bool(self.get_parameter("use_cuda").get_parameter_value().bool_value)
        self.allow_cpu_fallback = bool(
            self.get_parameter("allow_cpu_fallback").get_parameter_value().bool_value)

        model_path = self._resolve_model_path(model_path_param)

        self.get_logger().info(f"Subscribing to: {image_topic}")
        self.get_logger().info(f"Using model: {model_path}")

        self.bridge = CvBridge()
        self.model  = YOLO(model_path)

        self.last_lane_cx   = None
        self.last_lane_seen = 0
        self.imgsz  = imgsz
        self.device = select_yolo_device(
            self.get_logger(),
            requested_device=device,
            use_cuda=use_cuda,
            context="Lane YOLO",
            allow_cpu_fallback=self.allow_cpu_fallback,
        )
        self.get_logger().info(f"YOLO inference device: {self.device}")

        self.pub_overlay       = self.create_publisher(Image,  "/lane_detection/overlay",          qos_profile_sensor_data)
        self.pub_lane_selected = self.create_publisher(Image,  "/lane_detection/lane_selected",    qos_profile_sensor_data)
        self.pub_debug         = self.create_publisher(String, "/lane_detection/debug_detections", 10)

        self.sub = self.create_subscription(Image, image_topic, self.image_cb, qos_profile_sensor_data)

    def _resolve_model_path(self, model_path_param):
        if os.path.isabs(model_path_param):
            return model_path_param
        for p in os.environ.get("AMENT_PREFIX_PATH", "").split(":"):
            cand = os.path.join(p, "share", model_path_param)
            if os.path.exists(cand):
                return cand
        return os.path.abspath(model_path_param)

    def _predict(self, img_bgr):
        try:
            return self.model.predict(
                img_bgr,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )[0]
        except RuntimeError as e:
            if self.device == "cpu" or not is_cuda_runtime_error(e):
                raise
            if not self.allow_cpu_fallback:
                self.get_logger().error(
                    f"YOLO CUDA inference failed on device={self.device}: {e}. "
                    "CPU fallback is disabled for lane detection.")
                raise

            self.get_logger().error(
                f"YOLO CUDA inference failed on device={self.device}: {e}. "
                "Falling back to CPU for lane detection.")
            self.device = "cpu"
            clear_cuda_cache()
            return self.model.predict(
                img_bgr,
                imgsz=self.imgsz,
                device="cpu",
                verbose=False,
            )[0]

    def image_cb(self, msg: Image):
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge failed: encoding={msg.encoding} error={e}")
            return

        if img_bgr is None:
            return

        h, w = img_bgr.shape[:2]

        res = self._predict(img_bgr)

        union     = {LANE: np.zeros((h, w), dtype=bool)}
        dbg_lines = []

        if res.masks is not None and res.boxes is not None and len(res.boxes) == len(res.masks.data):
            masks = res.masks.data.cpu().numpy()
            clss  = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()

            for i, (m, c) in enumerate(zip(masks, clss)):
                if c not in union:
                    continue
                m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                union[c] |= m_resized
                dbg_lines.append(f"i={i} class={c} conf={float(confs[i]):.3f} area_px={int(m_resized.sum())}")

        if not dbg_lines:
            dbg_lines = ["no detections"]

        dbg      = String()
        dbg.data = " | ".join(dbg_lines)
        self.pub_debug.publish(dbg)

        lane_bool = union[LANE]
        lane_sel  = np.zeros((h, w), dtype=bool)

        if lane_bool.any():
            num_labels, labels = cv2.connectedComponents(lane_bool.astype(np.uint8))
            comps = []
            for lab in range(1, num_labels):
                ys, xs = np.where(labels == lab)
                if xs.size < 200:
                    continue
                comps.append((lab, float(xs.mean()), float(xs.size)))

            if comps:
                seed_x    = float(self.get_parameter("lane_seed_x_frac").value) * w
                target_cx = self.last_lane_cx if self.last_lane_cx is not None else seed_x
                comps.sort(key=lambda t: (abs(t[1] - target_cx), -t[2]))
                best_lab, best_cx, _ = comps[0]
                lane_sel            = (labels == best_lab)
                self.last_lane_cx   = best_cx
                self.last_lane_seen = 0
            else:
                self.last_lane_seen += 1
        else:
            self.last_lane_seen += 1

        if self.last_lane_seen > int(self.get_parameter("lane_lost_frames").value):
            self.last_lane_cx   = None
            self.last_lane_seen = 0

        out = overlay_mask(img_bgr.copy(), lane_sel, COLORS_BGR[LANE], ALPHA)
        overlay_msg = self.bridge.cv2_to_imgmsg(out, encoding="bgr8")
        overlay_msg.header = msg.header
        self.pub_overlay.publish(overlay_msg)

        lane_sel_msg = self.bridge.cv2_to_imgmsg(lane_sel.astype(np.uint8) * 255, encoding="mono8")
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
