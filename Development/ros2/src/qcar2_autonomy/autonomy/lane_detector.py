#!/usr/bin/env python3
"""
Lane Detector (CSI) - Yellow-only + Drift CTE Reference

Subscribes:
  /camera/csi_image   (sensor_msgs/Image)

Publishes:
  /vision/yellow_mask     (sensor_msgs/Image, mono8)
  /vision/lanes/cte       (std_msgs/Float64)   # meters, ZEROED to first valid frame
  /vision/yellow_debug    (sensor_msgs/Image, bgr8) overlay with dots + text

Logic:
  - Threshold "any yellow" in HSV (loose)
  - Compute centroid of the yellow mask
  - Dot A (red): camera center
  - Dot B (green): lane centroid
  - Compute raw CTE (meters) from centroid x-offset
  - On first valid frame: lock cte_ref
  - Publish cte_out = cte_raw - cte_ref  (drift-only CTE)
"""

import rclpy
from rclpy.node import Node

import cv2
import numpy as np
import time
import math

from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge


class LaneDetector(Node):
    def __init__(self):
        super().__init__('lane_detector')

        # -------------------- ROS Params --------------------
        self.declare_parameter('image_topic', '/camera/csi_image')

        # Yellow HSV (loose)
        self.declare_parameter('h_low', 15)
        self.declare_parameter('h_high', 50)
        self.declare_parameter('s_low', 40)
        self.declare_parameter('s_high', 255)
        self.declare_parameter('v_low', 40)
        self.declare_parameter('v_high', 255)

        # Morph cleanup (helps gaps + anti-aliasing)
        self.declare_parameter('use_morph', True)
        self.declare_parameter('kernel_size', 5)
        self.declare_parameter('close_iters', 2)
        self.declare_parameter('open_iters', 1)
        self.declare_parameter('dilate_iters', 1)

        # Centroid validity (avoid locking reference on noise)
        self.declare_parameter('min_mask_pixels', 300)

        # Convert pixels -> meters (CTE in meters)
        # Keep this consistent with your earlier scaling; tune later if needed.
        self.declare_parameter('xm_per_pix', 3.7 / 700.0)

        # Reference behavior
        self.declare_parameter('auto_zero_reference', True)   # lock first valid frame as reference
        self.declare_parameter('ref_reset_seconds', 0.0)      # 0 = never reset; >0 = periodic reset

        # Output topics
        self.declare_parameter('mask_topic', '/vision/yellow_mask')
        self.declare_parameter('cte_topic', '/vision/lanes/cte')
        self.declare_parameter('debug_topic', '/vision/yellow_debug')

        # -------------------- Internal State --------------------
        self.bridge = CvBridge()

        self._ref_set = False
        self._cte_ref_m = 0.0
        self._last_ref_time_s = 0.0

        # -------------------- ROS IO --------------------
        self.sub = self.create_subscription(
            Image,
            self.get_parameter('image_topic').value,
            self.on_image,
            10
        )

        self.pub_mask = self.create_publisher(Image, self.get_parameter('mask_topic').value, 1)
        self.pub_cte = self.create_publisher(Float64, self.get_parameter('cte_topic').value, 1)
        self.pub_dbg = self.create_publisher(Image, self.get_parameter('debug_topic').value, 1)

        self.get_logger().info(
            f"lane_detector up. sub={self.get_parameter('image_topic').value} "
            f"pub_cte={self.get_parameter('cte_topic').value} pub_dbg={self.get_parameter('debug_topic').value}"
        )

    # ------------------------------------------------------------
    # Yellow threshold
    # ------------------------------------------------------------
    def yellow_mask(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        lower = np.array([
            int(self.get_parameter('h_low').value),
            int(self.get_parameter('s_low').value),
            int(self.get_parameter('v_low').value)
        ], dtype=np.uint8)

        upper = np.array([
            int(self.get_parameter('h_high').value),
            int(self.get_parameter('s_high').value),
            int(self.get_parameter('v_high').value)
        ], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)

        if bool(self.get_parameter('use_morph').value):
            k = int(self.get_parameter('kernel_size').value)
            if k < 1:
                k = 1
            if k % 2 == 0:
                k += 1
            kernel = np.ones((k, k), np.uint8)

            close_iters = int(self.get_parameter('close_iters').value)
            open_iters = int(self.get_parameter('open_iters').value)
            dil_iters = int(self.get_parameter('dilate_iters').value)

            # Close gaps -> open noise -> optional thicken
            if close_iters > 0:
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iters)
            if open_iters > 0:
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iters)
            if dil_iters > 0:
                mask = cv2.dilate(mask, kernel, iterations=dil_iters)

        return mask

    # ------------------------------------------------------------
    # Centroid + CTE
    # ------------------------------------------------------------
    def centroid_cte_m(self, mask: np.ndarray, W: int, H: int):
        # Require enough yellow pixels so centroid is meaningful
        if int(np.count_nonzero(mask)) < int(self.get_parameter('min_mask_pixels').value):
            return None, None, None

        M = cv2.moments(mask, binaryImage=True)
        if M["m00"] <= 1e-6:
            return None, None, None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        x_center = W // 2
        cte_pix = float(cx - x_center)
        xm_per_pix = float(self.get_parameter('xm_per_pix').value)
        cte_m = cte_pix * xm_per_pix

        return cx, cy, cte_m

    # ------------------------------------------------------------
    # Main callback
    # ------------------------------------------------------------
    def on_image(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge decode failed: {e} | encoding={getattr(msg,'encoding','?')}")
            return

        if bgr is None:
            return

        H, W = bgr.shape[:2]

        mask = self.yellow_mask(bgr)

        # Publish mask (mono8)
        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        mask_msg.header = msg.header
        self.pub_mask.publish(mask_msg)

        # Compute centroid + raw CTE
        cx, cy, cte_raw_m = self.centroid_cte_m(mask, W, H)

        # Reference logic: lock first valid frame
        cte_out = float('nan')
        now_s = time.time()

        # Optional periodic reset
        reset_period = float(self.get_parameter('ref_reset_seconds').value)
        if reset_period > 0.0 and self._ref_set and ((now_s - self._last_ref_time_s) > reset_period):
            self._ref_set = False

        if cte_raw_m is not None:
            auto_zero = bool(self.get_parameter('auto_zero_reference').value)

            if auto_zero and not self._ref_set:
                self._cte_ref_m = float(cte_raw_m)
                self._ref_set = True
                self._last_ref_time_s = now_s
                self.get_logger().info(f"[lane] Reference locked: cte_ref_m={self._cte_ref_m:+.4f} m")

            if auto_zero and self._ref_set:
                # ✅ drift-only CTE (goal is zero)
                cte_out = float(cte_raw_m - self._cte_ref_m)
            else:
                # raw CTE if auto_zero_reference disabled
                cte_out = float(cte_raw_m)

        # Publish CTE
        cte_msg = Float64()
        cte_msg.data = float(cte_out)
        self.pub_cte.publish(cte_msg)

        # Debug overlay (dots + info)
        overlay = bgr.copy()

        # Dot A: camera center (red)
        cv2.circle(overlay, (W // 2, H // 2), 6, (0, 0, 255), -1)

        # Dot B: lane centroid (green)
        if cx is not None:
            cv2.circle(overlay, (cx, cy), 6, (0, 255, 0), -1)
            cv2.line(overlay, (W // 2, H // 2), (cx, cy), (255, 255, 255), 2)

        # Text
        if cte_raw_m is None:
            txt = "cte_raw=nan  cte_zeroed=nan  (no centroid)"
        else:
            txt = f"cte_raw={cte_raw_m:+.3f}m  ref={self._cte_ref_m:+.3f}m  cte_zeroed={cte_out:+.3f}m"

        cv2.putText(
            overlay, txt,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        dbg_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
        dbg_msg.header = msg.header
        self.pub_dbg.publish(dbg_msg)


def main():
    rclpy.init()
    node = LaneDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()