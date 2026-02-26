#!/usr/bin/env python3
"""
Lane Detector (CSI)
- Subscribes: /camera/csi_image  (sensor_msgs/Image)
- Detects ONLY the specific yellow lane color (HSV threshold tuned to your screenshot)
- Publishes:
    /vision/lanes/binary        sensor_msgs/Image (mono8)
    /vision/lanes/bev_binary    sensor_msgs/Image (mono8)
    /vision/lanes/cte           std_msgs/Float64  (meters)
    /vision/lanes/heading_error std_msgs/Float64  (radians)

Pipeline:
  BGR -> HSV yellow mask -> morph clean -> BEV warp -> sliding windows -> poly fit -> CTE + heading
"""

import rclpy
from rclpy.node import Node

import numpy as np
import cv2

from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge


class LaneDetector(Node):
    def __init__(self):
        super().__init__('lane_detector')

        # ===================== Params =====================
        self.declare_parameter('image_topic', '/camera/csi_image')
        self.declare_parameter('frame_width', 820)
        self.declare_parameter('frame_height', 410)

        # ROI points in IMAGE PIXELS (820x410)
        # Order: top-left, bottom-left, bottom-right, top-right
        # Tune these if BEV looks off.
        self.declare_parameter('roi_src', [
            270.0, 240.0,   # TL
            60.0,  400.0,   # BL
            760.0, 400.0,   # BR
            560.0, 240.0    # TR
        ])

        self.declare_parameter('bev_margin', 120.0)

        # Sliding window
        self.declare_parameter('n_windows', 9)
        self.declare_parameter('margin', 70)
        self.declare_parameter('minpix', 60)

        # Yellow-only HSV thresholds (OpenCV HSV: H 0-179)
        # Tuned from your screenshot: Hue ~31, high V, decent S
        self.declare_parameter('yellow_h_low', 26)
        self.declare_parameter('yellow_h_high', 36)
        self.declare_parameter('yellow_s_low', 110)
        self.declare_parameter('yellow_s_high', 255)
        self.declare_parameter('yellow_v_low', 200)
        self.declare_parameter('yellow_v_high', 255)

        # CTE/heading scaling (tune for real meters)
        self.declare_parameter('xm_per_pix', 3.7 / 700.0)
        self.declare_parameter('ym_per_pix', 10.0 / 410.0)

        # Lookahead row fraction (0 top -> 1 bottom)
        self.declare_parameter('lookahead_y_frac', 0.90)

        # Output topics
        self.declare_parameter('binary_topic', '/vision/lanes/binary')
        self.declare_parameter('bev_binary_topic', '/vision/lanes/bev_binary')
        self.declare_parameter('cte_topic', '/vision/lanes/cte')
        self.declare_parameter('heading_topic', '/vision/lanes/heading_error')

        # ===================== Read Params =====================
        self.image_topic = self.get_parameter('image_topic').value
        self.W = int(self.get_parameter('frame_width').value)
        self.H = int(self.get_parameter('frame_height').value)

        roi_src = self.get_parameter('roi_src').value
        if len(roi_src) != 8:
            raise RuntimeError("roi_src must have 8 values: tlx,tly, blx,bly, brx,bry, trx,try")

        self.roi_src = np.float32([
            (roi_src[0], roi_src[1]),
            (roi_src[2], roi_src[3]),
            (roi_src[4], roi_src[5]),
            (roi_src[6], roi_src[7]),
        ])

        bev_margin = float(self.get_parameter('bev_margin').value)
        self.roi_dst = np.float32([
            (bev_margin, 0.0),
            (bev_margin, float(self.H)),
            (float(self.W) - bev_margin, float(self.H)),
            (float(self.W) - bev_margin, 0.0),
        ])

        self.M = cv2.getPerspectiveTransform(self.roi_src, self.roi_dst)

        # Sliding window params
        self.n_windows = int(self.get_parameter('n_windows').value)
        self.win_margin = int(self.get_parameter('margin').value)
        self.minpix = int(self.get_parameter('minpix').value)

        # Yellow HSV thresholds
        self.h_lo = int(self.get_parameter('yellow_h_low').value)
        self.h_hi = int(self.get_parameter('yellow_h_high').value)
        self.s_lo = int(self.get_parameter('yellow_s_low').value)
        self.s_hi = int(self.get_parameter('yellow_s_high').value)
        self.v_lo = int(self.get_parameter('yellow_v_low').value)
        self.v_hi = int(self.get_parameter('yellow_v_high').value)

        # Meters/pixel for CTE
        self.xm_per_pix = float(self.get_parameter('xm_per_pix').value)
        self.ym_per_pix = float(self.get_parameter('ym_per_pix').value)

        self.lookahead_y_frac = float(self.get_parameter('lookahead_y_frac').value)

        # ===================== ROS IO =====================
        self.bridge = CvBridge()

        self.sub = self.create_subscription(Image, self.image_topic, self.on_image, 10)

        self.pub_bin = self.create_publisher(Image, self.get_parameter('binary_topic').value, 1)
        self.pub_bev = self.create_publisher(Image, self.get_parameter('bev_binary_topic').value, 1)
        self.pub_cte = self.create_publisher(Float64, self.get_parameter('cte_topic').value, 1)
        self.pub_head = self.create_publisher(Float64, self.get_parameter('heading_topic').value, 1)

        self._last_info_ns = 0

        self.get_logger().info(
            f"lane_detector up. Sub={self.image_topic} "
            f"H/S/V=[{self.h_lo}-{self.h_hi}, {self.s_lo}-{self.s_hi}, {self.v_lo}-{self.v_hi}] "
            f"WxH={self.W}x{self.H}"
        )

    # ============================================================
    # Yellow-only binary mask
    # ============================================================
    def make_binary(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        lower = np.array([self.h_lo, self.s_lo, self.v_lo], dtype=np.uint8)
        upper = np.array([self.h_hi, self.s_hi, self.v_hi], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)  # 0/255 mono

        # Clean noise / fill small gaps
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return mask

    # ============================================================
    # BEV warp
    # ============================================================
    def warp_bev(self, binary: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(binary, self.M, (self.W, self.H), flags=cv2.INTER_NEAREST)

    # ============================================================
    # Sliding window fit (single lane or both lanes)
    # If only one yellow lane exists, we still fit that lane and treat it as "reference".
    # ============================================================
    def sliding_window_fit(self, bev_bin: np.ndarray):
        bin01 = (bev_bin > 0).astype(np.uint8)

        nonzero = bin01.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        if nonzerox.size < 200:
            return None

        # Histogram base search in bottom half
        histogram = np.sum(bin01[self.H // 2:, :], axis=0)

        # Since you only want the yellow lane, just find the single strongest peak.
        x_current = int(np.argmax(histogram))

        window_height = int(self.H / self.n_windows)
        lane_inds = []

        for window in range(self.n_windows):
            win_y_low = self.H - (window + 1) * window_height
            win_y_high = self.H - window * window_height

            win_x_low = x_current - self.win_margin
            win_x_high = x_current + self.win_margin

            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
            lane_inds.append(good_inds)

            if len(good_inds) > self.minpix:
                x_current = int(np.mean(nonzerox[good_inds]))

        lane_inds = np.concatenate(lane_inds) if len(lane_inds) else np.array([], dtype=int)

        if lane_inds.size < 200:
            return None

        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        # Fit x = f(y)
        fit = np.polyfit(y, x, 2)
        return fit

    # ============================================================
    # CTE + heading error from x=f(y) in BEV
    # ============================================================
    def compute_cte_heading(self, fit):
        if fit is None:
            return None, None

        y_la = int(np.clip(self.lookahead_y_frac * self.H, 0, self.H - 1))

        a, b, c = fit
        x_lane = a * (y_la ** 2) + b * y_la + c

        # Vehicle reference pixel x at BEV center
        x_vehicle = self.W / 2.0

        # NOTE: With only ONE lane line, this is CTE to that line, not to lane center.
        # If this is the left lane boundary, lane center would be (x_lane + lane_width_px/2).
        cte_pix = x_lane - x_vehicle
        cte_m = float(cte_pix * self.xm_per_pix)

        dx_dy = 2.0 * a * y_la + b
        heading_err = float(np.arctan(dx_dy))  # radians

        return cte_m, heading_err

    # ============================================================
    # Callback
    # ============================================================
    def on_image(self, msg: Image):
        # Decode
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge decode failed: {e} | encoding={getattr(msg,'encoding','?')}")
            return

        if bgr is None:
            self.get_logger().warn(
                f"cv_bridge returned None frame. encoding={getattr(msg,'encoding','?')} "
                f"data_len={len(msg.data) if hasattr(msg,'data') else 'NA'}"
            )
            return

        # Throttled info print (every ~2s)
        now = self.get_clock().now().nanoseconds
        if now - self._last_info_ns > int(2e9):
            self._last_info_ns = now
            self.get_logger().info(f"Image OK: shape={bgr.shape} msg.encoding={msg.encoding}")

        # Resize to expected size so BEV matrix stays consistent
        if bgr.shape[1] != self.W or bgr.shape[0] != self.H:
            bgr = cv2.resize(bgr, (self.W, self.H), interpolation=cv2.INTER_LINEAR)

        # Binary + BEV
        binary = self.make_binary(bgr)
        bev = self.warp_bev(binary)

        # Fit + CTE/heading
        fit = self.sliding_window_fit(bev)
        cte_m, heading_err = self.compute_cte_heading(fit)

        # Publish images
        bin_msg = self.bridge.cv2_to_imgmsg(binary, encoding='mono8')
        bin_msg.header = msg.header
        self.pub_bin.publish(bin_msg)

        bev_msg = self.bridge.cv2_to_imgmsg(bev, encoding='mono8')
        bev_msg.header = msg.header
        self.pub_bev.publish(bev_msg)

        # Publish CTE + heading (NaN if invalid)
        m_cte = Float64()
        m_head = Float64()
        m_cte.data = float(cte_m) if cte_m is not None else float('nan')
        m_head.data = float(heading_err) if heading_err is not None else float('nan')
        self.pub_cte.publish(m_cte)
        self.pub_head.publish(m_head)


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