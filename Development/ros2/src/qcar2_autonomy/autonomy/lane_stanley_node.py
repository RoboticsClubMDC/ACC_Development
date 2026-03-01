#!/usr/bin/env python3
"""
lane_stanley_node.py  (OVERALL — NO BEV VERSION)
===============================================
✅ Removes ALL BEV dependency.
✅ Uses raw binary lane mask center from: /lane_detection/lane_selected (mono8 0/255).
✅ Publishes SAME topics (do not change):
    /lane_stanley/cte            (Float32)  — normalized [-1..1] (NOT meters anymore)
    /lane_stanley/heading_error  (Float32)  — normalized [-1..1]
    /lane_stanley/delta          (Float32)  — steering command (radians)
    /lane_stanley/trust          (Float32)  — 0.0 / 1.0

How it works (raw image logic):
- Use only a near-field band at the bottom of the mask (stable region).
- Compute lane center (centroid x) within band -> CTE in pixels -> normalized to [-1..1].
- Compute heading error from slope of lane center vs row (centroid per strip) -> normalized.
- Steering is a weighted sum:
    delta = k_cte * cte_norm + k_head * heading_norm
  (Then clipped to max_steer)
- Trust becomes 0 if insufficient pixels (min_lane_px).
- CTE and heading are EMA-smoothed to reduce jitter/oscillation.

NOTE:
- This does NOT use BEV, no homography, no meters conversion.
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32
from cv_bridge import CvBridge


class LaneStanleyNode(Node):
    def __init__(self):
        super().__init__('lane_stanley_node')

        # -----------------------------
        # Parameters (RAW MASK MODE)
        # -----------------------------
        # Use only a stable near-field band of the image:
        #   band_y0_frac=0.65 means start at 65% of image height (from top)
        #   band_y1_frac=0.95 means end at 95% of image height (near bottom)
        self.declare_parameter('band_y0_frac', 0.65)
        self.declare_parameter('band_y1_frac', 0.95)

        # Minimum number of lane pixels required to trust detection
        self.declare_parameter('min_lane_px', 400)

        # Smoothing factors (EMA). Higher = follows changes faster, lower = smoother.
        self.declare_parameter('cte_alpha', 0.2)
        self.declare_parameter('head_alpha', 0.2)

        # How many horizontal strips to sample for heading slope fit
        self.declare_parameter('n_strips', 10)

        # Steering weights (normalized domain)
        self.declare_parameter('k_cte', 0.35)
        self.declare_parameter('k_head', 0.65)

        # Steering clamp
        self.declare_parameter('max_steer', 0.25)

        # -----------------------------
        # Internal state
        # -----------------------------
        self.bridge = CvBridge()
        self.speed  = 0.0

        # EMA filtered values (start at zero)
        self.cte_f = 0.0
        self.he_f  = 0.0
        self.have_filter_state = False

        # -----------------------------
        # Publishers (KEEP EXACT TOPICS)
        # -----------------------------
        qos = QoSProfile(depth=2)
        self.pub_cte   = self.create_publisher(Float32, '/lane_stanley/cte', qos)
        self.pub_he    = self.create_publisher(Float32, '/lane_stanley/heading_error', qos)
        self.pub_delta = self.create_publisher(Float32, '/lane_stanley/delta', qos)
        self.pub_trust = self.create_publisher(Float32, '/lane_stanley/trust', qos)

        # -----------------------------
        # Subscribers
        # -----------------------------
        # RAW binary mask (mono8) from lane_detection.py
        self.create_subscription(
            Image, '/lane_detection/lane_selected', self._mask_cb, qos_profile_sensor_data
        )
        # Speed from QCar joint state (unchanged)
        self.create_subscription(
            JointState, '/qcar2_joint', self._joint_cb, 1
        )

        self.get_logger().info('Lane Stanley node ready (RAW MASK mode, NO BEV).')

    # ------------------------------------------------------------------
    def _joint_cb(self, msg: JointState):
        """Extract measured speed from wheel encoder (unchanged)."""
        if msg.velocity:
            raw = msg.velocity[0]
            self.speed = (raw / (720.0 * 4.0)) * ((13.0 * 19.0) / (70.0 * 30.0)) * (2.0 * math.pi) * 0.033

    # ------------------------------------------------------------------
    def _mask_cb(self, msg: Image):
        # 1) Convert ROS Image -> CV2 mono8
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge: {e}')
            return

        if mask is None or mask.ndim != 2:
            self._publish(0.0, 0.0, 0.0, 0.0)
            return

        H, W = mask.shape

        # 2) Select the near-field band (most stable for lane centering)
        y0 = int(np.clip(self.get_parameter('band_y0_frac').value, 0.0, 1.0) * H)
        y1 = int(np.clip(self.get_parameter('band_y1_frac').value, 0.0, 1.0) * H)
        if y1 <= y0:
            # fall back to bottom half if user misconfigures
            y0, y1 = int(0.5 * H), H

        band = mask[y0:y1, :]
        binary = (band > 127).astype(np.uint8)
        total_px = int(binary.sum())

        min_px = int(self.get_parameter('min_lane_px').value)
        trust = 1.0 if total_px >= min_px else 0.0

        if trust < 0.5:
            # If detection is weak, stop trusting and avoid steering noise
            self.have_filter_state = False
            self._publish(0.0, 0.0, 0.0, 0.0)
            return

        # 3) Compute CTE from centroid x in the band
        ys, xs = np.where(binary > 0)  # ys/xs are within the BAND coordinates
        if xs.size == 0:
            self.have_filter_state = False
            self._publish(0.0, 0.0, 0.0, 0.0)
            return

        cx = float(xs.mean())
        car_cx = W / 2.0
        cte_px = cx - car_cx

        # Normalize CTE to roughly [-1, +1]
        # Positive means lane center is to the RIGHT of car center in image
        cte_norm = float(cte_px / (W / 2.0))
        cte_norm = float(np.clip(cte_norm, -1.0, 1.0))

        # 4) Compute heading error from lane center shift vs row (strip centroids)
        n_strips = int(self.get_parameter('n_strips').value)
        n_strips = max(4, min(n_strips, 30))

        band_h = (y1 - y0)
        strip_edges = np.linspace(0, band_h, n_strips + 1, dtype=int)

        y_vals = []
        x_vals = []

        for i in range(n_strips):
            s0 = strip_edges[i]
            s1 = strip_edges[i + 1]
            if s1 <= s0:
                continue

            strip = binary[s0:s1, :]
            if int(strip.sum()) < 10:
                continue

            ys_s, xs_s = np.where(strip > 0)
            if xs_s.size == 0:
                continue

            # centroid x for this strip
            cx_s = float(xs_s.mean())

            # y coordinate (in full-image row coordinates) at center of this strip
            y_center_full = float(y0 + (s0 + s1) / 2.0)

            y_vals.append(y_center_full)
            x_vals.append(cx_s)

        # If not enough strips, heading estimate is unreliable
        if len(x_vals) < 2:
            heading_norm = 0.0
        else:
            y_arr = np.array(y_vals, dtype=np.float32)
            x_arr = np.array(x_vals, dtype=np.float32)

            # Fit line: x = m*y + b
            m, b = np.polyfit(y_arr, x_arr, 1)

            # In image coordinates:
            # - y increases downward
            # - x increases to the right
            #
            # If m is positive, x increases as y increases (lane leans right as it goes down).
            # We want a heading-like signal in [-1,1].
            heading_rad = math.atan(m)  # small angle approx is okay, atan keeps bounded
            heading_norm = float(heading_rad / (math.pi / 4.0))  # normalize by 45°
            heading_norm = float(np.clip(heading_norm, -1.0, 1.0))

        # 5) Smooth (EMA) to match stable behavior (avoid oscillation/jitter)
        cte_alpha  = float(np.clip(self.get_parameter('cte_alpha').value, 0.0, 1.0))
        head_alpha = float(np.clip(self.get_parameter('head_alpha').value, 0.0, 1.0))

        if not self.have_filter_state:
            self.cte_f = cte_norm
            self.he_f  = heading_norm
            self.have_filter_state = True
        else:
            self.cte_f = (1.0 - cte_alpha) * self.cte_f + cte_alpha * cte_norm
            self.he_f  = (1.0 - head_alpha) * self.he_f  + head_alpha * heading_norm

        # 6) Steering law (NO BEV Stanley):
        # delta = weighted sum of normalized errors
        k_cte  = float(self.get_parameter('k_cte').value)
        k_head = float(self.get_parameter('k_head').value)
        max_steer = float(self.get_parameter('max_steer').value)

        delta = k_cte * self.cte_f + k_head * self.he_f
        delta = float(np.clip(delta, -max_steer, +max_steer))

        # Publish normalized signals + delta
        self._publish(self.cte_f, self.he_f, delta, trust)

    # ------------------------------------------------------------------
    def _publish(self, cte, he, delta, trust):
        def f32(v):
            m = Float32()
            m.data = float(v)
            return m

        self.pub_cte.publish(f32(cte))
        self.pub_he.publish(f32(he))
        self.pub_delta.publish(f32(delta))
        self.pub_trust.publish(f32(trust))


def main(args=None):
    rclpy.init(args=args)
    node = LaneStanleyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()