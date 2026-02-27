#!/usr/bin/env python3
"""Lane / sidewalk guardrails for nav_to_pose.

This node sits between `nav_to_pose.py` (publishing `/cmd_vel_raw`) and the
motor command converter (subscribing to `/cmd_vel_nav`).

Goal
----
Make nav_to_pose *stay in the selected lane* and *avoid no-go/sidewalk*
WITHOUT "fighting" the planner/controller.

Key stability choices (to reduce oscillation)
--------------------------------------------
1) Lane center is computed as a **row-wise lane center** in a near-field band
   (median of per-row centers). This is usually smoother than an area centroid
   when the lane mask contains holes or perspective changes.

2) The no-go mask is used as a **repulsive term** (left vs right overlap inside
   the lane band), rather than carving holes from the lane mask.

3) A **deadband + ramp** on lane error prevents hunting around the center.

4) A **bias rate limiter** and an **optional output steering slew limiter**
   smooth both the guardrail bias and nav_to_pose oscillations.

Topics
------
Inputs:
- /cmd_vel_raw (Twist)                       : nav_to_pose output
- /lane_detection/lane_selected (Image mono8): selected lane mask
- /sidewalk_detection/no_go_margin (Image mono8): no-go mask

Output:
- /cmd_vel_nav (Twist)

"""

from __future__ import annotations

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String
from cv_bridge import CvBridge


class LaneKeeping(Node):
    def __init__(self):
        super().__init__("lane_keeping")

        # -------------------- Topics --------------------
        self.declare_parameter("input_cmd_topic", "/cmd_vel_raw")
        self.declare_parameter("output_cmd_topic", "/cmd_vel_nav")
        self.declare_parameter("mask_topic", "/sidewalk_detection/no_go_margin")
        self.declare_parameter("lane_mask_topic", "/lane_detection/lane_selected")

        # -------------------- Measurement band --------------------
        # Near-field band used for measurements (fractions of image height)
        self.declare_parameter("band_y0_frac", 0.80)
        self.declare_parameter("band_y1_frac", 0.98)

        # Desired lane-center x position in image (fraction of width)
        # (keep compatibility with previous `seed_x_frac`)
        self.declare_parameter("target_x_frac", 0.55)
        self.declare_parameter("seed_x_frac", 0.55)

        # No-go-only fallback ROI (legacy)
        self.declare_parameter("roi_half_width_frac", 0.20)

        # Legacy (kept for compatibility; unused)
        self.declare_parameter("inside_lane_bias_frac", 0.15)
        self.declare_parameter("min_free_pixels", 120)

        # -------------------- Steering bias tuning --------------------
        # These defaults are *intentionally calmer* than the previous version.
        self.declare_parameter("steer_bias_gain", 0.65)
        self.declare_parameter("steer_bias_max", 0.18)
        self.declare_parameter("steer_bias_rate", 0.60)  # rad/s
        self.declare_parameter("max_steering", 0.50)

        # Deadband + ramp to avoid hunting near center
        self.declare_parameter("err_deadband", 0.04)  # normalized
        self.declare_parameter("err_full", 0.22)      # normalized

        # Repulsion from no-go overlap inside lane band
        self.declare_parameter("repulse_gain", 0.18)
        self.declare_parameter("repulse_max", 0.10)

        # Optional output steering slew-rate limiter (helps nav_to_pose oscillation)
        # Set very high (e.g., 999) to effectively disable.
        self.declare_parameter("steer_output_rate", 2.5)  # rad/s

        # -------------------- Speed handling around no-go --------------------
        self.declare_parameter("panic_red_occ", 0.08)
        self.declare_parameter("panic_speed", 0.10)
        self.declare_parameter("slow_red_occ", 0.03)
        self.declare_parameter("min_speed_scale", 0.75)

        # -------------------- Mask robustness --------------------
        self.declare_parameter("mask_timeout_sec", 0.35)
        self.declare_parameter("min_lane_pixels", 250)
        self.declare_parameter("min_row_width_px", 18)  # for row-wise lane center

        # Lane center smoothing
        self.declare_parameter("cx_alpha", 0.20)  # EMA alpha (smaller => more smoothing)

        # Debug
        self.declare_parameter("publish_debug", False)

        self.bridge = CvBridge()

        # Latest inputs
        self.last_cmd = Twist()
        self.have_cmd = False

        self.last_red_u8: np.ndarray | None = None
        self.have_red = False
        self.last_red_t = 0.0

        self.last_lane_u8: np.ndarray | None = None
        self.have_lane = False
        self.last_lane_t = 0.0

        # State
        now = self.get_clock().now().nanoseconds * 1e-9
        self.last_time = now
        self.bias_state = 0.0
        self.steer_state = 0.0
        self.cx_filt: float | None = None

        # ROS pub/sub
        in_cmd = self.get_parameter("input_cmd_topic").value
        out_cmd = self.get_parameter("output_cmd_topic").value
        red_topic = self.get_parameter("mask_topic").value
        lane_topic = self.get_parameter("lane_mask_topic").value

        self.pub = self.create_publisher(Twist, out_cmd, 10)

        # Publish only on command updates (consistent timing)
        self.sub_cmd = self.create_subscription(Twist, in_cmd, self.cmd_cb, 10)
        self.sub_red = self.create_subscription(Image, red_topic, self.red_cb, qos_profile_sensor_data)
        self.sub_lane = self.create_subscription(Image, lane_topic, self.lane_cb, qos_profile_sensor_data)

        self.debug_enabled = bool(self.get_parameter("publish_debug").value)
        if self.debug_enabled:
            self.pub_bias = self.create_publisher(Float32, "/lane_keeping/bias", 10)
            self.pub_occ = self.create_publisher(Float32, "/lane_keeping/no_go_overlap", 10)
            self.pub_cx = self.create_publisher(Float32, "/lane_keeping/cx", 10)
            self.pub_status = self.create_publisher(String, "/lane_keeping/status", 10)

        self.get_logger().info(f"Input cmd:  {in_cmd}")
        self.get_logger().info(f"Output cmd: {out_cmd}")
        self.get_logger().info(f"No-go mask: {red_topic}")
        self.get_logger().info(f"Lane mask:  {lane_topic}")

    # ------------------------- Callbacks -------------------------
    def cmd_cb(self, msg: Twist):
        self.last_cmd = msg
        self.have_cmd = True
        self._compute_and_publish()

    def red_cb(self, msg: Image):
        try:
            m = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception:
            return
        self.last_red_u8 = m
        self.have_red = True
        self.last_red_t = self.get_clock().now().nanoseconds * 1e-9

    def lane_cb(self, msg: Image):
        try:
            m = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception:
            return
        self.last_lane_u8 = m
        self.have_lane = True
        self.last_lane_t = self.get_clock().now().nanoseconds * 1e-9

    # ------------------------- Helpers -------------------------
    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return float(np.clip(x, lo, hi))

    def _rate_limit(self, current: float, target: float, rate: float, dt: float) -> float:
        """Rate-limit current -> target by |d/dt| <= rate."""
        if dt <= 0.0 or rate <= 0.0:
            return float(target)
        step = float(rate) * float(dt)
        delta = self._clamp(float(target) - float(current), -step, step)
        return float(current) + float(delta)

    def _band_indices(self, h: int) -> tuple[int, int]:
        y0 = int(float(self.get_parameter("band_y0_frac").value) * h)
        y1 = int(float(self.get_parameter("band_y1_frac").value) * h)
        y0 = max(0, min(h - 1, y0))
        y1 = max(0, min(h, y1))
        if y1 <= y0:
            y1 = min(h, y0 + 1)
        return y0, y1

    def _desired_x(self, w: int) -> float:
        # Backward-compat: prefer seed_x_frac if target_x_frac is default but seed is overridden.
        default_x = 0.55
        target_x_frac = float(self.get_parameter("target_x_frac").value)
        seed_x_frac = float(self.get_parameter("seed_x_frac").value)
        if abs(target_x_frac - default_x) < 1e-6 and abs(seed_x_frac - default_x) >= 1e-6:
            x_frac = seed_x_frac
        else:
            x_frac = target_x_frac
        return float(x_frac) * float(w)

    def _compute_lane_center(self, lane_band: np.ndarray) -> float | None:
        """Compute a stable lane center x from a band using per-row edges."""
        min_row_w = int(self.get_parameter("min_row_width_px").value)
        centers = []
        for r in range(lane_band.shape[0]):
            xs = np.flatnonzero(lane_band[r, :])
            if xs.size < min_row_w:
                continue
            centers.append(0.5 * (float(xs[0]) + float(xs[-1])))
        if not centers:
            return None
        return float(np.median(np.asarray(centers)))

    def _deadband_ramp(self, err_n: float) -> float:
        """Apply deadband + ramp to normalized error to reduce hunting."""
        dead = float(self.get_parameter("err_deadband").value)
        full = float(self.get_parameter("err_full").value)
        aerr = abs(float(err_n))
        if aerr <= dead:
            return 0.0
        s = (aerr - dead) / max(1e-6, (full - dead))
        s = self._clamp(s, 0.0, 1.0)
        # At s=1, output equals original err_n; near deadband, output is much smaller.
        return float(np.sign(err_n)) * float(s) * float(aerr)

    # ------------------------- Core logic -------------------------
    def _compute_and_publish(self):
        if not self.have_cmd:
            return

        cmd_in = self.last_cmd

        now = self.get_clock().now().nanoseconds * 1e-9
        dt = max(1e-3, now - self.last_time)
        self.last_time = now

        timeout = float(self.get_parameter("mask_timeout_sec").value)

        lane_u8 = self.last_lane_u8 if (self.have_lane and (now - self.last_lane_t) <= timeout) else None
        red_u8 = self.last_red_u8 if (self.have_red and (now - self.last_red_t) <= timeout) else None

        steer_bias_rate = float(self.get_parameter("steer_bias_rate").value)
        steer_bias_gain = float(self.get_parameter("steer_bias_gain").value)
        steer_bias_max = float(self.get_parameter("steer_bias_max").value)
        max_steer = float(self.get_parameter("max_steering").value)
        steer_out_rate = float(self.get_parameter("steer_output_rate").value)

        # If we're essentially stopped, don't fight nav_to_pose.
        if abs(cmd_in.linear.x) < 0.02:
            self.bias_state = self._rate_limit(self.bias_state, 0.0, steer_bias_rate, dt)
            steer_target = self._clamp(float(cmd_in.angular.z) + self.bias_state, -max_steer, max_steer)
            self.steer_state = self._rate_limit(self.steer_state, steer_target, steer_out_rate, dt)

            out = Twist()
            out.linear.x = float(cmd_in.linear.x)
            out.angular.z = float(self.steer_state)
            self.pub.publish(out)
            return

        # Determine image size from whichever mask we have
        if lane_u8 is not None:
            h, w = lane_u8.shape[:2]
        elif red_u8 is not None:
            h, w = red_u8.shape[:2]
        else:
            h, w = 0, 0

        # Defaults
        speed_out = float(cmd_in.linear.x)
        desired_bias = 0.0
        occ_metric = 0.0
        status = "pass"

        # -------------------- Lane-based correction --------------------
        lane_ok = False
        if lane_u8 is not None and h > 0 and w > 0:
            y0, y1 = self._band_indices(h)
            lane_band = (lane_u8[y0:y1, :] > 127)
            lane_px = int(lane_band.sum())

            if lane_px >= int(self.get_parameter("min_lane_pixels").value):
                cx = self._compute_lane_center(lane_band)
                if cx is not None:
                    lane_ok = True

                    x_des = self._desired_x(w)

                    # EMA smooth cx
                    alpha = float(self.get_parameter("cx_alpha").value)
                    if self.cx_filt is None:
                        self.cx_filt = float(cx)
                    else:
                        self.cx_filt = alpha * float(cx) + (1.0 - alpha) * float(self.cx_filt)
                    cx_use = float(self.cx_filt)

                    # Normalized pixel error
                    err_n = (cx_use - x_des) / max(1.0, 0.5 * float(w))
                    err_scaled = self._deadband_ramp(float(err_n))

                    # Lane centering bias (sign: +steer left)
                    bias_lane = -steer_bias_gain * float(err_scaled)

                    # No-go repulsion inside lane (optional)
                    bias_repulse = 0.0
                    if red_u8 is not None and red_u8.shape[:2] == (h, w):
                        red_band = (red_u8[y0:y1, :] > 127)
                        overlap = lane_band & red_band
                        overlap_px = int(overlap.sum())
                        occ_metric = float(overlap_px / max(1, lane_px))

                        mid = w // 2
                        lane_L = lane_band[:, :mid]
                        lane_R = lane_band[:, mid:]
                        ov_L = overlap[:, :mid]
                        ov_R = overlap[:, mid:]

                        left_occ = float(ov_L.sum() / max(1, lane_L.sum()))
                        right_occ = float(ov_R.sum() / max(1, lane_R.sum()))

                        diff = self._clamp(right_occ - left_occ, -1.0, 1.0)
                        rep_gain = float(self.get_parameter("repulse_gain").value)
                        rep_max = float(self.get_parameter("repulse_max").value)

                        # Scale repulsion with overlap so tiny/noisy detections don't jitter
                        occ_scale = float(np.clip(occ_metric * 3.0, 0.0, 1.0))
                        bias_repulse = rep_gain * diff * occ_scale
                        bias_repulse = self._clamp(bias_repulse, -rep_max, rep_max)

                    desired_bias = bias_lane + bias_repulse
                    desired_bias = self._clamp(desired_bias, -steer_bias_max, steer_bias_max)
                    status = "lane" if occ_metric < 1e-6 else "lane+no_go"

        # -------------------- No-go-only fallback --------------------
        if (not lane_ok) and (red_u8 is not None) and h > 0 and w > 0:
            y0, y1 = self._band_indices(h)
            red_band = (red_u8[y0:y1, :] > 127)

            x_des = self._desired_x(w)
            half_w = int(float(self.get_parameter("roi_half_width_frac").value) * w)
            cx0 = int(np.clip(x_des, 0, w - 1))
            x0 = max(0, cx0 - half_w)
            x1 = min(w, cx0 + half_w)

            roi = red_band[:, x0:x1]
            occ_metric = float(roi.mean()) if roi.size > 0 else 0.0

            mid = (x1 - x0) // 2
            left_occ = float(roi[:, :mid].mean()) if mid > 0 else 0.0
            right_occ = float(roi[:, mid:].mean()) if (x1 - x0 - mid) > 0 else 0.0

            diff = self._clamp(right_occ - left_occ, -1.0, 1.0)
            k_repulse = 0.28
            # scale by occupancy so tiny detections don't jitter
            desired_bias = k_repulse * diff * float(np.clip(occ_metric * 2.0, 0.0, 1.0))
            desired_bias = self._clamp(desired_bias, -steer_bias_max, steer_bias_max)
            status = "no_go_only"

        # -------------------- Speed handling --------------------
        panic_occ = float(self.get_parameter("panic_red_occ").value)
        slow_occ = float(self.get_parameter("slow_red_occ").value)
        min_scale = float(self.get_parameter("min_speed_scale").value)

        if occ_metric > panic_occ:
            speed_out = min(speed_out, float(self.get_parameter("panic_speed").value))
            status += "+panic"
        elif occ_metric > slow_occ:
            t = (occ_metric - slow_occ) / max(1e-6, (panic_occ - slow_occ))
            t = self._clamp(t, 0.0, 1.0)
            scale = 1.0 - (1.0 - min_scale) * t
            speed_out = speed_out * scale
            status += "+slow"

        # -------------------- Apply bias + output smoothing --------------------
        self.bias_state = self._rate_limit(self.bias_state, desired_bias, steer_bias_rate, dt)

        steer_target = float(cmd_in.angular.z) + float(self.bias_state)
        steer_target = self._clamp(steer_target, -max_steer, max_steer)

        self.steer_state = self._rate_limit(self.steer_state, steer_target, steer_out_rate, dt)

        out = Twist()
        out.linear.x = float(speed_out)
        out.angular.z = float(self.steer_state)
        self.pub.publish(out)

        if self.debug_enabled:
            self.pub_bias.publish(Float32(data=float(self.bias_state)))
            self.pub_occ.publish(Float32(data=float(occ_metric)))
            if self.cx_filt is not None:
                self.pub_cx.publish(Float32(data=float(self.cx_filt)))
            self.pub_status.publish(String(data=str(status)))


def main():
    rclpy.init()
    node = LaneKeeping()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()