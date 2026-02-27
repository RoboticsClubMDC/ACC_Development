#!/usr/bin/env python3
"""Sidewalk-repulsion guardrail for nav_to_pose.

Replaces the original lane_keeping node.  Uses ONLY the sidewalk/no-go mask
(which the user trusts) — no lane-detection model required.

Algorithm
---------
1. Extract a near-field band from the no-go mask (bottom 55-95 % of image).
2. Compute left-vs-right occupancy imbalance inside that band.
3. Steer AWAY from whichever side has more no-go pixels (repulsive bias).
4. Scale the bias by overall occupancy so tiny noise blobs don't jitter.
5. Rate-limit both the bias and the final steering output.

If no no-go mask is available (timeout), the node passes the steering through
unchanged — nav_to_pose's Stanley controller keeps the car on the path.

Topics
------
Input : /cmd_vel_raw  (Twist)  — from nav_to_pose
Input : /sidewalk_detection/no_go_margin (Image mono8)
Output: /cmd_vel_nav   (Twist)  — to command converter
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
    """Sidewalk-only repulsive guardrail."""

    def __init__(self):
        super().__init__("lane_keeping")

        # ---- Topics ----
        self.declare_parameter("input_cmd_topic",  "/cmd_vel_raw")
        self.declare_parameter("output_cmd_topic", "/cmd_vel_nav")
        self.declare_parameter("mask_topic",
                               "/sidewalk_detection/no_go_margin")

        # ---- Measurement band (fraction of image height) ----
        self.declare_parameter("band_y0_frac", 0.55)
        self.declare_parameter("band_y1_frac", 0.95)

        # ---- Repulsion tuning ----
        self.declare_parameter("repulse_gain", 0.35)    # steer per unit imbalance
        self.declare_parameter("repulse_max",  0.25)    # max repulsive bias [rad]
        self.declare_parameter("repulse_rate", 0.80)    # bias slew [rad/s]

        # ---- Output ----
        self.declare_parameter("max_steering",       0.55)
        self.declare_parameter("steer_output_rate",  5.0)   # [rad/s]

        # ---- Speed ----
        self.declare_parameter("panic_occ",    0.10)
        self.declare_parameter("panic_speed",  0.10)
        self.declare_parameter("slow_occ",     0.03)
        self.declare_parameter("min_speed_scale", 0.75)

        # ---- Mask robustness ----
        self.declare_parameter("mask_timeout_sec", 0.40)

        # ---- Debug ----
        self.declare_parameter("publish_debug", False)

        self.bridge = CvBridge()

        # State
        self.last_cmd = Twist()
        self.have_cmd = False

        self.last_mask: np.ndarray | None = None
        self.have_mask = False
        self.last_mask_t = 0.0

        now = self.get_clock().now().nanoseconds * 1e-9
        self.last_time = now
        self.bias_state = 0.0
        self.steer_state = 0.0

        # ROS wiring
        in_cmd  = self.get_parameter("input_cmd_topic").value
        out_cmd = self.get_parameter("output_cmd_topic").value
        mask_topic = self.get_parameter("mask_topic").value

        self.pub = self.create_publisher(Twist, out_cmd, 10)
        self.sub_cmd  = self.create_subscription(Twist, in_cmd,
                                                  self.cmd_cb, 10)
        self.sub_mask = self.create_subscription(Image, mask_topic,
                                                  self.mask_cb,
                                                  qos_profile_sensor_data)

        self.debug_enabled = bool(self.get_parameter("publish_debug").value)
        if self.debug_enabled:
            self.pub_bias = self.create_publisher(Float32,
                                                   "/lane_keeping/bias", 10)
            self.pub_occ  = self.create_publisher(Float32,
                                                   "/lane_keeping/occ", 10)

        self.get_logger().info(f"Sidewalk guard: {in_cmd} → {out_cmd}")
        self.get_logger().info(f"No-go mask: {mask_topic}")

    # ---- Callbacks ----
    def cmd_cb(self, msg: Twist):
        self.last_cmd = msg
        self.have_cmd = True
        self._compute_and_publish()

    def mask_cb(self, msg: Image):
        try:
            self.last_mask = self.bridge.imgmsg_to_cv2(msg,
                                                        desired_encoding="mono8")
        except Exception:
            return
        self.have_mask = True
        self.last_mask_t = self.get_clock().now().nanoseconds * 1e-9

    # ---- Helpers ----
    @staticmethod
    def _clamp(x, lo, hi):
        return float(np.clip(x, lo, hi))

    def _rate_limit(self, cur, tgt, rate, dt):
        if dt <= 0 or rate <= 0:
            return float(tgt)
        step = rate * dt
        return cur + self._clamp(tgt - cur, -step, step)

    # ---- Core ----
    def _compute_and_publish(self):
        if not self.have_cmd:
            return

        cmd_in = self.last_cmd
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = max(1e-3, now - self.last_time)
        self.last_time = now

        timeout   = float(self.get_parameter("mask_timeout_sec").value)
        max_steer = float(self.get_parameter("max_steering").value)
        out_rate  = float(self.get_parameter("steer_output_rate").value)
        rep_rate  = float(self.get_parameter("repulse_rate").value)

        mask_ok = (self.have_mask
                   and (now - self.last_mask_t) <= timeout
                   and self.last_mask is not None)

        desired_bias = 0.0
        occ_metric   = 0.0
        speed_out    = float(cmd_in.linear.x)

        # --- Compute repulsion from no-go mask ---
        if mask_ok and abs(cmd_in.linear.x) > 0.02:
            mask = self.last_mask
            h, w = mask.shape[:2]

            y0 = int(float(self.get_parameter("band_y0_frac").value) * h)
            y1 = int(float(self.get_parameter("band_y1_frac").value) * h)
            y0 = max(0, min(h - 1, y0))
            y1 = max(y0 + 1, min(h, y1))

            band = (mask[y0:y1, :] > 127)
            total_px = max(1, band.size)
            occ_metric = float(band.sum()) / total_px

            # Left-vs-right imbalance
            mid = w // 2
            left_occ  = float(band[:, :mid].mean()) if mid > 0 else 0.0
            right_occ = float(band[:, mid:].mean()) if (w - mid) > 0 else 0.0

            imbalance = self._clamp(right_occ - left_occ, -1.0, 1.0)

            rep_gain = float(self.get_parameter("repulse_gain").value)
            rep_max  = float(self.get_parameter("repulse_max").value)

            # Scale by occupancy so tiny/noisy blobs don't cause jitter
            occ_scale = self._clamp(occ_metric * 5.0, 0.0, 1.0)
            desired_bias = self._clamp(rep_gain * imbalance * occ_scale,
                                       -rep_max, rep_max)

            # Speed reduction near sidewalk
            panic = float(self.get_parameter("panic_occ").value)
            slow  = float(self.get_parameter("slow_occ").value)
            if occ_metric > panic:
                speed_out = min(speed_out,
                                float(self.get_parameter("panic_speed").value))
            elif occ_metric > slow:
                t = (occ_metric - slow) / max(1e-6, panic - slow)
                scale = 1.0 - (1.0 - float(
                    self.get_parameter("min_speed_scale").value)) * t
                speed_out *= scale

        # --- Apply bias + slew ---
        self.bias_state = self._rate_limit(self.bias_state, desired_bias,
                                           rep_rate, dt)

        steer_target = self._clamp(float(cmd_in.angular.z) + self.bias_state,
                                   -max_steer, max_steer)
        self.steer_state = self._rate_limit(self.steer_state, steer_target,
                                            out_rate, dt)

        out = Twist()
        out.linear.x = float(speed_out)
        out.angular.z = float(self.steer_state)
        self.pub.publish(out)

        if self.debug_enabled:
            self.pub_bias.publish(Float32(data=float(self.bias_state)))
            self.pub_occ.publish(Float32(data=float(occ_metric)))


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