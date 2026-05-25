#!/usr/bin/env python3

import math
from typing import Optional

import numpy as np

import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, Float32


class LaneKeeping(Node):
    def __init__(self):
        super().__init__("lane_keeping")

        self.declare_parameter("input_cmd_topic", "/cmd_vel_nav")
        self.declare_parameter("output_cmd_topic", "/cmd_vel_nav")
        self.declare_parameter("mask_topic", "/lane_detection/road_mask")
        self.declare_parameter("publish_cmd", False)

        self.declare_parameter("band_y0_frac", 0.58)
        self.declare_parameter("band_y1_frac", 0.96)
        self.declare_parameter("n_strips", 12)
        self.declare_parameter("min_lane_px", 350)
        self.declare_parameter("min_strip_px", 12)
        self.declare_parameter("target_x_frac", 0.50)
        self.declare_parameter("target_offset_px", 0.0)

        self.declare_parameter("k_cte", 1.15)
        self.declare_parameter("k_heading", 0.95)
        self.declare_parameter("speed_softening", 0.06)
        self.declare_parameter("max_steering", 0.55)
        self.declare_parameter("steer_output_rate", 5.0)
        self.declare_parameter("cte_alpha", 0.35)
        self.declare_parameter("heading_alpha", 0.35)

        self.declare_parameter("mask_timeout_sec", 0.40)
        self.declare_parameter("require_forward_motion", True)
        self.declare_parameter("publish_debug", True)
        self.declare_parameter("use_lane_assist_toggle", False)
        self.declare_parameter("enabled_on_start", True)

        self.bridge = CvBridge()

        self.last_cmd = Twist()
        self.have_cmd = False
        self.last_mask: Optional[np.ndarray] = None
        self.have_mask = False
        self.last_mask_t = 0.0

        now = self.get_clock().now().nanoseconds * 1e-9
        self.last_time = now
        self.steer_state = 0.0
        self.cte_state = 0.0
        self.heading_state = 0.0
        self.have_filter_state = False
        self.enabled = bool(self.get_parameter("enabled_on_start").value)

        in_cmd = self.get_parameter("input_cmd_topic").value
        out_cmd = self.get_parameter("output_cmd_topic").value
        mask_topic = self.get_parameter("mask_topic").value

        qos = QoSProfile(depth=2)
        self.publish_cmd = bool(self.get_parameter("publish_cmd").value)
        self.pub = self.create_publisher(Twist, out_cmd, 10) if self.publish_cmd else None
        self.sub_cmd = self.create_subscription(Twist, in_cmd, self.cmd_cb, 10)
        self.sub_mask = self.create_subscription(Image, mask_topic, self.mask_cb, qos_profile_sensor_data)

        self.pub_delta = self.create_publisher(Float32, "/lane_keeping/delta", qos)
        self.pub_trust = self.create_publisher(Float32, "/lane_keeping/trust", qos)
        self.debug_enabled = bool(self.get_parameter("publish_debug").value)
        if self.debug_enabled:
            self.pub_cte = self.create_publisher(Float32, "/lane_keeping/cte", qos)
            self.pub_heading = self.create_publisher(Float32, "/lane_keeping/heading_error", qos)

        if bool(self.get_parameter("use_lane_assist_toggle").value):
            self.create_subscription(Empty, "/lane_assist/toggle", self.toggle_cb, 2)

        mode = f"{in_cmd} -> {out_cmd}" if self.publish_cmd else f"{in_cmd} -> /lane_keeping/delta"
        self.get_logger().info(f"Stanley lane keeping: {mode}")
        self.get_logger().info(f"Lane mask: {mask_topic}")

    def cmd_cb(self, msg: Twist):
        self.last_cmd = msg
        self.have_cmd = True
        self._compute_and_publish()

    def mask_cb(self, msg: Image):
        try:
            self.last_mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as exc:
            self.get_logger().warn(f"Could not convert lane mask: {exc}", throttle_duration_sec=2.0)
            return
        self.have_mask = True
        self.last_mask_t = self.get_clock().now().nanoseconds * 1e-9

    def toggle_cb(self, _msg: Empty):
        self.enabled = not self.enabled
        state = "enabled" if self.enabled else "disabled"
        self.get_logger().info(f"Lane keeping {state}")

    @staticmethod
    def _clamp(value, low, high):
        return float(np.clip(value, low, high))

    def _rate_limit(self, current, target, rate, dt):
        if dt <= 0.0 or rate <= 0.0:
            return float(target)
        step = rate * dt
        return current + self._clamp(target - current, -step, step)

    def _lane_measurement(self, mask):
        h, w = mask.shape[:2]

        y0 = int(self._clamp(float(self.get_parameter("band_y0_frac").value), 0.0, 0.98) * h)
        y1 = int(self._clamp(float(self.get_parameter("band_y1_frac").value), 0.02, 1.0) * h)
        if y1 <= y0:
            y0, y1 = int(0.55 * h), h

        band = (mask[y0:y1, :] > 127).astype(np.uint8)
        total_px = int(band.sum())
        min_lane_px = int(self.get_parameter("min_lane_px").value)
        if total_px < min_lane_px:
            return None

        ys, xs = np.where(band > 0)
        if xs.size == 0:
            return None

        target_x = (
            float(self.get_parameter("target_x_frac").value) * w
            + float(self.get_parameter("target_offset_px").value)
        )

        cte_px = float(xs.mean()) - target_x
        cte_norm = self._clamp(cte_px / max(1.0, w / 2.0), -1.0, 1.0)

        n_strips = int(self._clamp(int(self.get_parameter("n_strips").value), 4, 30))
        min_strip_px = int(self.get_parameter("min_strip_px").value)
        band_h = y1 - y0
        strip_edges = np.linspace(0, band_h, n_strips + 1, dtype=int)

        y_centers = []
        x_centers = []
        for i in range(n_strips):
            s0, s1 = strip_edges[i], strip_edges[i + 1]
            if s1 <= s0:
                continue
            strip = band[s0:s1, :]
            if int(strip.sum()) < min_strip_px:
                continue
            _, xs_s = np.where(strip > 0)
            if xs_s.size == 0:
                continue
            y_centers.append(float(y0 + (s0 + s1) / 2.0))
            x_centers.append(float(xs_s.mean()))

        if len(x_centers) >= 2:
            slope_dx_dy, _ = np.polyfit(
                np.array(y_centers, dtype=np.float32),
                np.array(x_centers, dtype=np.float32),
                1,
            )
            heading_error = math.atan(float(slope_dx_dy))
        else:
            heading_error = 0.0

        trust = self._clamp(total_px / max(float(min_lane_px), 1.0), 0.0, 1.0)
        return cte_norm, heading_error, trust

    def _compute_and_publish(self):
        if not self.have_cmd:
            return

        cmd_in = self.last_cmd
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = max(1e-3, now - self.last_time)
        self.last_time = now

        timeout = float(self.get_parameter("mask_timeout_sec").value)
        mask_ok = (
            self.have_mask
            and self.last_mask is not None
            and (now - self.last_mask_t) <= timeout
        )

        speed = float(cmd_in.linear.x)
        forward_ok = (speed > 0.02) or not bool(self.get_parameter("require_forward_motion").value)
        measurement = self._lane_measurement(self.last_mask) if mask_ok and self.enabled and forward_ok else None

        delta = 0.0
        trust = 0.0

        if measurement is None:
            self.have_filter_state = False
            steer_target = float(cmd_in.angular.z)
        else:
            cte_norm, heading_error, trust = measurement
            cte_alpha = self._clamp(float(self.get_parameter("cte_alpha").value), 0.0, 1.0)
            heading_alpha = self._clamp(float(self.get_parameter("heading_alpha").value), 0.0, 1.0)

            if not self.have_filter_state:
                self.cte_state = cte_norm
                self.heading_state = heading_error
                self.have_filter_state = True
            else:
                self.cte_state = (1.0 - cte_alpha) * self.cte_state + cte_alpha * cte_norm
                self.heading_state = (
                    (1.0 - heading_alpha) * self.heading_state
                    + heading_alpha * heading_error
                )

            k_cte = float(self.get_parameter("k_cte").value)
            k_heading = float(self.get_parameter("k_heading").value)
            softening = max(1e-3, float(self.get_parameter("speed_softening").value))
            speed_for_gain = abs(speed) + softening

            cross_track_term = math.atan2(k_cte * self.cte_state, speed_for_gain)
            delta = k_heading * self.heading_state - cross_track_term
            max_steer = float(self.get_parameter("max_steering").value)
            steer_target = self._clamp(delta, -max_steer, max_steer)

        max_steer = float(self.get_parameter("max_steering").value)
        out_rate = float(self.get_parameter("steer_output_rate").value)
        self.steer_state = self._rate_limit(
            self.steer_state,
            self._clamp(steer_target, -max_steer, max_steer),
            out_rate,
            dt,
        )

        if self.pub is not None:
            out = Twist()
            out.linear.x = speed
            out.angular.z = float(self.steer_state)
            self.pub.publish(out)

        self.pub_delta.publish(Float32(data=float(self.steer_state)))
        self.pub_trust.publish(Float32(data=float(trust)))
        if self.debug_enabled:
            self.pub_cte.publish(Float32(data=float(self.cte_state if measurement else 0.0)))
            self.pub_heading.publish(Float32(data=float(self.heading_state if measurement else 0.0)))


def main(args=None):
    rclpy.init(args=args)
    node = LaneKeeping()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
