#!/usr/bin/env python3
"""controller_watchdog — health monitor for the path_follower's PD controller.

Subscribes to /nav/* diagnostic topics produced by path_follower and
classifies the controller's behavior into a small set of named health
states. Output is /nav/controller_health (std_msgs/String), suitable for
binding to a Foxglove Indicator panel with color rules.

States (priority order — checked in this order, first match wins):
    SATURATED      — steering_saturation_rate > 0.5 for the last 1s window
    WIGGLING       — std(blended_delta) over last 1s > 0.20 rad
    LATE_REACTION  — |psi| > 0.5 rad AND |blended_delta| < 0.10 rad sustained
    HEALTHY        — otherwise

Intentionally NOT implemented in this version (you said you don't need them):
    OFF_PATH       — would require a cross-track-error computation
    STUCK          — would require a longer time window + speed comparison

Architecture: this is a passive observer. Subscribes only. Publishes only a
String. No control commands. Cannot break the controller.
"""

from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String


HEALTHY       = "healthy"
SATURATED     = "saturated"
WIGGLING      = "wiggling"
LATE_REACTION = "late_reaction"
WARMING_UP    = "warming_up"   # not enough samples yet


class ControllerWatchdog(Node):

    def __init__(self):
        super().__init__("controller_watchdog")

        # Tunable thresholds (as ROS params so they can be adjusted live).
        self.declare_parameter("window_seconds",      1.0)
        self.declare_parameter("rate_hz",             80.0)
        self.declare_parameter("publish_rate_hz",     2.0)
        self.declare_parameter("sat_threshold",       0.5)
        self.declare_parameter("wiggle_std_threshold", 0.20)
        self.declare_parameter("late_psi_threshold",  0.5)
        self.declare_parameter("late_delta_threshold", 0.10)
        self.declare_parameter("late_min_duration",   0.5)   # seconds

        self.rate_hz          = float(self.get_parameter("rate_hz").value)
        self.window_s         = float(self.get_parameter("window_seconds").value)
        self.publish_rate     = float(self.get_parameter("publish_rate_hz").value)
        self.sat_thr          = float(self.get_parameter("sat_threshold").value)
        self.wiggle_std_thr   = float(self.get_parameter("wiggle_std_threshold").value)
        self.late_psi_thr     = float(self.get_parameter("late_psi_threshold").value)
        self.late_delta_thr   = float(self.get_parameter("late_delta_threshold").value)
        self.late_min_dur     = float(self.get_parameter("late_min_duration").value)

        window_size = max(int(self.window_s * self.rate_hz), 10)
        self.delta_window = deque(maxlen=window_size)
        self.psi_window   = deque(maxlen=window_size)

        self.latest_sat_rate = 0.0
        self.late_streak_start_t = None   # set when LATE condition first triggers

        self.create_subscription(Float32, "/nav/blended_delta",
                                 self._delta_cb, 50)
        self.create_subscription(Float32, "/nav/psi",
                                 self._psi_cb, 50)
        self.create_subscription(Float32, "/nav/steering_saturation_rate",
                                 self._sat_cb, 10)

        self.pub_health = self.create_publisher(String, "/nav/controller_health", 10)
        self.create_timer(1.0 / self.publish_rate, self._publish_health)

        self.get_logger().info(
            f"controller_watchdog up | window={self.window_s}s | "
            f"sat>{self.sat_thr}=SATURATED | std>{self.wiggle_std_thr}=WIGGLING | "
            f"|psi|>{self.late_psi_thr} & |delta|<{self.late_delta_thr} "
            f"for>{self.late_min_dur}s=LATE_REACTION"
        )

    def _delta_cb(self, msg):
        self.delta_window.append(float(msg.data))

    def _psi_cb(self, msg):
        self.psi_window.append(float(msg.data))

    def _sat_cb(self, msg):
        self.latest_sat_rate = float(msg.data)

    def _publish_health(self):
        msg = String()

        # Warming up — wait until we have at least 1/4 of the window filled.
        if len(self.delta_window) < self.delta_window.maxlen // 4:
            msg.data = WARMING_UP
            self.pub_health.publish(msg)
            return

        # 1) SATURATED takes priority — controller is fighting hardware limits.
        if self.latest_sat_rate > self.sat_thr:
            msg.data = SATURATED
            self.late_streak_start_t = None
            self.pub_health.publish(msg)
            return

        # 2) WIGGLING — high-frequency oscillation in commanded steering.
        delta_std = float(np.std(list(self.delta_window)))
        if delta_std > self.wiggle_std_thr:
            msg.data = WIGGLING
            self.late_streak_start_t = None
            self.pub_health.publish(msg)
            return

        # 3) LATE_REACTION — psi has been demanding a turn that the controller
        #    is not producing. Requires the condition to hold for late_min_dur
        #    seconds (otherwise normal target-passing transients trigger it).
        last_psi   = self.psi_window[-1] if self.psi_window else 0.0
        last_delta = self.delta_window[-1]
        if abs(last_psi) > self.late_psi_thr and abs(last_delta) < self.late_delta_thr:
            now = self.get_clock().now().nanoseconds * 1e-9
            if self.late_streak_start_t is None:
                self.late_streak_start_t = now
            elif (now - self.late_streak_start_t) > self.late_min_dur:
                msg.data = LATE_REACTION
                self.pub_health.publish(msg)
                return
        else:
            self.late_streak_start_t = None

        msg.data = HEALTHY
        self.pub_health.publish(msg)


def main():
    rclpy.init()
    node = ControllerWatchdog()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
