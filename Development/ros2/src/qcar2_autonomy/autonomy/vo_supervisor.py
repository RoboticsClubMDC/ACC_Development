#!/usr/bin/env python3
"""vo_supervisor.py — Part 3 "Track A-lite" supervisor node.

Subscribes to the VO monitor's numeric topics and publishes
actionable decisions that a downstream controller can use.

This is the bridge between "monitoring" (Part 2) and "acting on
faults" (Part 3). The supervisor classifies the monitor's output
into operational modes and recommends actions.

CRITICAL DESIGN: stop_advised is driven by /vo/healthy (Bool),
NOT by state_id alone. This is because state_id can be
odom_suspect even when pending (not yet confirmed). Only when
healthy=False (confirmed odom fault, 3+ consecutive) does the
supervisor recommend stopping.

TOPICS SUBSCRIBED:
  /vo/state_id   (UInt8)   — 0=init 1=warming 2=vo_suspect 3=agree 4=odom_suspect
  /vo/delta_trans (Float32) — residual rho in meters
  /vo/vo_weight  (Float32) — VO trust weight [0..1]
  /vo/healthy    (Bool)    — False ONLY when odom_suspect is CONFIRMED

TOPICS PUBLISHED:
  /vo/supervisor/mode          (String)  — operational mode name
  /vo/supervisor/stop_advised  (Bool)    — True when confirmed odom fault
  /vo/supervisor/trust_level   (Float32) — continuous 0..1 navigation trust

MODES:
  NORMAL           — agree, both channels consistent
  VO_UNTRUSTED     — vo_suspect, VO abstaining (turns, poor quality)
  ODOM_WARNING     — odom_suspect pending (not yet confirmed)
  ODOM_FAULT       — confirmed odom_suspect (healthy=False)
  INITIALIZING     — system starting up or re-anchoring
  DEGRADED         — prolonged vo_suspect with no recent agree

SETUP.PY:  'vo_supervisor=autonomy.vo_supervisor:main'
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32, UInt8, Bool


class VOSupervisor(Node):

    def __init__(self):
        super().__init__('vo_supervisor')

        # ── Parameters ──
        # How many seconds of continuous vo_suspect before we
        # consider the system "degraded" (no redundancy coverage)
        self.declare_parameter('degraded_timeout_s', 30.0)

        self._degraded_timeout = float(
            self.get_parameter('degraded_timeout_s').value)

        # ── State ──
        self._state_id = 0       # last received state
        self._rho = 0.0          # last received residual
        self._vo_weight = 0.0    # last received VO weight
        self._healthy = True     # last received healthy flag
        self._mode = 'INITIALIZING'
        self._trust_level = 1.0

        # Track how long vo_suspect has been active
        self._vo_suspect_start = None
        self._last_agree_time = self.get_clock().now().nanoseconds / 1e9

        # ── Subscribers ──
        self.create_subscription(
            UInt8, '/vo/state_id', self._state_cb, 10)
        self.create_subscription(
            Float32, '/vo/delta_trans', self._rho_cb, 10)
        self.create_subscription(
            Float32, '/vo/vo_weight', self._weight_cb, 10)
        self.create_subscription(
            Bool, '/vo/healthy', self._healthy_cb, 10)

        # ── Publishers ──
        self._pub_mode = self.create_publisher(
            String, '/vo/supervisor/mode', 1)
        self._pub_stop = self.create_publisher(
            Bool, '/vo/supervisor/stop_advised', 1)
        self._pub_trust = self.create_publisher(
            Float32, '/vo/supervisor/trust_level', 1)

        # ── Decision timer (10 Hz) ──
        self.create_timer(0.1, self._decide)

        self.get_logger().info(
            'VO Supervisor started (degraded_timeout=%.0fs)'
            % self._degraded_timeout)

    # ── Subscriber Callbacks ──
    def _state_cb(self, msg):
        self._state_id = msg.data

    def _rho_cb(self, msg):
        self._rho = msg.data

    def _weight_cb(self, msg):
        self._vo_weight = msg.data

    def _healthy_cb(self, msg):
        self._healthy = msg.data

    # ── Decision Logic ──
    def _decide(self):
        now = self.get_clock().now().nanoseconds / 1e9

        # State IDs: 0=init, 1=warming, 2=vo_suspect,
        #            3=agree, 4=odom_suspect
        sid = self._state_id

        # ── CRITICAL: healthy flag is the TRUE fault indicator ──
        # healthy=False means CONFIRMED odom_suspect (3+ consecutive).
        # state_id=4 with healthy=True means PENDING (not confirmed).
        # The supervisor uses healthy as the stop trigger, not state_id.

        if not self._healthy:
            # Confirmed odom fault — this is the real deal
            self._mode = 'ODOM_FAULT'
            rho_severity = min(self._rho / 1.0, 1.0)
            self._trust_level = max(0.1, 0.5 - (0.4 * rho_severity))
            self._vo_suspect_start = None

        elif sid == 0 or sid == 1:
            # System initializing or warming up after re-anchor
            self._mode = 'INITIALIZING'
            self._trust_level = 0.8
            self._vo_suspect_start = None

        elif sid == 3:
            # Agree — both channels consistent
            self._mode = 'NORMAL'
            rho_ratio = min(self._rho / 0.35, 1.0) if self._rho > 0 else 0.0
            self._trust_level = 1.0 - (0.15 * rho_ratio)
            self._last_agree_time = now
            self._vo_suspect_start = None

        elif sid == 4 and self._healthy:
            # Odom suspect but only pending (not confirmed yet)
            # This is a "watch out" not a "fault"
            self._mode = 'ODOM_WARNING'
            self._trust_level = 0.65
            self._vo_suspect_start = None

        elif sid == 2:
            # VO suspect — VO is abstaining (turns, poor quality)
            if self._vo_suspect_start is None:
                self._vo_suspect_start = now

            time_since_agree = now - self._last_agree_time

            if time_since_agree > self._degraded_timeout:
                # No agreement for a long time — redundancy is gone
                self._mode = 'DEGRADED'
                self._trust_level = 0.5
            else:
                self._mode = 'VO_UNTRUSTED'
                # Odom is probably fine, we just can't verify
                self._trust_level = 0.75

        # ── Publish ──
        mode_msg = String()
        mode_msg.data = self._mode
        self._pub_mode.publish(mode_msg)

        # stop_advised is driven by healthy, not state_id
        stop_msg = Bool()
        stop_msg.data = (not self._healthy)
        self._pub_stop.publish(stop_msg)

        trust_msg = Float32()
        trust_msg.data = float(self._trust_level)
        self._pub_trust.publish(trust_msg)


def main(args=None):
    rclpy.init(args=args)
    node = VOSupervisor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Supervisor shutting down...')
    except Exception:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()