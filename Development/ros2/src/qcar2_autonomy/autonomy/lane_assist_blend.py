#!/usr/bin/env python3
"""
lane_assist_blend — blends lane-centering corrections into manual controller steering.

Toggle with RB (button 5) via /lane_assist/toggle (published by controller_drive).
Correction is disabled automatically when trust=0 (lane not detected) or delta is stale.

Sign convention: delta > 0 means car drifted LEFT of lane center → correction steers RIGHT
  → output.angular.z -= gain * delta  (negative because ROS +z = CCW = left)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Empty
import numpy as np


class LaneAssistBlend(Node):
    def __init__(self):
        super().__init__('lane_assist_blend')

        self.declare_parameter('input_cmd_topic',   '/cmd_vel_raw')
        self.declare_parameter('output_cmd_topic',  '/cmd_vel_nav')
        self.declare_parameter('gain',              3.0)    # delta (±0.25) → angular.z scale
        self.declare_parameter('max_turn',          2.0)    # rad/s output clamp
        self.declare_parameter('trust_threshold',   0.5)
        self.declare_parameter('delta_timeout_sec', 0.3)

        in_topic  = self.get_parameter('input_cmd_topic').value
        out_topic = self.get_parameter('output_cmd_topic').value

        self._enabled      = False
        self._delta        = 0.0
        self._trust        = 0.0
        self._last_delta_t = 0.0

        self.pub = self.create_publisher(Twist, out_topic, 10)

        self.create_subscription(Twist,   in_topic,               self._cmd_cb,    10)
        self.create_subscription(Float32, '/lane_stanley/delta',  self._delta_cb,  10)
        self.create_subscription(Float32, '/lane_stanley/trust',  self._trust_cb,  10)
        self.create_subscription(Empty,   '/lane_assist/toggle',  self._toggle_cb, 10)

        self.get_logger().info(
            f'Lane Assist Blend: {in_topic} → {out_topic}  [DISABLED — press RB to enable]'
        )

    def _toggle_cb(self, _: Empty):
        self._enabled = not self._enabled
        state = 'ON' if self._enabled else 'OFF'
        self.get_logger().info(f'Lane assist {state}')
        print(f'\n  [LANE ASSIST] {state}\n', flush=True)

    def _delta_cb(self, msg: Float32):
        self._delta        = float(msg.data)
        self._last_delta_t = self.get_clock().now().nanoseconds * 1e-9

    def _trust_cb(self, msg: Float32):
        self._trust = float(msg.data)

    def _cmd_cb(self, msg: Twist):
        now     = self.get_clock().now().nanoseconds * 1e-9
        timeout = float(self.get_parameter('delta_timeout_sec').value)
        fresh   = (now - self._last_delta_t) < timeout

        out = Twist()
        out.linear.x = msg.linear.x

        if (self._enabled
                and fresh
                and self._trust >= float(self.get_parameter('trust_threshold').value)):
            gain     = float(self.get_parameter('gain').value)
            max_turn = float(self.get_parameter('max_turn').value)
            # Subtract because delta > 0 → car left of center → need negative angular.z (right)
            correction      = -gain * self._delta
            out.angular.z   = float(np.clip(msg.angular.z + correction, -max_turn, max_turn))
        else:
            out.angular.z = msg.angular.z

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = LaneAssistBlend()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
