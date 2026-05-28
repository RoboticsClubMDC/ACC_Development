#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class LaneStanleyController(Node):
    def __init__(self):
        super().__init__('lane_stanley_controller')

        self.declare_parameter('cmd_topic', '/cmd_vel_nav')
        self.declare_parameter('cte_topic', '/lane_keeping/cross_track_error')
        self.declare_parameter('heading_topic', '/lane_keeping/heading_error')
        self.declare_parameter('detected_topic', '/lane_keeping/lane_detected')

        self.declare_parameter('speed_mps', 0.20)
        self.declare_parameter('stanley_gain', 1.0)
        self.declare_parameter('heading_gain', 1.0)
        self.declare_parameter('steering_sign', 1.0)
        self.declare_parameter('max_steer_rad', 0.35)
        self.declare_parameter('min_speed_for_control', 0.05)
        # 2026-05-28: use MEASURED speed (from /odom) in the Stanley
        # cross-track denominator, not the fixed speed_mps param. The
        # Stanley law atan2(k·cte, v) is only speed-adaptive (gentler at
        # speed, sharper when slow) if v is the REAL speed. With a fixed
        # 0.20 denominator the controller was tuned for exactly one speed
        # and over-corrected whenever the blended/path speed was higher.
        # Set use_measured_speed:=false to revert to the speed_mps param.
        self.declare_parameter('use_measured_speed', True)
        self.declare_parameter('odom_topic', '/odom')
        # 2026-05-27: raised 0.35 → 1.0 to survive the gap between dashed
        # lane segments. At competition speed (~0.5 m/s) the gap between
        # dashes is well under 1 s, so Stanley keeps integrating CTE
        # using the last-known value through the gap rather than going
        # silent and handing control back to PP every dash boundary.
        self.declare_parameter('lane_timeout_sec', 1.0)
        self.declare_parameter('publish_stop_when_lost', False)
        self.declare_parameter('control_rate_hz', 30.0)

        self.cte = 0.0
        self.heading = 0.0
        self.detected = False
        self.measured_speed = 0.0
        self.last_lane_time = self.get_clock().now()

        self.use_measured_speed = bool(self.get_parameter('use_measured_speed').value)

        self.cmd_pub = self.create_publisher(
            Twist, self.get_parameter('cmd_topic').value, 1)
        self.create_subscription(
            Float64, self.get_parameter('cte_topic').value, self._cte_cb, 1)
        self.create_subscription(
            Float64, self.get_parameter('heading_topic').value, self._heading_cb, 1)
        self.create_subscription(
            Bool, self.get_parameter('detected_topic').value, self._detected_cb, 1)
        self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value, self._odom_cb, 10)

        rate = float(self.get_parameter('control_rate_hz').value)
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self._timer_cb)

        self.get_logger().info(
            'Lane Stanley controller ready. Make sure path_follower/manual_drive '
            'are not also publishing /cmd_vel_nav during this test.')

    def _cte_cb(self, msg):
        self.cte = float(msg.data)
        self.last_lane_time = self.get_clock().now()

    def _heading_cb(self, msg):
        self.heading = float(msg.data)
        self.last_lane_time = self.get_clock().now()

    def _detected_cb(self, msg):
        self.detected = bool(msg.data)
        if self.detected:
            self.last_lane_time = self.get_clock().now()

    def _odom_cb(self, msg):
        v = float(msg.twist.twist.linear.x)
        if math.isfinite(v):
            self.measured_speed = v

    def _timer_cb(self):
        now = self.get_clock().now()
        lane_age = (now - self.last_lane_time).nanoseconds * 1e-9
        lane_ok = self.detected and lane_age <= float(
            self.get_parameter('lane_timeout_sec').value)

        cmd = Twist()
        if lane_ok:
            speed_param = float(self.get_parameter('speed_mps').value)
            min_speed = float(self.get_parameter('min_speed_for_control').value)
            k = float(self.get_parameter('stanley_gain').value)
            heading_gain = float(self.get_parameter('heading_gain').value)
            steering_sign = float(self.get_parameter('steering_sign').value)
            max_steer = float(self.get_parameter('max_steer_rad').value)

            # Speed-adaptive Stanley: use the MEASURED speed in the cross-track
            # denominator so the correction is gentle at speed and sharp when
            # slow (the whole point of Stanley). Fall back to speed_mps param
            # if measured speed is unavailable / use_measured_speed=false.
            v_denom = self.measured_speed if self.use_measured_speed else speed_param
            v_denom = max(abs(v_denom), min_speed)

            steer = heading_gain * self.heading + math.atan2(
                k * self.cte, v_denom)
            steer = steering_sign * steer
            steer = max(-max_steer, min(max_steer, steer))

            # linear.x is the speed Stanley WOULD drive at; the blender uses
            # linear_source='path' so path_follower's speed wins downstream.
            cmd.linear.x = speed_param
            cmd.angular.z = steer
            self.cmd_pub.publish(cmd)
            return

        if bool(self.get_parameter('publish_stop_when_lost').value):
            self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = LaneStanleyController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
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
