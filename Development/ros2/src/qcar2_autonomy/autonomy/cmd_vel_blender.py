#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


class CmdVelBlender(Node):
    def __init__(self):
        super().__init__('cmd_vel_blender')

        self.declare_parameter('path_cmd_topic', '/cmd_vel_path')
        self.declare_parameter('lane_cmd_topic', '/cmd_vel_lane')
        self.declare_parameter('cmd_topic', '/cmd_vel_nav')
        self.declare_parameter('path_weight', 0.40)
        self.declare_parameter('lane_weight', 0.60)
        self.declare_parameter('cmd_timeout_sec', 0.35)
        self.declare_parameter('control_rate_hz', 30.0)
        self.declare_parameter('linear_source', 'path')  # path, lane, blend, min
        self.declare_parameter('stop_topic', '/car_stop')

        self.path_cmd = Twist()
        self.lane_cmd = Twist()
        self.path_time = None
        self.lane_time = None
        self.emergency_stop = False
        self._last_status = None

        self.pub = self.create_publisher(
            Twist, self.get_parameter('cmd_topic').value, 1)
        self.create_subscription(
            Twist, self.get_parameter('path_cmd_topic').value,
            self._path_cb, 1)
        self.create_subscription(
            Twist, self.get_parameter('lane_cmd_topic').value,
            self._lane_cb, 1)
        self.create_subscription(
            Bool, self.get_parameter('stop_topic').value,
            self._stop_cb, 1)

        rate = float(self.get_parameter('control_rate_hz').value)
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self._timer_cb)

        self.get_logger().info(
            'CmdVel blender ready: lane_weight=%.2f path_weight=%.2f'
            % (self.get_parameter('lane_weight').value,
               self.get_parameter('path_weight').value))

    def _path_cb(self, msg):
        self.path_cmd = msg
        self.path_time = self.get_clock().now()

    def _lane_cb(self, msg):
        self.lane_cmd = msg
        self.lane_time = self.get_clock().now()

    def _stop_cb(self, msg):
        stop = bool(msg.data)
        if stop != self.emergency_stop:
            self.get_logger().warn(
                'Emergency stop %s' % ('ENGAGED' if stop else 'released'))
        self.emergency_stop = stop

    def _fresh(self, stamp, now, timeout):
        if stamp is None:
            return False
        return (now - stamp).nanoseconds * 1e-9 <= timeout

    def _timer_cb(self):
        if self.emergency_stop:
            self.pub.publish(Twist())
            return

        now = self.get_clock().now()
        timeout = float(self.get_parameter('cmd_timeout_sec').value)
        path_ok = self._fresh(self.path_time, now, timeout)
        lane_ok = self._fresh(self.lane_time, now, timeout)
        status = (path_ok, lane_ok)
        if status != self._last_status:
            self._last_status = status
            self.get_logger().info(
                'Blend inputs: path=%s lane=%s'
                % ('fresh' if path_ok else 'missing',
                   'fresh' if lane_ok else 'missing'))

        cmd = Twist()
        if not path_ok and not lane_ok:
            self.pub.publish(cmd)
            return

        if path_ok and lane_ok:
            path_w = float(self.get_parameter('path_weight').value)
            lane_w = float(self.get_parameter('lane_weight').value)
            total = max(path_w + lane_w, 1e-6)
            path_w /= total
            lane_w /= total

            cmd.angular.z = (
                path_w * self.path_cmd.angular.z +
                lane_w * self.lane_cmd.angular.z)

            linear_source = str(self.get_parameter('linear_source').value)
            if linear_source == 'lane':
                cmd.linear.x = self.lane_cmd.linear.x
            elif linear_source == 'blend':
                cmd.linear.x = (
                    path_w * self.path_cmd.linear.x +
                    lane_w * self.lane_cmd.linear.x)
            elif linear_source == 'min':
                cmd.linear.x = min(self.path_cmd.linear.x, self.lane_cmd.linear.x)
            else:
                cmd.linear.x = self.path_cmd.linear.x
        elif path_ok:
            cmd = self.path_cmd
        else:
            cmd = self.lane_cmd

        self.pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelBlender()
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
