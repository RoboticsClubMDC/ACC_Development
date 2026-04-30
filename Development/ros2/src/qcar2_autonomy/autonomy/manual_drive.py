#!/usr/bin/env python3

import select
import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


def get_key(timeout_s=0.1):
    """Read one key from stdin in raw mode, or return '' on timeout."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
        if ready:
            return sys.stdin.read(1)
        return ""
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class ManualDrive(Node):
    """Simple keyboard teleop for QCar2 via /cmd_vel_nav."""

    def __init__(self):
        super().__init__("manual_drive")

        # Conservative defaults for first physical tests.
        self.declare_parameter("forward_speed", 0.10)
        self.declare_parameter("reverse_speed", 0.08)
        self.declare_parameter("turn_rate", 0.25)
        self.declare_parameter("cmd_topic", "/cmd_vel_nav")

        self.forward_speed = float(self.get_parameter("forward_speed").value)
        self.reverse_speed = float(self.get_parameter("reverse_speed").value)
        self.turn_rate = float(self.get_parameter("turn_rate").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)

        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)

    def publish_cmd(self, linear_x, angular_z):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ManualDrive()

    print("Keyboard manual drive (WASD style)")
    print("  w: forward")
    print("  s: reverse")
    print("  a: turn left in place")
    print("  d: turn right in place")
    print("  space/x: stop")
    print("  q: stop and quit")
    print(
        f"Using topic={node.cmd_topic}, "
        f"forward={node.forward_speed:.2f}, "
        f"reverse={node.reverse_speed:.2f}, "
        f"turn={node.turn_rate:.2f}"
    )

    linear_x = 0.0
    angular_z = 0.0

    try:
        while rclpy.ok():
            key = get_key().lower()

            if not key:
                continue

            if key == "q":
                linear_x = 0.0
                angular_z = 0.0
                node.publish_cmd(linear_x, angular_z)
                print("\nStopped. Exiting manual drive.")
                break
            if key == "w":
                linear_x = node.forward_speed
                angular_z = 0.0
            elif key == "s":
                linear_x = -node.reverse_speed
                angular_z = 0.0
            elif key == "a":
                linear_x = 0.0
                angular_z = node.turn_rate
            elif key == "d":
                linear_x = 0.0
                angular_z = -node.turn_rate
            elif key in (" ", "x"):
                linear_x = 0.0
                angular_z = 0.0
            else:
                continue

            node.publish_cmd(linear_x, angular_z)
            print(
                f"\rlinear={linear_x:+.2f} angular={angular_z:+.2f}  ",
                end="",
                flush=True,
            )
    finally:
        node.publish_cmd(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
