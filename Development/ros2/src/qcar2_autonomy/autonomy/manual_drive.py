#!/usr/bin/env python3

import select
import subprocess
import sys
import termios
import time
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

_fd = None
_old_settings = None

def setup_terminal():
    """Set terminal to raw mode once."""
    global _fd, _old_settings
    _fd = sys.stdin.fileno()
    _old_settings = termios.tcgetattr(_fd)
    tty.setraw(_fd)

def restore_terminal():
    """Restore terminal settings."""
    global _fd, _old_settings
    if _old_settings:
        termios.tcsetattr(_fd, termios.TCSADRAIN, _old_settings)

def get_key(timeout_s=0.01):
    """Read one key from stdin in raw mode, or return '' on timeout."""
    ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if ready:
        return sys.stdin.read(1)
    return ""


class ManualDrive(Node):
    """Simple keyboard teleop for QCar2 via /cmd_vel_nav."""

    def __init__(self):
        super().__init__("manual_drive")

        # Higher defaults for faster response.
        self.declare_parameter("forward_speed", 0.25)
        self.declare_parameter("reverse_speed", 0.20)
        self.declare_parameter("turn_rate", 0.50)
        self.declare_parameter("cmd_topic", "/cmd_vel_nav")
        self.declare_parameter("speed_step", 0.05)
        self.declare_parameter("auto_start_converter", True)

        self.forward_speed = float(self.get_parameter("forward_speed").value)
        self.reverse_speed = float(self.get_parameter("reverse_speed").value)
        self.turn_rate = float(self.get_parameter("turn_rate").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.speed_step = float(self.get_parameter("speed_step").value)
        self.auto_start_converter = bool(
            self.get_parameter("auto_start_converter").value
        )

        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.converter_process = None

        if self.auto_start_converter and self.cmd_topic == "/cmd_vel_nav":
            self.ensure_nav2_qcar_converter()

    def converter_is_running(self):
        """Return true if a known nav2_qcar2 converter node is already in graph."""
        converter_names = {
            "nav2_qcar2_converter",
            "nav2_qcar2_command_converter",
        }
        for name, _namespace in self.get_node_names_and_namespaces():
            if name in converter_names:
                return True
        return False

    def ensure_nav2_qcar_converter(self):
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.converter_is_running():
                self.get_logger().info("nav2_qcar2_converter already running.")
                return

        # Added 2026-05-22 22:18:54 EDT:
        # manual_drive publishes /cmd_vel_nav, but QCar hardware consumes
        # qcar2_motor_speed_cmd. Start the converter only when it is missing so
        # this teleop command works during RTAB-only sessions without Nav2.
        command = [
            "ros2",
            "run",
            "qcar2_nodes",
            "nav2_qcar2_converter",
            "--ros-args",
            "-r",
            "__node:=nav2_qcar2_converter",
        ]
        try:
            self.converter_process = subprocess.Popen(command)
            time.sleep(0.5)
        except OSError as exc:
            self.get_logger().error(f"Failed to start nav2_qcar2_converter: {exc}")
            self.converter_process = None
            return

        if self.converter_process.poll() is not None:
            self.get_logger().error(
                "nav2_qcar2_converter exited immediately. "
                "Build/source qcar2_nodes before driving."
            )
            self.converter_process = None
            return

        self.get_logger().info("Started nav2_qcar2_converter for manual_drive.")

    def publish_cmd(self, linear_x, angular_z):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def stop_converter_if_started(self):
        if self.converter_process is None:
            return
        if self.converter_process.poll() is None:
            self.get_logger().info("Stopping nav2_qcar2_converter started by manual_drive.")
            self.converter_process.terminate()
            try:
                self.converter_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.converter_process.kill()
                self.converter_process.wait(timeout=1.0)
        self.converter_process = None


def main(args=None):
    rclpy.init(args=args)
    node = ManualDrive()
    setup_terminal()

    print("Keyboard manual drive (WASD style)")
    print("  w: forward")
    print("  s: reverse")
    print("  a: steer left (keep current speed)")
    print("  d: steer right (keep current speed)")
    print("  space/x: stop")
    print("  +/=: increase speed")
    print("  -/_: decrease speed")
    print("  {/[: decrease turn rate")
    print("  }/]: increase turn rate")
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

            # Republish current state on idle ticks so nav2_qcar_command_convert
            # (0.25 s /cmd_vel_nav safety timeout, added 2026-05-04) does not
            # zero throttle and steering between key presses.
            if not key:
                node.publish_cmd(linear_x, angular_z)
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
                # Keep the currently latched speed and only steer.
                angular_z = node.turn_rate
            elif key == "d":
                # Keep the currently latched speed and only steer.
                angular_z = -node.turn_rate
            elif key in (" ", "x"):
                linear_x = 0.0
                angular_z = 0.0
            elif key in ("+", "="):
                node.forward_speed = min(1.0, node.forward_speed + node.speed_step)
                node.reverse_speed = min(1.0, node.reverse_speed + node.speed_step)
                print(
                    f"\nSpeed up: forward={node.forward_speed:.2f} reverse={node.reverse_speed:.2f}  "
                )
                continue
            elif key in ("-", "_"):
                node.forward_speed = max(0.05, node.forward_speed - node.speed_step)
                node.reverse_speed = max(0.05, node.reverse_speed - node.speed_step)
                print(
                    f"\nSpeed down: forward={node.forward_speed:.2f} reverse={node.reverse_speed:.2f}  "
                )
                continue
            elif key in ("{", "["):
                node.turn_rate = max(0.1, node.turn_rate - node.speed_step)
                print(
                    f"\nTurn rate down: {node.turn_rate:.2f}  "
                )
                continue
            elif key in ("}", "]"):
                node.turn_rate = min(1.0, node.turn_rate + node.speed_step)
                print(
                    f"\nTurn rate up: {node.turn_rate:.2f}  "
                )
                continue
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
        node.stop_converter_if_started()
        node.destroy_node()
        rclpy.shutdown()
        restore_terminal()


if __name__ == "__main__":
    main()
