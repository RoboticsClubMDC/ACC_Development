#!/usr/bin/env python3

import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node


DEFAULT_SPEED = 0.5
DEFAULT_TURN = 0.2


def get_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


class ManualDrive(Node):
    def __init__(self):
        super().__init__('manual_drive')
        self.declare_parameter('speed', DEFAULT_SPEED)
        self.declare_parameter('turn', DEFAULT_TURN)
        self.declare_parameter('cmd_topic', '/cmd_vel_nav')

        self.speed = float(self.get_parameter('speed').get_parameter_value().double_value)
        self.turn = float(self.get_parameter('turn').get_parameter_value().double_value)
        self.cmd_topic = self.get_parameter('cmd_topic').get_parameter_value().string_value

        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.add_on_set_parameters_callback(self.parameter_update_callback)
        self.get_logger().info(
            f'ManualDrive ready: speed={self.speed:.3f} turn={self.turn:.3f} '
            f'topic={self.cmd_topic}')

    def parameter_update_callback(self, params):
        for param in params:
            if param.name == 'speed' and param.type_ in (param.Type.DOUBLE, param.Type.INTEGER):
                self.speed = float(param.value)
                self.get_logger().info(f'speed updated: {self.speed:.3f}')
            elif param.name == 'turn' and param.type_ in (param.Type.DOUBLE, param.Type.INTEGER):
                self.turn = float(param.value)
                self.get_logger().info(f'turn updated: {self.turn:.3f}')
        return SetParametersResult(successful=True)

    def publish_cmd(self, speed, turn):
        msg = Twist()
        msg.linear.x = float(speed)
        msg.angular.z = float(turn)
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ManualDrive()

    print('Controls: W=forward  S=back  A=left  D=right  SPACE=stop  Q=quit')
    print(f'Using: speed={node.speed:.3f}  turn={node.turn:.3f}  topic={node.cmd_topic}')
    print('----------------------------------------------------------------')

    speed = 0.0
    turn = 0.0

    try:
        while rclpy.ok():
            key = get_key().lower()

            if key == 'q':
                speed = 0.0
                turn = 0.0
                node.publish_cmd(speed, turn)
                print('\nStopped. Bye.')
                break
            elif key == 'w':
                speed = node.speed
                turn = 0.0
            elif key == 's':
                speed = -node.speed
                turn = 0.0
            elif key == 'a':
                speed = node.speed
                turn = node.turn
            elif key == 'd':
                speed = node.speed
                turn = -node.turn
            elif key == ' ':
                speed = 0.0
                turn = 0.0
            else:
                continue

            node.publish_cmd(speed, turn)
            print(f'\r  speed={speed:+.2f}  turn={turn:+.2f}   ', end='', flush=True)
    finally:
        node.publish_cmd(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
