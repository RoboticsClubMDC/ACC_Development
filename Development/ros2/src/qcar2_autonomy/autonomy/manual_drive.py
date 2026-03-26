#!/usr/bin/env python3

import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


SPEED = 0.15
TURN = 0.4


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
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_nav', 10)

    def publish_cmd(self, speed, turn):
        msg = Twist()
        msg.linear.x = float(speed)
        msg.angular.z = float(turn)
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ManualDrive()

    print('Controls: W=forward  S=back  A=left  D=right  SPACE=stop  Q=quit')
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
                speed = SPEED
                turn = 0.0
            elif key == 's':
                speed = -SPEED
                turn = 0.0
            elif key == 'a':
                speed = SPEED
                turn = TURN
            elif key == 'd':
                speed = SPEED
                turn = -TURN
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
