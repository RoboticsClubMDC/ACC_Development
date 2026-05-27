#!/usr/bin/env python3

import argparse
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from std_msgs.msg import Bool


class CarStopCommand(Node):
    def __init__(self, release=False):
        super().__init__('car_stop')
        self.release = release
        self.stop_pub = self.create_publisher(Bool, '/car_stop', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_nav', 10)

    def publish_once(self):
        stop = Bool()
        stop.data = not self.release
        self.stop_pub.publish(stop)
        if not self.release:
            self.cmd_pub.publish(Twist())


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description='Engage or release the QCar command stop override.')
    parser.add_argument(
        '--release',
        action='store_true',
        help='Release the /car_stop override so autonomy can drive again.')
    parser.add_argument(
        '--seconds',
        type=float,
        default=1.0,
        help='How long to repeat the stop/release command before exiting.')
    return parser.parse_args(remove_ros_args(args=argv)[1:])


def main(args=None):
    parsed = _parse_args(args)
    rclpy.init(args=args)
    node = CarStopCommand(release=parsed.release)

    end_time = time.time() + max(0.1, parsed.seconds)
    try:
        while rclpy.ok() and time.time() < end_time:
            node.publish_once()
            rclpy.spin_once(node, timeout_sec=0.01)
            time.sleep(0.03)
    finally:
        action = 'released' if parsed.release else 'engaged'
        node.get_logger().warn(f'Car stop {action}.')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
