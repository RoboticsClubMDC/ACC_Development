#!/usr/bin/env python3

import select
import sys
import termios
import time
import tty

import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node


MAX_SPEED    = 0.5   # m/s   (ROS param: 'speed')
MAX_TURN     = 0.3   # rad/s (ROS param: 'turn')
ACCEL        = 0.05  # m/s   per tick while W/S held
DECEL        = 0.03  # m/s   per tick coast when released
STEER_RATE   = 0.06  # rad/s per tick while A/D held
STEER_RETURN = 0.10  # rad/s per tick return to centre when released
LOOP_HZ      = 20

# Must be longer than the OS key-repeat initial delay.
# Ubuntu default is 500 ms; 600 ms gives safe margin.
KEY_HOLD_S   = 0.60


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


def _decay(val, step):
    if abs(val) <= step:
        return 0.0
    return val - step * (1.0 if val > 0 else -1.0)


class ManualDrive(Node):
    def __init__(self):
        super().__init__('manual_drive')
        self.declare_parameter('speed', MAX_SPEED)
        self.declare_parameter('turn',  MAX_TURN)
        self.declare_parameter('cmd_topic', '/cmd_vel_nav')
        self.declare_parameter('accel_per_tick', ACCEL)
        self.declare_parameter('decel_per_tick', DECEL)
        self.declare_parameter('steer_rate_per_tick', STEER_RATE)
        self.declare_parameter('steer_return_per_tick', STEER_RETURN)
        self.declare_parameter('key_hold_s', KEY_HOLD_S)

        self.max_speed = float(self.get_parameter('speed').get_parameter_value().double_value)
        self.max_turn  = float(self.get_parameter('turn').get_parameter_value().double_value)
        self.cmd_topic = self.get_parameter('cmd_topic').get_parameter_value().string_value
        self.accel = float(self.get_parameter('accel_per_tick').get_parameter_value().double_value)
        self.decel = float(self.get_parameter('decel_per_tick').get_parameter_value().double_value)
        self.steer_rate = float(self.get_parameter('steer_rate_per_tick').get_parameter_value().double_value)
        self.steer_return = float(self.get_parameter('steer_return_per_tick').get_parameter_value().double_value)
        self.key_hold_s = float(self.get_parameter('key_hold_s').get_parameter_value().double_value)

        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.undo_pub = self.create_publisher(Empty, '/undo_last_node', 2)
        self.add_on_set_parameters_callback(self._on_param_update)
        self.get_logger().info(
            f'ManualDrive ready — max_speed={self.max_speed:.2f}  '
            f'max_turn={self.max_turn:.2f}  topic={self.cmd_topic}  '
            f'accel={self.accel:.3f} decel={self.decel:.3f} '
            f'steer_rate={self.steer_rate:.3f} key_hold={self.key_hold_s:.2f}s')

    def _on_param_update(self, params):
        for p in params:
            if p.name == 'speed' and p.type_ in (p.Type.DOUBLE, p.Type.INTEGER):
                self.max_speed = float(p.value)
            elif p.name == 'turn' and p.type_ in (p.Type.DOUBLE, p.Type.INTEGER):
                self.max_turn = float(p.value)
            elif p.name == 'accel_per_tick' and p.type_ in (p.Type.DOUBLE, p.Type.INTEGER):
                self.accel = float(p.value)
            elif p.name == 'decel_per_tick' and p.type_ in (p.Type.DOUBLE, p.Type.INTEGER):
                self.decel = float(p.value)
            elif p.name == 'steer_rate_per_tick' and p.type_ in (p.Type.DOUBLE, p.Type.INTEGER):
                self.steer_rate = float(p.value)
            elif p.name == 'steer_return_per_tick' and p.type_ in (p.Type.DOUBLE, p.Type.INTEGER):
                self.steer_return = float(p.value)
            elif p.name == 'key_hold_s' and p.type_ in (p.Type.DOUBLE, p.Type.INTEGER):
                self.key_hold_s = float(p.value)
        return SetParametersResult(successful=True)

    def publish_cmd(self, speed, turn):
        msg = Twist()
        msg.linear.x  = float(speed)
        msg.angular.z = float(turn)
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ManualDrive()

    print('Controls: W=forward  S=reverse  A=left  D=right  SPACE=stop  Z=undo last node  Q=quit')
    print(f'max_speed={node.max_speed:.2f}  max_turn={node.max_turn:.2f}  topic={node.cmd_topic}')
    print('Hold W/S to move and A/D to steer. Release keys to coast and re-centre smoothly.')
    print('----------------------------------------------------------------')

    speed = 0.0
    turn  = 0.0
    dt    = 1.0 / LOOP_HZ

    NEG_INF = -(node.key_hold_s + 1.0)
    last_press = {'w': NEG_INF, 's': NEG_INF, 'a': NEG_INF, 'd': NEG_INF}

    OPPOSITE = {'w': 's', 's': 'w', 'a': 'd', 'd': 'a'}

    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    running = True
    try:
        tty.setraw(fd)
        while rclpy.ok() and running:
            t0  = time.monotonic()
            now = t0

            stop_requested = False
            while select.select([sys.stdin], [], [], 0)[0]:
                k = sys.stdin.read(1).lower()
                if k == 'q':
                    running = False
                    break
                elif k == ' ':
                    stop_requested = True
                elif k == 'z':
                    node.undo_pub.publish(Empty())
                elif k in last_press:
                    last_press[k] = now
                    # pressing a key immediately cancels its opposite so
                    # direction changes feel instant instead of waiting for timeout
                    last_press[OPPOSITE[k]] = NEG_INF

            if not running:
                break

            if stop_requested:
                speed = 0.0
                turn  = 0.0
                for key in last_press:
                    last_press[key] = NEG_INF
            else:
                w = (now - last_press['w']) < node.key_hold_s
                s = (now - last_press['s']) < node.key_hold_s
                a = (now - last_press['a']) < node.key_hold_s
                d = (now - last_press['d']) < node.key_hold_s

                if w and not s:
                    speed = _clamp(speed + node.accel, -node.max_speed, node.max_speed)
                elif s and not w:
                    speed = _clamp(speed - node.accel, -node.max_speed, node.max_speed)
                else:
                    speed = _decay(speed, node.decel)

                if a and not d:
                    turn = _clamp(turn + node.steer_rate, -node.max_turn, node.max_turn)
                elif d and not a:
                    turn = _clamp(turn - node.steer_rate, -node.max_turn, node.max_turn)
                else:
                    turn = _decay(turn, node.steer_return)

            node.publish_cmd(speed, turn)
            sys.stdout.write(f'\r  speed={speed:+.3f}  turn={turn:+.3f}   ')
            sys.stdout.flush()

            leftover = dt - (time.monotonic() - t0)
            if leftover > 0:
                time.sleep(leftover)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        node.publish_cmd(0.0, 0.0)
        print('\nStopped. Bye.')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
