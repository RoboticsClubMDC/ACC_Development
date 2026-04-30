#!/usr/bin/env python3
"""vo_terminal_dashboard.py — Scrolling VO redundancy log v3.

v3: labels yaw as (diag) since decisions are translation-only now.
    Also shows drift and psi error as separate line.

RUN:  ros2 run qcar2_autonomy vo_dashboard
SETUP.PY:  'vo_dashboard=autonomy.vo_terminal_dashboard:main'
"""

import math
import re
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def _f(d, k, default=0.0):
    try:
        return float(d.get(k, default))
    except Exception:
        return default

def _i(d, k, default=0):
    try:
        return int(float(d.get(k, default)))
    except Exception:
        return default

def _s(d, k, default=''):
    return str(d.get(k, default))

def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))

def _deg(a):
    return a * 180.0 / math.pi


class Dashboard(Node):
    _COMPACT_RE = re.compile(
        r'^(?P<state>\w+)\s+rho=(?P<rho>[-0-9.]+)\s+w=(?P<w>[-0-9.]+)\s+\|\s+'
        r'vo\((?P<vo_x>[-0-9.]+),(?P<vo_y>[-0-9.]+),(?P<vo_psi_deg>[-0-9.]+)\)\s+'
        r'ct\((?P<ct_x>[-0-9.]+),(?P<ct_y>[-0-9.]+),(?P<ct_psi_deg>[-0-9.]+)\)\s+\|\s+'
        r'dx=(?P<dx>[-0-9.]+)\s+dy=(?P<dy>[-0-9.]+)\s+dpsi=(?P<dpsi_deg>[-0-9.]+)\s+'
        r'inl=(?P<inl>\d+)\s+sp=(?P<sp>[-0-9.]+)$'
    )

    def __init__(self):
        super().__init__('vo_dashboard')
        self._data = {}
        self._last_rx = 0.0
        self._block_num = 0
        self.create_subscription(
            String, '/vo/fault_status', self._cb, 10)
        self.create_timer(1.5, self._print_block)

        self.get_logger().info(
            'VO Dashboard v3 started (1.5s interval)')

    def _cb(self, msg):
        text = msg.data.strip()
        d = {}

        # Preferred path: parse current compact fault_status string.
        m = self._COMPACT_RE.match(text)
        if m:
            g = m.groupdict()
            d = {
                'state': g['state'],
                'rho': float(g['rho']),
                'w': float(g['w']),
                'vo_x': float(g['vo_x']),
                'vo_y': float(g['vo_y']),
                'vo_psi': math.radians(float(g['vo_psi_deg'])),
                'cart_x': float(g['ct_x']),
                'cart_y': float(g['ct_y']),
                'cart_psi': math.radians(float(g['ct_psi_deg'])),
                'dx': float(g['dx']),
                'dy': float(g['dy']),
                'dpsi': math.radians(float(g['dpsi_deg'])),
                'inliers': int(g['inl']),
                'spread': float(g['sp']),
            }
        else:
            # Fallback: legacy key=value parser.
            for part in text.split():
                if '=' not in part:
                    continue
                k, v = part.split('=', 1)
                d[k] = v

        self._data = d
        self._last_rx = time.time()

    def _print_block(self):
        d = self._data
        if not d:
            print('[VO DASHBOARD] Waiting for /vo/fault_status ...')
            return

        self._block_num += 1

        state = _s(d, 'state', _s(d, 'flag', 'init'))
        inliers = _i(d, 'inliers')
        spread = _f(d, 'spread')
        rho = _f(d, 'rho')
        weight = _f(d, 'w')

        vo_x = _f(d, 'vo_x')
        vo_y = _f(d, 'vo_y')
        vo_psi = _f(d, 'vo_psi')

        cart_x = _f(d, 'cart_x')
        cart_y = _f(d, 'cart_y')
        cart_psi = _f(d, 'cart_psi')

        dx = _f(d, 'dx')
        dy = _f(d, 'dy')
        dpsi = _f(d, 'dpsi')
        vo_speed = _f(d, 'vo_speed')
        vo_dist = _f(d, 'vo_dist')

        drift = math.sqrt(
            (vo_x - cart_x)**2 + (vo_y - cart_y)**2)
        psi_err = abs(_deg(_wrap(vo_psi - cart_psi)))

        c = math.cos(dpsi)
        s = math.sin(dpsi)

        sep = '=' * 58

        print(f'\n{sep}')
        print(f'  VO REDUNDANCY  block #{self._block_num}')
        print(sep)

        print(f'  T_frame (3x3):')
        print(f'    [{c:+.5f} {-s:+.5f} {dx:+.6f}]')
        print(f'    [{s:+.5f} {c:+.5f} {dy:+.6f}]')
        print(f'    [ 0.00000  0.00000  1.000000]')

        print(f'  ---')
        print(f'  {"":8s} {"Vis Odom":>12s}  '
              f'{"Cartograph":>12s}  {"Delta":>8s}')
        print(f'  {"x":8s} {vo_x:+12.4f}m  '
              f'{cart_x:+12.4f}m  {abs(vo_x-cart_x):8.4f}m')
        print(f'  {"y":8s} {vo_y:+12.4f}m  '
              f'{cart_y:+12.4f}m  {abs(vo_y-cart_y):8.4f}m')
        print(f'  {"psi":8s} {_deg(vo_psi):+11.2f}dg  '
              f'{_deg(cart_psi):+11.2f}dg  {psi_err:7.2f}dg')
        print(f'  {"speed":8s} {vo_speed:+12.4f}m/s')
        print(f'  {"dist":8s} {vo_dist:+12.4f}m')

        print(f'  ---')
        print(f'  State: {state}  rho={rho:.3f}m  weight={weight:.2f}')
        print(f'  Drift: {drift:.4f}m  PsiErr: {psi_err:.2f}dg')
        print(f'  Inliers: {inliers}  Spread: {spread:.1f}')
        print(sep)


def main(args=None):
    rclpy.init(args=args)
    node = Dashboard()
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
