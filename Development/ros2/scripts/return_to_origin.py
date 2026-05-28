#!/usr/bin/env python3
"""
return_to_origin.py — drive the QCar to the cartographer map origin POSE
(0, 0, 0) — position AND heading — for a bullet-proof AMCL seed.

The map origin is the only drift-free point (it's Cartographer's reference),
and (0,0,0) is the boot pose. AMCL must be seeded with the FULL pose, so the
car has to arrive both AT the origin AND facing map-yaw 0. Arriving at the
right spot but the wrong heading gives AMCL a wrong orientation → corrections
get rejected → the node-8-style mislocalization bug.

Controller — kinematic bicycle pose regulation + 3-point-turn alignment:

  APPROACH  (rho > pos_tol): pose-regulation law steers the car toward the
            target POSE, naturally aligning heading as it nears. Within
            straight_in_radius (~2 m) the heading term dominates so the last
            stretch is a near-straight, aligned run-in.
                rho   = ||(x, y)||
                gamma = atan2(-y, -x)         # world bearing to origin
                alpha = wrap(gamma - theta)   # heading error toward origin
                beta  = wrap(target_yaw - gamma)  # final-heading error
                omega = k_alpha*alpha + k_beta*beta
                delta = atan2(omega*L, v)     # bicycle: omega = v*tan(delta)/L
            speed tapers with rho; cut hard if alpha large (turn first).

  ALIGN     (rho <= pos_tol but |theta - target_yaw| > yaw_tol): Ackermann
            can't spin in place, so do a 3-point turn — forward with full
            steer one way, reverse with full steer the other way → net
            rotation, ~zero net translation. Repeat until heading is within
            yaw_tol. After each cycle, re-check: if position drifted out of
            pos_tol, go back to APPROACH.

  DONE      rho <= pos_tol AND |theta - target_yaw| <= yaw_tol → stop, exit 0.

Reads map->base_link TF; publishes /cmd_vel_nav (steer in angular.z, speed in
linear.x; reverse = negative linear.x).
"""

import argparse
import math
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.utilities import remove_ros_args
from tf2_ros import Buffer, TransformListener

L = 0.256   # wheelbase (QCar 2)


def wrap_to_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class ReturnToOrigin(Node):
    def __init__(self, pos_tol, yaw_tol, vmax, target_yaw, timeout):
        super().__init__('return_to_origin')
        self.pos_tol = pos_tol
        self.yaw_tol = yaw_tol
        self.vmax = vmax
        self.target_yaw = target_yaw
        self.timeout = timeout

        # Pose-regulation gains (forward-only run-in). k_beta<0, k_alpha>k_rho.
        self.k_alpha = 1.3
        self.k_beta = -0.45
        self.straight_in_radius = 2.0   # within this, prioritise alignment
        self.v_min = 0.06
        self.max_steer = 0.50

        # 3-point-turn alignment params. 2026-05-28: shorter segments + lower
        # speed so the FORWARD→BACKWARD cycle fine-tunes to the exact node-0
        # pose instead of over-rotating past it. Tighter yaw_tol (see main())
        # ensures a near-but-not-exact arrival (e.g. 5.5°) does NOT short-circuit
        # to DONE — it runs the backward phase and seats precisely.
        self.wiggle_v = 0.10            # forward/back speed during alignment
        self.wiggle_seg_t = 0.7         # seconds per forward (or back) segment
        self.wiggle_pause_t = 0.3       # brief stop between segments

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_nav', 10)

        self.state = 'APPROACH'
        self.done = False
        self.t0 = time.time()
        # wiggle sub-state
        self._wig_phase = None          # 'FWD' | 'BACK' | 'PAUSE'
        self._wig_t0 = 0.0
        self._wig_dir = 1.0             # +1 = net CCW, -1 = net CW

        self.timer = self.create_timer(0.05, self._tick)   # 20 Hz

    def _pose(self):
        try:
            tf = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
        except Exception:
            return None
        x = tf.transform.translation.x
        y = tf.transform.translation.y
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return float(x), float(y), float(yaw)

    def _stop(self):
        self.cmd_pub.publish(Twist())

    def _tick(self):
        if time.time() - self.t0 > self.timeout:
            self.get_logger().error('return_to_origin TIMEOUT — stopping.')
            self._stop()
            self.done = True
            return

        p = self._pose()
        if p is None:
            self.get_logger().warn('waiting for map->base_link TF...',
                                   throttle_duration_sec=2.0)
            return
        x, y, yaw = p
        rho = math.hypot(x, y)
        yaw_err = wrap_to_pi(self.target_yaw - yaw)

        # ── Terminal check ──
        if rho <= self.pos_tol and abs(yaw_err) <= self.yaw_tol:
            self._stop()
            self.get_logger().info(
                f'SEED POSE reached: x={x:.3f} y={y:.3f} yaw={math.degrees(yaw):.1f}° '
                f'(pos={rho:.3f}≤{self.pos_tol}, yaw_err={math.degrees(yaw_err):.1f}°≤'
                f'{math.degrees(self.yaw_tol):.1f}°).')
            self.done = True
            return

        # ── State transitions ──
        if self.state == 'APPROACH' and rho <= self.pos_tol:
            self.state = 'ALIGN'
            self._wig_phase = None
            self.get_logger().info(
                f'At origin (pos={rho:.3f}) but yaw_err={math.degrees(yaw_err):.1f}° '
                f'> {math.degrees(self.yaw_tol):.1f}° → 3-point-turn ALIGN.')
        elif self.state == 'ALIGN' and rho > self.pos_tol + 0.05:
            self.state = 'APPROACH'   # wiggle drifted us out; re-approach
            self.get_logger().info(f'Alignment drifted (pos={rho:.3f}) → re-APPROACH.')

        if self.state == 'APPROACH':
            self._approach(x, y, yaw, rho)
        else:
            self._align(yaw_err)

    def _approach(self, x, y, yaw, rho):
        gamma = math.atan2(-y, -x)             # world bearing to origin
        alpha = wrap_to_pi(gamma - yaw)        # heading error toward origin
        beta = wrap_to_pi(self.target_yaw - gamma)

        # Speed: taper with distance; floor so we keep creeping.
        if rho > self.straight_in_radius:
            v = self.vmax
        else:
            v = max(self.v_min, self.vmax * (rho / self.straight_in_radius))
        # Turn toward goal before charging at it.
        if abs(alpha) > 0.5:
            v *= 0.4

        omega = self.k_alpha * alpha + self.k_beta * beta
        delta = math.atan2(omega * L, max(v, self.v_min))
        delta = float(np.clip(delta, -self.max_steer, self.max_steer))

        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = delta
        self.cmd_pub.publish(cmd)
        self.get_logger().info(
            f'APPROACH rho={rho:.2f} a={math.degrees(alpha):.0f}° '
            f'b={math.degrees(beta):.0f}° v={v:.2f} steer={delta:.2f}',
            throttle_duration_sec=0.5)

    def _align(self, yaw_err):
        """3-point turn: net-rotate toward target_yaw with ~zero net translation.
        Forward + full steer one way, reverse + full steer the other way."""
        now = time.time()
        # Start / restart a cycle if not mid-phase.
        if self._wig_phase is None:
            self._wig_dir = 1.0 if yaw_err > 0 else -1.0   # +1 = CCW
            self._wig_phase = 'FWD'
            self._wig_t0 = now
            self.get_logger().info(
                f'ALIGN cycle: yaw_err={math.degrees(yaw_err):.1f}° '
                f'dir={"CCW" if self._wig_dir > 0 else "CW"}')

        cmd = Twist()
        # CCW (dir +1): forward+left(+steer), reverse+right(-steer).
        # CW  (dir -1): forward+right(-steer), reverse+left(+steer).
        if self._wig_phase == 'FWD':
            cmd.linear.x = self.wiggle_v
            cmd.angular.z = self._wig_dir * self.max_steer
            if now - self._wig_t0 >= self.wiggle_seg_t:
                self._wig_phase = 'PAUSE'
                self._wig_t0 = now
        elif self._wig_phase == 'BACK':
            cmd.linear.x = -self.wiggle_v
            cmd.angular.z = -self._wig_dir * self.max_steer
            if now - self._wig_t0 >= self.wiggle_seg_t:
                self._wig_phase = 'PAUSE_END'
                self._wig_t0 = now
        elif self._wig_phase == 'PAUSE':
            cmd = Twist()
            if now - self._wig_t0 >= self.wiggle_pause_t:
                self._wig_phase = 'BACK'
                self._wig_t0 = now
        else:  # PAUSE_END — finished one full FWD/BACK cycle; re-evaluate next tick
            cmd = Twist()
            if now - self._wig_t0 >= self.wiggle_pause_t:
                self._wig_phase = None   # terminal check / next cycle decides
        self.cmd_pub.publish(cmd)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos-tol', type=float, default=0.06)
    parser.add_argument('--yaw-tol', type=float, default=0.045)  # rad (~2.6°) — exact seat
    parser.add_argument('--vmax', type=float, default=0.35)
    parser.add_argument('--target-yaw', type=float, default=0.0) # map-frame heading at origin
    # 40 s budget: mapping lap is ~2:40, the return+align gets 40 s. If it
    # can't fully seat (0,0,0) in time it stops and seeds the best pose it has.
    parser.add_argument('--timeout', type=float, default=40.0)
    parsed = parser.parse_args(remove_ros_args(args=sys.argv)[1:])

    rclpy.init(args=argv)
    node = ReturnToOrigin(parsed.pos_tol, parsed.yaw_tol, parsed.vmax,
                          parsed.target_yaw, parsed.timeout)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
        node.cmd_pub.publish(Twist())
        time.sleep(0.2)
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
