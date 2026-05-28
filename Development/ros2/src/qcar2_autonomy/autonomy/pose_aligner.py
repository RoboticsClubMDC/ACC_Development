#!/usr/bin/env python3
"""
pose_aligner — reusable kinematic-bicycle "seat exactly on a target POSE"
maneuver, extracted from return_to_origin.py so path_follower (HUB / node
arrival) and the carto→AMCL return can share ONE implementation.

An Ackermann car cannot spin in place, so reaching a full pose (x, y, yaw) has
two phases:

  APPROACH (rho > pos_tol): pose-regulation steering law drives toward the
           target POSE, naturally aligning heading on the run-in.
               rho   = ||(x - tx, y - ty)||
               gamma = atan2(ty - y, tx - x)     # world bearing to target
               alpha = wrap(gamma - theta)        # heading error toward target
               beta  = wrap(target_yaw - gamma)   # final-heading error
               omega = k_alpha*alpha + k_beta*beta
               delta = atan2(omega*L, v)          # bicycle: omega = v*tan(d)/L

  ALIGN    (rho <= pos_tol, |yaw_err| > yaw_tol): 3-point turn — forward with
           full steer one way, reverse with full steer the other → net
           rotation, ~zero net translation. Repeat until heading is seated.

The class is pure control: feed it the current pose + target each tick, it
returns (linear_x, steer, done). linear_x may be NEGATIVE (reverse) during the
3-point turn — the caller must publish it without a non-negative clamp.
"""

import math
import time

L = 0.256   # QCar 2 wheelbase


def wrap_to_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class PoseAligner:
    def __init__(self, pos_tol=0.06, yaw_tol=0.045, vmax=0.30,
                 k_alpha=1.3, k_beta=-0.45, straight_in_radius=1.0,
                 v_min=0.06, max_steer=0.50,
                 wiggle_v=0.10, wiggle_seg_t=0.7, wiggle_pause_t=0.3):
        self.pos_tol = pos_tol
        self.yaw_tol = yaw_tol
        self.vmax = vmax
        self.k_alpha = k_alpha
        self.k_beta = k_beta
        self.straight_in_radius = straight_in_radius
        self.v_min = v_min
        self.max_steer = max_steer
        self.wiggle_v = wiggle_v
        self.wiggle_seg_t = wiggle_seg_t
        self.wiggle_pause_t = wiggle_pause_t
        self.reset()

    def reset(self):
        self.state = 'APPROACH'
        self._wig_phase = None       # 'PUSH' | 'PAUSE'
        self._wig_t0 = 0.0
        self._wig_dir = 1.0          # +1 = net CCW, -1 = net CW
        self._last_push_dir = -1     # so the first push prefers FORWARD
        self._cur_push_fwd = True
        self.stuck = False           # set True when boxed in by walls both ways

    def tick(self, x, y, theta, tx, ty, target_yaw, now=None,
             front_clear=float('inf'), rear_clear=float('inf'), wall_min=0.0):
        """Return (linear_x, steer, done) for the current pose vs target pose.

        front_clear / rear_clear are lidar clearances (m) ahead/behind; with
        wall_min>0 the maneuver avoids pushing toward a wall closer than
        wall_min. Defaults (inf, inf, 0.0) = no wall awareness (original
        behavior, used by return_to_origin.py)."""
        if now is None:
            now = time.time()
        dx = x - tx
        dy = y - ty
        rho = math.hypot(dx, dy)
        yaw_err = wrap_to_pi(target_yaw - theta)

        # ── Terminal ──
        if rho <= self.pos_tol and abs(yaw_err) <= self.yaw_tol:
            return 0.0, 0.0, True

        # ── State transitions ──
        if self.state == 'APPROACH' and rho <= self.pos_tol:
            self.state = 'ALIGN'
            self._wig_phase = None
        elif self.state == 'ALIGN' and rho > self.pos_tol + 0.05:
            self.state = 'APPROACH'   # wiggle drifted us out; re-approach

        if self.state == 'APPROACH':
            v, steer = self._approach(x, y, theta, tx, ty, target_yaw, rho)
            # Don't drive into a close front wall during the run-in.
            if v > 0.0 and front_clear < wall_min:
                v = 0.0
        else:
            v, steer = self._align(yaw_err, now, front_clear, rear_clear, wall_min)
        return v, steer, False

    def _approach(self, x, y, theta, tx, ty, target_yaw, rho):
        gamma = math.atan2(ty - y, tx - x)       # world bearing to target
        alpha = wrap_to_pi(gamma - theta)        # heading error toward target
        beta = wrap_to_pi(target_yaw - gamma)

        if rho > self.straight_in_radius:
            v = self.vmax
        else:
            v = max(self.v_min, self.vmax * (rho / self.straight_in_radius))
        if abs(alpha) > 0.5:
            v *= 0.4

        omega = self.k_alpha * alpha + self.k_beta * beta
        delta = math.atan2(omega * L, max(v, self.v_min))
        delta = max(-self.max_steer, min(self.max_steer, delta))
        return float(v), float(delta)

    def _align(self, yaw_err, now, front_clear, rear_clear, wall_min):
        """3-point turn: net-rotate toward target_yaw. A forward push and a
        reverse push BOTH rotate the same way, so when a wall blocks one
        direction we bias to the other — which also backs the car off the wall.
        Alternates directions when both are clear (≈zero net translation)."""
        if self._wig_phase is None:
            self._wig_dir = 1.0 if yaw_err > 0 else -1.0   # +1 = CCW
            fwd_ok = front_clear >= wall_min
            back_ok = rear_clear >= wall_min
            if not fwd_ok and not back_ok:
                self.stuck = True          # boxed in — caller will time out
                return 0.0, 0.0
            self.stuck = False
            # Prefer to alternate from the last push (zero net drift); if the
            # preferred direction is wall-blocked, take the open one.
            prefer_fwd = (self._last_push_dir <= 0)
            if prefer_fwd and fwd_ok:
                self._cur_push_fwd = True
            elif (not prefer_fwd) and back_ok:
                self._cur_push_fwd = False
            else:
                self._cur_push_fwd = fwd_ok
            self._last_push_dir = 1 if self._cur_push_fwd else -1
            self._wig_phase = 'PUSH'
            self._wig_t0 = now

        if self._wig_phase == 'PUSH':
            if self._cur_push_fwd:
                v = self.wiggle_v
                steer = self._wig_dir * self.max_steer
            else:
                v = -self.wiggle_v
                steer = -self._wig_dir * self.max_steer
            if now - self._wig_t0 >= self.wiggle_seg_t:
                self._wig_phase = 'PAUSE'
                self._wig_t0 = now
            return float(v), float(steer)

        # PAUSE — brief stop, then re-evaluate (terminal / next push direction).
        if now - self._wig_t0 >= self.wiggle_pause_t:
            self._wig_phase = None
        return 0.0, 0.0
