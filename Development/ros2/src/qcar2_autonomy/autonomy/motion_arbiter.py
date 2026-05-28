#!/usr/bin/env python3
"""
motion_arbiter — the SINGLE owner of /cmd_vel_nav during competition.

Replaces cmd_vel_blender. Instead of a blind 60/40 average, it enforces a
PRIORITY LADDER and applies gyro D-damping ONCE on the final command. This
fixes three problems at once:

  1. PP-vs-Stanley fight through the IMU. path_follower used to apply
     `−kd·gyro` itself, but the gyro reflects the BLENDED motion (mostly
     Stanley), so PP damped against a turn it didn't command. Now path_follower
     emits raw kp·pp (apply_gyro_damping:=false) and the arbiter is the single
     authority that damps the final blended steering — legitimate, no fight.

  2. Double-publisher risk on /cmd_vel_nav. The arbiter is the only publisher.

  3. Disconnected stop pipeline. perception publishes /perception/stop_required
     and nobody consumed it; /motion_enable had no publisher. The arbiter
     consumes BOTH (plus /car_stop e-stop) and turns them into an actual stop.

Priority ladder (highest first):
  1. ESTOP            (/car_stop True)                  → zero command
  2. STOP_REQUIRED    (/perception/stop_required True
                       OR /motion_enable False)         → zero speed, zero steer
  3. TURN/INTERSECTION (path |steer| large, lane stale) → PATH ONLY
  4. BLEND            (both fresh)                       → w_path·path + w_lane·lane
  5. PATH_ONLY        (lane stale)                       → path
  6. LANE_ONLY        (path stale, require_path=false)   → lane
  7. IDLE             (nothing fresh / require_path)     → zero

Then: final_steer = clip(blend_steer − kd·gyro_filtered, ±max_steer)
      final_speed = path speed (linear_source=path) [or blend/min]

Inputs:
  /cmd_vel_path           Twist  (path_follower, RAW kp·pp — set its
                                  apply_gyro_damping:=false)
  /cmd_vel_lane           Twist  (lane_stanley_controller)
  /qcar2_imu              Imu    (gyro yaw rate for D-damping)
  /car_stop               Bool   (emergency stop)
  /perception/stop_required Bool (stop sign / red light)
  /motion_enable          Bool   (legacy stop gate; True=go)
Output:
  /cmd_vel_nav            Twist  (THE command to nav2_qcar_command_convert)
  /arbiter/mode           String (diagnostic: current ladder rung)
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, String


class MotionArbiter(Node):
    def __init__(self):
        super().__init__('motion_arbiter')

        # Topics
        self.declare_parameter('path_cmd_topic', '/cmd_vel_path')
        self.declare_parameter('lane_cmd_topic', '/cmd_vel_lane')
        self.declare_parameter('out_cmd_topic', '/cmd_vel_nav')
        self.declare_parameter('imu_topic', '/qcar2_imu')
        self.declare_parameter('estop_topic', '/car_stop')
        self.declare_parameter('stop_required_topic', '/perception/stop_required')
        self.declare_parameter('motion_enable_topic', '/motion_enable')

        # Blend + damping. 2026-05-28: LANE-PRIMARY on open road (lane leads,
        # path is a gentle bias), PATH-ONLY at junctions (gate closed). This
        # inverts the old path-primary blend to match how SDCS driving works:
        # follow the painted lane between nodes, let A*/PP make the turns at
        # marked junction nodes where the lane is ambiguous/absent.
        self.declare_parameter('path_weight', 0.25)   # was 0.40
        self.declare_parameter('lane_weight', 0.75)   # was 0.60 — lane leads
        self.declare_parameter('kd_steering', 0.20)      # moved from path_follower
        self.declare_parameter('max_steer_rad', 0.55)
        self.declare_parameter('gyro_filter_alpha', 0.2)  # 1st-order LP on gyro
        # /nav/lane_gate (Bool): True (default) = lane allowed → lane-primary
        # blend; False = at/near a junction node → PATH ONLY (ignore lane so a
        # straight-lane reading can't fight the turn). Published by the
        # dispatcher / path_follower based on proximity to junction_nodes.
        self.declare_parameter('lane_gate_topic', '/nav/lane_gate')

        # Behavior
        self.declare_parameter('require_path', True)
        self.declare_parameter('linear_source', 'path')   # path | blend | min
        self.declare_parameter('cmd_timeout_sec', 0.35)
        self.declare_parameter('control_rate_hz', 50.0)
        # If path |steer| exceeds this AND lane is stale, treat as a turn and
        # go PATH-ONLY (don't let a hallucinated straight lane fight the turn).
        self.declare_parameter('turn_steer_thresh', 0.30)

        self.path_cmd = Twist()
        self.lane_cmd = Twist()
        self.path_time = None
        self.lane_time = None
        self.estop = False
        self.stop_required = False
        self.motion_enabled = True
        self.lane_gate_open = True   # True = lane allowed; False = junction → path-only
        self.gyro_yaw = 0.0
        self.gyro_filt = 0.0
        self._last_mode = None

        self.pub = self.create_publisher(
            Twist, self.get_parameter('out_cmd_topic').value, 1)
        self.pub_mode = self.create_publisher(String, '/arbiter/mode', 10)

        self.create_subscription(
            Twist, self.get_parameter('path_cmd_topic').value, self._path_cb, 1)
        self.create_subscription(
            Twist, self.get_parameter('lane_cmd_topic').value, self._lane_cb, 1)
        self.create_subscription(
            Imu, self.get_parameter('imu_topic').value, self._imu_cb, 20)
        self.create_subscription(
            Bool, self.get_parameter('estop_topic').value, self._estop_cb, 10)
        self.create_subscription(
            Bool, self.get_parameter('stop_required_topic').value,
            self._stop_required_cb, 10)
        self.create_subscription(
            Bool, self.get_parameter('motion_enable_topic').value,
            self._motion_enable_cb, 10)
        self.create_subscription(
            Bool, self.get_parameter('lane_gate_topic').value,
            self._lane_gate_cb, 10)

        rate = float(self.get_parameter('control_rate_hz').value)
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self._tick)

        self.get_logger().info(
            'motion_arbiter ready — SINGLE owner of /cmd_vel_nav. '
            'Run path_follower with apply_gyro_damping:=false and do NOT '
            'run cmd_vel_blender alongside this node.')

    # ── callbacks ──────────────────────────────────────────────────────────
    def _path_cb(self, msg):
        self.path_cmd = msg
        self.path_time = self.get_clock().now()

    def _lane_cb(self, msg):
        self.lane_cmd = msg
        self.lane_time = self.get_clock().now()

    def _imu_cb(self, msg):
        z = float(msg.angular_velocity.z)
        if math.isfinite(z):
            self.gyro_yaw = z

    def _estop_cb(self, msg):
        e = bool(msg.data)
        if e != self.estop:
            self.get_logger().warn(f'E-STOP {"ENGAGED" if e else "released"}')
        self.estop = e

    def _stop_required_cb(self, msg):
        self.stop_required = bool(msg.data)

    def _motion_enable_cb(self, msg):
        self.motion_enabled = bool(msg.data)

    def _lane_gate_cb(self, msg):
        self.lane_gate_open = bool(msg.data)

    # ── helpers ─────────────────────────────────────────────────────────────
    def _fresh(self, stamp, now, timeout):
        return stamp is not None and (now - stamp).nanoseconds * 1e-9 <= timeout

    def _set_mode(self, mode):
        if mode != self._last_mode:
            self._last_mode = mode
            self.pub_mode.publish(String(data=mode))

    # ── main tick ────────────────────────────────────────────────────────────
    def _tick(self):
        now = self.get_clock().now()
        timeout = float(self.get_parameter('cmd_timeout_sec').value)
        path_ok = self._fresh(self.path_time, now, timeout)
        lane_ok = self._fresh(self.lane_time, now, timeout)

        # Low-pass the gyro (matches path_follower's filtered D-term feel).
        a = float(self.get_parameter('gyro_filter_alpha').value)
        self.gyro_filt = (1.0 - a) * self.gyro_filt + a * self.gyro_yaw

        cmd = Twist()

        # ── PRIORITY LADDER ──
        # 1. E-STOP
        if self.estop:
            self._set_mode('ESTOP')
            self.pub.publish(cmd)
            return

        # 2. STOP REQUIRED (perception stop sign / red light / motion gate)
        if self.stop_required or (not self.motion_enabled):
            self._set_mode('STOP_REQUIRED')
            self.pub.publish(cmd)   # zero speed, zero steer
            return

        # 3. require_path gate — don't move on lane alone
        require_path = bool(self.get_parameter('require_path').value)
        if require_path and not path_ok:
            self._set_mode('IDLE_NO_PATH')
            self.pub.publish(cmd)
            return

        if not path_ok and not lane_ok:
            self._set_mode('IDLE')
            self.pub.publish(cmd)
            return

        path_w = float(self.get_parameter('path_weight').value)
        lane_w = float(self.get_parameter('lane_weight').value)
        total = max(path_w + lane_w, 1e-6)
        path_w /= total
        lane_w /= total

        path_steer = float(self.path_cmd.angular.z)
        lane_steer = float(self.lane_cmd.angular.z)
        turn_thresh = float(self.get_parameter('turn_steer_thresh').value)

        # 4/5/6. Steering source selection — gated by /nav/lane_gate.
        if not self.lane_gate_open:
            # JUNCTION: gate closed by the dispatcher (near a junction node).
            # Lane is ambiguous/absent here — let PP make the geometric turn.
            steer_blend = path_steer
            self._set_mode('JUNCTION_PATH_ONLY')
        elif path_ok and lane_ok:
            # Open road, both fresh → LANE-PRIMARY blend (lane leads, path bias).
            # Safety override: if PP is turning hard while the lane reads nearly
            # straight, trust PP — a stale/hallucinated lane shouldn't fight a
            # real turn even with the gate open.
            if abs(path_steer) >= turn_thresh and abs(lane_steer) < 0.5 * turn_thresh:
                steer_blend = path_steer
                self._set_mode('TURN_PATH_ONLY')
            else:
                steer_blend = path_w * path_steer + lane_w * lane_steer
                self._set_mode('LANE_PRIMARY_BLEND')
        elif path_ok:
            steer_blend = path_steer
            self._set_mode('PATH_ONLY')
        else:
            steer_blend = lane_steer
            self._set_mode('LANE_ONLY')

        # ── SINGLE gyro D-damp on the final command (the whole point) ──
        kd = float(self.get_parameter('kd_steering').value)
        max_steer = float(self.get_parameter('max_steer_rad').value)
        steer = float(np.clip(steer_blend - kd * self.gyro_filt,
                              -max_steer, max_steer))

        # ── Speed ──
        linear_source = str(self.get_parameter('linear_source').value)
        if linear_source == 'lane':
            speed = self.lane_cmd.linear.x
        elif linear_source == 'blend':
            speed = path_w * self.path_cmd.linear.x + lane_w * self.lane_cmd.linear.x
        elif linear_source == 'min':
            speed = min(self.path_cmd.linear.x, self.lane_cmd.linear.x)
        else:
            speed = self.path_cmd.linear.x

        cmd.linear.x = float(speed)
        cmd.angular.z = steer
        self.pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = MotionArbiter()
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
