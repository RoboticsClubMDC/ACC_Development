#!/usr/bin/env python3
"""
ride_dispatch.py — interactive ACC 2026 ride driver (raw ros2 param-set).

A RIDE is always:   HUB → PICKUP → DROPOFF → HUB   (start and end at the HUB).
The program ASKS which ride to run (A–X), runs it, then asks again.

Each leg is dispatched as a TEMP-START route `node_values=[-1, dest]`: the
leading -1 makes path_follower read the live AMCL-corrected /qcar2_pose_fused,
drop a temp node at the current pose, and A* a path to `dest`, then the
arrival-align seats the car on `dest`. (So this script never needs node poses
for driving — path_follower owns that.)

STOP nodes (rides with a 3rd entry, e.g. L: 20,7,STOP=1): per the rules, when
the car PASSES that node during the ride it must stop — /motion_enable OFF,
LED RED, wait 3 s, /motion_enable ON, continue. We detect "passing" by
distance from /robot_pose to the STOP node's map pose (computed from the
SDCSRoadMap with the same rotation/translation offsets path_follower uses).

LED loop (qcar2_hardware led_color_id) — GREEN = driving; color changes only
AT events:
    MAGENTA(5) = parked at HUB (start + after each ride)
    GREEN(1)   = a ride is active / driving any leg
    BLUE(2)    = stopped at PICKUP
    ORANGE(6)  = stopped at DROPOFF
    RED(0)     = STOP node (motion gate off, 3 s)
Loop: MAGENTA → [GREEN→BLUE→GREEN→ORANGE→GREEN→MAGENTA] per ride, repeat.

Usage (after the AMCL stack + path_follower are up, e.g. via carto_to_amcl.sh):
    python3 Development/ros2/scripts/ride_dispatch.py
"""

import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterType
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped

from hal.products.mats import SDCSRoadMap


HUB               = 10
PAUSE_PICKUP_SEC  = 2.0
PAUSE_DROPOFF_SEC = 2.0
PARK_SETTLE_SEC   = 2.0
LEG_TIMEOUT_SEC   = 180.0
STOP_PAUSE_SEC    = 3.0     # /motion_enable OFF dwell at a STOP node
STOP_RADIUS_M     = 0.40    # "passing" the STOP node
STOP_REARM_M      = 0.65    # must leave this radius before STOP can re-fire

# On HUB arrival, reset AMCL to the HUB's IDEAL node pose to bound per-lap
# drift — but ONLY if the car is already within this range of it (trust but
# verify: don't snap-lie if AMCL ended up far off).
RESEED_HUB           = True
RESEED_MAX_RANGE_M   = 0.40

# Re-approach safety: if dist_to_final is still beyond tol after arrival,
# re-dispatch the same node. pickup/dropoff just need to GET there (align is
# OFF on those legs), so a loose tol avoids churn; the HUB seats precisely.
REACH_TOL_M            = 0.35    # pickup / dropoff — position loose (heading-only align does the squaring)
REAPPROACH_TOL_HUB_M   = 0.30    # HUB (wide anchor, then arrival-align seats it)
MAX_REAPPROACH         = 3

# LED IDs (qcar2_hardware.cpp lines 333-346)
LED_RED     = 0
LED_GREEN   = 1
LED_BLUE    = 2
LED_MAGENTA = 5
LED_ORANGE  = 6

# Official ACC 2026 ride list. 2-entry = (pickup, dropoff); 3-entry adds a
# STOP node the car must halt at when it passes during the ride.
RIDES = {
    'A': {'pickup': 1,  'dropoff': 8},
    'B': {'pickup': 6,  'dropoff': 18},
    'C': {'pickup': 2,  'dropoff': 6},
    'D': {'pickup': 13, 'dropoff': 6},
    'E': {'pickup': 1,  'dropoff': 11},
    'F': {'pickup': 14, 'dropoff': 11},
    'G': {'pickup': 8,  'dropoff': 18},
    'H': {'pickup': 4,  'dropoff': 0},
    'I': {'pickup': 14, 'dropoff': 22},
    'J': {'pickup': 14, 'dropoff': 24},
    'K': {'pickup': 24, 'dropoff': 22},
    'L': {'pickup': 20, 'dropoff': 7,  'stop': 1},
    'M': {'pickup': 4,  'dropoff': 1,  'stop': 8},
    'N': {'pickup': 13, 'dropoff': 3},
    'O': {'pickup': 24, 'dropoff': 20},
    'P': {'pickup': 9,  'dropoff': 15},
    'Q': {'pickup': 7,  'dropoff': 23, 'stop': 11},
    'R': {'pickup': 18, 'dropoff': 5},
    'S': {'pickup': 15, 'dropoff': 22},
    'T': {'pickup': 10, 'dropoff': 0,  'stop': 23},
    'U': {'pickup': 15, 'dropoff': 7,  'stop': 13},
    'V': {'pickup': 23, 'dropoff': 14, 'stop': 2},
    'W': {'pickup': 25, 'dropoff': 8},
    'X': {'pickup': 24, 'dropoff': 0,  'stop': 25},
}


class RideDispatcher(Node):
    def __init__(self):
        super().__init__('ride_dispatch')

        # Same QLabs→map transform path_follower/trip_planner use. Defaults
        # match path_follower's rotation_offset=83 / translation_offset=[0,0];
        # override here if you change path_follower's.
        self.declare_parameter('rotation_offset', 83.0)
        self.declare_parameter('translation_offset', [0.0, 0.0])
        self.rot = float(self.get_parameter('rotation_offset').value)
        self.trans = list(self.get_parameter('translation_offset').value)
        self.roadmap = SDCSRoadMap()

        self.pf_cli = self.create_client(
            SetParameters, '/path_follower/set_parameters')
        self.hw_cli = self.create_client(
            SetParameters, '/qcar2_hardware/set_parameters')
        self._wait_for_svc(self.pf_cli, 'path_follower')
        self._wait_for_svc(self.hw_cli, 'qcar2_hardware')

        self.path_complete = False
        self.dist_to_final = float('inf')
        self.robot_xy = None
        self._stop_armed = True

        self.motion_pub = self.create_publisher(Bool, '/motion_enable', 1)
        self.initpose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 1)
        self.create_subscription(Bool, '/path_status', self._path_cb, 10)
        self.create_subscription(Float32, '/nav/distance_to_final',
                                 self._dist_cb, 10)
        self.create_subscription(PoseStamped, '/robot_pose', self._pose_cb, 10)
        time.sleep(0.5)   # DDS discovery settle
        self._publish_motion_enable(True)   # default: motion allowed

    # ── callbacks ─────────────────────────────────────────────────────────
    def _path_cb(self, msg):  self.path_complete = bool(msg.data)
    def _dist_cb(self, msg):  self.dist_to_final = float(msg.data)
    def _pose_cb(self, msg):
        self.robot_xy = (float(msg.pose.position.x), float(msg.pose.position.y))

    def _wait_for_svc(self, cli, name):
        self.get_logger().info(f'Waiting for /{name}/set_parameters...')
        while not cli.wait_for_service(timeout_sec=3.0):
            self.get_logger().info(f'  still waiting for {name}...')
        self.get_logger().info(f'  /{name}/set_parameters ready.')

    # ── transforms ────────────────────────────────────────────────────────
    def _node_map_pose(self, node_id):
        """Node (x, y, yaw) in the ROS map frame: p_ros = (p_qlabs + t) @ R(-a),
        yaw_ros = yaw_qlabs + a (same convention as path_follower)."""
        a = self.rot * math.pi / 180.0
        R = np.array([[math.cos(-a), -math.sin(-a)],
                      [math.sin(-a),  math.cos(-a)]])
        pose = np.array(self.roadmap.nodes[int(node_id)].pose).reshape(-1)
        t = np.array([self.trans[0], self.trans[1]])
        p = (np.array([pose[0], pose[1]]) + t) @ R
        yaw = (float(pose[2]) + a + math.pi) % (2.0 * math.pi) - math.pi
        return float(p[0]), float(p[1]), yaw

    def _node_map_xy(self, node_id):
        x, y, _ = self._node_map_pose(node_id)
        return x, y

    def _reseed_hub(self):
        """Reset AMCL to the HUB's IDEAL node pose to bound per-lap drift —
        but ONLY if the car is already within RESEED_MAX_RANGE_M of it (trust
        but verify; never snap-lie if AMCL ended up far off)."""
        if not RESEED_HUB:
            return
        hx, hy, hyaw = self._node_map_pose(HUB)
        if self.robot_xy is None:
            self.get_logger().warn('  HUB reseed skipped — no /robot_pose yet.')
            return
        d = math.hypot(self.robot_xy[0] - hx, self.robot_xy[1] - hy)
        if d > RESEED_MAX_RANGE_M:
            self.get_logger().warn(
                f'  HUB reseed SKIPPED — {d:.2f} m off ideal (> {RESEED_MAX_RANGE_M}); '
                f'not snapping (AMCL may be lost — consider re-mapping).')
            return
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = hx
        msg.pose.pose.position.y = hy
        msg.pose.pose.orientation.z = math.sin(hyaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(hyaw / 2.0)
        msg.pose.covariance = [
            0.05, 0.0,  0.0, 0.0, 0.0, 0.0,
            0.0,  0.05, 0.0, 0.0, 0.0, 0.0,
            0.0,  0.0,  0.0, 0.0, 0.0, 0.0,
            0.0,  0.0,  0.0, 0.0, 0.0, 0.0,
            0.0,  0.0,  0.0, 0.0, 0.0, 0.0,
            0.0,  0.0,  0.0, 0.0, 0.0, 0.02]
        for _ in range(5):
            self.initpose_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info(
            f'  HUB reseed → ideal node {HUB} pose ({hx:.3f},{hy:.3f}); '
            f'was {d:.2f} m off — drift reset.')

    # ── raw param-set helpers ─────────────────────────────────────────────
    def _set_node_values(self, dest, label):
        """Temp-start route [-1, dest] — drive from the current pose to dest."""
        p = Parameter()
        p.name = 'node_values'
        p.value.type = ParameterType.PARAMETER_INTEGER_ARRAY
        p.value.integer_array_value = [-1, int(dest)]
        req = SetParameters.Request(); req.parameters = [p]
        future = self.pf_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is None or not future.result().results \
                or not future.result().results[0].successful:
            self.get_logger().error(f'  [{label}] node_values set FAILED')
            return False
        self.get_logger().info(f'  [{label}] node_values = [-1, {dest}]')
        return True

    def _set_led(self, color_id, label):
        p = Parameter()
        p.name = 'led_color_id'
        p.value.type = ParameterType.PARAMETER_INTEGER
        p.value.integer_value = int(color_id)
        req = SetParameters.Request(); req.parameters = [p]
        future = self.hw_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        self.get_logger().info(f'  LED → {label} ({color_id})')

    def _set_pf_bool(self, name, val, label):
        p = Parameter()
        p.name = name
        p.value.type = ParameterType.PARAMETER_BOOL
        p.value.bool_value = bool(val)
        req = SetParameters.Request(); req.parameters = [p]
        future = self.pf_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        self.get_logger().info(f'  [{label}] {name} = {val}')

    def _publish_motion_enable(self, allow):
        self.motion_pub.publish(Bool(data=bool(allow)))

    # ── STOP-node handling ────────────────────────────────────────────────
    def _maybe_stop(self, stop_xy, led_id, led_label):
        """If the car is passing the STOP node, halt via /motion_enable for
        STOP_PAUSE_SEC (LED RED), then resume + restore the leg LED."""
        if stop_xy is None or self.robot_xy is None:
            return
        d = math.hypot(self.robot_xy[0] - stop_xy[0],
                       self.robot_xy[1] - stop_xy[1])
        if d < STOP_RADIUS_M and self._stop_armed:
            self._stop_armed = False
            self.get_logger().info(
                f'  STOP node reached (d={d:.2f}) → motion OFF, RED, {STOP_PAUSE_SEC}s.')
            self._publish_motion_enable(False)
            self._set_led(LED_RED, 'RED (STOP)')
            t0 = time.time()
            while rclpy.ok() and time.time() - t0 < STOP_PAUSE_SEC:
                rclpy.spin_once(self, timeout_sec=0.1)
            self._publish_motion_enable(True)
            self._set_led(led_id, led_label)
            self.get_logger().info('  STOP done → motion ON, resuming.')
        elif d > STOP_REARM_M:
            self._stop_armed = True

    # ── leg driving ───────────────────────────────────────────────────────
    def _wait_arrival(self, stop_xy, led_id, led_label):
        # New path started: wait for path_status to drop to False.
        deadline = time.time() + 5.0
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            self._maybe_stop(stop_xy, led_id, led_label)
            if not self.path_complete:
                break
        # Then wait for True (arrived + aligned), monitoring the STOP node.
        deadline = time.time() + LEG_TIMEOUT_SEC
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)
            self._maybe_stop(stop_xy, led_id, led_label)
            if self.path_complete:
                return True
        self.get_logger().error('  TIMEOUT waiting for path_status=True')
        return False

    def _drive_to(self, dest, label, tol_m, led_id, led_label, stop_xy):
        self.dist_to_final = float('inf')
        self._set_node_values(dest, label)
        if not self._wait_arrival(stop_xy, led_id, led_label):
            return
        for attempt in range(MAX_REAPPROACH):
            for _ in range(5):
                rclpy.spin_once(self, timeout_sec=0.1)
                self._maybe_stop(stop_xy, led_id, led_label)
            d = self.dist_to_final
            if d <= tol_m:
                self.get_logger().info(
                    f'  [{label}] arrived node {dest} (dist_to_final={d:.3f} ≤ {tol_m}).')
                return
            self.get_logger().warn(
                f'  [{label}] offset {d:.3f} > {tol_m} — re-approach {attempt+1}/{MAX_REAPPROACH}')
            self.dist_to_final = float('inf')
            self._set_node_values(dest, f'{label} re-approach')
            self._wait_arrival(stop_xy, led_id, led_label)
        self.get_logger().warn(
            f'  [{label}] gave up balancing (dist_to_final={self.dist_to_final:.3f}). Proceeding.')

    # ── ride protocol ─────────────────────────────────────────────────────
    def _run_one_ride(self, letter):
        ride = RIDES[letter]
        pickup, dropoff = ride['pickup'], ride['dropoff']
        stop_node = ride.get('stop')
        stop_xy = self._node_map_xy(stop_node) if stop_node is not None else None
        self._stop_armed = True

        print('\n' + '═' * 60)
        msg = f'  START RIDE {letter} — pickup={pickup}, dropoff={dropoff}, hub={HUB}'
        if stop_node is not None:
            msg += f', STOP@{stop_node}'
        print(msg)
        print('═' * 60)

        # LED loop (per spec): GREEN = driving (a ride is active). The color
        # only changes AT events — BLUE at pickup, ORANGE at dropoff, MAGENTA
        # at the HUB, RED during a STOP. So every driving leg restores to GREEN
        # after a STOP.
        DRIVE = (LED_GREEN, 'GREEN (driving)')

        # Pickup/dropoff: HEADING-ONLY align — PP gets there, then it just
        # squares to the node's heading (≈0° error) without seating the exact
        # (x,y). Only the HUB does the full position+heading seat.
        self._set_pf_bool('arrival_align', True, f'{letter} pickup/dropoff')
        self._set_pf_bool('arrival_align_heading_only', True, f'{letter} pickup/dropoff')

        # HUB → PICKUP (GREEN driving) → arrive + heading-align → BLUE dwell
        self._set_led(*DRIVE)
        self._drive_to(pickup, f'{letter}.1 HUB→PICKUP', REACH_TOL_M,
                       DRIVE[0], DRIVE[1], stop_xy)
        self._set_led(LED_BLUE, 'BLUE (at PICKUP)')
        time.sleep(PAUSE_PICKUP_SEC)

        # PICKUP → DROPOFF (GREEN driving) → arrive + heading-align → ORANGE dwell
        self._set_led(*DRIVE)
        self._drive_to(dropoff, f'{letter}.2 PICKUP→DROPOFF', REACH_TOL_M,
                       DRIVE[0], DRIVE[1], stop_xy)
        self._set_led(LED_ORANGE, 'ORANGE (at DROPOFF)')
        time.sleep(PAUSE_DROPOFF_SEC)

        # DROPOFF → HUB — FULL seat (position + heading) at the HUB.
        self._set_pf_bool('arrival_align_heading_only', False, f'{letter} HUB')
        self._set_led(*DRIVE)
        self._drive_to(HUB, f'{letter}.3 DROPOFF→HUB', REAPPROACH_TOL_HUB_M,
                       DRIVE[0], DRIVE[1], stop_xy)
        # Reset AMCL to the ideal HUB pose (bounds per-lap drift; gated to
        # RESEED_MAX_RANGE_M so we never snap-lie if AMCL is far off).
        self._reseed_hub()
        # AT HUB — MAGENTA (parked, ready for the next ride)
        self._set_led(LED_MAGENTA, 'MAGENTA (parked at HUB)')

        print('═' * 60)
        print(f'  END RIDE {letter} — parked at HUB ({HUB}), MAGENTA.')
        print('═' * 60)

    def run(self):
        # Already parked at HUB from carto_to_amcl.sh / amcl_load.sh; settle.
        self._set_led(LED_MAGENTA, 'MAGENTA (ready at HUB)')
        time.sleep(PARK_SETTLE_SEC)
        print('\nReady at HUB. Enter a ride letter to dispatch.')

        while rclpy.ok():
            try:
                choice = input(
                    '\n>>> Which ride? [A-X] (or "quit"): ').strip().upper()
            except EOFError:
                break
            if choice in ('QUIT', 'EXIT', 'Q!'):
                break
            if choice not in RIDES:
                print(f'  "{choice}" is not a valid ride. Options: '
                      f'{", ".join(sorted(RIDES))}')
                continue
            self._run_one_ride(choice)


def main():
    rclpy.init()
    node = RideDispatcher()
    try:
        node.run()
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
