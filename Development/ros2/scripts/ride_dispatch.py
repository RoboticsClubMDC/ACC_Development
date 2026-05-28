#!/usr/bin/env python3
"""
ride_dispatch.py — raw ros2 param-set ride driver for ACC 2026.

Loops the ride order forever:   A → C → F → G → H → A → C → …
Every ride: HUB → PICKUP → DROPOFF → HUB.

LED scheme (per user spec 2026-05-27):
    MAGENTA (5)  = parked at HUB, AND driving DROPOFF→HUB
    GREEN   (1)  = driving HUB → PICKUP
    BLUE    (2)  = full stop at PICKUP node (3 s pause)
    ORANGE  (6)  = driving PICKUP → DROPOFF

Arrival detection: edge on /path_status (False → True).
Ride dispatch: raw `ros2 param set /path_follower node_values [N]`.
LED dispatch:   raw `ros2 param set /qcar2_hardware led_color_id <id>`.

Usage:
    python3 Development/ros2/scripts/ride_dispatch.py
    # Optional: pass a custom ride order (e.g., A C only):
    python3 Development/ros2/scripts/ride_dispatch.py A C
"""

import sys
import time

import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterType
from std_msgs.msg import Bool, Float32


HUB              = 10
PAUSE_PICKUP_SEC  = 3.0
PAUSE_DROPOFF_SEC = 3.0
PARK_SETTLE_SEC  = 2.0
LEG_TIMEOUT_SEC  = 180.0

# Fork-B "balance myself" re-approach: after path_follower reports arrival,
# if its /nav/distance_to_final is still beyond this, re-dispatch the same
# node so it re-plans from the (offset) current pose and nudges closer.
# This is closed-loop — trust the pose, physically reach the node, DON'T
# snap-lie. HUB uses the loose value (it's a wide anchor); pickup/dropoff
# use the tight value (judges measure these).
REAPPROACH_TOL_TIGHT_M = 0.10    # pickup / dropoff
REAPPROACH_TOL_HUB_M   = 0.30    # HUB
MAX_REAPPROACH         = 3       # give up after this many nudges (avoid loop)

# LED IDs (qcar2_hardware.cpp lines 333-346)
LED_GREEN   = 1
LED_BLUE    = 2
LED_ORANGE  = 6
LED_MAGENTA = 5

# Official ACC 2026 ride list (pickup, dropoff) — first/last node of each ride.
# Multi-stop rides (L, M, N, Q, T, U, V, X) are not in the default loop;
# their middle nodes are dynamic stops via /motion_enable per the rules.
RIDES = {
    'A': [1, 8],     'B': [6, 18],    'C': [2, 6],     'D': [13, 6],
    'E': [1, 11],    'F': [14, 11],   'G': [8, 18],    'H': [4, 0],
    'I': [14, 22],   'J': [14, 24],   'K': [24, 22],   'L': [20, 1],
    'M': [4, 8],     'N': [13, 3],    'O': [24, 20],   'P': [9, 15],
    'Q': [7, 11],    'R': [18, 5],    'S': [15, 22],   'T': [10, 23],
    'U': [15, 13],   'V': [23, 2],    'W': [25, 8],    'X': [24, 25],
}

# Default loop order: rides A, C, F, G, H.
DEFAULT_RIDE_ORDER = ['A', 'C', 'F', 'G', 'H']


class RideDispatcher(Node):
    def __init__(self):
        super().__init__('ride_dispatch')

        self.pf_cli = self.create_client(
            SetParameters, '/path_follower/set_parameters')
        self.hw_cli = self.create_client(
            SetParameters, '/qcar2_hardware/set_parameters')

        self._wait_for_svc(self.pf_cli, 'path_follower')
        self._wait_for_svc(self.hw_cli, 'qcar2_hardware')

        self.path_complete = False
        self.dist_to_final = float('inf')   # path_follower's frame-correct distance
        self.create_subscription(Bool, '/path_status', self._path_cb, 10)
        # /nav/distance_to_final is path_follower's Euclidean distance from the
        # current pose to the LAST waypoint, in the ROS map frame. We use it
        # for the Fork-B "balance myself" re-approach — no need for this script
        # to know node poses or frame offsets.
        self.create_subscription(Float32, '/nav/distance_to_final',
                                 self._dist_cb, 10)
        time.sleep(0.5)   # DDS discovery settle

    def _wait_for_svc(self, cli, name):
        self.get_logger().info(f'Waiting for /{name}/set_parameters service...')
        while not cli.wait_for_service(timeout_sec=3.0):
            self.get_logger().info(f'  still waiting for {name}...')
        self.get_logger().info(f'  /{name}/set_parameters ready.')

    def _path_cb(self, msg: Bool):
        self.path_complete = bool(msg.data)

    def _dist_cb(self, msg: Float32):
        self.dist_to_final = float(msg.data)

    # ── raw param-set helpers ─────────────────────────────────────────────

    def _set_node_values(self, node_id: int, label: str) -> bool:
        p = Parameter()
        p.name                       = 'node_values'
        p.value.type                 = ParameterType.PARAMETER_INTEGER_ARRAY
        p.value.integer_array_value  = [int(node_id)]
        req                          = SetParameters.Request()
        req.parameters               = [p]
        future = self.pf_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is None or not future.result().results:
            self.get_logger().error(f'  [{label}] param set FAILED')
            return False
        res = future.result().results[0]
        if not res.successful:
            self.get_logger().error(f'  [{label}] param rejected: {res.reason}')
            return False
        self.get_logger().info(f'  [{label}] node_values = [{node_id}]')
        return True

    def _set_led(self, color_id: int, label: str) -> bool:
        p = Parameter()
        p.name                  = 'led_color_id'
        p.value.type            = ParameterType.PARAMETER_INTEGER
        p.value.integer_value   = int(color_id)
        req                     = SetParameters.Request()
        req.parameters          = [p]
        future = self.hw_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        self.get_logger().info(f'  LED → {label} ({color_id})')
        return True

    def _wait_arrival(self) -> bool:
        # Wait for path_status to drop to False (new path started)
        deadline = time.time() + 5.0
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if not self.path_complete:
                break
        # Then wait for True (arrived)
        deadline = time.time() + LEG_TIMEOUT_SEC
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.5)
            if self.path_complete:
                return True
        self.get_logger().error('  TIMEOUT waiting for path_status=True')
        return False

    # ── ride protocol ────────────────────────────────────────────────────

    def _drive_to(self, node_id: int, label: str, tol_m: float):
        """Drive to a node, then Fork-B 'balance myself': if path_follower's
        distance_to_final is still beyond tol_m, re-dispatch the same node so
        it re-plans from the current (offset) pose and nudges closer. We trust
        the pose and physically reach the node — we do NOT snap-lie.
        """
        self.dist_to_final = float('inf')
        self._set_node_values(node_id, label)
        if not self._wait_arrival():
            return

        # Re-approach loop — balance until actually within tolerance.
        for attempt in range(MAX_REAPPROACH):
            # Let a couple of distance_to_final samples land.
            for _ in range(5):
                rclpy.spin_once(self, timeout_sec=0.1)
            d = self.dist_to_final
            if d <= tol_m:
                self.get_logger().info(
                    f'  [{label}] arrived at node {node_id} '
                    f'(dist_to_final={d:.3f} m ≤ {tol_m:.2f}).')
                return
            self.get_logger().warn(
                f'  [{label}] offset {d:.3f} m > {tol_m:.2f} — '
                f're-approach {attempt + 1}/{MAX_REAPPROACH}')
            self.dist_to_final = float('inf')
            self._set_node_values(node_id, f'{label} re-approach')
            self._wait_arrival()

        self.get_logger().warn(
            f'  [{label}] gave up balancing after {MAX_REAPPROACH} tries '
            f'(dist_to_final={self.dist_to_final:.3f} m). Proceeding anyway.')

    def _run_one_ride(self, ride_letter: str):
        pickup, dropoff = RIDES[ride_letter]
        print('')
        print('╔══════════════════════════════════════════════════════════╗')
        print(f'║  START RIDE {ride_letter}  —  pickup={pickup}, dropoff={dropoff}, hub={HUB}')
        print('╚══════════════════════════════════════════════════════════╝')
        self.get_logger().info(
            f'>>> START RIDE {ride_letter} (pickup={pickup}, dropoff={dropoff}) <<<')

        # HUB → PICKUP — GREEN (tight tolerance: judges measure pickup)
        self._set_led(LED_GREEN, 'GREEN (HUB→PICKUP)')
        self._drive_to(pickup, f'{ride_letter}.1 HUB→PICKUP', REAPPROACH_TOL_TIGHT_M)

        # At PICKUP — BLUE, pause
        self._set_led(LED_BLUE, 'BLUE (at PICKUP)')
        time.sleep(PAUSE_PICKUP_SEC)

        # PICKUP → DROPOFF — ORANGE (tight tolerance)
        self._set_led(LED_ORANGE, 'ORANGE (PICKUP→DROPOFF)')
        self._drive_to(dropoff, f'{ride_letter}.2 PICKUP→DROPOFF', REAPPROACH_TOL_TIGHT_M)

        # After DROPOFF → flip to MAGENTA, brief pause, then drive home on MAGENTA
        self._set_led(LED_MAGENTA, 'MAGENTA (DROPOFF→HUB)')
        time.sleep(PAUSE_DROPOFF_SEC)

        # HUB is a wide anchor — loose tolerance is fine.
        self._drive_to(HUB, f'{ride_letter}.3 DROPOFF→HUB', REAPPROACH_TOL_HUB_M)

        print('')
        print('╔══════════════════════════════════════════════════════════╗')
        print(f'║  END RIDE {ride_letter}  —  parked at HUB ({HUB}), MAGENTA')
        print('╚══════════════════════════════════════════════════════════╝')
        self.get_logger().info(f'<<< END RIDE {ride_letter} — parked at HUB. >>>')

    def run(self, order):
        # Park at HUB to start (MAGENTA).
        self._set_led(LED_MAGENTA, 'MAGENTA (startup → HUB)')
        self._drive_to(HUB, 'STARTUP → HUB', REAPPROACH_TOL_HUB_M)
        time.sleep(PARK_SETTLE_SEC)

        print('')
        print(f'Startup complete. Parked at HUB (node {HUB}), MAGENTA.')
        print(f'Ride order (will repeat forever, ENTER between rides): {order}')

        # Loop forever, gated by ENTER between rides.
        i = 0
        while rclpy.ok():
            ride_letter = order[i % len(order)]
            if ride_letter not in RIDES:
                self.get_logger().error(
                    f'Unknown ride "{ride_letter}". Skipping.')
                i += 1
                continue

            # Wait for human GO signal before dispatching this ride.
            try:
                prompt = (f'\n>>> Press ENTER to START ride {ride_letter} '
                          f'(or Ctrl-C to quit) <<< ')
                input(prompt)
            except EOFError:
                # stdin closed; treat as quit
                self.get_logger().info('stdin closed — exiting.')
                break

            self._run_one_ride(ride_letter)
            i += 1


def main():
    args = sys.argv[1:]
    order = [a.upper() for a in args] if args else DEFAULT_RIDE_ORDER

    rclpy.init()
    node = RideDispatcher()
    node.get_logger().info(f'Ride order (loops forever): {order}')
    try:
        node.run(order)
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
