#!/usr/bin/env python3
"""bo_pd_tune.py — Bayesian Optimization of path_follower PD gains.

Wires INTO the existing system as a peer node (no controller/EKF/SLAM
modifications needed):

    ┌──────────────────┐     publish (Float32)         ┌──────────────────┐
    │  bo_pd_tune.py   │ ─── /nav/kp_steering_set ──▶ │  path_follower   │
    │  (this script)   │ ─── /nav/kd_steering_set ──▶ │                  │
    │                  │ ─── /cmd_waypoints (Path) ──▶│                  │
    │                  │                              └──────────────────┘
    │   skopt          │
    │   gp_minimize    │
    │   surrogate(GP)  │     subscribe                   path_follower
    │   acquisition(EI)│ ◀── /nav/blended_delta ──────  publishes
    │                  │ ◀── /nav/steering_saturation_rate
    │                  │ ◀── /nav/distance_to_final
    │                  │ ◀── /path_status (Bool)
    │                  │ ◀── /qcar2_pose_fused (PoseWithCovarianceStamped)
    └──────────────────┘

Each BO iteration runs ONE trial:
    1. Publish new (Kp, Kd) to the live-tuning topics.
    2. Publish a fixed test path via /cmd_waypoints.
    3. Subscribe to /nav/* + /path_status for trial telemetry.
    4. Wait for path completion or timeout.
    5. Compute scalar cost J from the trial:
         J = w_arr * arrival_error
           + w_sat * mean(steering_saturation_rate)
           + w_std * std(blended_delta)
           + w_dur * trial_duration
           + w_dist * mean(distance_to_final - dist_to_final_min)
       (weights at top of file, tune as needed)
    6. Hand J back to skopt's gp_minimize.
    7. Repeat n_calls times (15 by default).
    8. Save best (Kp, Kd) + full trial log to /tmp/bo_pd_tune_result.json.

Assumptions:
    - path_follower, ekf_fusor, and Cartographer (or AMCL) are already running.
    - The car is in a free area; the test path is small (~3 m loop).
    - You're OK with the car driving 15+ short loops back-to-back.

Dependencies:
    pip install scikit-optimize numpy
    (rclpy comes from your ROS 2 install)

Usage:
    python3 /workspaces/isaac_ros-dev/ros2/scripts/bo_pd_tune.py

Outputs:
    /tmp/bo_pd_tune_result.json — best (Kp, Kd) and per-trial metrics
    Logs to console with each trial's cost
"""

import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path

try:
    from skopt import gp_minimize
    from skopt.space import Real
except ImportError:
    raise SystemExit(
        "scikit-optimize not installed.\n"
        "Install with:  pip install scikit-optimize"
    )


# ====== Cost function weights — tune to your preference ======
W_ARRIVAL    = 1.0      # m; per-meter penalty on final-position error
W_SATURATION = 2.0      # 0..1; per-fraction penalty (fighting steering limit is bad)
W_DELTA_STD  = 0.5      # rad; per-rad penalty on steering wiggle
W_DURATION   = 0.10     # s; per-second penalty (don't reward lazy slow drives)
W_DIST_MEAN  = 0.3      # m; how close to "straight to target" the path was

# ====== Test path — offsets relative to a FIXED TEST_ORIGIN ======
# Two-phase trial design (handles drift + crashes + instant-complete bugs):
#   Phase 1 — APPROACH: drive back to TEST_ORIGIN using SAFE gains. This
#             ensures every trial starts from the same physical position no
#             matter where the previous trial ended (or crashed).
#   Phase 2 — MEASUREMENT: drive the test path using the BO-suggested (Kp,Kd)
#             and measure cost J only during this phase.
#
# TEST_ORIGIN is CAPTURED FROM THE CAR'S POSE WHEN YOU PRESS ENTER, not
# hardcoded. This means you drive the car to a safe spot during warm-up,
# press ENTER, and BO uses that spot as its test origin for all 15 trials.
# As long as the test path geometry fits in the open space around your
# chosen origin, no walls get hit.
#
# Design constraints for the path offsets (must hold for ALL trial Kp/Kd):
#   1. Consecutive waypoints must be > lookahead_dist (~0.7 m at 0.4 m/s)
#   2. Final waypoint must be > 0.25 m from TEST_ORIGIN (else instant complete)
#   3. Path must fit in the open area around TEST_ORIGIN (else crash)
#
# Current path: open square ending 0.3 m north of origin (segments all ≥ 0.7 m).
TEST_PATH_OFFSETS = [
    (1.0, 0.0),     # forward 1 m
    (1.0, 1.0),     # left  1 m   → upper-right corner
    (0.0, 1.0),     # back  1 m   → upper-left corner
    (0.0, 0.3),     # FINAL → 0.3 m north of origin (above 0.25 threshold)
]

# Safe gains used during the approach phase. Not too aggressive so the
# approach doesn't crash; not too sluggish so it doesn't time out.
APPROACH_KP = 1.0
APPROACH_KD = 0.30

APPROACH_TIMEOUT_S = 20.0   # max seconds to drive back to TEST_ORIGIN

# ====== BO settings ======
KP_BOUNDS = (0.5, 1.8)
KD_BOUNDS = (0.0, 0.8)
N_CALLS         = 15        # total trials
N_INITIAL       = 5         # random samples before GP guides
TRIAL_TIMEOUT_S = 45.0      # max wall-clock per trial
RANDOM_SEED     = 42

RESULT_PATH = "/tmp/bo_pd_tune_result.json"


class BOTuner(Node):
    """Drives one trial: publishes gains + path, accumulates telemetry,
    waits for completion or timeout, returns scalar cost J."""

    def __init__(self):
        super().__init__("bo_pd_tuner")

        # Publishers
        self.pub_kp   = self.create_publisher(Float32, "/nav/kp_steering_set", 5)
        self.pub_kd   = self.create_publisher(Float32, "/nav/kd_steering_set", 5)
        self.pub_path = self.create_publisher(Path,    "/cmd_waypoints",      5)

        # Subscribers — accumulate per-trial telemetry
        self.create_subscription(Float32, "/nav/blended_delta",
                                 self._delta_cb, 100)
        self.create_subscription(Float32, "/nav/steering_saturation_rate",
                                 self._sat_cb, 10)
        self.create_subscription(Float32, "/nav/distance_to_final",
                                 self._dist_cb, 20)
        self.create_subscription(Bool, "/path_status", self._status_cb, 5)
        self.create_subscription(PoseWithCovarianceStamped, "/qcar2_pose_fused",
                                 self._pose_cb, 10)

        # TEST_ORIGIN — captured from the car's pose when the user presses
        # ENTER (see main()). Every trial drives back here in the approach
        # phase before measuring with the BO-suggested gains.
        self.test_origin = None  # (x, y) tuple, set by main() after warm-up

        self._reset_trial()
        self.get_logger().info("BO tuner ready. Waiting before first trial...")

    def _reset_trial(self):
        self.delta_history = []
        self.sat_history   = []
        self.dist_history  = []
        self.start_time    = None
        self.path_complete = False
        # NOTE: do NOT reset self.current_pose here — pose subscription is
        # continuous and we want the latest known pose available at all times.
        self.last_dist     = None

    def wait_for_pose(self, timeout_s=10.0):
        """Spin until a /qcar2_pose_fused message has been received.
        Returns True if pose available, False if timed out."""
        t0 = time.time()
        while self.current_pose is None:
            if time.time() - t0 > timeout_s:
                return False
            rclpy.spin_once(self, timeout_sec=0.1)
        return True

    def _delta_cb(self, msg):
        if self.start_time is not None:
            self.delta_history.append(float(msg.data))

    def _sat_cb(self, msg):
        if self.start_time is not None:
            self.sat_history.append(float(msg.data))

    def _dist_cb(self, msg):
        d = float(msg.data)
        if self.start_time is not None:
            self.dist_history.append(d)
        self.last_dist = d

    def _status_cb(self, msg):
        if bool(msg.data) and self.start_time is not None:
            self.path_complete = True

    def _pose_cb(self, msg):
        self.current_pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        )

    def _publish_gains(self, kp, kd):
        m = Float32(); m.data = float(kp); self.pub_kp.publish(m)
        m = Float32(); m.data = float(kd); self.pub_kd.publish(m)

    def _publish_test_path(self):
        # Anchor the path to the car's CURRENT pose so each trial drives the
        # same shape regardless of where the previous trial ended.
        if self.current_pose is None:
            self.get_logger().warn(
                "No fused pose received yet — using (0,0) as trial anchor. "
                "This is fine for first trial but means the path is in absolute frame."
            )
            x0, y0 = 0.0, 0.0
        else:
            x0, y0 = self.current_pose
            self.get_logger().info(
                f"Anchoring test path at car pose ({x0:.2f}, {y0:.2f})"
            )

        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        for dx, dy in TEST_PATH_OFFSETS:
            ps = PoseStamped()
            ps.header.frame_id = "map"
            ps.pose.position.x = float(x0 + dx)
            ps.pose.position.y = float(y0 + dy)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.pub_path.publish(msg)

    def run_trial(self, kp, kd):
        # Reset state, set gains, give path_follower a moment to apply them.
        self._reset_trial()
        self._publish_gains(kp, kd)
        time.sleep(0.5)
        rclpy.spin_once(self, timeout_sec=0.1)

        self._publish_test_path()
        self.start_time = time.time()

        # Spin until complete or timeout
        while not self.path_complete:
            elapsed = time.time() - self.start_time
            if elapsed > TRIAL_TIMEOUT_S:
                self.get_logger().warn(
                    f"TIMEOUT (kp={kp:.3f}, kd={kd:.3f}) at {elapsed:.1f}s"
                )
                break
            rclpy.spin_once(self, timeout_sec=0.1)

        duration = time.time() - self.start_time

        # Cost function
        if not self.delta_history:
            self.get_logger().warn("No telemetry collected — huge penalty J=100")
            return 100.0

        arrival_error = self.dist_history[-1] if self.dist_history else 5.0
        sat_mean      = float(np.mean(self.sat_history)) if self.sat_history else 0.0
        delta_std     = float(np.std(self.delta_history))
        dist_mean     = float(np.mean(self.dist_history)) if self.dist_history else 5.0

        J = (W_ARRIVAL    * arrival_error +
             W_SATURATION * sat_mean +
             W_DELTA_STD  * delta_std +
             W_DURATION   * duration +
             W_DIST_MEAN  * dist_mean)

        self.get_logger().info(
            f"Trial (kp={kp:.3f}, kd={kd:.3f}) → "
            f"arr={arrival_error:.3f} sat={sat_mean:.3f} "
            f"std={delta_std:.3f} dur={duration:.1f} dmean={dist_mean:.3f} "
            f"| J={J:.3f}"
        )

        return J


def main():
    rclpy.init()
    tuner = BOTuner()
    trial_log = []

    def objective(params):
        kp, kd = params
        J = tuner.run_trial(kp, kd)
        trial_log.append({
            "kp": float(kp), "kd": float(kd), "J": float(J),
            "timestamp": time.time(),
        })
        return J

    # Pubs/subs discovery warm-up.
    time.sleep(2.0)
    for _ in range(20):
        rclpy.spin_once(tuner, timeout_sec=0.05)

    # ─── Manual warm-up gate ─────────────────────────────────────────────
    # Cartographer needs scan data to stabilize its map before the BO trials
    # start measuring cost. Recommended: manually drive the car around the
    # test area for ~30 seconds with `manual_drive`, then come back here
    # and press ENTER to begin the 15-trial BO run.
    print("")
    print("=" * 70)
    print("  BO is READY but PAUSED.")
    print("")
    print("  Before pressing ENTER:")
    print("    1. Run `ros2 run qcar2_autonomy manual_drive` in another terminal.")
    print("    2. Drive the car around the test area for ~30 s so")
    print("       Cartographer builds a stable local map.")
    print("    3. (Optional) Drive the car back near (0, 0) in map frame.")
    print("    4. Stop manual_drive (Ctrl-C) so the steering bus is free.")
    print("")
    print("  Once you press ENTER:")
    print(f"    - {N_CALLS} BO trials will run back-to-back (~12 min total).")
    print("    - Each trial drives the test triangle in map frame.")
    print("    - Do NOT interfere unless something goes wrong (Ctrl-C this script).")
    print("=" * 70)
    print("")
    try:
        input("  Press ENTER when ready (or Ctrl-C to abort)... ")
    except (EOFError, KeyboardInterrupt):
        print("\nAborted before any trial ran.")
        tuner.destroy_node()
        rclpy.shutdown()
        return

    tuner.get_logger().info(
        f"Starting BO: n_calls={N_CALLS}, n_initial={N_INITIAL}, "
        f"Kp∈{KP_BOUNDS}, Kd∈{KD_BOUNDS}, acq=EI"
    )

    result = gp_minimize(
        objective,
        [Real(KP_BOUNDS[0], KP_BOUNDS[1], name="kp"),
         Real(KD_BOUNDS[0], KD_BOUNDS[1], name="kd")],
        n_calls=N_CALLS,
        n_initial_points=N_INITIAL,
        acq_func="EI",
        random_state=RANDOM_SEED,
    )

    best = {
        "best_kp": float(result.x[0]),
        "best_kd": float(result.x[1]),
        "best_J":  float(result.fun),
        "n_calls": N_CALLS,
        "weights": {
            "W_ARRIVAL": W_ARRIVAL, "W_SATURATION": W_SATURATION,
            "W_DELTA_STD": W_DELTA_STD, "W_DURATION": W_DURATION,
            "W_DIST_MEAN": W_DIST_MEAN,
        },
        "trial_log": trial_log,
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(best, f, indent=2)

    tuner.get_logger().info(
        "=" * 60 + "\n"
        f"BO complete. Best: Kp={best['best_kp']:.3f}, "
        f"Kd={best['best_kd']:.3f}, J={best['best_J']:.3f}\n"
        f"Full result saved to {RESULT_PATH}\n"
        + "=" * 60
    )

    # Leave the best gains applied so you can drive with them immediately.
    tuner._publish_gains(best["best_kp"], best["best_kd"])
    time.sleep(0.5)

    tuner.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
