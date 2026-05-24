#!/usr/bin/env python3
"""Stress test for ekf_fusor's outlier rejection gate.

Publishes a sequence of PoseWithCovarianceStamped messages to /amcl_pose,
alternating between "good" (close to current EKF estimate) and "bad"
(deliberate ~3 m offset) poses. While ekf_fusor is running in
correction_source=amcl_pose mode, the Mahalanobis distance on
/qcar2_ekf/innovation_mahalanobis should:

  - sit near 0 for "good" messages (accepted)
  - spike past the gate (default 11.345) for "bad" messages (rejected,
    visible warning in ekf_fusor's log)

Watch /qcar2_ekf/innovation_mahalanobis and the ekf_fusor terminal log
during the run.

Usage (in dev container, sourced + ROS_DOMAIN_ID=69):

  # Terminal A — ekf_fusor in amcl_pose mode (NOT tf mode)
  ros2 run qcar2_autonomy ekf_fusor --ros-args -p correction_source:=amcl_pose

  # Terminal B — this script
  python3 /workspaces/isaac_ros-dev/ros2/scripts/stress_test_for_EKF_and_mahalanobis.py

Expected output in Foxglove on /qcar2_ekf/innovation_mahalanobis:
  alternating low → high → low → high pattern, with the high samples
  always sitting above 11.345 (the χ²_3 99% outlier gate).

Expected log lines from ekf_fusor:
  [INFO] Bootstrap from amcl_pose: ... (very first message)
  [WARN] Outlier rejected (amcl_pose): maha=XX.XX > gate=11.34 (streak=N)
"""

import sys
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node

Test = "Hard" # type of test to see how good system works.
#Try "Hard" and "Easy".
GOOD_OFFSET = (0.0, 0.0)        # near bootstrap origin → small innovation
BAD_OFFSET  = (3.0, 3.0)        # ~4.2 m diagonal offset → large innovation
PERIOD_SEC  = 1.0               # one message per second
if Test == "Hard":
    PATTERN     = ["good"] * 5 + ["bad"] * 10  # 2 good then 2 bad, repeat
elif Test == "Easy":
    PATTERN     = ["good", "good", "bad", "bad"]   # alternate good and bad, repeat

class StressTester(Node):

    def __init__(self):
        super().__init__("ekf_stress_tester")
        self.pub = self.create_publisher(PoseWithCovarianceStamped, "/amcl_pose", 10)
        self.counter = 0
        self.timer = self.create_timer(PERIOD_SEC, self.tick)

        self.get_logger().info("=" * 60)
        self.get_logger().info("EKF stress test running.")
        self.get_logger().info(f"Pattern: {PATTERN} (1 msg/sec, repeats forever)")
        self.get_logger().info(f"Good offset: {GOOD_OFFSET} m")
        self.get_logger().info(f"Bad  offset: {BAD_OFFSET} m  (~4.2 m diagonal)")
        self.get_logger().info("Watch /qcar2_ekf/innovation_mahalanobis in Foxglove.")
        self.get_logger().info("Watch ekf_fusor terminal for 'Outlier rejected' warnings.")
        self.get_logger().info("Ctrl-C to stop.")
        self.get_logger().info("=" * 60)

    def tick(self):
        kind = PATTERN[self.counter % len(PATTERN)]
        offset = GOOD_OFFSET if kind == "good" else BAD_OFFSET

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.pose.position.x = float(offset[0])
        msg.pose.pose.position.y = float(offset[1])
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = 0.0
        msg.pose.pose.orientation.w = 1.0
        # 6x6 covariance with diag (var_x, var_y, _, _, _, var_yaw)
        cov = [0.0] * 36
        cov[0]  = 0.05   # var_x
        cov[7]  = 0.05   # var_y
        cov[35] = 0.03   # var_yaw
        msg.pose.covariance = cov

        self.pub.publish(msg)
        label = "GOOD" if kind == "good" else "BAD "
        self.get_logger().info(
            f"[{self.counter:4d}] {label}  published pose=({offset[0]:+.2f}, {offset[1]:+.2f})"
        )
        self.counter += 1


def main():
    rclpy.init()
    node = StressTester()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
