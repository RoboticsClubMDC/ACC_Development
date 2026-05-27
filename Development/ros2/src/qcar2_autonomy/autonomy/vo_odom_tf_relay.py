#!/usr/bin/env python3
"""vo_odom_tf_relay.py - make /vo/odometry usable as RTAB-Map external odom.

Our vo_node publishes /vo/odometry stamped frame_id='map',
child_frame_id='base_link' (to mirror Cartographer's framing). RTAB-Map
cannot ingest that as external odometry:
  - frame_id='map' collides with RTAB-Map's own SLAM map frame, and
  - nothing broadcasts the odom->base_link TF that RTAB-Map needs to
    interpolate the camera pose at each image timestamp.
Symptom: "Tf has two or more unconnected trees", map stuck at 1 node
(observed in Physical Test 12 variant B, 2026-05-21).

This relay subscribes to /vo/odometry and:
  1. republishes it on /vo/odom_relay with frame_id='odom',
     child_frame_id='base_link' (clean odom frame, no 'map' collision)
  2. broadcasts the matching TF odom -> base_link
so RTAB-Map sees the connected chain map -> odom -> base_link -> camera.

Standalone use (NO colcon build required; just source ROS first):
  python3 vo_odom_tf_relay.py --ros-args -p use_sim_time:=true
(use_sim_time:=true when driving it from a bag played with --clock.)

Params (all optional):
  in_topic   (default /vo/odometry)   source odometry
  out_topic  (default /vo/odom_relay) re-framed odometry for RTAB-Map
  odom_frame (default odom)           frame_id to stamp on output + TF
  base_frame (default base_link)      child_frame_id on output + TF
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

# 6x6 row-major covariance diagonal indices: x, y, z, roll, pitch, yaw.
_DIAG = (0, 7, 14, 21, 28, 35)
# Dimensions a planar VO does not observe -> give them a large but FINITE
# variance ("ignore", information ~ 0). vo_node leaves the twist Z/roll/pitch
# at exactly 0.0, and RTAB-Map inverts the covariance for its link
# information matrix -> 1/0 = inf -> FATAL Link.cpp setInfMatrix assertion.
_OFFPLANE = (14, 21, 28)
_BIG = 1e6      # finite "ignore" variance for unobserved dims
_FLOOR = 1e-6   # tiny floor so an observed dim never yields inf information


def _sanitize_cov(cov):
    """Return a 36-elem covariance with every diagonal entry finite and >0,
    so RTAB-Map's inverse (information matrix) never sees 0 or non-finite."""
    out = list(cov)
    for d in _DIAG:
        v = out[d]
        if (not math.isfinite(v)) or v <= 0.0:
            out[d] = _BIG if d in _OFFPLANE else _FLOOR
    return out


class VoOdomTfRelay(Node):
    def __init__(self):
        super().__init__('vo_odom_tf_relay')
        self.declare_parameter('in_topic', '/vo/odometry')
        self.declare_parameter('out_topic', '/vo/odom_relay')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.in_topic = str(self.get_parameter('in_topic').value)
        self.out_topic = str(self.get_parameter('out_topic').value)
        self.odom_frame = str(self.get_parameter('odom_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)

        qos = QoSProfile(depth=20)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.history = HistoryPolicy.KEEP_LAST

        self.pub = self.create_publisher(Odometry, self.out_topic, qos)
        self.br = TransformBroadcaster(self)
        self.sub = self.create_subscription(Odometry, self.in_topic, self.cb, qos)
        self.n = 0
        self.get_logger().info(
            f"relay {self.in_topic} -> {self.out_topic} "
            f"(frame_id={self.odom_frame}, child={self.base_frame}) + TF broadcast")

    def cb(self, msg: Odometry):
        # Re-framed odometry topic (same pose/twist, clean frame ids), with
        # both covariances sanitized so RTAB-Map's information-matrix inverse
        # never hits 1/0 = inf (vo_node leaves twist Z/roll/pitch at 0.0).
        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.odom_frame
        out.child_frame_id = self.base_frame
        out.pose = msg.pose
        out.twist = msg.twist
        out.pose.covariance = _sanitize_cov(msg.pose.covariance)
        out.twist.covariance = _sanitize_cov(msg.twist.covariance)
        self.pub.publish(out)

        # Matching odom -> base_link TF so RTAB-Map can build a connected tree.
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        p = msg.pose.pose
        t.transform.translation.x = p.position.x
        t.transform.translation.y = p.position.y
        t.transform.translation.z = p.position.z
        t.transform.rotation = p.orientation
        self.br.sendTransform(t)

        self.n += 1
        if self.n % 50 == 1:
            self.get_logger().info(f"relayed {self.n} odom msgs")


def main():
    rclpy.init()
    node = VoOdomTfRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
