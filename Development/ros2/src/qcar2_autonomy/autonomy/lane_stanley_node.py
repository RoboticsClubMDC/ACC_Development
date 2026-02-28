#!/usr/bin/env python3
"""
lane_stanley_node.py
====================
Stanley controller that runs on the BEV binary lane mask.

Subscribes to:
  /csi/front/lane_bev      (mono8)  — from bev_csi_node
  /qcar2_joint             (JointState) — for measured speed

Publishes:
  /lane_stanley/cte            (Float32) — cross-track error in meters
  /lane_stanley/heading_error  (Float32) — heading error in radians
  /lane_stanley/delta          (Float32) — stanley steering output
  /lane_stanley/trust          (Float32) — 0.0 if no lane, 1.0 if good

The Stanley formula:
  δ = θ_e + arctan(k · e_fa / max(v, v_min))

  θ_e  = heading error (lane angle vs car forward axis)
  e_fa = lateral CTE at the front axle lookahead row
  k    = gain (parameter)
  v    = measured speed

BEV coordinate system (from bev_csi_node defaults):
  row 0   → x_max (far forward)
  row H-1 → x_min (near/car)
  col 0   → y_max (left)
  col W-1 → y_min (right)
  car sits at approximately (row H-1, col W/2)
"""

import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32
from cv_bridge import CvBridge


class StanleyController:
    """
    Pure Stanley steering law.
    δ = clip(θ_e + arctan(k · e / max(v, v_min)), -max_steer, max_steer)
    """
    def __init__(self, k=1.0, v_min=0.05, max_steer=0.6):
        self.k        = k
        self.v_min    = v_min
        self.max_steer = max_steer

    def compute(self, cte_m: float, heading_err_rad: float, speed: float) -> float:
        v = max(abs(speed), self.v_min)
        delta = heading_err_rad + math.atan2(self.k * cte_m, v)
        return float(np.clip(delta, -self.max_steer, self.max_steer))


class LaneStanleyNode(Node):

    def __init__(self):
        super().__init__('lane_stanley_node')

        # ---- Parameters ----
        self.declare_parameter('bev_x_min',        0.0)
        self.declare_parameter('bev_x_max',       20.0)
        self.declare_parameter('bev_y_min',       -6.0)
        self.declare_parameter('bev_y_max',        6.0)
        self.declare_parameter('stanley_k',        0.20)   # cross-track gain
        self.declare_parameter('stanley_v_min',    0.05)
        self.declare_parameter('max_steer',        0.25)
        # Row fraction from bottom to sample front-axle CTE (0=car, 1=far)
        self.declare_parameter('fa_row_frac',      0.22)
        # Minimum lane pixels to trust the detection
        self.declare_parameter('min_lane_px',      300)

        self.x_min     = self.get_parameter('bev_x_min').value
        self.x_max     = self.get_parameter('bev_x_max').value
        self.y_min     = self.get_parameter('bev_y_min').value
        self.y_max     = self.get_parameter('bev_y_max').value
        self.fa_frac   = self.get_parameter('fa_row_frac').value
        self.min_px    = self.get_parameter('min_lane_px').value

        self.stanley = StanleyController(
            k          = self.get_parameter('stanley_k').value,
            v_min      = self.get_parameter('stanley_v_min').value,
            max_steer  = self.get_parameter('max_steer').value)

        self.bridge = CvBridge()
        self.speed  = 0.0   # updated from /qcar2_joint

        # ---- Publishers ----
        qos = QoSProfile(depth=2)
        self.pub_cte     = self.create_publisher(Float32, '/lane_stanley/cte',           qos)
        self.pub_he      = self.create_publisher(Float32, '/lane_stanley/heading_error',  qos)
        self.pub_delta   = self.create_publisher(Float32, '/lane_stanley/delta',          qos)
        self.pub_trust   = self.create_publisher(Float32, '/lane_stanley/trust',          qos)

        # ---- Subscribers ----
        self.create_subscription(
            Image, '/csi/front/lane_bev', self._bev_cb, qos_profile_sensor_data)
        self.create_subscription(
            JointState, '/qcar2_joint', self._joint_cb, 1)

        self.get_logger().info('Lane Stanley node ready.')

    # ------------------------------------------------------------------
    def _joint_cb(self, msg: JointState):
        """Extract measured speed from wheel encoder."""
        if msg.velocity:
            raw = msg.velocity[0]
            self.speed = (raw / (720.0 * 4.0)) * ((13.0 * 19.0) / (70.0 * 30.0)) * (2.0 * math.pi) * 0.033

    # ------------------------------------------------------------------
    def _bev_cb(self, msg: Image):
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge: {e}')
            return

        bev_h, bev_w = mask.shape

        # Pixel -> meter conversion
        m_per_pix_y = (self.y_max - self.y_min) / bev_w
        m_per_pix_x = (self.x_max - self.x_min) / bev_h

        # Threshold and check minimum pixels
        binary = (mask > 127).astype(np.uint8)
        total_px = int(binary.sum())

        trust = 1.0 if total_px >= self.min_px else 0.0

        if trust < 0.5:
            self._publish(0.0, 0.0, 0.0, 0.0)
            return

        # ---- Fit a line through lane centroids at multiple row slices ----
        n_slices   = 10
        row_starts = np.linspace(bev_h - 1, 0, n_slices + 1, dtype=int)
        centroids  = []  # (row, col) pairs

        for i in range(n_slices):
            r0 = min(row_starts[i + 1], row_starts[i])
            r1 = max(row_starts[i + 1], row_starts[i]) + 1
            strip = binary[r0:r1, :]
            if strip.sum() < 10:
                continue
            cols_idx = np.where(strip.sum(axis=0) > 0)[0]
            if cols_idx.size == 0:
                continue
            cx = float(cols_idx.mean())
            cr = float((r0 + r1) / 2.0)
            centroids.append((cr, cx))

        if len(centroids) < 2:
            self._publish(0.0, 0.0, 0.0, 0.5)
            return

        centroids = np.array(centroids)  # (N, 2): col 0=row, col 1=col

        # Fit a line: col = a*row + b  (row is independent variable)
        rows_c = centroids[:, 0]
        cols_c = centroids[:, 1]
        coeffs = np.polyfit(rows_c, cols_c, 1)   # [slope, intercept]
        slope, intercept = coeffs[0], coeffs[1]

        # ---- CTE at front-axle row ----
        # front-axle row = some fraction above the car (bottom of BEV)
        fa_row = bev_h - 1 - int(self.fa_frac * bev_h)
        fa_row = max(0, min(fa_row, bev_h - 1))

        lane_col_at_fa = slope * fa_row + intercept
        car_col        = bev_w / 2.0

        # CTE in meters: positive = lane center is to the LEFT of car
        # BEV: col 0 = y_max (left), col W-1 = y_min (right)
        # so increasing col = more right = smaller Y
        cte_px = lane_col_at_fa - car_col
        cte_m  = -cte_px * m_per_pix_y   # negative sign: left deviation is positive CTE

        # ---- Heading error ----
        # slope is in (col/row) units. Convert to angle vs vertical.
        # In BEV: row decreases = forward, col increases = right.
        # lane_heading = angle of lane in image coords
        # car heading = straight up (row direction), so heading_error = atan(slope * m_per_pix_y / m_per_pix_x)
        heading_err = math.atan2(slope * m_per_pix_y, m_per_pix_x)
        # Negate: positive heading error should steer right when lane goes right
        heading_err = -heading_err

        # ---- Stanley delta ----
        delta = self.stanley.compute(cte_m, heading_err, self.speed)

        self._publish(cte_m, heading_err, delta, trust)

    # ------------------------------------------------------------------
    def _publish(self, cte, he, delta, trust):
        def f32(v):
            m = Float32(); m.data = float(v); return m

        self.pub_cte.publish(f32(cte))
        self.pub_he.publish(f32(he))
        self.pub_delta.publish(f32(delta))
        self.pub_trust.publish(f32(trust))


def main(args=None):
    rclpy.init(args=args)
    node = LaneStanleyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()