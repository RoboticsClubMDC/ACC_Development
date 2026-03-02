#!/usr/bin/env python3

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32
from cv_bridge import CvBridge


class LaneStanleyNode(Node):
    def __init__(self):
        super().__init__('lane_stanley_node')

        self.declare_parameter('band_y0_frac', 0.65)
        self.declare_parameter('band_y1_frac', 0.95)
        self.declare_parameter('min_lane_px',  400)
        self.declare_parameter('cte_alpha',    0.2)
        self.declare_parameter('head_alpha',   0.2)
        self.declare_parameter('n_strips',     10)
        self.declare_parameter('k_cte',        0.35)
        self.declare_parameter('k_head',       0.65)
        self.declare_parameter('max_steer',    0.25)

        self.bridge = CvBridge()
        self.speed  = 0.0
        self.cte_f  = 0.0
        self.he_f   = 0.0
        self.have_filter_state = False

        qos = QoSProfile(depth=2)
        self.pub_cte   = self.create_publisher(Float32, '/lane_stanley/cte',           qos)
        self.pub_he    = self.create_publisher(Float32, '/lane_stanley/heading_error',  qos)
        self.pub_delta = self.create_publisher(Float32, '/lane_stanley/delta',          qos)
        self.pub_trust = self.create_publisher(Float32, '/lane_stanley/trust',          qos)

        self.create_subscription(
            Image, '/lane_detection/lane_selected', self._mask_cb, qos_profile_sensor_data)
        self.create_subscription(
            JointState, '/qcar2_joint', self._joint_cb, 1)

        self.get_logger().info('Lane Stanley node ready (RAW MASK mode, NO BEV).')

    def _joint_cb(self, msg: JointState):
        if msg.velocity:
            raw = msg.velocity[0]
            self.speed = (raw / (720.0 * 4.0)) * ((13.0 * 19.0) / (70.0 * 30.0)) * (2.0 * math.pi) * 0.033

    def _mask_cb(self, msg: Image):
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge: {e}')
            return

        if mask is None or mask.ndim != 2:
            self._publish(0.0, 0.0, 0.0, 0.0)
            return

        H, W = mask.shape

        y0 = int(np.clip(self.get_parameter('band_y0_frac').value, 0.0, 1.0) * H)
        y1 = int(np.clip(self.get_parameter('band_y1_frac').value, 0.0, 1.0) * H)
        if y1 <= y0:
            y0, y1 = int(0.5 * H), H

        band   = mask[y0:y1, :]
        binary = (band > 127).astype(np.uint8)
        total_px = int(binary.sum())

        min_px = int(self.get_parameter('min_lane_px').value)
        trust  = 1.0 if total_px >= min_px else 0.0

        if trust < 0.5:
            self.have_filter_state = False
            self._publish(0.0, 0.0, 0.0, 0.0)
            return

        ys, xs = np.where(binary > 0)
        if xs.size == 0:
            self.have_filter_state = False
            self._publish(0.0, 0.0, 0.0, 0.0)
            return

        cx       = float(xs.mean())
        cte_px   = cx - W / 2.0
        cte_norm = float(np.clip(cte_px / (W / 2.0), -1.0, 1.0))

        n_strips   = int(np.clip(self.get_parameter('n_strips').value, 4, 30))
        band_h     = y1 - y0
        strip_edges = np.linspace(0, band_h, n_strips + 1, dtype=int)

        y_vals, x_vals = [], []
        for i in range(n_strips):
            s0, s1 = strip_edges[i], strip_edges[i + 1]
            if s1 <= s0:
                continue
            strip = binary[s0:s1, :]
            if int(strip.sum()) < 10:
                continue
            _, xs_s = np.where(strip > 0)
            if xs_s.size == 0:
                continue
            y_vals.append(float(y0 + (s0 + s1) / 2.0))
            x_vals.append(float(xs_s.mean()))

        if len(x_vals) < 2:
            heading_norm = 0.0
        else:
            m, _ = np.polyfit(np.array(y_vals, dtype=np.float32),
                              np.array(x_vals, dtype=np.float32), 1)
            heading_norm = float(np.clip(math.atan(m) / (math.pi / 4.0), -1.0, 1.0))

        cte_alpha  = float(np.clip(self.get_parameter('cte_alpha').value,  0.0, 1.0))
        head_alpha = float(np.clip(self.get_parameter('head_alpha').value, 0.0, 1.0))

        if not self.have_filter_state:
            self.cte_f = cte_norm
            self.he_f  = heading_norm
            self.have_filter_state = True
        else:
            self.cte_f = (1.0 - cte_alpha)  * self.cte_f + cte_alpha  * cte_norm
            self.he_f  = (1.0 - head_alpha) * self.he_f  + head_alpha * heading_norm

        k_cte     = float(self.get_parameter('k_cte').value)
        k_head    = float(self.get_parameter('k_head').value)
        max_steer = float(self.get_parameter('max_steer').value)

        delta = float(np.clip(k_cte * self.cte_f + k_head * self.he_f, -max_steer, max_steer))

        self._publish(self.cte_f, self.he_f, delta, trust)

    def _publish(self, cte, he, delta, trust):
        def f32(v):
            m = Float32()
            m.data = float(v)
            return m
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