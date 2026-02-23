#!/usr/bin/env python3
"""
lane_detector.py  –  ROS 2 node for single-yellow-lane detection on QCar2.

Subscribes to /camera/color_image (raw bgr8 Image) using the same
decoding approach as yolo_detector.py.

Lane approach:
  - Row-by-row centroid scan of yellow pixels in BEV.
  - Yellow stripe center + half lane width (5 inches) → driving target.
  - CTE = car center − target at lookahead row.
  - Heading = two-point slope of target trajectory.
  - OUTPUT CLAMPING: CTE and heading are clamped to safe maximums
    so steering can never push past the sidewalk.
  - DEADBAND: tiny CTE values are zeroed to prevent hunting.
"""

import numpy as np
import cv2
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy,
                        DurabilityPolicy, HistoryPolicy)
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float64
from cv_bridge import CvBridge


class LaneDetector(Node):

    def __init__(self):
        super().__init__('lane_detector')

        # ───────────── Parameters ─────────────
        self.declare_parameter('image_topic', '/camera/color_image')

        # BEV source polygon
        self.declare_parameter('src_top_left',      [190, 200])
        self.declare_parameter('src_top_right',     [450, 200])
        self.declare_parameter('src_bottom_right',  [620, 470])
        self.declare_parameter('src_bottom_left',   [20,  470])
        self.declare_parameter('bev_width',  400)
        self.declare_parameter('bev_height', 400)
        self.declare_parameter('bev_world_width_m', 1.5)

        # Yellow HSV
        self.declare_parameter('hsv_h_low',  18)
        self.declare_parameter('hsv_h_high', 40)
        self.declare_parameter('hsv_s_low',  80)
        self.declare_parameter('hsv_s_high', 255)
        self.declare_parameter('hsv_v_low',  120)
        self.declare_parameter('hsv_v_high', 255)

        # Lane geometry — 10 inch lane, drive in the center
        self.declare_parameter('lane_width_m', 0.254)
        self.declare_parameter('lane_side', 1)

        self.declare_parameter('lookahead_row', 200)
        self.declare_parameter('heading_row_gap', 80)

        # Row scanning
        self.declare_parameter('min_row_pixels', 3)
        self.declare_parameter('min_valid_rows', 15)
        self.declare_parameter('row_scan_step', 4)

        # Filtering
        self.declare_parameter('ema_alpha', 0.3)
        self.declare_parameter('no_detect_max_frames', 15)
        self.declare_parameter('morph_kernel_size', 5)

        # ──── OUTPUT CLAMPING / BANDPASS ────
        # Max CTE published (metres).  Anything beyond this is
        # almost certainly noise or the lane is lost.  The sidewalk
        # is ~5 inches (0.127 m) from lane center, so clamping at
        # ~0.10 m keeps corrections inside the lane.
        self.declare_parameter('max_cte_m', 0.10)

        # Max heading error published (radians).
        # 15 deg ≈ 0.26 rad is plenty for lane keeping corrections.
        self.declare_parameter('max_heading_rad', 0.26)

        # Deadband: ignore CTE smaller than this (metres).
        # Prevents micro-corrections that cause hunting/oscillation.
        self.declare_parameter('cte_deadband_m', 0.008)

        # Rate limit: max change in CTE per frame (metres).
        # Prevents sudden jumps from noisy detections.
        self.declare_parameter('max_cte_rate_m', 0.015)

        # Rate limit: max change in heading per frame (radians).
        self.declare_parameter('max_heading_rate_rad', 0.05)

        # Debug
        self.declare_parameter('publish_debug_images', True)

        # ───────────── State ─────────────
        self.bridge = CvBridge()
        self._cte_filtered = 0.0
        self._heading_filtered = 0.0
        self._cte_prev = 0.0
        self._heading_prev = 0.0
        self._frames_without_lane = 0
        self._last_good_cte = 0.0
        self._last_good_heading = 0.0
        self._got_first = False
        self._empty_count = 0

        # ───────────── Homography ─────────────
        self._rebuild_homography()

        # ───────────── Publishers ─────────────
        self.pub_cte = self.create_publisher(
            Float64, '/lane_keeping/cross_track_error', 1)
        self.pub_heading = self.create_publisher(
            Float64, '/lane_keeping/heading_error', 1)
        self.pub_detected = self.create_publisher(
            Bool, '/lane_keeping/lane_detected', 1)
        self.pub_debug_overlay = self.create_publisher(
            Image, '/lane_keeping/debug_overlay', 10)
        self.pub_debug_bev = self.create_publisher(
            Image, '/lane_keeping/debug_bev', 10)

        # ───────────── Subscriber ─────────────
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        image_topic = self.get_parameter(
            'image_topic').get_parameter_value().string_value
        self.create_subscription(
            Image, image_topic, self._image_cb, qos)

        self.get_logger().info(
            f'LaneDetector subscribing to "{image_topic}" (raw bgr8)')
        self.get_logger().info(
            f'Output clamps: CTE ±{self._pd("max_cte_m"):.3f} m, '
            f'heading ±{math.degrees(self._pd("max_heading_rad")):.1f} deg, '
            f'deadband {self._pd("cte_deadband_m")*1000:.1f} mm')

    # ── Param helpers ──
    def _pi(self, n):
        return self.get_parameter(n).get_parameter_value().integer_value

    def _pd(self, n):
        return self.get_parameter(n).get_parameter_value().double_value

    def _pia(self, n):
        return list(self.get_parameter(
            n).get_parameter_value().integer_array_value)

    def _pb(self, n):
        return self.get_parameter(n).get_parameter_value().bool_value

    # ───────────── Decode raw bgr8 (same as yolo_detector) ─────────────
    def _decode_bgr8(self, msg: Image):
        if msg.height == 0 or msg.width == 0 or msg.step == 0:
            self._empty_count += 1
            if self._empty_count % 30 == 0:
                self.get_logger().warn(
                    f'Empty frames: {self._empty_count}')
            return None

        if msg.encoding.lower() != 'bgr8':
            self.get_logger().warn(
                f'Unexpected encoding: {msg.encoding}')
            return None

        buf = np.frombuffer(msg.data, dtype=np.uint8)
        expected = msg.height * msg.step
        if buf.size < expected:
            return None

        img = buf[:expected].reshape(
            (msg.height, msg.step // 3, 3))[:, :msg.width, :]
        return np.ascontiguousarray(img)

    # ───────────── Homography ─────────────
    def _rebuild_homography(self):
        tl = self._pia('src_top_left')
        tr = self._pia('src_top_right')
        br = self._pia('src_bottom_right')
        bl = self._pia('src_bottom_left')
        self.bev_w = self._pi('bev_width')
        self.bev_h = self._pi('bev_height')

        src = np.array([tl, tr, br, bl], dtype=np.float32)
        dst = np.array([
            [0, 0], [self.bev_w, 0],
            [self.bev_w, self.bev_h], [0, self.bev_h]
        ], dtype=np.float32)

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.m_per_pix = self._pd('bev_world_width_m') / self.bev_w
        self.get_logger().info(
            f'Homography: bev={self.bev_w}x{self.bev_h} '
            f'm/pix={self.m_per_pix:.5f}')

    def _to_bev(self, img):
        return cv2.warpPerspective(
            img, self.M, (self.bev_w, self.bev_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # ───────────── Yellow mask ─────────────
    def _yellow_mask(self, bev_bgr):
        hsv = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HSV)
        lo = np.array([self._pi('hsv_h_low'),
                       self._pi('hsv_s_low'),
                       self._pi('hsv_v_low')])
        hi = np.array([self._pi('hsv_h_high'),
                       self._pi('hsv_s_high'),
                       self._pi('hsv_v_high')])
        mask = cv2.inRange(hsv, lo, hi)
        ks = self._pi('morph_kernel_size')
        kern = np.ones((ks, ks), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kern)
        return mask

    # ───────────── Row centroid scan ─────────────
    def _scan_row_centroids(self, mask):
        centroids = {}
        min_px = self._pi('min_row_pixels')
        step = self._pi('row_scan_step')
        for row in range(0, self.bev_h, step):
            cols = np.where(mask[row, :] > 0)[0]
            if len(cols) >= min_px:
                centroids[row] = float(np.mean(cols))
        return centroids

    # ───────────── Interpolate centroid ─────────────
    def _interp(self, centroids, sorted_rows, target_row):
        if not sorted_rows:
            return None
        if target_row in centroids:
            return centroids[target_row]

        below = above = None
        for r in sorted_rows:
            if r <= target_row:
                below = r
            if r >= target_row and above is None:
                above = r

        if below is not None and above is not None and below != above:
            frac = (target_row - below) / (above - below)
            return centroids[below] + frac * (
                centroids[above] - centroids[below])
        if below is not None:
            return centroids[below]
        if above is not None:
            return centroids[above]
        return None

    # ================================================================
    #  OUTPUT CONDITIONING: clamp, deadband, rate-limit
    # ================================================================
    def _condition_output(self, cte_raw, heading_raw):
        max_cte = self._pd('max_cte_m')
        max_hdg = self._pd('max_heading_rad')
        deadband = self._pd('cte_deadband_m')
        max_cte_rate = self._pd('max_cte_rate_m')
        max_hdg_rate = self._pd('max_heading_rate_rad')

        # 1. Hard clamp — never exceed these
        cte = np.clip(cte_raw, -max_cte, max_cte)
        hdg = np.clip(heading_raw, -max_hdg, max_hdg)

        # 2. Deadband — zero out tiny CTE to prevent hunting
        if abs(cte) < deadband:
            cte = 0.0

        # 3. Rate limit — max change per frame
        cte_delta = cte - self._cte_prev
        cte_delta = np.clip(cte_delta, -max_cte_rate, max_cte_rate)
        cte = self._cte_prev + cte_delta

        hdg_delta = hdg - self._heading_prev
        hdg_delta = np.clip(hdg_delta, -max_hdg_rate, max_hdg_rate)
        hdg = self._heading_prev + hdg_delta

        # 4. Re-clamp after rate limiting
        cte = np.clip(cte, -max_cte, max_cte)
        hdg = np.clip(hdg, -max_hdg, max_hdg)

        self._cte_prev = cte
        self._heading_prev = hdg

        return float(cte), float(hdg)

    # ================================================================
    #  IMAGE CALLBACK
    # ================================================================
    def _image_cb(self, msg: Image):
        bgr = self._decode_bgr8(msg)
        if bgr is None:
            return

        if not self._got_first:
            h, w = bgr.shape[:2]
            self.get_logger().info(f'First image: {w}x{h}')
            self._got_first = True

        bev = self._to_bev(bgr)
        yellow = self._yellow_mask(bev)
        centroids = self._scan_row_centroids(yellow)

        min_rows = self._pi('min_valid_rows')
        lookahead = self._pi('lookahead_row')
        heading_gap = self._pi('heading_row_gap')
        ema = self._pd('ema_alpha')
        max_lost = self._pi('no_detect_max_frames')
        lane_w = self._pd('lane_width_m')
        lane_side = self._pi('lane_side')

        offset_px = (lane_w / 2.0) / self.m_per_pix * lane_side

        detected = len(centroids) >= min_rows
        cte_raw = 0.0
        heading_raw = 0.0
        sorted_rows = sorted(centroids.keys())

        if detected:
            la_cx = self._interp(centroids, sorted_rows, lookahead)

            if la_cx is not None:
                target_x = la_cx + offset_px
                car_x = self.bev_w / 2.0
                cte_raw = (car_x - target_x) * self.m_per_pix

                near_row = lookahead
                far_row = max(lookahead - heading_gap, sorted_rows[0])
                near_cx = self._interp(centroids, sorted_rows, near_row)
                far_cx = self._interp(centroids, sorted_rows, far_row)

                if near_cx is not None and far_cx is not None:
                    tn = near_cx + offset_px
                    tf = far_cx + offset_px
                    dx = tf - tn
                    dy = float(far_row - near_row)
                    if abs(dy) > 1e-3:
                        heading_raw = math.atan2(-dx, -dy)
                else:
                    detected = False
            else:
                detected = False

        # ── Lane-lost ──
        if detected:
            self._frames_without_lane = 0
            self._last_good_cte = cte_raw
            self._last_good_heading = heading_raw
        else:
            self._frames_without_lane += 1
            if self._frames_without_lane <= max_lost:
                cte_raw = self._last_good_cte
                heading_raw = self._last_good_heading
            else:
                cte_raw = 0.0
                heading_raw = 0.0

        # ── EMA ──
        self._cte_filtered = ema * cte_raw + (1 - ema) * self._cte_filtered
        self._heading_filtered = (
            ema * heading_raw + (1 - ema) * self._heading_filtered)

        # ── BANDPASS: clamp + deadband + rate limit ──
        cte_out, hdg_out = self._condition_output(
            self._cte_filtered, self._heading_filtered)

        # ── Publish ──
        m = Float64(); m.data = cte_out
        self.pub_cte.publish(m)

        m = Float64(); m.data = hdg_out
        self.pub_heading.publish(m)

        m = Bool()
        m.data = detected or (self._frames_without_lane <= max_lost)
        self.pub_detected.publish(m)

        if self._pb('publish_debug_images'):
            self._debug(bev, yellow, centroids, sorted_rows,
                        offset_px, lookahead, heading_gap,
                        detected, cte_out, hdg_out)

    # ───────────── Debug ─────────────
    def _debug(self, bev, yellow, centroids, sorted_rows,
               offset_px, lookahead, heading_gap,
               detected, cte_out, hdg_out):
        try:
            self.pub_debug_bev.publish(
                self.bridge.cv2_to_imgmsg(yellow, encoding='mono8'))
        except Exception:
            pass

        ov = bev.copy()
        ov[yellow > 0] = (0, 255, 0)

        for row, cx in centroids.items():
            cv2.circle(ov, (int(cx), row), 2, (255, 0, 0), -1)
            cv2.circle(ov, (int(cx + offset_px), row), 2,
                       (255, 0, 255), -1)

        cv2.line(ov, (0, lookahead), (self.bev_w, lookahead),
                 (0, 255, 255), 1)
        far = max(lookahead - heading_gap,
                  sorted_rows[0] if sorted_rows else 0)
        cv2.line(ov, (0, far), (self.bev_w, far), (255, 255, 0), 1)

        car_cx = self.bev_w // 2
        cv2.line(ov, (car_cx, 0), (car_cx, self.bev_h),
                 (128, 128, 128), 1)

        la_cx = self._interp(centroids, sorted_rows, lookahead)
        if la_cx is not None:
            tx = int(la_cx + offset_px)
            cv2.circle(ov, (tx, lookahead), 8, (255, 0, 255), 2)
            cv2.line(ov, (car_cx, self.bev_h),
                     (tx, lookahead), (0, 200, 200), 1)

        # Show CLAMPED values so you see what actually gets published
        s = 'DETECTED' if detected else 'LOST'
        c = (0, 255, 0) if detected else (0, 0, 255)
        cv2.putText(ov, f'CTE: {cte_out:.3f} m',
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1)
        cv2.putText(ov,
                    f'Hdg: {math.degrees(hdg_out):.1f} deg',
                    (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1)
        cv2.putText(ov, s, (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
        cv2.putText(ov, f'Rows: {len(centroids)}',
                    (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1)

        # Visualize clamp limits as vertical red lines
        max_cte_px = int(self._pd('max_cte_m') / self.m_per_pix)
        cv2.line(ov, (car_cx - max_cte_px, 0),
                 (car_cx - max_cte_px, self.bev_h), (0, 0, 255), 1)
        cv2.line(ov, (car_cx + max_cte_px, 0),
                 (car_cx + max_cte_px, self.bev_h), (0, 0, 255), 1)

        try:
            self.pub_debug_overlay.publish(
                self.bridge.cv2_to_imgmsg(ov, encoding='bgr8'))
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetector()
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