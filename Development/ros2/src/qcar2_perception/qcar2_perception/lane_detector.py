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
import os
import sys
import inspect
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy,
                        DurabilityPolicy, HistoryPolicy)
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float64
from cv_bridge import CvBridge


def _resolve_lanenet_import(custom_path: str):
    """Find Quanser's PIT LaneNet package without requiring a global install."""
    candidates = []
    if custom_path:
        candidates.append(custom_path)
    env_path = os.getenv('MDC_PYTHON_PATH', '').strip()
    if env_path:
        candidates.append(env_path)

    home = Path.home()
    candidates.extend([
        str(home / 'Documents/GitHub/ACC_Development/Development/MDC_libraries/python'),
        str(home / 'Documents/ACC_Development/Development/MDC_libraries/python'),
        '/home/nvidia/Documents/ACC_Development/Development/MDC_libraries/python',
        '/workspaces/isaac_ros-dev/MDC_libraries/python',
    ])

    errors = []
    for candidate in [''] + candidates:
        if candidate and candidate not in sys.path:
            sys.path.insert(0, candidate)
        try:
            from pit.LaneNet.nets import LaneNet  # type: ignore
            return LaneNet, candidate or '<existing-pythonpath>'
        except Exception as exc:
            errors.append(f'{candidate or "<existing-pythonpath>"}: {exc}')

    raise ImportError('Could not import pit.LaneNet.nets.LaneNet. ' + ' | '.join(errors[-3:]))


class LaneDetector(Node):

    def __init__(self):
        super().__init__('lane_detector')

        # ───────────── Parameters ─────────────
        self.declare_parameter('image_topic', '/camera/csi_image')
        self.declare_parameter('detector_backend', 'hsv')  # hsv or lanenet

        # 2026-05-27 HYBRID: undistort enabled by default, BUT the
        # homography rebuild now applies cv2.undistortPoints to the
        # source points so Arturo's empirical wide-trapezoid (specified
        # in the distorted image) gets correctly mapped to the undistorted
        # image before warpPerspective. This combines Arturo's pixel-
        # density benefit AND the geometric correctness of undistorted
        # input → no more distortion-induced oscillation while still
        # having a usable near-field BEV.
        self.declare_parameter('undistort_enabled', True)
        self.declare_parameter('camera_matrix_fx',  318.86)
        self.declare_parameter('camera_matrix_fy',  312.14)
        self.declare_parameter('camera_matrix_cx',  401.34)
        self.declare_parameter('camera_matrix_cy',  201.50)
        # Distortion: [k1, k2, p1, p2, k3]
        self.declare_parameter('distortion_coeffs',
                               [-0.9033, 1.5314, -0.0173, 0.0080, -1.1659])

        # Quanser PIT LaneNet
        self.declare_parameter('pit_python_path', '')
        self.declare_parameter('lanenet_model_path', '')
        self.declare_parameter('lanenet_allow_download', False)
        self.declare_parameter('lanenet_row_upper_bound', 240)
        self.declare_parameter('lanenet_use_dbscan', False)
        self.declare_parameter('lanenet_dbscan_eps', 0.5)
        self.declare_parameter('lanenet_dbscan_min_samples', 250)
        self.declare_parameter('lanenet_dbscan_min_area', 100)

        # BEV source polygon
        # Front CSI default is 820x410. These are the old 640x480
        # homography points scaled into the CSI image so the bottom
        # points stay inside the frame.
        self.declare_parameter('src_top_left',      [243, 171])
        self.declare_parameter('src_top_right',     [576, 171])
        self.declare_parameter('src_bottom_right',  [794, 401])
        self.declare_parameter('src_bottom_left',   [26,  401])
        self.declare_parameter('bev_width',  400)
        self.declare_parameter('bev_height', 400)
        self.declare_parameter('bev_world_width_m', 1.5)
        self.declare_parameter('tracking_roi_top', 0)
        self.declare_parameter('tracking_roi_bottom', 0)
        self.declare_parameter('tracking_roi_left', 0)
        self.declare_parameter('tracking_roi_right', 0)

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
        self.declare_parameter('car_center_offset_m', -0.40)
        self.declare_parameter('front_axle_offset_m', 0.256)

        self.declare_parameter('lookahead_distance_m', 0.05)
        self.declare_parameter('heading_segment_m', 0.05)
        self.declare_parameter('heading_offset_deg', 14.9)
        self.declare_parameter('lookahead_row', 200)
        self.declare_parameter('heading_row_gap', 80)

        # Row scanning
        self.declare_parameter('min_row_pixels', 3)
        self.declare_parameter('min_valid_rows', 15)
        self.declare_parameter('row_scan_step', 4)

        # Intersection cleanup. LaneNet can produce thick connected blobs at
        # intersections, so we thin detections into trackable center markings.
        self.declare_parameter('use_centerline_skeleton', True)
        self.declare_parameter('min_lane_component_area', 30)
        self.declare_parameter('max_lane_blob_width_px', 80)
        self.declare_parameter('centerline_search_margin_px', 120)
        self.declare_parameter('intersection_branch', 'straight')  # straight, left, right
        self.declare_parameter('intersection_branch_bias_px', 110)

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
        self.declare_parameter('debug_crop_overlay', False)
        self.declare_parameter('debug_crop_top', 0)
        self.declare_parameter('debug_crop_bottom', 0)
        self.declare_parameter('debug_crop_left', 0)
        self.declare_parameter('debug_crop_right', 0)

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
        self._lanenet = None
        self._lanenet_disabled = False
        self._lanenet_fallback_warned = False
        self._homography_warning_logged = False

        self.add_on_set_parameters_callback(self._parameter_update_cb)

        # ───────────── Undistortion maps ─────────────
        # 2026-05-27: precompute the undistortion remap once for speed
        # (cv2.initUndistortRectifyMap → cv2.remap is much faster than
        # cv2.undistort each frame).
        self._undistort_maps = None
        self._undistort_image_shape = None

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
        # 2026-05-27: BEV publishers — needed to visually verify homography
        # AND to measure pixel distances for bev_world_width_m calibration
        # (Option 2 — yellow line + white road edge with known SDCS spacing).
        self.pub_bev = self.create_publisher(
            Image, '/lane_keeping/bev', 1)
        self.pub_bev_mask = self.create_publisher(
            Image, '/lane_keeping/bev_mask', 1)

        # ───────────── Subscriber ─────────────
        # 2026-05-15: BEST_EFFORT to match the bridge publishers.
        # Previous RELIABLE QoS half-dropped at the DDS layer for
        # high-rate large Image messages.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        image_topic = self.get_parameter(
            'image_topic').get_parameter_value().string_value
        self.create_subscription(
            Image, image_topic, self._image_cb, qos)

        self.get_logger().info(
            f'LaneDetector subscribing to "{image_topic}" (raw bgr8)')
        self.get_logger().info(
            f'Lane detector backend: {self._ps("detector_backend")}')
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

    def _ps(self, n):
        return self.get_parameter(n).get_parameter_value().string_value

    def _parameter_update_cb(self, params):
        # 2026-05-27: rebuild homography on the fly when any source point
        # or bev_world_width_m changes. Without this, ros2 param set on
        # those params is silently ignored — self.M / self.m_per_pix keep
        # whatever values were computed at startup. Caused hours of
        # confusion: user could "set" params but BEV never updated.
        homography_params = {
            'src_top_left', 'src_top_right',
            'src_bottom_right', 'src_bottom_left',
            'bev_width', 'bev_height', 'bev_world_width_m',
        }
        needs_rebuild = False
        for param in params:
            if param.name in (
                'heading_offset_deg',
                'max_heading_rad',
                'front_axle_offset_m',
                'lookahead_distance_m',
                'heading_segment_m',
                'car_center_offset_m',
            ):
                self.get_logger().info(
                    f'Parameter update: {param.name}={param.value}')
            if param.name in homography_params:
                needs_rebuild = True
                self.get_logger().info(
                    f'Homography param update: {param.name}={param.value}')

        if needs_rebuild:
            # Apply the new params first by re-reading via _pi/_pia
            # (declare_parameter stores the new value at this point).
            self._rebuild_homography()
            self._homography_warning_logged = False  # re-check next image
        return SetParametersResult(successful=True)

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

        # 2026-05-27 HYBRID: when undistortion is enabled, the source
        # points (which are conventionally specified in the DISTORTED
        # image's coords — e.g., Arturo's empirical calibration) must
        # be re-mapped to UNDISTORTED image coordinates before
        # getPerspectiveTransform. This gets us the best of both
        # worlds: Arturo's wide-trapezoid pixel-density benefit AND
        # geometric correctness (no distortion-induced oscillation).
        if self._pb('undistort_enabled'):
            try:
                K = np.array([
                    [self._pd('camera_matrix_fx'), 0.0, self._pd('camera_matrix_cx')],
                    [0.0, self._pd('camera_matrix_fy'), self._pd('camera_matrix_cy')],
                    [0.0, 0.0, 1.0],
                ], dtype=np.float32)
                dist = np.array(
                    list(self.get_parameter('distortion_coeffs')
                         .get_parameter_value().double_array_value),
                    dtype=np.float32,
                )
                # undistortPoints returns NORMALIZED coords; passing P=K
                # converts back to undistorted pixel coords.
                src_reshape = src.reshape(-1, 1, 2)
                src_undist = cv2.undistortPoints(src_reshape, K, dist, P=K)
                src_for_warp = src_undist.reshape(-1, 2).astype(np.float32)
                self.get_logger().info(
                    f'Source points remapped distorted→undistorted: '
                    f'TL={src[0].tolist()}→{src_for_warp[0].tolist()}'
                )
            except Exception as exc:
                self.get_logger().error(
                    f'undistortPoints failed; using raw source points. {exc}')
                src_for_warp = src
        else:
            src_for_warp = src

        dst = np.array([
            [0, 0], [self.bev_w, 0],
            [self.bev_w, self.bev_h], [0, self.bev_h]
        ], dtype=np.float32)

        self.M = cv2.getPerspectiveTransform(src_for_warp, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src_for_warp)
        self.m_per_pix = self._pd('bev_world_width_m') / self.bev_w
        self.get_logger().info(
            f'Homography: bev={self.bev_w}x{self.bev_h} '
            f'm/pix={self.m_per_pix:.5f} '
            f'undistort_aware={"yes" if self._pb("undistort_enabled") else "no"}'
        )

    def _warn_if_homography_outside_image(self, image_w, image_h):
        if self._homography_warning_logged:
            return
        pts = [
            self._pia('src_top_left'),
            self._pia('src_top_right'),
            self._pia('src_bottom_right'),
            self._pia('src_bottom_left'),
        ]
        bad = [
            p for p in pts
            if p[0] < 0 or p[0] >= image_w or p[1] < 0 or p[1] >= image_h
        ]
        if bad:
            self.get_logger().warn(
                'Homography source point outside image frame '
                f'{image_w}x{image_h}: {bad}. Lane CTE/heading will be wrong.'
            )
        self._homography_warning_logged = True

    def _ensure_undistort_maps(self, image_shape):
        """Lazily build undistort remap tables once per image size."""
        if not self._pb('undistort_enabled'):
            return False
        h, w = image_shape[:2]
        if (self._undistort_maps is not None
                and self._undistort_image_shape == (h, w)):
            return True
        try:
            K = np.array([
                [self._pd('camera_matrix_fx'), 0.0, self._pd('camera_matrix_cx')],
                [0.0, self._pd('camera_matrix_fy'), self._pd('camera_matrix_cy')],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)
            dist = np.array(
                list(self.get_parameter('distortion_coeffs')
                     .get_parameter_value().double_array_value),
                dtype=np.float32,
            )
            map1, map2 = cv2.initUndistortRectifyMap(
                K, dist, None, K, (w, h), cv2.CV_16SC2)
            self._undistort_maps = (map1, map2)
            self._undistort_image_shape = (h, w)
            self.get_logger().info(
                f'Undistort maps built for {w}x{h} (k1={dist[0]:.3f})')
            return True
        except Exception as exc:
            self.get_logger().error(
                f'Failed to build undistort maps; disabling. {exc}')
            self._undistort_maps = None
            return False

    def _undistort(self, img):
        """Apply precomputed undistort remap, or return img unchanged if disabled."""
        if not self._ensure_undistort_maps(img.shape):
            return img
        map1, map2 = self._undistort_maps
        return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    def _to_bev(self, img):
        # 2026-05-27: undistort BEFORE warp. The 4 source points were
        # derived from a pinhole camera model (no distortion); applying
        # them to a distorted image stretches barrel-bent ground lines
        # into a still-bent BEV. Undistorting first makes the IPM exact.
        return cv2.warpPerspective(
            self._undistort(img), self.M, (self.bev_w, self.bev_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    def _crop_rect(self, prefix, width, height):
        left = max(0, min(width - 1, self._pi(f'{prefix}_left')))
        top = max(0, min(height - 1, self._pi(f'{prefix}_top')))

        right = self._pi(f'{prefix}_right')
        bottom = self._pi(f'{prefix}_bottom')
        if right <= 0 or right > width:
            right = width
        if bottom <= 0 or bottom > height:
            bottom = height

        if right <= left:
            left, right = 0, width
        if bottom <= top:
            top, bottom = 0, height

        return left, top, right, bottom

    def _apply_tracking_roi(self, mask):
        left, top, right, bottom = self._crop_rect(
            'tracking_roi', self.bev_w, self.bev_h)
        if left == 0 and top == 0 and right == self.bev_w and bottom == self.bev_h:
            return mask

        cropped = np.zeros_like(mask)
        cropped[top:bottom, left:right] = mask[top:bottom, left:right]
        return cropped

    def _bev_points_to_image(self, points):
        if not points:
            return []
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pts, self.M_inv).reshape(-1, 2)
        return [(int(round(x)), int(round(y))) for x, y in mapped]

    def _lookahead_from_params(self):
        distance_m = self._pd('lookahead_distance_m')
        fallback_row = self._pi('lookahead_row')
        _, top, _, bottom = self._crop_rect(
            'tracking_roi', self.bev_w, self.bev_h)

        if distance_m > 0.0:
            row = int(round(bottom - distance_m / self.m_per_pix))
        else:
            row = fallback_row

        row = max(top, min(bottom - 1, row))
        actual_m = max(0.0, (bottom - row) * self.m_per_pix)
        return row, actual_m

    def _front_axle_row_from_params(self):
        distance_m = self._pd('front_axle_offset_m')
        lookahead_row = self._pi('lookahead_row')
        _, top, _, bottom = self._crop_rect(
            'tracking_roi', self.bev_w, self.bev_h)

        if distance_m > 0.0:
            row = int(round(bottom - distance_m / self.m_per_pix))
        else:
            row = lookahead_row

        row = max(top, min(bottom - 1, row))
        actual_m = max(0.0, (bottom - row) * self.m_per_pix)
        return row, actual_m

    def _heading_gap_from_params(self):
        segment_m = self._pd('heading_segment_m')
        if segment_m > 0.0:
            gap = int(round(segment_m / self.m_per_pix))
        else:
            gap = self._pi('heading_row_gap')

        gap = max(1, min(self.bev_h - 1, gap))
        actual_m = gap * self.m_per_pix
        return gap, actual_m

    def _car_center_x(self):
        offset_px = self._pd('car_center_offset_m') / self.m_per_pix
        return np.clip(self.bev_w / 2.0 + offset_px, 0.0, self.bev_w - 1.0)

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

    # ───────────── LaneNet backend ─────────────
    def _init_lanenet(self, image_shape):
        if self._lanenet is not None:
            return True
        if self._lanenet_disabled:
            return False

        try:
            LaneNet, pit_path = _resolve_lanenet_import(
                self._ps('pit_python_path').strip())
            h, w = image_shape[:2]
            model_path = self._ps('lanenet_model_path').strip() or None
            if model_path is None and not self._pb('lanenet_allow_download'):
                lanenet_file = Path(inspect.getfile(LaneNet)).resolve()
                default_engine = (
                    lanenet_file.parent /
                    '../../../resources/pretrained_models/lanenet.engine'
                ).resolve()
                if not default_engine.exists():
                    raise FileNotFoundError(
                        f'No LaneNet engine at {default_engine}. Set '
                        'lanenet_model_path or set lanenet_allow_download:=true.')
            self._lanenet = LaneNet(
                modelPath=model_path,
                imageHeight=h,
                imageWidth=w,
                rowUpperBound=self._pi('lanenet_row_upper_bound'))
            self.get_logger().info(
                f'LaneNet initialized from {pit_path}; model={model_path or "default"}')
            return True
        except Exception as exc:
            self._lanenet_disabled = True
            self.get_logger().error(
                f'LaneNet failed to initialize; falling back to HSV. {exc}')
            return False

    def _lanenet_camera_mask(self, bgr):
        if not self._init_lanenet(bgr.shape):
            return None

        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img_tensor = self._lanenet.pre_process(rgb)
            binary_pred, _ = self._lanenet.predict(img_tensor)
            mask = (binary_pred > 127).astype(np.uint8) * 255

            if self._pb('lanenet_use_dbscan'):
                clustered = self._lanenet.post_process(
                    eps=self._pd('lanenet_dbscan_eps'),
                    min_samples=self._pi('lanenet_dbscan_min_samples'),
                    min_area=self._pi('lanenet_dbscan_min_area'))
                clustered_gray = cv2.cvtColor(clustered, cv2.COLOR_RGB2GRAY)
                row_upper = self._pi('lanenet_row_upper_bound')
                full = np.zeros_like(mask)
                resized = cv2.resize(
                    clustered_gray,
                    (bgr.shape[1], bgr.shape[0] - row_upper),
                    interpolation=cv2.INTER_LINEAR)
                full[row_upper:, :] = resized
                if np.count_nonzero(full) > self._pi('min_lane_component_area'):
                    mask = (full > 0).astype(np.uint8) * 255

            return np.ascontiguousarray(mask)
        except Exception as exc:
            self._lanenet_disabled = True
            self.get_logger().error(
                f'LaneNet inference failed; falling back to HSV. {exc}')
            return None

    def _detect_lane_mask(self, bgr, bev_bgr):
        backend = self._ps('detector_backend').strip().lower()
        if backend in ('lanenet', 'lane_net'):
            cam_mask = self._lanenet_camera_mask(bgr)
            if cam_mask is not None:
                return self._to_bev(cam_mask)
            if not self._lanenet_fallback_warned:
                self._lanenet_fallback_warned = True
                self.get_logger().warn('Using HSV lane mask fallback.')

        return self._yellow_mask(bev_bgr)

    def _prepare_tracking_mask(self, mask):
        mask = (mask > 0).astype(np.uint8) * 255

        min_area = self._pi('min_lane_component_area')
        if min_area > 0:
            labels_ret = cv2.connectedComponentsWithStats(
                mask, connectivity=8, ltype=cv2.CV_32S)
            labels = labels_ret[1]
            stats = labels_ret[2]
            filtered = np.zeros_like(mask)
            for idx, stat in enumerate(stats):
                if idx > 0 and stat[4] >= min_area:
                    filtered[labels == idx] = 255
            mask = filtered

        ks = max(1, self._pi('morph_kernel_size'))
        kern = np.ones((ks, ks), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)

        if self._pb('use_centerline_skeleton'):
            mask = self._skeletonize(mask)

        return mask

    def _skeletonize(self, mask):
        img = (mask > 0).astype(np.uint8) * 255
        skel = np.zeros_like(img)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        # 400x400 BEV converges quickly; the guard keeps bad masks bounded.
        for _ in range(300):
            opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(img, opened)
            eroded = cv2.erode(img, element)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded
            if cv2.countNonZero(img) == 0:
                break
        return skel

    # ───────────── Row centroid scan ─────────────
    def _row_segments(self, row_mask):
        cols = np.where(row_mask > 0)[0]
        if len(cols) == 0:
            return []

        splits = np.where(np.diff(cols) > 1)[0] + 1
        groups = np.split(cols, splits)
        min_px = 1 if self._pb('use_centerline_skeleton') else self._pi('min_row_pixels')

        segments = []
        for group in groups:
            if len(group) < min_px:
                continue
            start = int(group[0])
            end = int(group[-1])
            segments.append({
                'start': start,
                'end': end,
                'center': float(0.5 * (start + end)),
                'width': end - start + 1,
                'count': len(group),
            })
        return segments

    def _scan_row_centroids(self, mask, reference_x=None):
        centroids = {}
        step = self._pi('row_scan_step')
        max_blob_w = self._pi('max_lane_blob_width_px')
        search_margin = self._pi('centerline_search_margin_px')
        branch_bias = self._pi('intersection_branch_bias_px')
        branch = self._ps('intersection_branch').strip().lower()
        branch_dir = -1 if branch == 'left' else 1 if branch == 'right' else 0

        prev_x = reference_x if reference_x is not None else self.bev_w / 2.0

        for row in range(self.bev_h - 1, -1, -step):
            segments = self._row_segments(mask[row, :])
            if not segments:
                continue

            horizon = (self.bev_h - row) / max(1.0, float(self.bev_h))
            expected_x = prev_x + branch_dir * branch_bias * horizon

            def score(seg):
                width_penalty = max(0, seg['width'] - max_blob_w) * 2.0
                return abs(seg['center'] - expected_x) + width_penalty

            best = min(segments, key=score)
            best_dist = abs(best['center'] - expected_x)
            if centroids and best_dist > search_margin:
                continue

            centroids[row] = best['center']
            prev_x = best['center']
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
            self._warn_if_homography_outside_image(w, h)
            self._got_first = True

        lane_w = self._pd('lane_width_m')
        lane_side = self._pi('lane_side')
        offset_px = (lane_w / 2.0) / self.m_per_pix * lane_side
        car_x = self._car_center_x()
        reference_x = car_x - offset_px

        bev = self._to_bev(bgr)
        lane_mask = self._detect_lane_mask(bgr, bev)
        lane_mask = self._apply_tracking_roi(lane_mask)

        # 2026-05-27: publish BEV + BEV mask for visual verification and
        # for the user to read off pixel coordinates needed for
        # bev_world_width_m calibration. Always-on; skip if subscribers
        # is a future optimization.
        try:
            self.pub_bev.publish(self.bridge.cv2_to_imgmsg(bev, encoding='bgr8'))
            self.pub_bev_mask.publish(
                self.bridge.cv2_to_imgmsg(lane_mask, encoding='mono8'))
        except Exception:
            pass
        tracking_mask = self._prepare_tracking_mask(lane_mask)
        centroids = self._scan_row_centroids(tracking_mask, reference_x)

        min_rows = self._pi('min_valid_rows')
        lookahead, lookahead_m = self._lookahead_from_params()
        front_axle_row, front_axle_m = self._front_axle_row_from_params()
        heading_gap, heading_segment_m = self._heading_gap_from_params()
        ema = self._pd('ema_alpha')
        max_lost = self._pi('no_detect_max_frames')

        detected = len(centroids) >= min_rows
        cte_raw = 0.0
        heading_raw = 0.0
        sorted_rows = sorted(centroids.keys())

        if detected:
            la_cx = self._interp(centroids, sorted_rows, front_axle_row)

            if la_cx is not None:
                target_x = la_cx + offset_px
                cte_raw = (car_x - target_x) * self.m_per_pix

                near_row = front_axle_row
                far_row = max(front_axle_row - heading_gap, sorted_rows[0])
                near_cx = self._interp(centroids, sorted_rows, near_row)
                far_cx = self._interp(centroids, sorted_rows, far_row)

                if near_cx is not None and far_cx is not None:
                    tn = near_cx + offset_px
                    tf = far_cx + offset_px
                    dx = tf - tn
                    dy = float(far_row - near_row)
                    if abs(dy) > 1e-3:
                        heading_raw = math.atan2(-dx, -dy)
                        heading_raw += math.radians(
                            self._pd('heading_offset_deg'))
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
            self._debug(bgr, tracking_mask, centroids, sorted_rows,
                        offset_px, car_x, lookahead, front_axle_row,
                        heading_gap, lookahead_m, front_axle_m,
                        heading_segment_m,
                        detected, cte_out, hdg_out, heading_raw)

    # ───────────── Debug ─────────────
    def _debug(self, bgr, yellow, centroids, sorted_rows,
               offset_px, car_x, lookahead, front_axle_row, heading_gap,
               lookahead_m, front_axle_m, heading_segment_m,
               detected, cte_out, hdg_out, heading_raw):
        ov = bgr.copy()
        h, w = ov.shape[:2]

        camera_mask = cv2.warpPerspective(
            yellow, self.M_inv, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        idx = camera_mask > 0
        if np.any(idx):
            green = np.array([0, 255, 0], dtype=np.uint8)
            ov[idx] = (0.45 * ov[idx] + 0.55 * green).astype(np.uint8)

        for row, cx in centroids.items():
            points = self._bev_points_to_image([
                (cx, row),
                (cx + offset_px, row),
            ])
            if len(points) == 2:
                cv2.circle(ov, points[0], 2, (255, 0, 0), -1)
                cv2.circle(ov, points[1], 2, (255, 0, 255), -1)

        points = self._bev_points_to_image([
            (0, lookahead),
            (self.bev_w, lookahead),
        ])
        if len(points) == 2:
            cv2.line(ov, points[0], points[1], (0, 255, 255), 1)
        points = self._bev_points_to_image([
            (0, front_axle_row),
            (self.bev_w, front_axle_row),
        ])
        if len(points) == 2:
            cv2.line(ov, points[0], points[1], (0, 165, 255), 1)
        far = max(front_axle_row - heading_gap,
                  sorted_rows[0] if sorted_rows else 0)
        points = self._bev_points_to_image([
            (0, far),
            (self.bev_w, far),
        ])
        if len(points) == 2:
            cv2.line(ov, points[0], points[1], (255, 255, 0), 1)

        points = self._bev_points_to_image([
            (car_x, 0),
            (car_x, self.bev_h),
        ])
        if len(points) == 2:
            cv2.line(ov, points[0], points[1], (128, 128, 128), 1)

        la_cx = self._interp(centroids, sorted_rows, front_axle_row)
        if la_cx is not None:
            points = self._bev_points_to_image([
                (la_cx + offset_px, front_axle_row),
                (car_x, self.bev_h),
            ])
            if len(points) == 2:
                cv2.circle(ov, points[0], 8, (255, 0, 255), 2)
                cv2.line(ov, points[1], points[0], (0, 200, 200), 1)

        # Visualize clamp limits as vertical red lines
        max_cte_px = int(self._pd('max_cte_m') / self.m_per_pix)
        for clamp_x in (car_x - max_cte_px, car_x + max_cte_px):
            points = self._bev_points_to_image([
                (clamp_x, 0),
                (clamp_x, self.bev_h),
            ])
            if len(points) == 2:
                cv2.line(ov, points[0], points[1], (0, 0, 255), 1)

        if self._pb('debug_crop_overlay'):
            h, w = ov.shape[:2]
            left, top, right, bottom = self._crop_rect(
                'debug_crop', w, h)
            ov = np.ascontiguousarray(ov[top:bottom, left:right])

        # Show CLAMPED values so you see what actually gets published.
        s = 'DETECTED' if detected else 'LOST'
        c = (0, 255, 0) if detected else (0, 0, 255)
        text_color = (0, 0, 0)
        cv2.putText(ov, f'CTE: {cte_out:.3f} m',
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    text_color, 1)
        cv2.putText(ov,
                    f'Hdg cmd/raw: {math.degrees(hdg_out):.1f}/{math.degrees(heading_raw):.1f} deg',
                    (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    text_color, 1)
        cv2.putText(ov, s, (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
        cv2.putText(ov,
                    f'Look: {lookahead_m:.2f}m Nose: {front_axle_m:.2f}m',
                    (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    text_color, 1)
        cv2.putText(ov,
                    f'Seg: {heading_segment_m:.2f}m Center: {self._pd("car_center_offset_m"):+.3f}m',
                    (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    text_color, 1)
        cv2.putText(ov, f'Rows: {len(centroids)}',
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    text_color, 1)
        cv2.putText(ov,
                    f'HdgOff: {self._pd("heading_offset_deg"):+.1f} deg',
                    (10, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    text_color, 1)

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
