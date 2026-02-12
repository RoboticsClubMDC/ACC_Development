#!/usr/bin/env python3
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# region : File Description and Imports
"""
lane_detector.py

ROS2 Lane Following Node for QCar2 (Physical) using Intel RealSense D435.
Integrates with qcar2_autonomy package — publishes Twist on /cmd_vel_nav
(same interface as nav_to_pose.py), compatible with nav2_qcar2_converter
and the qcar2_hardware node.

Pipeline (from Quanser Lane Keeping Lab Guide):
  Camera → LaneNet → BEV (IPM) → Preprocess → Isolate → Targets → Pure Pursuit

Published Topics:
    /cmd_vel_nav                 - Twist (linear.x=speed, angular.z=steering)
    /lane_keeping/binary_image   - Raw lane binary from LaneNet (mono8)
    /lane_keeping/bev_rgb        - Bird's-eye view of camera feed (bgr8)
    /lane_keeping/bev_binary     - BEV lane markings binary (mono8)
    /lane_keeping/debug          - Debug overlay with targets and ld arc (bgr8)

Subscribed Topics:
    /camera/color_image          - Intel RealSense D435 RGB (sensor_msgs/Image)
    /qcar2_joint                 - Motor encoder for measured speed
    /motion_enable               - Object detection flag (stop sign / traffic light)
"""
#! /usr/bin/env python3
import sys
sys.path.insert(0, "/workspaces/isaac_ros-dev/MDC_libraries/python")

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import os

# endregion


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# region : Try importing LaneNet from Quanser PIT
LANENET_AVAILABLE = False
try:
    from pit.LaneNet.nets import LaneNet
    LANENET_AVAILABLE = True
except ImportError:
    pass
# endregion


# ============================================================================
# region : Inverse Perspective Mapping (IPM)
# ============================================================================
class IPM:
    """Inverse Perspective Mapping: creates a bird's-eye view (BEV) image
    using the Intel RealSense D435 camera intrinsics and extrinsics on QCar2.

    Coordinate frames (from QCar2 Hardware Manual):
        Body {B}:   x → forward,  y → left,  z → up.
                     Origin is between the front and rear axles on the ground.
        Camera {C}: x → right,  y → down,  z → forward (depth).

    The hardware manual provides  realsense_to_body  (camera → body).
    For projection we need  body → camera, which is its inverse.
    """

    def __init__(self, bev_shape, bev_world_dims, fx, fy, cx, cy):
        """
        Args:
            bev_shape:      [width, height] of BEV output image in pixels.
            bev_world_dims: [x_min, x_max, y_min, y_max] in meters (body frame).
                            x_min > 0.15 avoids showing camera/lens artefacts.
            fx, fy, cx, cy: Intel RealSense D435 intrinsics (640×480 RGB).
        """
        self.bev_shape = bev_shape
        self.bev_world_dims = bev_world_dims
        self.m_per_pix = (bev_world_dims[1] - bev_world_dims[0]) / bev_shape[1]

        # ── Camera Intrinsics (4×4 homogeneous) ──────────────────────────
        self.K = np.array([
            [fx,  0,  cx, 0],
            [ 0, fy,  cy, 0],
            [ 0,  0,   1, 0],
            [ 0,  0,   0, 1]
        ], dtype=np.float64)

        # ── Camera Extrinsics: body → camera ─────────────────────────────
        # From hardware manual, realsense_to_body (cam → body):
        #   T_C^B = [ 0   0   1   0.095 ]
        #           [-1   0   0   0.032 ]
        #           [ 0  -1   0   0.172 ]
        #           [ 0   0   0   1     ]
        #
        # Inverse (body → camera):
        self.E = np.array([
            [ 0, -1,  0,  0.032],
            [ 0,  0, -1,  0.172],
            [ 1,  0,  0, -0.095],
            [ 0,  0,  0,  1.0  ]
        ], dtype=np.float64)

        self._compute_homography()

    def v2img(self, XYZ):
        """Project Nx3 points in body frame → Nx2 pixel coords (u, v)."""
        N = XYZ.shape[0]
        P_hom = np.hstack([XYZ, np.ones((N, 1))])
        P_cam = (self.E @ P_hom.T).T
        P_img = (self.K @ P_cam.T).T
        uv = np.zeros((N, 2), dtype=np.float64)
        uv[:, 0] = P_img[:, 0] / P_img[:, 2]
        uv[:, 1] = P_img[:, 1] / P_img[:, 2]
        return uv

    def _compute_homography(self):
        """Compute 3×3 perspective transform (camera image → BEV).

        Pick 4 corners of the desired BEV region in body frame, project
        them into camera-image pixels, compute corresponding BEV pixel
        positions, then use cv2.getPerspectiveTransform for the homography.
        """
        xmin, xmax, ymin, ymax = self.bev_world_dims
        w, h = self.bev_shape

        # 4 corners in body frame (z=0, ground plane)
        corners_body = np.array([
            [xmin, ymax, 0.0],   # near-left
            [xmin, ymin, 0.0],   # near-right
            [xmax, ymin, 0.0],   # far-right
            [xmax, ymax, 0.0],   # far-left
        ])

        img_pts = self.v2img(corners_body).astype(np.float32)

        # BEV pixel positions:
        #   column 0 = left (ymax), column w-1 = right (ymin)
        #   row 0 = far (xmax), row h-1 = near (xmin)
        bev_pts = np.float32([
            [0,     h - 1],     # near-left  → bottom-left
            [w - 1, h - 1],     # near-right → bottom-right
            [w - 1, 0],         # far-right  → top-right
            [0,     0],         # far-left   → top-left
        ])

        self.M = cv2.getPerspectiveTransform(img_pts, bev_pts)

    def to_bev(self, img):
        """Warp a camera image (or binary mask) into BEV."""
        w, h = self.bev_shape
        return cv2.warpPerspective(img, self.M, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)

# endregion


# ============================================================================
# region : Lane Processing Helpers (Sections C, D, E from Lab Guide)
# ============================================================================
class LaneMarking:
    """Stores a single isolated lane marking blob and its look-ahead
    intersection point."""

    def __init__(self, bev_rear_wheel_pos, binary):
        self.bev_rear_wheel_pos = bev_rear_wheel_pos
        self.binary = binary
        self.intersection = None

    def find_intersection(self, ld_pix):
        """Find the point on this blob at distance ld_pix from rear wheel."""
        ys, xs = np.where(self.binary > 0)
        if len(xs) == 0:
            self.intersection = None
            return

        dx = xs.astype(np.float64) - self.bev_rear_wheel_pos[0]
        dy = ys.astype(np.float64) - self.bev_rear_wheel_pos[1]
        dists = np.sqrt(dx ** 2 + dy ** 2)

        dist_err = np.abs(dists - ld_pix)
        tolerance = max(10, ld_pix * 0.08)
        mask = dist_err < tolerance

        if np.any(mask):
            self.intersection = np.array([
                np.mean(xs[mask]),
                np.mean(ys[mask])
            ])
        else:
            self.intersection = None


def preprocess_lane_marking(bev_lane):
    """SECTION C — Threshold grey→white, morphological close to merge edges."""
    if len(bev_lane.shape) == 3:
        bev_lane = cv2.cvtColor(bev_lane, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(bev_lane, 50, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return binary


def isolate_lane_markings(binary, min_area=300):
    """SECTION D — Connected-component analysis to split individual markings."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    blobs = []
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        blob = np.zeros_like(binary)
        blob[labels == i] = 255
        blobs.append(blob)
    return blobs


def find_rot_mat(lane):
    """SECTION E.4 — Rotation matrix so that offset vector is ⊥ to the lane."""
    ys, xs = np.where(lane.binary > 0)
    if len(xs) < 5:
        return np.eye(2)

    inter = lane.intersection
    dx = xs.astype(float) - inter[0]
    dy = ys.astype(float) - inter[1]
    dists = np.sqrt(dx ** 2 + dy ** 2)
    near = dists < 40
    if np.sum(near) < 5:
        return np.eye(2)

    pts = np.column_stack([xs[near], ys[near]]).astype(np.float32)
    vx, vy, _, _ = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    angle = math.atan2(vy, vx)

    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s],
                     [s,  c]])


def ccw_key(rear_center):
    """Sort key for ordering lane markings in CCW order."""
    cx, cy = rear_center

    def _key(lane):
        px, py = lane.intersection
        return math.atan2(py - cy, px - cx)

    return _key

# endregion


# ============================================================================
# region : Pure Pursuit Controller (Section F from Lab Guide)
# ============================================================================
class PurePursuitController:
    """Pure pursuit in BEV pixel space.

    In BEV:
        column increases → right  (body −y direction)
        row    increases → closer (body −x direction)

    δ = arctan(2 L sin(α) / ld)
    """

    def __init__(self, m_per_pix, bev_rear_wheel, max_steer, wheelbase=0.256):
        self.m_per_pix = m_per_pix
        self.bev_rear_wheel = bev_rear_wheel
        self.max_steer = max_steer
        self.L = wheelbase   # QCar2 wheelbase

    def target2steer(self, point):
        """Compute steering angle from a BEV target point."""
        dx_pix = point[0] - self.bev_rear_wheel[0]
        dy_pix = point[1] - self.bev_rear_wheel[1]

        dx_m = dx_pix * self.m_per_pix
        dy_m = dy_pix * self.m_per_pix

        ld = math.sqrt(dx_m ** 2 + dy_m ** 2)
        if ld < 1e-4:
            return 0.0

        # sin(alpha): positive when target is LEFT of heading
        # Target left → dx_pix < 0 → positive steer → negate
        sin_alpha = -dx_m / ld

        steer = math.atan2(2.0 * self.L * sin_alpha, ld)
        return float(np.clip(steer, -self.max_steer, self.max_steer))

# endregion


# ============================================================================
# region : Main ROS 2 Node
# ============================================================================
class LaneDetectorNode(Node):

    def __init__(self):
        super().__init__('lane_detector')

        # ── Declare ROS parameters ────────────────────────────────────
        # Camera intrinsics (640×480 RGB, Intel RealSense D435)
        self.declare_parameter('fx', 615.0)
        self.declare_parameter('fy', 615.0)
        self.declare_parameter('cx', 320.0)
        self.declare_parameter('cy', 240.0)
        self.declare_parameter('image_width',  640)
        self.declare_parameter('image_height', 480)

        # BEV dimensions (metres, body frame). x_min > 0.15 avoids lens.
        self.declare_parameter('bev_width',  400)
        self.declare_parameter('bev_height', 400)
        self.declare_parameter('bev_xmin',  0.20)
        self.declare_parameter('bev_xmax',  2.50)
        self.declare_parameter('bev_ymin', -0.80)
        self.declare_parameter('bev_ymax',  0.80)

        # Lane keeping / pure pursuit
        self.declare_parameter('Kdd',       3.0)
        self.declare_parameter('ld_min',    0.10)
        self.declare_parameter('ld_max',    0.60)
        self.declare_parameter('max_steer', 0.5)

        # Speed
        self.declare_parameter('v_desire', 0.5)

        # LaneNet model path
        self.declare_parameter('lanenet_model',
            '/home/quanser/Documents/ACC_Development/Development/'
            'ros2/src/qcar2_autonomy/models/Lanenet.pt')

        # ── Read parameters ───────────────────────────────────────────
        fx = self.get_parameter('fx').value
        fy = self.get_parameter('fy').value
        cx = self.get_parameter('cx').value
        cy = self.get_parameter('cy').value
        self.img_w = self.get_parameter('image_width').value
        self.img_h = self.get_parameter('image_height').value

        bev_w = self.get_parameter('bev_width').value
        bev_h = self.get_parameter('bev_height').value
        bev_xmin = self.get_parameter('bev_xmin').value
        bev_xmax = self.get_parameter('bev_xmax').value
        bev_ymin = self.get_parameter('bev_ymin').value
        bev_ymax = self.get_parameter('bev_ymax').value

        self.Kdd      = self.get_parameter('Kdd').value
        self.ld_min   = self.get_parameter('ld_min').value
        self.ld_max   = self.get_parameter('ld_max').value
        max_steer     = self.get_parameter('max_steer').value

        self.v_desire = self.get_parameter('v_desire').value
        self.max_steering_angle = max_steer

        model_path = self.get_parameter('lanenet_model').value

        # ── Build IPM ─────────────────────────────────────────────────
        self.bev_shape = [bev_w, bev_h]
        self.bev_world_dims = [bev_xmin, bev_xmax, bev_ymin, bev_ymax]

        self.ipm = IPM(self.bev_shape, self.bev_world_dims, fx, fy, cx, cy)
        self.m_per_pix = self.ipm.m_per_pix

        # ── Reference positions in BEV ────────────────────────────────
        # Rear wheel is 0.128 m behind body origin (half of 0.256 m wheelbase)
        self.bev_camera_pos = np.array([bev_w // 2, bev_h])
        rear_offset_pix = int(0.128 / self.m_per_pix)
        self.bev_rear_wheel_pos = np.array([bev_w // 2,
                                            bev_h + rear_offset_pix])

        # ── Pure Pursuit Controller ───────────────────────────────────
        self.pp = PurePursuitController(
            self.m_per_pix, self.bev_rear_wheel_pos, max_steer, wheelbase=0.256
        )

        # ── State ─────────────────────────────────────────────────────
        self.current_v = 0.0
        self.current_steering = 0.0
        self.motion_flag = True

        # ── LaneNet ───────────────────────────────────────────────────
        self.lanenet = None
        if LANENET_AVAILABLE and os.path.isfile(model_path):
            try:
                self.lanenet = LaneNet(
                    rowUpperBound=self.img_h // 2,
                    imageWidth=self.img_w,
                    imageHeight=self.img_h,
                )
                self.get_logger().info(f'LaneNet loaded from {model_path}')
            except Exception as e:
                self.get_logger().warn(f'LaneNet init failed: {e}')
                self.lanenet = None
        else:
            self.get_logger().warn(
                'LaneNet NOT available — using Canny + HoughLinesP fallback.'
            )

        # ── CV Bridge ─────────────────────────────────────────────────
        self.bridge = CvBridge()

        # ── Subscribers ───────────────────────────────────────────────
        self.create_subscription(
            Image, '/camera/color_image', self._image_cb, 10
        )

        # Measured speed from motor encoder (same as nav_to_pose.py)
        self.create_subscription(
            JointState, '/qcar2_joint', self._joint_state_cb, 1
        )

        # Object detection flag (from yolo_detector or traffic_system_detector)
        self.create_subscription(
            Bool, '/motion_enable', self._motion_enable_cb, 1
        )

        # ── Publishers — control (Twist on /cmd_vel_nav) ──────────────
        # Same interface as nav_to_pose.py → nav2_qcar2_converter →
        # qcar2_hardware. linear.x = desired speed, angular.z = steering.
        self.cmd_publisher = self.create_publisher(
            Twist, '/cmd_vel_nav', 1
        )

        # ── Publishers — visualisation ────────────────────────────────
        self.pub_binary  = self.create_publisher(Image, '/lane_keeping/binary_image', 10)
        self.pub_bev_rgb = self.create_publisher(Image, '/lane_keeping/bev_rgb', 10)
        self.pub_bev_bin = self.create_publisher(Image, '/lane_keeping/bev_binary', 10)
        self.pub_debug   = self.create_publisher(Image, '/lane_keeping/debug', 10)

        # Timing
        self._prev_time = self.get_clock().now()
        self._ld_pix = 50   # initialise for debug drawing

        self.get_logger().info(
            f'Lane detector started  |  BEV {bev_w}x{bev_h}  '
            f'world [{bev_xmin:.2f}..{bev_xmax:.2f}] x '
            f'[{bev_ymin:.2f}..{bev_ymax:.2f}] m  |  '
            f'm/pix={self.m_per_pix:.5f}'
        )

    # ================================================================== #
    #  SUBSCRIBER CALLBACKS
    # ================================================================== #
    def _joint_state_cb(self, msg):
        """Extract measured linear speed from motor encoder.
        Formula from nav_to_pose.py:
          v = (encoder_vel / (720*4)) * ((13*19)/(70*30)) * (2π) * 0.033
        """
        if msg.velocity:
            self.current_v = (
                (msg.velocity[0] / (720.0 * 4.0))
                * ((13.0 * 19.0) / (70.0 * 30.0))
                * (2.0 * np.pi) * 0.033
            )

    def _motion_enable_cb(self, msg):
        """Stop sign / traffic light detection flag."""
        self.motion_flag = msg.data

    # ================================================================== #
    #  IMAGE CALLBACK — full lane keeping pipeline
    # ================================================================== #
    def _image_cb(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge: {e}')
            return

        now = self.get_clock().now()
        dt = (now - self._prev_time).nanoseconds * 1e-9
        self._prev_time = now
        if dt <= 0 or dt > 1.0:
            dt = 0.1

        # ── 1. Lane marking extraction ────────────────────────────────
        lane_binary = self._detect_lanes(img)

        # ── 2. Bird's-eye views ───────────────────────────────────────
        bev_rgb    = self.ipm.to_bev(img)
        bev_binary = self.ipm.to_bev(lane_binary)

        # ── 3. Preprocess BEV lane markings (Section C) ──────────────
        processed = preprocess_lane_marking(bev_binary)

        # ── 4. Isolate individual lane markings (Section D) ──────────
        blobs = isolate_lane_markings(processed, min_area=250)

        # ── 5. Find lane-center targets (Section E) ──────────────────
        targets = self._find_targets(blobs, self.current_v)

        # ── 6. Select best target & compute steering (Section F) ─────
        steer = 0.0
        selected_target = None
        if targets:
            best_id = min(targets,
                          key=lambda t: abs(targets[t][0] -
                                            self.bev_rear_wheel_pos[0]))
            selected_target = targets[best_id]
            steer = self.pp.target2steer(selected_target)

        self.current_steering = steer

        # ── 7. Publish control command (Twist on /cmd_vel_nav) ────────
        # Follows nav_to_pose.py convention:
        #   linear.x  = speed (scaled by cos²(steering) for turning slowdown)
        #   angular.z = steering angle (radians)
        cmd = Twist()
        if self.motion_flag:
            speed = self.v_desire * np.power(np.cos(steer), 2)
            speed = np.clip(speed, 0.05, 0.7)
            cmd.linear.x = float(speed)
            cmd.angular.z = float(steer)
        else:
            # Object detector says stop
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        self.cmd_publisher.publish(cmd)

        # ── 8. Publish visualisation topics ───────────────────────────
        self._publish_img(self.pub_binary,  lane_binary, 'mono8')
        self._publish_img(self.pub_bev_rgb, bev_rgb,     'bgr8')
        self._publish_img(self.pub_bev_bin, processed,   'mono8')

        debug = self._draw_debug(bev_rgb, blobs, targets, selected_target)
        self._publish_img(self.pub_debug, debug, 'bgr8')

    # ================================================================== #
    #  LANE DETECTION (LaneNet or fallback)
    # ================================================================== #
    def _detect_lanes(self, img):
        """Return a single-channel binary mask of lane markings."""
        if self.lanenet is not None:
            try:
                processed = self.lanenet.pre_process(img)
                self.lanenet.predict(processed)
                return self.lanenet.binaryPred
            except Exception as e:
                self.get_logger().warn(f'LaneNet inference failed: {e}',
                                       throttle_duration_sec=5.0)

        # ── Fallback: Canny + HoughLinesP ─────────────────────────────
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(grey, 100, 200)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=50, minLineLength=30,
                                maxLineGap=80)
        mask = np.zeros(grey.shape, dtype=np.uint8)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
        return mask

    # ================================================================== #
    #  FIND TARGETS (Sections E.1 – E.6)
    # ================================================================== #
    def _find_targets(self, blobs, v):
        """Return dict {id: np.array([x, y])} of lane-centre targets."""
        targets_list = []
        isolated_lanes = []

        # E.1 — look-ahead distance
        ld_m = np.clip(self.Kdd * abs(v), self.ld_min, self.ld_max)
        self._ld_pix = int(ld_m / self.m_per_pix)
        if self._ld_pix < 5:
            self._ld_pix = int(self.ld_min / self.m_per_pix)

        # Build LaneMarking objects and find intersections (E.2)
        for blob in blobs:
            lm = LaneMarking(self.bev_rear_wheel_pos, blob)
            lm.find_intersection(self._ld_pix)
            if lm.intersection is not None:
                isolated_lanes.append(lm)

        if not isolated_lanes:
            return {}

        # Sort CCW (leftmost first in image → highest column)
        center = (self.bev_shape[0] // 2, self.bev_shape[1])
        isolated_lanes.sort(key=ccw_key(center))

        # ── Case I: single lane marking (E.3–E.4) ────────────────────
        if len(isolated_lanes) == 1:
            lane = isolated_lanes[0]
            if lane.intersection[0] > self.bev_rear_wheel_pos[0]:
                vec = np.array([-75.0, 0.0])   # lane is right → target left
            else:
                vec = np.array([ 75.0, 0.0])   # lane is left  → target right

            rot = find_rot_mat(lane)
            target = lane.intersection + rot @ vec
            targets_list.append(target)

        # ── Multiple lane markings (E.5–E.6) ─────────────────────────
        else:
            prev_lane = isolated_lanes[0]
            for i in range(1, len(isolated_lanes)):
                lane = isolated_lanes[i]
                delta = lane.intersection - prev_lane.intersection
                dist = np.linalg.norm(delta)

                if 80 < dist < 300:
                    # Case II: adjacent markings → midpoint
                    target = (lane.intersection + prev_lane.intersection) / 2.0
                    targets_list.append(target)

                elif dist >= 300:
                    # Case III: two lanes far apart → two Case-I targets
                    rot1 = find_rot_mat(lane)
                    rot2 = find_rot_mat(prev_lane)

                    if lane.intersection[0] > prev_lane.intersection[0]:
                        v1 = np.array([-75.0, 0.0])
                        v2 = np.array([ 75.0, 0.0])
                    else:
                        v1 = np.array([ 75.0, 0.0])
                        v2 = np.array([-75.0, 0.0])

                    targets_list.append(lane.intersection      + rot1 @ v1)
                    targets_list.append(prev_lane.intersection + rot2 @ v2)

                prev_lane = lane

            # Fallback to Case I if no targets found from multi-lane logic
            if not targets_list and isolated_lanes:
                lane = isolated_lanes[0]
                if lane.intersection[0] > self.bev_rear_wheel_pos[0]:
                    vec = np.array([-75.0, 0.0])
                else:
                    vec = np.array([ 75.0, 0.0])
                rot = find_rot_mat(lane)
                targets_list.append(lane.intersection + rot @ vec)

        result = {}
        for idx, pt in enumerate(targets_list):
            result[10 + idx] = pt
        return result

    # ================================================================== #
    #  DEBUG DRAWING
    # ================================================================== #
    def _draw_debug(self, bev_rgb, blobs, targets, selected):
        """Draw look-ahead arc, intersection points, and targets on BEV."""
        debug = bev_rgb.copy()
        if len(debug.shape) == 2:
            debug = cv2.cvtColor(debug, cv2.COLOR_GRAY2BGR)

        # Blue arc = look-ahead distance from rear wheel
        ld = self._ld_pix
        cv2.circle(debug, tuple(self.bev_rear_wheel_pos), ld, (255, 0, 0), 1)

        # White: lane blobs overlaid
        for blob in blobs:
            b3 = cv2.cvtColor(blob, cv2.COLOR_GRAY2BGR) if len(blob.shape) == 2 else blob
            debug = cv2.addWeighted(debug, 1.0, b3, 0.4, 0)

        # Green circles = all targets
        if targets:
            for tid, pt in targets.items():
                cv2.circle(debug, pt.astype(int), 6, (0, 255, 0), -1)

        # Red circle = selected target
        if selected is not None:
            cv2.circle(debug, selected.astype(int), 8, (0, 0, 255), -1)

        return debug

    # ================================================================== #
    #  UTILITY
    # ================================================================== #
    def _publish_img(self, pub, img, encoding):
        try:
            pub.publish(self.bridge.cv2_to_imgmsg(img, encoding=encoding))
        except Exception:
            pass

# endregion


# ============================================================================
# region : Entry Point
# ============================================================================
def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down lane detector.')
    finally:
        # Send zero command on shutdown
        cmd = Twist()
        node.cmd_publisher.publish(cmd)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

# endregion