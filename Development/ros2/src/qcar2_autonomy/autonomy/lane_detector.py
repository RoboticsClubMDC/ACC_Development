#!/usr/bin/env python3
"""
lane_detector.py

ROS2 Lane Detection Node for QCar2 — single yellow lane mode.
Bypasses SDK find_target (flaky with 1 lane) and computes CTE/heading
directly from BEV lane mask using line fitting.

Case I from lab (1 lane marking):
  - Fit a line through the detected lane pixels in BEV
  - Offset perpendicular to the lane line (drive to right of yellow center line)
  - CTE = lateral offset from desired position
  - Heading = angle of lane line vs straight ahead

Published Topics:
    /lane_keeping/cross_track_error  (Float64) filtered lateral offset [m]
    /lane_keeping/heading_error      (Float64) filtered heading offset [rad]
    /lane_keeping/lane_detected      (Bool)    whether lane line is visible
    /lane_keeping/bev_binary         (Image)   BEV lane binary with CTE/heading text
    /lane_keeping/debug              (Image)   BEV RGB with center line, lane line, target
    /lane_keeping/camera_image       (Image)   raw camera with lane overlay

Subscribed Topics:
    /camera/color_image              Intel RealSense D435 RGB
    /qcar2_joint                     Motor encoder for velocity
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64, Bool
from cv_bridge import CvBridge
import sys
sys.path.insert(0, "/workspaces/isaac_ros-dev/MDC_libraries/python")
import cv2
import numpy as np
import math
import os

from hal.content.qcar_functions import LaneKeeping, LaneSelection

import torch
import torchvision.transforms as transforms


# ============================================================================
#  PureTorchLaneNet
# ============================================================================
class PureTorchLaneNet:

    def __init__(self, modelPath, imageHeight=480, imageWidth=640,
                 rowUpperBound=240):
        self.imageHeight = imageHeight
        self.imageWidth = imageWidth
        self.rowUpperBound = rowUpperBound
        self.imgTransforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 512)),
            transforms.ToTensor(),
        ])
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(modelPath, map_location=self.device)
        self.model.eval()
        self.binaryPred = np.zeros(
            (imageHeight, imageWidth), dtype=np.uint8)

    def pre_process(self, inputImg):
        if inputImg.shape[:2] != (self.imageHeight, self.imageWidth):
            inputImg = cv2.resize(
                inputImg, (self.imageWidth, self.imageHeight),
                interpolation=cv2.INTER_LINEAR)
        self.imgClone = inputImg
        rgb = cv2.cvtColor(
            inputImg[self.rowUpperBound:, :, :], cv2.COLOR_BGR2RGB)
        self.imgTensor = self.imgTransforms(rgb)
        return self.imgTensor

    @torch.no_grad()
    def predict(self, inputImg):
        x = inputImg.unsqueeze(0).to(self.device)
        outputs = self.model(x)
        binary_np = (outputs['binary_seg_pred']
                     .squeeze().cpu().numpy() * 255).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        binary_np = cv2.morphologyEx(binary_np, cv2.MORPH_OPEN, kernel)

        # CCA to remove small blobs
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_np, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 200:
                binary_np[labels == i] = 0

        self.binaryPred = np.zeros(
            (self.imageHeight, self.imageWidth), dtype=np.uint8)
        resized = cv2.resize(
            binary_np,
            (self.imageWidth, self.imageHeight - self.rowUpperBound),
            interpolation=cv2.INTER_LINEAR)
        _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
        self.binaryPred[self.rowUpperBound:, :] = resized
        return self.binaryPred


# ============================================================================
#  LaneDetectorNode
# ============================================================================
class LaneDetectorNode(Node):

    def __init__(self):
        super().__init__('lane_detector')

        # ── BEV Configuration ────────────────────────────────────────
        self.bevShape = [800, 800]
        bevWorldDims = [0, 20, -10, 10]
        self.m_per_pix = (bevWorldDims[1] - bevWorldDims[0]) / self.bevShape[0]

        # ── Vehicle reference in BEV ─────────────────────────────────
        # Rear wheel at vehicle origin (0,0) → BEV pixel:
        #   col = (world_max_y - 0) / m_per_pix = 10 / 0.025 = 400 (center)
        #   row = bevShape[0] - (0 - world_min_x) / m_per_pix = 800 (bottom)
        self.car_center_col = self.bevShape[1] // 2  # 400
        self.car_bottom_row = self.bevShape[0]       # 800

        # ── Single lane (Case I) offset ──────────────────────────────
        # Yellow center line → drive to RIGHT of it
        # 75px offset at 0.025 m/px = 1.875m (half lane width)
        self.LANE_OFFSET_PX = 75  # pixels to the right of detected lane

        # ── Look-ahead for CTE measurement ───────────────────────────
        # Measure lane position at this row (lower = closer to car, more stable)
        # Row 600 = ~5m ahead, Row 400 = ~10m ahead
        self.LOOKAHEAD_ROW = 500  # ~7.5m ahead

        # ── EMA filter ────────────────────────────────────────────────
        self._ema_alpha = 0.3
        self._filtered_cte = 0.0
        self._filtered_heading = 0.0
        self._no_detect_count = 0
        self._no_detect_max = 5

        model_path = ('/workspaces/isaac_ros-dev/'
                      'ros2/src/qcar2_autonomy/models/lanenet.pt')

        # ── Build SDK (only for IPM — we skip find_target) ───────────
        self.lane_keeping = LaneKeeping(
            Kdd=2, ldMin=10, ldMax=20, maxSteer=0.05,
            bevShape=self.bevShape, bevWorldDims=bevWorldDims)

        # ── Patch IPM (stub has zero intrinsics) ─────────────────────
        ipm = self.lane_keeping.ipm

        ipm.camera_intrinsics = np.array([
            [483.671,       0, 321.188, 0],
            [      0, 483.579, 238.462, 0],
            [      0,       0,       1, 0],
            [      0,       0,       0, 1]])

        phi, theta, psi, height = np.pi/2, 0.0, np.pi/2, 1.72
        cx, sx = np.cos(phi), np.sin(phi)
        cy, sy = np.cos(theta), np.sin(theta)
        cz, sz = np.cos(psi), np.sin(psi)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        R_v2cam = Rx @ Ry @ Rz
        t_v2cam = np.array([[0, height, 0]]).T
        T_v2cam = np.vstack((np.hstack((R_v2cam, t_v2cam)),
                             np.array([[0, 0, 0, 1]])))
        ipm.T_v2cam = T_v2cam
        ipm.camera_extrinsics = T_v2cam

        def v2img(XYZ):
            XYZ1 = np.hstack((XYZ, np.ones((XYZ.shape[0], 1))))
            img_coords = ipm.camera_intrinsics @ T_v2cam @ XYZ1.T
            img_coords /= img_coords[2]
            return img_coords[:2, :].astype(int).T
        ipm.v2img = v2img

        world_max_x, world_max_y = 15, 3
        world_min_x, world_min_y = 3, -3
        rgb_corners = v2img(np.array([
            [world_max_x, world_max_y, 0],
            [world_min_x, world_max_y, 0],
            [world_max_x, world_min_y, 0],
            [world_min_x, world_min_y, 0]]))

        m = ipm.m_per_pix
        bev_corners = np.array([
            [(bevWorldDims[3] - world_max_y) / m,
             (bevWorldDims[1] - world_max_x) / m],
            [(bevWorldDims[3] - world_max_y) / m,
             self.bevShape[0] - (world_min_x - bevWorldDims[0]) / m],
            [self.bevShape[0] - (world_min_y - bevWorldDims[2]) / m,
             (bevWorldDims[1] - world_max_x) / m],
            [self.bevShape[0] - (world_min_y - bevWorldDims[2]) / m,
             self.bevShape[0] - (world_min_x - bevWorldDims[0]) / m]])

        ipm.M = cv2.getPerspectiveTransform(
            rgb_corners.astype(np.float32),
            bev_corners.astype(np.float32))
        ipm.bevShape = self.bevShape
        ipm.world_dims = bevWorldDims
        ipm.bevWorldDims = bevWorldDims

        self.get_logger().info(
            f'IPM patched | m_per_pix={ipm.m_per_pix} | '
            f'car_center_col={self.car_center_col} | '
            f'lane_offset={self.LANE_OFFSET_PX}px = '
            f'{self.LANE_OFFSET_PX * self.m_per_pix:.2f}m')

        # ── LaneNet ──────────────────────────────────────────────────
        self.lanenet = None
        if os.path.isfile(model_path):
            try:
                self.lanenet = PureTorchLaneNet(
                    modelPath=model_path, rowUpperBound=240,
                    imageWidth=640, imageHeight=480)
                self.get_logger().info(f'LaneNet loaded from {model_path}')
            except Exception as e:
                self.get_logger().error(f'LaneNet init failed: {e}')
        else:
            self.get_logger().error(f'Model not found: {model_path}')

        # ── State ─────────────────────────────────────────────────────
        self.current_v = 0.0
        self.bridge = CvBridge()

        # ── Subscribers ───────────────────────────────────────────────
        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(
            Image, '/camera/csi_image', self._image_cb, img_qos)
        self.create_subscription(
            JointState, '/qcar2_joint', self._joint_state_cb, 1)

        # ── Publishers ────────────────────────────────────────────────
        self.pub_cte = self.create_publisher(
            Float64, '/lane_keeping/cross_track_error', 1)
        self.pub_heading = self.create_publisher(
            Float64, '/lane_keeping/heading_error', 1)
        self.pub_detected = self.create_publisher(
            Bool, '/lane_keeping/lane_detected', 1)
        self.pub_debug = self.create_publisher(
            Image, '/lane_keeping/debug', 10)
        self.pub_bev_bin = self.create_publisher(
            Image, '/lane_keeping/bev_binary', 10)
        self.pub_camera = self.create_publisher(
            Image, '/lane_keeping/camera_image', 10)

        self.get_logger().info('Lane detector started (single yellow lane mode)')

    # ================================================================ #
    def _joint_state_cb(self, msg):
        if msg.velocity:
            self.current_v = msg.velocity[0] * 10

    # ================================================================ #
    #  YELLOW COLOR FILTER — reject white lanes / sidewalks
    # ================================================================ #
    def _filter_yellow(self, bgr_img, binary_mask):
        """AND LaneNet binary with HSV yellow mask. Keeps only largest blob."""
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

        # Wide yellow range for QLabs sim
        yellow_mask = cv2.inRange(
            hsv,
            np.array([5, 20, 60], dtype=np.uint8),
            np.array([55, 255, 255], dtype=np.uint8))

        # Generous dilate so lane edges aren't clipped
        yellow_mask = cv2.dilate(
            yellow_mask, np.ones((11, 11), np.uint8), iterations=2)

        # AND: only lane pixels that are also yellow
        filtered = cv2.bitwise_and(binary_mask, yellow_mask)

        # Keep only the single largest blob (one lane at a time)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            filtered, connectivity=8)
        if num_labels <= 1:
            return filtered
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        result = np.zeros_like(filtered)
        result[labels == largest] = 255
        return result

    # ================================================================ #
    #  LINE FITTING — stable CTE/heading from BEV lane mask
    # ================================================================ #
    def _fit_lane_line(self, bev_binary):
        """Fit a line through the lane pixels in BEV.
        Returns (lane_vx, lane_vy, lane_x0, lane_y0, success)
        where (vx,vy) is the direction vector and (x0,y0) is a point on the line.
        """
        # Find all white pixels (lane pixels)
        pts = cv2.findNonZero(bev_binary)
        if pts is None or len(pts) < 50:
            return 0, 0, 0, 0, False

        # cv2.fitLine returns [vx, vy, x0, y0]
        # vx,vy = normalized direction vector
        # x0,y0 = a point on the fitted line
        line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        vx = float(line[0])
        vy = float(line[1])
        x0 = float(line[2])
        y0 = float(line[3])

        return vx, vy, x0, y0, True

    def _compute_errors(self, bev_binary):
        """Compute CTE and heading error from BEV lane mask.

        Case I (single lane):
        1. Fit line through lane pixels
        2. Find lane x-position at look-ahead row
        3. Target = lane_x + LANE_OFFSET_PX (offset to right of center line)
        4. CTE = (target_x - car_center_x) * m_per_pix
        5. Heading = angle of lane line vs vertical (straight ahead)
        """
        vx, vy, x0, y0, success = self._fit_lane_line(bev_binary)
        if not success:
            return 0.0, 0.0, False, None

        # Lane x-position at the look-ahead row
        # Parametric: x = x0 + t*vx, y = y0 + t*vy
        # At row = LOOKAHEAD_ROW: t = (LOOKAHEAD_ROW - y0) / vy
        if abs(vy) < 1e-6:
            # Lane is perfectly horizontal — shouldn't happen in BEV
            return 0.0, 0.0, False, None

        t_la = (self.LOOKAHEAD_ROW - y0) / vy
        lane_x_at_la = x0 + t_la * vx

        # Case I: single lane — offset to RIGHT of yellow center line
        target_x = lane_x_at_la + self.LANE_OFFSET_PX

        # CTE: positive = target is right of car center → steer right
        cte = (target_x - self.car_center_col) * self.m_per_pix

        # Heading error: angle of lane line vs vertical (straight ahead)
        # In BEV, straight ahead = negative y direction (up the image)
        # Lane direction: (vx, vy). Straight ahead: (0, -1)
        # Ensure direction vector points "forward" (upward in BEV = negative y)
        if vy > 0:
            vx, vy = -vx, -vy
        heading = math.atan2(vx, -vy)  # angle from straight-ahead

        # Line info for visualization
        line_info = {
            'vx': vx, 'vy': vy, 'x0': x0, 'y0': y0,
            'lane_x_at_la': lane_x_at_la,
            'target_x': target_x,
        }

        return cte, heading, True, line_info

    # ================================================================ #
    #  IMAGE CALLBACK
    # ================================================================ #
    def _image_cb(self, msg: Image):
        if msg.width == 0 or msg.height == 0:
            return
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return
        if img is None or img.size == 0:
            return

        # ── LaneNet inference ─────────────────────────────────────────
        if self.lanenet is None:
            return
        # Resize to 640x480 (LaneNet + IPM expect this)
        if img.shape[:2] != (480, 640):
            img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
        try:
            rgbProcessed = self.lanenet.pre_process(img)
            self.lanenet.predict(rgbProcessed)
            laneMarking = self.lanenet.binaryPred
        except Exception:
            return

        # ── Yellow-only filter (keeps largest blob only) ──────────────
        laneMarking = self._filter_yellow(img, laneMarking)

        # ── Camera image with lane overlay ────────────────────────────
        cam_overlay = img.copy()
        cam_overlay[laneMarking > 0] = [0, 255, 0]
        try:
            self.pub_camera.publish(
                self.bridge.cv2_to_imgmsg(cam_overlay, encoding='bgr8'))
        except Exception:
            pass

        # ── BEV transform ─────────────────────────────────────────────
        bev = self.lane_keeping.ipm.create_bird_eye_view(img)
        bevLM = self.lane_keeping.ipm.create_bird_eye_view(laneMarking)

        # ── Preprocess BEV lane mask (SDK cleanup) ────────────────────
        if len(bevLM.shape) == 3:
            bevLM = cv2.cvtColor(bevLM, cv2.COLOR_BGR2GRAY)
        processedLM = self.lane_keeping.preprocess(bevLM)

        # ── Compute errors via line fitting (bypasses SDK find_target) ─
        raw_cte, raw_heading, detected, line_info = \
            self._compute_errors(processedLM)

        # ── EMA filter ────────────────────────────────────────────────
        if detected:
            self._no_detect_count = 0
            a = self._ema_alpha
            self._filtered_cte = a * raw_cte + (1 - a) * self._filtered_cte
            self._filtered_heading = (a * raw_heading
                                      + (1 - a) * self._filtered_heading)
        else:
            self._no_detect_count += 1
            if self._no_detect_count > self._no_detect_max:
                self._filtered_cte *= 0.9
                self._filtered_heading *= 0.9

        # ── Publish errors ────────────────────────────────────────────
        cte_msg = Float64()
        cte_msg.data = float(self._filtered_cte)
        self.pub_cte.publish(cte_msg)

        heading_msg = Float64()
        heading_msg.data = float(self._filtered_heading)
        self.pub_heading.publish(heading_msg)

        detected_msg = Bool()
        detected_msg.data = detected
        self.pub_detected.publish(detected_msg)

        # ── BEV binary with text overlay ──────────────────────────────
        bev_bin_vis = cv2.cvtColor(processedLM, cv2.COLOR_GRAY2BGR) \
            if len(processedLM.shape) == 2 else processedLM.copy()
        self._draw_text_overlay(bev_bin_vis)
        try:
            self.pub_bev_bin.publish(
                self.bridge.cv2_to_imgmsg(bev_bin_vis, encoding='bgr8'))
        except Exception:
            pass

        # ── Debug overlay ─────────────────────────────────────────────
        debug = bev.copy() if bev is not None else \
            np.zeros((800, 800, 3), dtype=np.uint8)
        if len(debug.shape) == 2:
            debug = cv2.cvtColor(debug, cv2.COLOR_GRAY2BGR)

        # RED vertical center line (vehicle heading direction)
        cv2.line(debug,
                 (self.car_center_col, self.bevShape[0]),  # bottom center
                 (self.car_center_col, 0),                  # top center
                 (0, 0, 255), 2)

        # Fitted lane line + target (if detected)
        if detected and line_info is not None:
            vx = line_info['vx']
            vy = line_info['vy']
            x0 = line_info['x0']
            y0 = line_info['y0']
            lane_x = line_info['lane_x_at_la']
            target_x = line_info['target_x']

            # YELLOW fitted lane line (extend across full BEV height)
            t_top = (0 - y0) / vy if abs(vy) > 1e-6 else 0
            t_bot = (self.bevShape[0] - y0) / vy if abs(vy) > 1e-6 else 0
            pt_top = (int(x0 + t_top * vx), 0)
            pt_bot = (int(x0 + t_bot * vx), self.bevShape[0])
            cv2.line(debug, pt_top, pt_bot, (0, 255, 255), 2)

            # GREEN circle = lane intersection at look-ahead
            cv2.circle(debug,
                       (int(lane_x), self.LOOKAHEAD_ROW),
                       8, (0, 255, 0), -1)

            # BLUE circle = target (lane center = lane + offset)
            cv2.circle(debug,
                       (int(target_x), self.LOOKAHEAD_ROW),
                       10, (255, 100, 0), -1)

            # Look-ahead horizontal line
            cv2.line(debug,
                     (0, self.LOOKAHEAD_ROW),
                     (self.bevShape[1], self.LOOKAHEAD_ROW),
                     (100, 100, 100), 1)

        # Text overlay
        self._draw_text_overlay(debug)

        try:
            self.pub_debug.publish(
                self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))
        except Exception:
            pass

    # ================================================================ #
    def _draw_text_overlay(self, img):
        """Draw CTE/heading text on an image."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 0, 255)  # red text
        cv2.putText(img,
                    f'CTE: {self._filtered_cte:+.3f} m',
                    (10, 30), font, 0.8, color, 2)
        cv2.putText(img,
                    f'Heading: {math.degrees(self._filtered_heading):+.1f} deg',
                    (10, 65), font, 0.8, color, 2)


# ============================================================================
def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down lane detector.')
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()