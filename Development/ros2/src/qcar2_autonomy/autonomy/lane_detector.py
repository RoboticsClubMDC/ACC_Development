#!/usr/bin/env python3
"""
lane_detector.py

ROS2 Lane Detection (Perception-Only) Node for QCar2.
Publishes EMA-filtered cross-track error and heading error for nav_to_pose.

Pipeline (from Quanser Lane Keeping Lab Guide):
  Camera -> LaneNet -> BEV (IPM) -> Preprocess -> Isolate -> Targets -> Errors

Published Topics:
    /lane_keeping/cross_track_error  (Float64) filtered lateral offset [m]
    /lane_keeping/heading_error      (Float64) filtered heading offset [rad]
    /lane_keeping/lane_detected      (Bool)    whether lanes are visible
    /lane_keeping/debug              (Image)   debug overlay (optional viz)

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
#  PureTorchLaneNet — drop-in for pit.LaneNet.nets.LaneNet (no TensorRT)
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

        # Minimal cleanup for FP32 speckles
        kernel = np.ones((5, 5), np.uint8)
        binary_np = cv2.morphologyEx(binary_np, cv2.MORPH_OPEN, kernel)

        # Remove small blobs (non-lane noise)
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

        # ── Configuration (matches lab code) ──────────────────────────
        bevShape = [800, 800]
        bevWorldDims = [0, 20, -10, 10]
        Kdd = 2.0
        ldMin = 10.0
        ldMax = 20.0
        maxSteer = 0.05

        model_path = ('/workspaces/isaac_ros-dev/'
                      'ros2/src/qcar2_autonomy/models/lanenet.pt')

        # ── EMA filter (anti-oscillation) ─────────────────────────────
        # alpha = 0.0 → full smoothing; alpha = 1.0 → no filter
        # 0.25 gives ~4-frame smoothing window — kills jitter,
        # responds within ~0.5s at camera rate
        self._ema_alpha = 0.25
        self._filtered_cte = 0.0
        self._filtered_heading = 0.0
        self._no_detect_count = 0
        self._no_detect_max = 5

        # ── Build SDK objects ─────────────────────────────────────────
        self.lane_keeping = LaneKeeping(
            Kdd=Kdd, ldMin=ldMin, ldMax=ldMax,
            maxSteer=maxSteer, bevShape=bevShape,
            bevWorldDims=bevWorldDims)
        self.selector = LaneSelection()

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
             bevShape[0] - (world_min_x - bevWorldDims[0]) / m],
            [bevShape[0] - (world_min_y - bevWorldDims[2]) / m,
             (bevWorldDims[1] - world_max_x) / m],
            [bevShape[0] - (world_min_y - bevWorldDims[2]) / m,
             bevShape[0] - (world_min_x - bevWorldDims[0]) / m]])

        ipm.M = cv2.getPerspectiveTransform(
            rgb_corners.astype(np.float32),
            bev_corners.astype(np.float32))
        ipm.bevShape = bevShape
        ipm.world_dims = bevWorldDims
        ipm.bevWorldDims = bevWorldDims

        self.get_logger().info(
            f'IPM patched | m_per_pix={ipm.m_per_pix}')

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
            Image, '/camera/color_image', self._image_cb, img_qos)
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

        self.get_logger().info(
            f'Lane detector started | BEV {bevShape} '
            f'world {bevWorldDims} | EMA alpha={self._ema_alpha}')

    # ================================================================ #
    #  CALLBACKS
    # ================================================================ #
    def _joint_state_cb(self, msg):
        if msg.velocity:
            self.current_v = msg.velocity[0] * 10

    def _image_cb(self, msg: Image):
        if msg.width == 0 or msg.height == 0:
            return
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return
        if img is None or img.size == 0:
            return

        v = self.current_v

        # ── LaneNet ───────────────────────────────────────────────────
        if self.lanenet is None:
            return
        try:
            rgbProcessed = self.lanenet.pre_process(img)
            self.lanenet.predict(rgbProcessed)
            laneMarking = self.lanenet.binaryPred

            # ── Yellow-only filter ────────────────────────────────────
            # Keep only lane pixels that are yellow in the original image
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            yellow_mask = cv2.inRange(
                hsv,
                (15, 50, 100),   # lower HSV bound (tune if needed)
                (35, 255, 255))  # upper HSV bound
            # Dilate the yellow mask so it covers the full lane width
            yellow_mask = cv2.dilate(
                yellow_mask, np.ones((15, 15), np.uint8), iterations=1)
            laneMarking = cv2.bitwise_and(laneMarking, yellow_mask)
        except Exception:
            return

        # ── BEV ───────────────────────────────────────────────────────
        # Publish camera image with lane markings overlaid
        cam_overlay = img.copy()
        cam_overlay[laneMarking > 0] = [0, 255, 0]  # green overlay
        try:
            self.pub_camera.publish(
                self.bridge.cv2_to_imgmsg(cam_overlay, encoding='bgr8'))
        except Exception:
            pass

        bev = self.lane_keeping.ipm.create_bird_eye_view(img)
        bevLM = self.lane_keeping.ipm.create_bird_eye_view(laneMarking)

        # ── SDK pipeline ──────────────────────────────────────────────
        if len(bevLM.shape) == 3:
            bevLM = cv2.cvtColor(bevLM, cv2.COLOR_BGR2GRAY)
        processedLM = self.lane_keeping.preprocess(bevLM)
        isolated = self.lane_keeping.isolate_lane_markings(processedLM)
        try:
            targets = self.lane_keeping.find_target(isolated, v)
        except (IndexError, ValueError):
            targets = {}

        # ── Raw errors ────────────────────────────────────────────────
        lane_detected = len(targets) > 0
        raw_cte = 0.0
        raw_heading = 0.0
        selected = None

        if lane_detected:
            self._no_detect_count = 0
            bev_cx = bev.shape[1] // 2
            best_id = min(targets,
                          key=lambda t: abs(targets[t][0] - bev_cx))
            selected = targets[best_id]
            raw_heading = self.lane_keeping.pp.target2steer(selected)
            raw_cte = (selected[0] - bev_cx) * self.lane_keeping.ipm.m_per_pix
        else:
            self._no_detect_count += 1

        # ── EMA filter ────────────────────────────────────────────────
        if lane_detected:
            a = self._ema_alpha
            self._filtered_cte = (a * raw_cte
                                  + (1 - a) * self._filtered_cte)
            self._filtered_heading = (a * raw_heading
                                      + (1 - a) * self._filtered_heading)
        elif self._no_detect_count > self._no_detect_max:
            # Decay to zero when lanes gone for several frames
            self._filtered_cte *= 0.9
            self._filtered_heading *= 0.9

        # ── Publish ───────────────────────────────────────────────────
        cte_msg = Float64()
        cte_msg.data = float(self._filtered_cte)
        self.pub_cte.publish(cte_msg)

        heading_msg = Float64()
        heading_msg.data = float(self._filtered_heading)
        self.pub_heading.publish(heading_msg)

        detected_msg = Bool()
        detected_msg.data = lane_detected
        self.pub_detected.publish(detected_msg)

        # ── Debug overlay ─────────────────────────────────────────────
        # Publish BEV binary lane markings
        try:
            if processedLM is not None and processedLM.size > 0:
                pub_lm = processedLM
                if len(pub_lm.shape) == 3:
                    pub_lm = cv2.cvtColor(pub_lm, cv2.COLOR_BGR2GRAY)
                self.pub_bev_bin.publish(
                    self.bridge.cv2_to_imgmsg(pub_lm, encoding='mono8'))
        except Exception:
            pass

        debug = bev.copy() if bev is not None else \
            np.zeros((800, 800, 3), dtype=np.uint8)
        if len(debug.shape) == 2:
            debug = cv2.cvtColor(debug, cv2.COLOR_GRAY2BGR)
        for tid in targets:
            cv2.circle(debug, targets[tid].astype(int), 10, (0, 255, 0), -1)
        if selected is not None:
            cv2.circle(debug, selected.astype(int), 10, (0, 0, 255), -1)
        cv2.putText(debug, f'CTE: {self._filtered_cte:+.3f}m',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 1)
        cv2.putText(debug,
                    f'Hdg: {math.degrees(self._filtered_heading):+.1f}deg',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 1)
        try:
            self.pub_debug.publish(
                self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))
        except Exception:
            pass


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