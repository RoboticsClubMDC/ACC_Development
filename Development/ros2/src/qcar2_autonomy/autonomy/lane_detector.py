#!/usr/bin/env python3
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# region : File Description and Imports
"""
lane_detector.py

ROS2 Lane Detection (Perception-Only) Node for QCar2.
Thin ROS2 wrapper around Quanser's LaneKeeping, LaneSelection, and LaneNet
from hal.content.qcar2_functions and pit.LaneNet.nets.

This node does NOT control the car. It publishes lane errors that
nav_to_pose.py (path_follower) can subscribe to and blend into its
pure-pursuit steering as a lane-keeping correction.

Pipeline (from Quanser Lane Keeping Lab Guide):
  Camera -> LaneNet -> BEV (IPM) -> Preprocess -> Isolate -> Targets -> Errors

Published Topics:
    /lane_keeping/cross_track_error  - Lateral offset from lane centre (Float64, metres)
                                       positive = car is RIGHT of lane centre
    /lane_keeping/heading_error      - Angle between car heading and lane direction
                                       (Float64, radians) positive = car aimed right of lane
    /lane_keeping/lane_detected      - Whether valid lane markings are visible (Bool)
    /lane_keeping/binary_image       - Raw lane binary from LaneNet (mono8)
    /lane_keeping/bev_rgb            - Bird's-eye view of camera feed (bgr8)
    /lane_keeping/bev_binary         - BEV lane markings binary (mono8)
    /lane_keeping/debug              - Debug overlay with targets and ld arc (bgr8)

Subscribed Topics:
    /camera/color_image              - Intel RealSense D435 RGB (sensor_msgs/Image)
    /qcar2_joint                     - Motor encoder for speed-dependent look-ahead
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

# endregion


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# region : Quanser SDK Imports

# LaneKeeping + LaneSelection from qcar_functions
# (has IPM, preprocess, isolate, find_target, pp, and all camera
#  intrinsics/extrinsics/BEV transforms built in)
from hal.content.qcar_functions import LaneKeeping, LaneSelection

# LaneNet — pure PyTorch replacement (no TensorRT)
# Original: from pit.LaneNet.nets import LaneNet
# Replaced due to TensorRT/CUDA version mismatch in container
import torch
import torchvision.transforms as transforms


class PureTorchLaneNet:
    """Drop-in replacement for pit.LaneNet.nets.LaneNet using pure PyTorch.
    Same pre_process / predict / binaryPred interface, zero TensorRT."""

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

        # Morphological kernel for denoising binary output
        self._morph_kernel = np.ones((5, 5), np.uint8)

    def pre_process(self, inputImg):
        if inputImg.shape[:2] != (self.imageHeight, self.imageWidth):
            inputImg = cv2.resize(
                inputImg, (self.imageWidth, self.imageHeight),
                interpolation=cv2.INTER_LINEAR)
        self.imgClone = inputImg
        self.imgTensor = self.imgTransforms(
            inputImg[self.rowUpperBound:, :, :])
        return self.imgTensor

    @torch.no_grad()
    def predict(self, inputImg):
        x = inputImg.unsqueeze(0).to(self.device)
        outputs = self.model(x)
        # Model returns dict with keys:
        #   binary_seg_pred:    [1, 1, 256, 512] int64
        #   binary_seg_logits:  [1, 2, 256, 512] float32
        #   instance_seg_logits:[1, 3, 256, 512] float32
        binary_np = (outputs['binary_seg_pred']
                     .squeeze().cpu().numpy() * 255).astype(np.uint8)

        # Denoise: morphological open (erode then dilate) removes small
        # noise blobs while preserving lane line structure
        binary_np = cv2.morphologyEx(
            binary_np, cv2.MORPH_OPEN, self._morph_kernel)

        # Resize to match original image dimensions
        self.binaryPred = np.zeros(
            (self.imageHeight, self.imageWidth), dtype=np.uint8)
        resized = cv2.resize(
            binary_np,
            (self.imageWidth, self.imageHeight - self.rowUpperBound),
            interpolation=cv2.INTER_LINEAR)

        # Threshold after resize to keep binary clean
        _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

        self.binaryPred[self.rowUpperBound:, :] = resized
        return self.binaryPred

# endregion


# ============================================================================
# region : Main ROS 2 Node
# ============================================================================
class LaneDetectorNode(Node):

    def __init__(self):
        super().__init__('lane_detector')

        # ── Configuration ──────────────────────────────────────────────
        # Matches original lab code parameters exactly.
        # LaneKeeping SDK (qcar2_functions) handles all camera intrinsics,
        # extrinsics, and IPM (bird's-eye view) transforms internally.

        # Bird's-Eye View parameters (passed to LaneKeeping SDK)
        bevShape = [800, 800]
        bevWorldDims = [0, 20, -10, 10]

        # Lane Keeping parameters
        Kdd = 2.0
        ldMin = 10.0
        ldMax = 20.0
        maxSteer = 0.05

        # Image dimensions (RealSense D435 RGB)
        self.img_w = 640
        self.img_h = 480

        model_path = ('/workspaces/isaac_ros-dev/'
                      'ros2/src/qcar2_autonomy/models/lanenet.pt')

        # ── Build SDK objects (same as original lab code) ──────────────
        self.lane_keeping = LaneKeeping(
            Kdd=Kdd,
            ldMin=ldMin,
            ldMax=ldMax,
            maxSteer=maxSteer,
            bevShape=bevShape,
            bevWorldDims=bevWorldDims,
        )
        self.selector = LaneSelection()

        # ── Patch IPM with correct camera params ──────────────────────
        # The installed qcar_functions has a stub IPM with zero intrinsics.
        # We patch it here with the correct values from the lab source.
        ipm = self.lane_keeping.ipm

        # Section B.1 — Camera Intrinsics (Intel RealSense D435)
        ipm.camera_intrinsics = np.array([
            [483.671,       0, 321.188, 0],
            [      0, 483.579, 238.462, 0],
            [      0,       0,       1, 0],
            [      0,       0,       0, 1]
        ])

        # Section B.2 — Camera Extrinsics (vehicle to camera transform)
        phi = np.pi / 2
        theta = 0.0
        psi = np.pi / 2
        height = 1.72

        cx, sx = np.cos(phi), np.sin(phi)
        cy, sy = np.cos(theta), np.sin(theta)
        cz, sz = np.cos(psi), np.sin(psi)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        R_v2cam = Rx @ Ry @ Rz
        t_v2cam = np.array([[0, height, 0]]).T
        T_v2cam = np.vstack((
            np.hstack((R_v2cam, t_v2cam)),
            np.array([[0, 0, 0, 1]])
        ))
        ipm.T_v2cam = T_v2cam
        ipm.camera_extrinsics = T_v2cam

        # Section B.3 — v2img: vehicle coords to image pixel coords
        def v2img(XYZ):
            XYZ1 = np.hstack((XYZ, np.ones((XYZ.shape[0], 1))))
            img_coords = ipm.camera_intrinsics @ T_v2cam @ XYZ1.T
            img_coords /= img_coords[2]
            uv = img_coords[:2, :].astype(int)
            return uv.T
        ipm.v2img = v2img

        # Section B.4 — Compute homography
        world_max_x, world_max_y = 15, 3
        world_min_x, world_min_y = 3, -3
        corners = np.array([
            [world_max_x, world_max_y, 0],
            [world_min_x, world_max_y, 0],
            [world_max_x, world_min_y, 0],
            [world_min_x, world_min_y, 0],
        ])
        rgb_corners = v2img(corners)

        bev_world_min_x = ipm.world_dims[0]
        bev_world_max_x = ipm.world_dims[1]
        bev_world_min_y = ipm.world_dims[2]
        bev_world_max_y = ipm.world_dims[3]
        m = ipm.m_per_pix
        bev_corners = np.array([
            [(bev_world_max_y - world_max_y) / m,
             (bev_world_max_x - world_max_x) / m],
            [(bev_world_max_y - world_max_y) / m,
             ipm.bevShape[0] - (world_min_x - bev_world_min_x) / m],
            [ipm.bevShape[0] - (world_min_y - bev_world_min_y) / m,
             (bev_world_max_x - world_max_x) / m],
            [ipm.bevShape[0] - (world_min_y - bev_world_min_y) / m,
             ipm.bevShape[0] - (world_min_x - bev_world_min_x) / m],
        ])

        ipm.M = cv2.getPerspectiveTransform(
            rgb_corners.astype(np.float32),
            bev_corners.astype(np.float32))

        self.get_logger().info(
            f'LaneKeeping loaded + IPM patched  |  '
            f'm_per_pix={ipm.m_per_pix}')
        self.get_logger().info(
            f'IPM intrinsics fx={ipm.camera_intrinsics[0,0]:.1f} '
            f'fy={ipm.camera_intrinsics[1,1]:.1f}')
        self.get_logger().info(
            f'rgb_corners:\n{rgb_corners}')
        self.get_logger().info(
            f'bev_corners:\n{bev_corners}')
        self.get_logger().info(
            f'Homography M:\n{ipm.M}')

        # ── LaneNet (pure PyTorch, no TensorRT) ──────────────────────
        self.lanenet = None
        if not os.path.isfile(model_path) or \
                os.path.getsize(model_path) < 1000:
            model_dir = os.path.dirname(model_path)
            os.makedirs(model_dir, exist_ok=True)
            self.get_logger().info(
                f'Downloading LaneNet model to {model_path}...')
            try:
                import urllib.request
                urllib.request.urlretrieve(
                    'https://quanserinc.box.com/shared/static/'
                    'c19pjultyikcgzlbzu6vs8tu5vuqhl2n.pt',
                    model_path
                )
                self.get_logger().info('Download complete.')
            except Exception as e:
                self.get_logger().warn(f'Model download failed: {e}')

        if os.path.isfile(model_path):
            try:
                self.lanenet = PureTorchLaneNet(
                    modelPath=model_path,
                    rowUpperBound=240,
                    imageWidth=640,
                    imageHeight=480,
                )
                self.get_logger().info(
                    f'LaneNet loaded (pure PyTorch) from {model_path}')
            except Exception as e:
                self.get_logger().error(f'LaneNet init failed: {e}')
                self.lanenet = None
        else:
            self.get_logger().error(f'LaneNet model not found: {model_path}')

        # ── State ──────────────────────────────────────────────────────
        self.current_v = 0.0

        # ── CV Bridge ──────────────────────────────────────────────────
        self.bridge = CvBridge()

        # ── Subscribers ────────────────────────────────────────────────
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.create_subscription(
            Image, '/camera/color_image',
            self._image_cb, image_qos
        )
        self.create_subscription(
            JointState, '/qcar2_joint', self._joint_state_cb, 1
        )

        # ── Publishers — perception outputs ────────────────────────────
        self.pub_cte = self.create_publisher(
            Float64, '/lane_keeping/cross_track_error', 1)
        self.pub_heading = self.create_publisher(
            Float64, '/lane_keeping/heading_error', 1)
        self.pub_detected = self.create_publisher(
            Bool, '/lane_keeping/lane_detected', 1)

        # ── Publishers — visualisation ─────────────────────────────────
        self.pub_binary = self.create_publisher(
            Image, '/lane_keeping/binary_image', 10)
        self.pub_bev_rgb = self.create_publisher(
            Image, '/lane_keeping/bev_rgb', 10)
        self.pub_bev_bin = self.create_publisher(
            Image, '/lane_keeping/bev_binary', 10)
        self.pub_debug = self.create_publisher(
            Image, '/lane_keeping/debug', 10)

        self.get_logger().info(
            f'Lane detector started (perception only)  |  '
            f'BEV {bevShape}  world {bevWorldDims}')

    # ================================================================== #
    #  SUBSCRIBER CALLBACKS
    # ================================================================== #
    def _joint_state_cb(self, msg):
        """Extract measured linear speed from motor encoder."""
        if msg.velocity:
            # motor tach output is in 1/10th scale (same as original lab)
            self.current_v = msg.velocity[0] * 10

    # ================================================================== #
    #  IMAGE CALLBACK — full lane detection pipeline
    #  Mirrors original lab code sections A through visualization
    # ================================================================== #
    def _image_cb(self, msg: Image):
        # Quanser RealSense driver sends periodic empty heartbeat frames
        if msg.width == 0 or msg.height == 0:
            return

        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge: {e}')
            return

        if img is None or img.size == 0:
            return

        v = self.current_v

        # ── SECTION A: Lane marking detection via LaneNet ─────────────
        if self.lanenet is None:
            self.get_logger().warn(
                'LaneNet not loaded — no lane detection available',
                throttle_duration_sec=10.0)
            return

        try:
            rgbProcessed = self.lanenet.pre_process(img)
            self.lanenet.predict(rgbProcessed)
            laneMarking = self.lanenet.binaryPred
        except Exception as e:
            self.get_logger().error(
                f'LaneNet inference failed: {e}',
                throttle_duration_sec=5.0)
            return

        # ── BEV of camera feed and lane markings ──────────────────────
        bev = self.lane_keeping.ipm.create_bird_eye_view(img)
        bevLaneMarking = self.lane_keeping.ipm.create_bird_eye_view(
            laneMarking)

        if not hasattr(self, '_bev_logged'):
            self._bev_logged = True
            self.get_logger().info(
                f'BEV rgb: shape={bev.shape} dtype={bev.dtype} '
                f'sum={np.sum(bev)} max={np.max(bev)} | '
                f'BEV lane: shape={bevLaneMarking.shape} '
                f'sum={np.sum(bevLaneMarking)} max={np.max(bevLaneMarking)} | '
                f'M:\n{self.lane_keeping.ipm.M}')

        if not hasattr(self, '_bev_logged'):
            self._bev_logged = True
            self.get_logger().info(
                f'First BEV: rgb shape={bev.shape} sum={np.sum(bev)} | '
                f'lane shape={bevLaneMarking.shape} sum={np.sum(bevLaneMarking)} | '
                f'input img shape={img.shape} laneMarking shape={laneMarking.shape}')

        # ── Process lane markings and extract targets ─────────────────
        # Ensure bevLaneMarking is single-channel for CCA
        if len(bevLaneMarking.shape) == 3:
            bevLaneMarking = cv2.cvtColor(bevLaneMarking, cv2.COLOR_BGR2GRAY)
        processedLaneMarking = self.lane_keeping.preprocess(bevLaneMarking)
        isolated = self.lane_keeping.isolate_lane_markings(
            processedLaneMarking)
        targets = self.lane_keeping.find_target(isolated, v)

        # ── Select target and compute errors ──────────────────────────
        lane_detected = len(targets) > 0
        cte = 0.0
        heading_err = 0.0
        selected = None

        if lane_detected:
            # Auto-select: pick the target closest to BEV centre
            bev_center_x = bev.shape[1] // 2
            best_id = min(
                targets,
                key=lambda t: abs(targets[t][0] - bev_center_x))
            selected = targets[best_id]

            # Heading error via pure pursuit (same as original lab)
            heading_err = self.lane_keeping.pp.target2steer(selected)

            # Cross-track error: lateral pixel offset * metres/pixel
            dx_pix = selected[0] - bev_center_x
            cte = dx_pix * self.lane_keeping.ipm.m_per_pix

        # ── Publish perception outputs ─────────────────────────────────
        cte_msg = Float64()
        cte_msg.data = float(cte)
        self.pub_cte.publish(cte_msg)

        heading_msg = Float64()
        heading_msg.data = float(heading_err)
        self.pub_heading.publish(heading_msg)

        detected_msg = Bool()
        detected_msg.data = lane_detected
        self.pub_detected.publish(detected_msg)

        # ── Publish visualisation topics ───────────────────────────────
        self._publish_img(self.pub_binary, laneMarking, 'mono8')

        # BEV RGB — ensure 3-channel
        if bev is not None and bev.size > 0:
            if len(bev.shape) == 2:
                bev = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
            self._publish_img(self.pub_bev_rgb, bev, 'bgr8')

        # BEV binary lane markings
        if processedLaneMarking is not None and processedLaneMarking.size > 0:
            if len(processedLaneMarking.shape) == 3:
                processedLaneMarking = cv2.cvtColor(
                    processedLaneMarking, cv2.COLOR_BGR2GRAY)
            self._publish_img(self.pub_bev_bin, processedLaneMarking,
                              'mono8')

        # Debug overlay (same as original lab visualization)
        debug = bev.copy() if bev is not None else \
            np.zeros((800, 800, 3), dtype=np.uint8)
        if len(debug.shape) == 2:
            debug = cv2.cvtColor(debug, cv2.COLOR_GRAY2BGR)
        for tid in targets:
            point = targets[tid]
            cv2.circle(debug, point.astype(int), 10, (0, 255, 0), -1)
        if selected is not None:
            cv2.circle(debug, selected.astype(int), 10, (0, 0, 255), -1)
        cv2.putText(debug, f'CTE: {cte:+.3f} m',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 1)
        cv2.putText(debug,
                    f'Heading: {math.degrees(heading_err):+.1f} deg',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 1)
        cv2.putText(debug, f'Targets: {len(targets)}',
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 1)
        self._publish_img(self.pub_debug, debug, 'bgr8')

    # ================================================================== #
    #  UTILITY
    # ================================================================== #
    def _publish_img(self, pub, img, encoding):
        try:
            pub.publish(self.bridge.cv2_to_imgmsg(img, encoding=encoding))
        except Exception as e:
            self.get_logger().error(
                f'Publish failed: {e} | shape={img.shape} enc={encoding}',
                throttle_duration_sec=5.0)

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
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()

# endregion