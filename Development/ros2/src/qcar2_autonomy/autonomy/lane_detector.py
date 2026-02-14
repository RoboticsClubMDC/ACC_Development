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
    /camera/csi_image                QCar camera (in your setup)
    /qcar2_joint                     Motor encoder for velocity
"""

import sys
sys.path.insert(0, "/workspaces/isaac_ros-dev/MDC_libraries/python")

import cv2
import numpy as np
import math
import os
import time
import traceback
from pathlib import Path
import urllib.request
import shutil
import torch
import torchvision.transforms as transforms

from hal.content.qcar_functions import LaneKeeping, LaneSelection  # noqa: F401

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64, Bool
from cv_bridge import CvBridge

LANENET_URL = "https://quanserinc.box.com/shared/static/c19pjultyikcgzlbzu6vs8tu5vuqhl2n.pt"


# ============================================================================
#  PureTorchLaneNet
# ============================================================================
class PureTorchLaneNet:
    def __init__(self, modelPath, imageHeight=480, imageWidth=640, rowUpperBound=240):
        self.imageHeight = imageHeight
        self.imageWidth = imageWidth
        self.rowUpperBound = rowUpperBound
        self.imgTransforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 512)),
            transforms.ToTensor(),
        ])
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(modelPath, map_location=self.device)
        self.model.eval()
        self.binaryPred = np.zeros((imageHeight, imageWidth), dtype=np.uint8)

    def pre_process(self, inputImg):
        if inputImg.shape[:2] != (self.imageHeight, self.imageWidth):
            inputImg = cv2.resize(
                inputImg, (self.imageWidth, self.imageHeight),
                interpolation=cv2.INTER_LINEAR
            )
        self.imgClone = inputImg
        rgb = cv2.cvtColor(inputImg[self.rowUpperBound:, :, :], cv2.COLOR_BGR2RGB)
        self.imgTensor = self.imgTransforms(rgb)
        return self.imgTensor

    @torch.no_grad()
    def predict(self, inputImg):
        x = inputImg.unsqueeze(0).to(self.device)
        outputs = self.model(x)
        binary_np = (outputs['binary_seg_pred'].squeeze().cpu().numpy() * 255).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        binary_np = cv2.morphologyEx(binary_np, cv2.MORPH_OPEN, kernel)

        # CCA to remove small blobs
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_np, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 200:
                binary_np[labels == i] = 0

        self.binaryPred = np.zeros((self.imageHeight, self.imageWidth), dtype=np.uint8)
        resized = cv2.resize(
            binary_np,
            (self.imageWidth, self.imageHeight - self.rowUpperBound),
            interpolation=cv2.INTER_LINEAR
        )
        _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
        self.binaryPred[self.rowUpperBound:, :] = resized
        return self.binaryPred


def ensure_lanenet_model(logger, filename="lanenet.pt"):
    """
    Prefer storing the model in the *source workspace* so it persists across rebuilds,
    even if this node is executed from the installed entry point.
    Fallback to ~/.ros cache if needed.
    """
    here = Path(__file__).resolve()

    use_candidates = []
    dl_candidates = []

    # A) Walk upwards and look for: <something>/ros2/src/qcar2_autonomy/models/lanenet.pt
    for p in here.parents:
        cand = p / "ros2" / "src" / "qcar2_autonomy" / "models" / filename
        if cand.parent.is_dir():
            use_candidates.append(cand)
            dl_candidates.append(cand)
            break

    # B) Docker default workspace fallback
    ws_fallback = Path("/workspaces/isaac_ros-dev/ros2/src/qcar2_autonomy/models") / filename
    if ws_fallback.parent.is_dir():
        use_candidates.append(ws_fallback)
        dl_candidates.append(ws_fallback)

    # C) Installed share dir (ROS-style place)
    try:
        from ament_index_python.packages import get_package_share_directory
        share_dir = Path(get_package_share_directory("qcar2_autonomy"))
        share_cand = share_dir / "models" / filename
        use_candidates.append(share_cand)
        dl_candidates.append(share_cand)
    except Exception:
        pass

    # D) Always-writable cache
    cache = Path.home() / ".ros" / "qcar2_autonomy" / "models" / filename
    use_candidates.append(cache)
    dl_candidates.append(cache)

    def uniq(paths):
        out, seen = [], set()
        for x in paths:
            if x is None:
                continue
            sx = str(x)
            if sx not in seen:
                out.append(x)
                seen.add(sx)
        return out

    use_candidates = uniq(use_candidates)
    dl_candidates = uniq(dl_candidates)

    # 1) If any candidate exists and is non-trivial, use it
    for p in use_candidates:
        try:
            if p.is_file() and p.stat().st_size > 1_000_000:
                return str(p)
        except Exception:
            pass

    # 2) Otherwise download to the first location that works
    for p in dl_candidates:
        tmp = p.with_suffix(p.suffix + ".part")
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"LaneNet model missing, downloading to: {p}")

            with urllib.request.urlopen(LANENET_URL, timeout=300) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)

            tmp.replace(p)
            try:
                os.chmod(p, 0o644)
            except Exception:
                pass

            logger.info(f"LaneNet model ready: {p}")
            return str(p)

        except PermissionError:
            logger.warn(f"No write permission for {p.parent}. Trying next location.")
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Failed downloading LaneNet model to {p}: {e}")
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

    return None


# ============================================================================
#  LaneDetectorNode
# ============================================================================
class LaneDetectorNode(Node):
    def __init__(self):
        super().__init__('lane_detector')

        # ===== Observability (no more silent failures) =====
        self._log_last = {}  # key -> last time.monotonic()
        self._warn_throttle_s = 3.0
        self._stats_period_s = 2.0

        self._frames_in = 0
        self._frames_decoded = 0
        self._frames_inferred = 0
        self._frames_detected = 0
        self._last_stats_t = time.monotonic()
        self._last_frame_t = None

        # Track which topic we are actually subscribing to
        self._image_topic = '/camera/csi_image'
        self._status_timer = self.create_timer(2.0, self._status_timer_cb)

        # ── BEV Configuration ────────────────────────────────────────
        self.bevShape = [800, 800]
        bevWorldDims = [0, 20, -10, 10]
        self.m_per_pix = (bevWorldDims[1] - bevWorldDims[0]) / self.bevShape[0]

        # ── Vehicle reference in BEV ─────────────────────────────────
        self.car_center_col = self.bevShape[1] // 2  # 400
        self.car_bottom_row = self.bevShape[0]       # 800

        # ── Single lane (Case I) offset ──────────────────────────────
        self.LANE_OFFSET_PX = 75  # pixels to the right of detected lane

        # ── Look-ahead (kept for reference drawing; compute uses dynamic look_row) ──
        self.LOOKAHEAD_ROW = 500

        # ── EMA filter ────────────────────────────────────────────────
        self._ema_alpha = 0.3
        self._filtered_cte = 0.0
        self._filtered_heading = 0.0
        self._no_detect_count = 0
        self._no_detect_max = 5

        model_path = ensure_lanenet_model(self.get_logger())

        # ── Build SDK (only for IPM — we skip find_target) ───────────
        self.lane_keeping = LaneKeeping(
            Kdd=2, ldMin=10, ldMax=20, maxSteer=0.05,
            bevShape=self.bevShape, bevWorldDims=bevWorldDims
        )

        # ── Patch IPM (stub has zero intrinsics) ─────────────────────
        ipm = self.lane_keeping.ipm

        ipm.camera_intrinsics = np.array([
            [483.671,       0, 321.188, 0],
            [      0, 483.579, 238.462, 0],
            [      0,       0,       1, 0],
            [      0,       0,       0, 1]
        ])

        phi, theta, psi, height = np.pi/2, 0.0, np.pi/2, 1.72
        cx, sx = np.cos(phi), np.sin(phi)
        cy, sy = np.cos(theta), np.sin(theta)
        cz, sz = np.cos(psi), np.sin(psi)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        R_v2cam = Rx @ Ry @ Rz
        t_v2cam = np.array([[0, height, 0]]).T
        T_v2cam = np.vstack((np.hstack((R_v2cam, t_v2cam)), np.array([[0, 0, 0, 1]])))
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
            [world_min_x, world_min_y, 0]
        ]))

        m = ipm.m_per_pix
        bev_corners = np.array([
            [(bevWorldDims[3] - world_max_y) / m, (bevWorldDims[1] - world_max_x) / m],
            [(bevWorldDims[3] - world_max_y) / m, self.bevShape[0] - (world_min_x - bevWorldDims[0]) / m],
            [self.bevShape[0] - (world_min_y - bevWorldDims[2]) / m, (bevWorldDims[1] - world_max_x) / m],
            [self.bevShape[0] - (world_min_y - bevWorldDims[2]) / m, self.bevShape[0] - (world_min_x - bevWorldDims[0]) / m]
        ])

        ipm.M = cv2.getPerspectiveTransform(rgb_corners.astype(np.float32), bev_corners.astype(np.float32))
        ipm.bevShape = self.bevShape
        ipm.world_dims = bevWorldDims
        ipm.bevWorldDims = bevWorldDims

        self.get_logger().info(
            f"IPM patched | m_per_pix={ipm.m_per_pix} | "
            f"car_center_col={self.car_center_col} | "
            f"lane_offset={self.LANE_OFFSET_PX}px = {self.LANE_OFFSET_PX * self.m_per_pix:.2f}m"
        )

        # ── LaneNet ──────────────────────────────────────────────────
        self.lanenet = None
        if model_path and os.path.isfile(model_path):
            try:
                self.lanenet = PureTorchLaneNet(
                    modelPath=model_path, rowUpperBound=240,
                    imageWidth=640, imageHeight=480
                )
                self.get_logger().info(f"LaneNet loaded from {model_path}")
                self.get_logger().info(
                    f"LaneNet device={self.lanenet.device} | torch.cuda.is_available()={torch.cuda.is_available()} | "
                    f"model_type={type(self.lanenet.model)}"
                )
            except Exception as e:
                self.get_logger().error(f"LaneNet init failed: {e}")
        else:
            self.get_logger().error(f"LaneNet model unavailable. Tried download from: {LANENET_URL}")

        # ── State ─────────────────────────────────────────────────────
        self.current_v = 0.0
        self.bridge = CvBridge()

        # ── Subscribers ───────────────────────────────────────────────
        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=1
        )
        self.create_subscription(Image, self._image_topic, self._image_cb, img_qos)
        self.create_subscription(JointState, '/qcar2_joint', self._joint_state_cb, 1)

        # ── Publishers ────────────────────────────────────────────────
        self.pub_cte = self.create_publisher(Float64, '/lane_keeping/cross_track_error', 1)
        self.pub_heading = self.create_publisher(Float64, '/lane_keeping/heading_error', 1)
        self.pub_detected = self.create_publisher(Bool, '/lane_keeping/lane_detected', 1)
        self.pub_debug = self.create_publisher(Image, '/lane_keeping/debug', 10)
        self.pub_bev_bin = self.create_publisher(Image, '/lane_keeping/bev_binary', 10)
        self.pub_camera = self.create_publisher(Image, '/lane_keeping/camera_image', 10)

        self.get_logger().info("Lane detector started (single yellow lane mode)")

    # ================================================================ #
    # Step 1 helpers
    # ================================================================ #
    def _log_throttled(self, level: str, key: str, msg: str, period_s: float = None):
        if period_s is None:
            period_s = self._warn_throttle_s
        now = time.monotonic()
        last = self._log_last.get(key)
        if last is not None and (now - last) < period_s:
            return
        self._log_last[key] = now
        log = self.get_logger()
        if level == "debug":
            log.debug(msg)
        elif level == "info":
            log.info(msg)
        elif level == "warn":
            log.warn(msg)
        else:
            log.error(msg)

    def _status_timer_cb(self):
        now = time.monotonic()

        # No images yet => likely topic mismatch
        if self._last_frame_t is None:
            self._log_throttled(
                "warn",
                "no_images_yet",
                f"No image messages received yet. If this persists, your image topic is likely wrong "
                f"(currently subscribing to '{self._image_topic}').",
                period_s=3.0,
            )
            return

        # Stalled stream
        dt = now - self._last_frame_t
        if dt > 2.0:
            self._log_throttled(
                "warn",
                "image_stalled",
                f"Image stream stalled: last image {dt:.1f}s ago on '{self._image_topic}'.",
                period_s=3.0,
            )

        # Periodic stats
        if (now - self._last_stats_t) >= self._stats_period_s:
            self._last_stats_t = now
            self.get_logger().info(
                f"stats | in={self._frames_in} decoded={self._frames_decoded} "
                f"inferred={self._frames_inferred} detected={self._frames_detected} "
                f"cte={self._filtered_cte:+.3f}m heading={math.degrees(self._filtered_heading):+.1f}deg"
            )

    # ================================================================ #
    def _joint_state_cb(self, msg):
        if msg.velocity:
            self.current_v = msg.velocity[0] * 10
            
    def _fit_lane_line(self, bev_binary):
        """
        Fit a line through the lane pixels in BEV.

        Returns (vx, vy, x0, y0, success)
        where (vx,vy) is direction and (x0,y0) is a point on the line.
        """

        if bev_binary is None or bev_binary.size == 0:
            return 0, 0, 0, 0, False

        # Use most of the image, but ignore a small strip at the very bottom (closest to car)
        # because it can include artifacts.
        mask = np.zeros_like(bev_binary)
        mask[0:self.bevShape[0] - 80, :] = bev_binary[0:self.bevShape[0] - 80, :]

        pts = cv2.findNonZero(mask)
        if pts is None or len(pts) < 30:
            return 0, 0, 0, 0, False

        line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        vx = float(line[0])
        vy = float(line[1])
        x0 = float(line[2])
        y0 = float(line[3])

        return vx, vy, x0, y0, True

    def _compute_errors(self, bev_binary):
        """Compute CTE and heading error from BEV lane mask."""
        vx, vy, x0, y0, success = self._fit_lane_line(bev_binary)
        if not success:
            return 0.0, 0.0, False, None

        ys = np.where(bev_binary > 0)[0]
        if ys.size == 0:
            return 0.0, 0.0, False, None

        # Pick lookahead near lane pixels closer to the car (stable)
        look_row = int(np.percentile(ys, 80))
        look_row = max(50, min(self.bevShape[0] - 50, look_row))

        # Band for lane_x measurement
        band = 10
        y0b = max(0, look_row - band)
        y1b = min(self.bevShape[0], look_row + band)

        roi = bev_binary[y0b:y1b, :]
        roi_bin = (roi > 0).astype(np.uint8)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(roi_bin, connectivity=8)
        if num <= 1:
            self._log_throttled(
                "warn", "no_cc_in_band",
                f"No CC in CTE band at look_row={look_row}. mask_y_range=[{int(ys.min())},{int(ys.max())}]",
                period_s=2.0
            )
            return 0.0, 0.0, False, None

        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = int(stats[largest, cv2.CC_STAT_AREA])
        if area < 15:
            self._log_throttled(
                "warn", "cc_too_small",
                f"CTE band CC too small at look_row={look_row}: area={area}",
                period_s=2.0
            )
            return 0.0, 0.0, False, None

        xs = np.where(labels == largest)[1]
        lane_x_at_la = float(np.median(xs))

        # Heading window (taller) for stability
        hwin = 160
        hy0 = max(0, look_row - hwin)
        hy1 = min(self.bevShape[0], look_row + hwin)

        roi_h = bev_binary[hy0:hy1, :]
        roi_h_bin = (roi_h > 0).astype(np.uint8)
        numh, labelsh, statsh, _ = cv2.connectedComponentsWithStats(roi_h_bin, connectivity=8)

        heading = 0.0
        if numh > 1:
            largest_h = 1 + int(np.argmax(statsh[1:, cv2.CC_STAT_AREA]))
            yh, xh = np.where(labelsh == largest_h)

            if xh.size >= 30:
                pts_h = np.stack([xh.astype(np.float32), yh.astype(np.float32)], axis=1)
                lineh = cv2.fitLine(pts_h, cv2.DIST_L2, 0, 0.01, 0.01)
                lvx = float(lineh[0])
                lvy = float(lineh[1])

                heading = math.atan2(lvx, -lvy)

                # Normalize to [-pi/2, +pi/2]
                if heading > math.pi / 2:
                    heading -= math.pi
                elif heading < -math.pi / 2:
                    heading += math.pi

                # Reject near-horizontal stripes (crosswalks)
                if abs(heading) > math.radians(60):
                    return 0.0, 0.0, False, None

        # Update temporal track (NO need to modify __init__)
        if not hasattr(self, "_last_lane_x") or self._last_lane_x is None:
            self._last_lane_x = lane_x_at_la
        else:
            # Smooth tracking so we don't jump on noisy frames
            self._last_lane_x = 0.7 * float(self._last_lane_x) + 0.3 * lane_x_at_la

        target_x = lane_x_at_la + self.LANE_OFFSET_PX
        cte = (target_x - self.car_center_col) * self.m_per_pix

        line_info = {
            'vx': vx, 'vy': vy, 'x0': x0, 'y0': y0,
            'lane_x_at_la': lane_x_at_la,
            'target_x': target_x,
            'look_row': look_row,
        }
        return cte, heading, True, line_info

    # ================================================================ #
    #  IMAGE CALLBACK
    # ================================================================ #
    def _image_cb(self, msg: Image):
        self._frames_in += 1
        self._last_frame_t = time.monotonic()

        if msg.width == 0 or msg.height == 0:
            self._log_throttled("warn", "empty_ros_image", "Received an empty image message (0x0).")
            return

        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self._frames_decoded += 1
        except Exception:
            self._log_throttled(
                "error",
                "imgmsg_to_cv2_failed",
                "cv_bridge imgmsg_to_cv2 failed:\n" + traceback.format_exc(),
                period_s=3.0,
            )
            return

        if img is None or img.size == 0:
            self._log_throttled("warn", "cv_image_empty", "Decoded OpenCV image is empty.")
            return

        if self.lanenet is None:
            self._log_throttled(
                "error",
                "lanenet_none",
                "LaneNet is not initialized; cannot run lane detection. Check earlier model load logs.",
                period_s=3.0,
            )
            return

        if img.shape[:2] != (480, 640):
            try:
                img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
            except Exception:
                self._log_throttled(
                    "error",
                    "resize_failed",
                    "Failed to resize image to 640x480:\n" + traceback.format_exc(),
                    period_s=3.0,
                )
                return

        # ── LaneNet inference ─────────────────────────────────────────
        try:
            rgbProcessed = self.lanenet.pre_process(img)
            self.lanenet.predict(rgbProcessed)
            laneMarking = self.lanenet.binaryPred
            self._frames_inferred += 1
        except Exception:
            self._log_throttled(
                "error",
                "lanenet_inference_failed",
                "LaneNet inference failed:\n" + traceback.format_exc(),
                period_s=3.0,
            )
            return

        # ── Publish camera overlay ─────────────────────────────────────
        try:
            cam_overlay = img.copy()
            cam_overlay[laneMarking > 0] = [0, 255, 0]
            self.pub_camera.publish(self.bridge.cv2_to_imgmsg(cam_overlay, encoding='bgr8'))
        except Exception:
            self._log_throttled(
                "warn",
                "publish_camera_failed",
                "Publishing /lane_keeping/camera_image failed:\n" + traceback.format_exc(),
                period_s=3.0,
            )

        # ── BEV transform ─────────────────────────────────────────────
        try:
            bev = self.lane_keeping.ipm.create_bird_eye_view(img)
            M = self.lane_keeping.ipm.M
            bevLM = cv2.warpPerspective(
                laneMarking, M,
                (self.bevShape[1], self.bevShape[0]),
                flags=cv2.INTER_NEAREST
            )
        except Exception:
            self._log_throttled(
                "error",
                "ipm_failed",
                "IPM (create_bird_eye_view) failed:\n" + traceback.format_exc(),
                period_s=3.0,
            )
            return

        # ── Preprocess BEV lane mask ──────────────────────────────────
        try:
            if len(bevLM.shape) == 3:
                bevLM = cv2.cvtColor(bevLM, cv2.COLOR_BGR2GRAY)

            processedLM = bevLM
            _, processedLM = cv2.threshold(processedLM, 127, 255, cv2.THRESH_BINARY)

            # ------------------------------------------------------------
            # Prefer YELLOW markings (color gate in BEV)
            # ------------------------------------------------------------
            bev_bgr = bev.copy()
            if bev_bgr is None or bev_bgr.size == 0:
                bev_bgr = np.zeros((self.bevShape[0], self.bevShape[1], 3), dtype=np.uint8)

            hsv = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HSV)
            yellow_mask = cv2.inRange(hsv, (15, 70, 70), (45, 255, 255))
            yellowLM = cv2.bitwise_and(processedLM, yellow_mask)

            yellow_count = int(cv2.countNonZero(yellowLM))
            if yellow_count > 120:
                processedLM = yellowLM

            # ------------------------------------------------------------
            # Connected-component selection: most lane-like, PLUS temporal tracking
            # ------------------------------------------------------------
            bin01 = (processedLM > 0).astype(np.uint8)
            num, labels, stats, centroids = cv2.connectedComponentsWithStats(bin01, connectivity=8)

            if num > 1:
                # Expected x position: track last lane x if we have it, else vehicle center
                expected_x = self.car_center_col
                if hasattr(self, "_last_lane_x") and self._last_lane_x is not None:
                    expected_x = float(self._last_lane_x)

                best_i = None
                best_score = -1e9

                for i in range(1, num):
                    area = int(stats[i, cv2.CC_STAT_AREA])
                    if area < 150:
                        continue

                    w = int(stats[i, cv2.CC_STAT_WIDTH])
                    h = int(stats[i, cv2.CC_STAT_HEIGHT])
                    aspect = (h + 1e-6) / (w + 1e-6)

                    # Crosswalks: wide horizontal blobs
                    horiz_penalty = 3000.0 if aspect < 0.8 else 0.0

                    cx = float(centroids[i][0])

                    # Prefer close to previous lane x (this is the key fix)
                    track_dist = abs(cx - expected_x)

                    # Keep a mild preference for being not insanely far from car center too
                    center_dist = abs(cx - self.car_center_col)

                    score = (
                        (area * 1.0) +
                        (aspect * 400.0) -
                        horiz_penalty -
                        (track_dist * 4.0) -
                        (center_dist * 0.5)
                    )

                    if score > best_score:
                        best_score = score
                        best_i = i

                if best_i is not None:
                    processedLM = np.where(labels == best_i, 255, 0).astype(np.uint8)

            # Mild dilation to connect broken segments
            processedLM = cv2.dilate(processedLM, np.ones((3, 3), np.uint8), iterations=1)

        except Exception:
            self._log_throttled(
                "error",
                "bev_preprocess_failed",
                "BEV preprocessing failed:\n" + traceback.format_exc(),
                period_s=3.0,
            )
            return

        # ── Compute errors ───────────────────────────────────────────
        try:
            raw_cte, raw_heading, detected, line_info = self._compute_errors(processedLM)
        except Exception:
            self._log_throttled(
                "error",
                "compute_errors_failed",
                "Computing CTE/heading failed:\n" + traceback.format_exc(),
                period_s=3.0,
            )
            return

        if detected:
            self._frames_detected += 1

        # ── EMA filter ────────────────────────────────────────────────
        if detected:
            self._no_detect_count = 0
            a = self._ema_alpha
            self._filtered_cte = a * raw_cte + (1 - a) * self._filtered_cte
            self._filtered_heading = a * raw_heading + (1 - a) * self._filtered_heading
        else:
            self._no_detect_count += 1
            if self._no_detect_count > self._no_detect_max:
                self._filtered_cte *= 0.9
                self._filtered_heading *= 0.9

        # ── Publish errors ────────────────────────────────────────────
        try:
            cte_msg = Float64()
            cte_msg.data = float(self._filtered_cte)
            self.pub_cte.publish(cte_msg)

            heading_msg = Float64()
            heading_msg.data = float(self._filtered_heading)
            self.pub_heading.publish(heading_msg)

            detected_msg = Bool()
            detected_msg.data = bool(detected)
            self.pub_detected.publish(detected_msg)
        except Exception:
            self._log_throttled(
                "error",
                "publish_errors_failed",
                "Publishing CTE/heading/detected failed:\n" + traceback.format_exc(),
                period_s=3.0,
            )

        # ── BEV binary with text overlay ──────────────────────────────
        try:
            bev_bin_vis = cv2.cvtColor(processedLM, cv2.COLOR_GRAY2BGR) if len(processedLM.shape) == 2 else processedLM.copy()
            self._draw_text_overlay(bev_bin_vis)
            self.pub_bev_bin.publish(self.bridge.cv2_to_imgmsg(bev_bin_vis, encoding='bgr8'))
        except Exception:
            self._log_throttled(
                "warn",
                "publish_bev_binary_failed",
                "Publishing /lane_keeping/bev_binary failed:\n" + traceback.format_exc(),
                period_s=3.0,
            )

        # ── Debug overlay ─────────────────────────────────────────────
        try:
            debug = bev.copy() if bev is not None else np.zeros((800, 800, 3), dtype=np.uint8)

            mask_vis = np.zeros_like(debug)
            mask_vis[:, :, 1] = processedLM
            debug = cv2.addWeighted(debug, 1.0, mask_vis, 0.8, 0.0)

            if len(debug.shape) == 2:
                debug = cv2.cvtColor(debug, cv2.COLOR_GRAY2BGR)

            cv2.line(
                debug,
                (self.car_center_col, self.bevShape[0]),
                (self.car_center_col, 0),
                (0, 0, 255), 2
            )

            draw_row = self.LOOKAHEAD_ROW
            if line_info is not None and 'look_row' in line_info:
                draw_row = int(line_info['look_row'])

            band = 10
            cv2.line(debug, (0, draw_row), (self.bevShape[1], draw_row), (255, 0, 255), 1)
            cv2.rectangle(debug, (0, draw_row - band), (self.bevShape[1], draw_row + band), (255, 0, 255), 1)

            if detected and line_info is not None:
                vx = line_info['vx']
                vy = line_info['vy']
                x0 = line_info['x0']
                y0 = line_info['y0']
                lane_x = line_info['lane_x_at_la']
                target_x = line_info['target_x']

                t_top = (0 - y0) / vy if abs(vy) > 1e-6 else 0
                t_bot = (self.bevShape[0] - y0) / vy if abs(vy) > 1e-6 else 0
                pt_top = (int(x0 + t_top * vx), 0)
                pt_bot = (int(x0 + t_bot * vx), self.bevShape[0])
                cv2.line(debug, pt_top, pt_bot, (0, 255, 255), 2)

                cv2.circle(debug, (int(lane_x), draw_row), 8, (0, 255, 0), -1)
                cv2.circle(debug, (int(target_x), draw_row), 10, (255, 100, 0), -1)

            self._draw_text_overlay(debug)
            self.pub_debug.publish(self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))
        except Exception:
            self._log_throttled(
                "warn",
                "publish_debug_failed",
                "Publishing /lane_keeping/debug failed:\n" + traceback.format_exc(),
                period_s=3.0,
            )
    
    # ================================================================ #
    def _draw_text_overlay(self, img):
        """Draw CTE/heading text on an image."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 0, 255)  # red text
        cv2.putText(img, f"CTE: {self._filtered_cte:+.3f} m", (10, 30), font, 0.8, color, 2)
        cv2.putText(img, f"Heading: {math.degrees(self._filtered_heading):+.1f} deg", (10, 65), font, 0.8, color, 2)


# ============================================================================
def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down lane detector.")
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
