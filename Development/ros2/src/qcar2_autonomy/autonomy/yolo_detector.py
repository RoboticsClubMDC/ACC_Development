#! /usr/bin/env python3
import sys
sys.path.insert(0, "/workspaces/isaac_ros-dev/MDC_libraries/python")

from pit.YOLO.utils import QCar2DepthAligned

import time
import numpy as np
import cv2
from pathlib import Path
import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from ultralytics import YOLO


def _canon(s: str) -> str:
    return str(s).strip().lower()


def depth_to_meters(depth):
    """Normalize depth to float32 meters, best-effort."""
    if depth is None:
        return None
    d = np.asarray(depth)
    if d.ndim == 3 and d.shape[2] == 1:
        d = d[:, :, 0]

    # uint16 usually mm
    if d.dtype == np.uint16:
        return d.astype(np.float32) * 0.001

    d = d.astype(np.float32, copy=False)

    # If float but looks like mm (very large), convert
    v = d[np.isfinite(d) & (d > 0)]
    if v.size > 100:
        p95 = float(np.percentile(v, 95))
        if p95 > 50.0:
            d = d * 0.001

    return d


def bbox_distance(depth_m, x1, y1, x2, y2,
                  clip_m=5.0,
                  shrink_x=0.25,
                  shrink_y=0.25,
                  top_ratio=1.0,
                  min_valid=50):
    """
    Robust distance inside bbox region on aligned depth.
    top_ratio < 1.0 uses only upper part of bbox (good for signs to avoid ground/pole base).
    Returns inf if no valid depth.
    """
    if depth_m is None:
        return float("inf")

    h, w = depth_m.shape[:2]

    x1 = int(max(0, min(w - 1, x1)))
    x2 = int(max(0, min(w - 1, x2)))
    y1 = int(max(0, min(h - 1, y1)))
    y2 = int(max(0, min(h - 1, y2)))

    if x2 <= x1 or y2 <= y1:
        return float("inf")

    bw = (x2 - x1)
    bh = (y2 - y1)

    # top crop (for signs/traffic lights)
    y2t = int(y1 + top_ratio * bh)
    y2t = max(y1 + 1, min(y2, y2t))

    # shrink to avoid edges (alignment artifacts)
    dx = int(bw * shrink_x)
    dy = int((y2t - y1) * shrink_y)

    xs1 = max(0, x1 + dx)
    xs2 = min(w - 1, x2 - dx)
    ys1 = max(0, y1 + dy)
    ys2 = min(h - 1, y2t - dy)

    if xs2 <= xs1 or ys2 <= ys1:
        xs1, ys1, xs2, ys2 = x1, y1, x2, y2t

    roi = depth_m[ys1:ys2, xs1:xs2]
    if roi.size == 0:
        return float("inf")

    valid = roi[np.isfinite(roi) & (roi > 0.0) & (roi < clip_m)]
    if valid.size >= min_valid:
        # median is safer for signs (less likely to pick ground)
        return float(np.median(valid))

    # fallback: center patch
    cx = int((x1 + x2) * 0.5)
    cy = int((y1 + y2t) * 0.5)
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))
    patch = depth_m[max(cy - 2, 0):cy + 3, max(cx - 2, 0):cx + 3]
    v2 = patch[np.isfinite(patch) & (patch > 0.0) & (patch < clip_m)]
    if v2.size == 0:
        return float("inf")
    return float(np.median(v2))


class ObjectDetector(Node):
    """
    Custom Ultralytics DETECTION model:
      Crosswalk, Green, Pedestrians, Red, Roundabout Sign, Stop Sign, Yellow, Yield Sign

    Publishes:
      /qcar_camera/rgb       (bgr8)
      /qcar_camera/depth     (32FC1 meters)
      /qcar_camera/rgb_yolo  (bgr8 annotated)
      /motion_enable         (Bool)
    """

    def __init__(self):
        super().__init__('yolo_detector')

        os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
        os.environ.setdefault("TORCH_HOME", "/tmp/torch")

        self.imageWidth = 640
        self.imageHeight = 480

        # Depth aligned to RGB (D435 / QLabs virtual 3D camera)
        self.QCarImg = QCar2DepthAligned()

        # Model path (NO URL)
        model_dir = Path("/workspaces/isaac_ros-dev/ros2/src/qcar2_autonomy/models")
        model_path = model_dir / "yoloObjDetBP01.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.get_logger().info(f"Loading model: {model_path}")
        self.model = YOLO(str(model_path))
        self.get_logger().info(f"Model names: {self.model.names}")

        # ---- tunables ----
        self.det_conf = 0.30
        self.depth_clip_m = 5.0

        # IMPORTANT: if depth looks too small (p95 ~0.3), set this to 5.5 manually
        self.depth_multiplier = 1.0

        # Stop / Yield desired trigger distance (to stop later -> LOWER this)
        self.stop_dist = 0.60
        self.yield_dist = 0.60

        self.stop_conf = 0.70
        self.yield_conf = 0.70

        self.stop_hold = 3.0
        self.yield_hold = 1.5

        # Minimum believable sign distance (filters alignment junk like 0.08m)
        self.sign_min_valid_dist = 0.20

        # Size gating (prevents stopping too early on tiny far detections)
        self.sign_min_bbox_h = 70
        self.sign_min_bbox_area = 3500

        # Traffic light (Red/Yellow/Green classes)
        self.tl_conf = 0.55
        self.tl_min_dist = 0.50
        self.tl_stop_dist = 2.50
        self.tl_hold = 0.25

        # log depth stats every N frames
        self.depth_stats_every = 60
        self.frame_count = 0

        # Stop window logic (same as original)
        self.sign_detected = False
        self.disable_until = 0.0
        self.detection_cooldown = 10.0
        self.t0 = time.time()

        # ROS pubs
        self.bridge = CvBridge()
        self.publish_rgb = self.create_publisher(Image, '/qcar_camera/rgb', 10)
        self.publish_depth = self.create_publisher(Image, '/qcar_camera/depth', 10)
        self.publish_rgb_yolo = self.create_publisher(Image, '/qcar_camera/rgb_yolo', 10)

        self.motion_publisher = self.create_publisher(Bool, '/motion_enable', 1)
        self.flag_value = True
        self.publish_motion_flag(True)

        self.timer = self.create_timer(1/30, self.on_timer)
        self.timer2 = self.create_timer(1/500, self.flag_publisher)

    def flag_publisher(self):
        self.publish_motion_flag(self.flag_value)

    def publish_motion_flag(self, enable: bool):
        msg = Bool()
        msg.data = enable
        self.motion_publisher.publish(msg)

    def on_timer(self):
        self.QCarImg.read()
        rgb = self.QCarImg.rgb
        depth_m = depth_to_meters(self.QCarImg.depth)

        if depth_m is not None:
            depth_m = depth_m * float(self.depth_multiplier)

        # publish raw feeds
        if rgb is not None:
            self.publish_rgb.publish(self.bridge.cv2_to_imgmsg(rgb, "bgr8"))
        if depth_m is not None:
            self.publish_depth.publish(self.bridge.cv2_to_imgmsg(depth_m.astype(np.float32), "32FC1"))

        # depth stats for diagnosing scale
        self.frame_count += 1
        if depth_m is not None and (self.frame_count % self.depth_stats_every == 0):
            v = depth_m[np.isfinite(depth_m) & (depth_m > 0)]
            if v.size > 500:
                p50 = float(np.percentile(v, 50))
                p95 = float(np.percentile(v, 95))
                self.get_logger().info(f"DEPTH STATS: p50={p50:.2f}m p95={p95:.2f}m (mult={self.depth_multiplier})")

        current_time = time.time() - self.t0

        delay = 0.0
        detected = False

        if not self.sign_detected:
            delay, detected = self.yolo_detect(rgb, depth_m)

            if detected and delay > 0.0:
                self.sign_detected = True
                self.disable_until = delay
                self.flag_value = False
            else:
                self.flag_value = True
        else:
            if current_time >= self.disable_until:
                if current_time >= self.detection_cooldown:
                    self.sign_detected = False
                self.flag_value = True

    def yolo_detect(self, rgb, depth_m):
        detected = False
        delay = 0.0

        if rgb is None:
            return 0.0, False

        results = self.model.predict(
            source=rgb,
            conf=self.det_conf,
            imgsz=(self.imageHeight, self.imageWidth),
            verbose=False
        )
        r0 = results[0]

        # annotated image
        try:
            annotated = r0.plot()
        except Exception:
            annotated = rgb.copy()
        annotated = np.ascontiguousarray(annotated)

        n_det = 0 if r0.boxes is None else len(r0.boxes)
        self.get_logger().info(f"YOLO detections: {n_det}")

        # Keep best detection per relevant class (reduces spam & double triggers)
        best = {}  # name -> (conf, dist, bbox)
        all_dets = []

        if r0.boxes is not None and len(r0.boxes) > 0:
            for b in r0.boxes:
                cls_id = int(b.cls[0].item())
                name = self.model.names.get(cls_id, str(cls_id)) if isinstance(self.model.names, dict) else str(self.model.names[cls_id])
                conf = float(b.conf[0].item())
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()

                # distance strategy:
                # For signs & TL colors: use TOP part of bbox to avoid ground
                n = _canon(name)
                if n in ("stop sign", "yield sign", "roundabout sign", "red", "yellow", "green"):
                    dist = bbox_distance(
                        depth_m, x1, y1, x2, y2,
                        clip_m=self.depth_clip_m,
                        shrink_x=0.30,
                        shrink_y=0.25,
                        top_ratio=0.60,
                        min_valid=60
                    )
                else:
                    dist = bbox_distance(
                        depth_m, x1, y1, x2, y2,
                        clip_m=self.depth_clip_m,
                        shrink_x=0.20,
                        shrink_y=0.20,
                        top_ratio=1.0,
                        min_valid=50
                    )

                all_dets.append((name, conf, dist, x1, y1, x2, y2))

                # LOG EVERYTHING (as you requested)
                if np.isfinite(dist):
                    self.get_logger().info(f"{name} @ {conf:.3f} conf. @ {dist:.3f} m")
                else:
                    self.get_logger().info(f"{name} @ {conf:.3f} conf. @ inf m")

                # save best per class by confidence
                prev = best.get(n)
                if prev is None or conf > prev[0]:
                    best[n] = (conf, dist, (x1, y1, x2, y2))

        # draw extra text with distance
        for (name, conf, dist, x1, y1, x2, y2) in all_dets:
            xi1, yi1 = int(x1), int(y1)
            txt = f"{name} {conf:.2f} {dist:.2f}m" if np.isfinite(dist) else f"{name} {conf:.2f} inf"
            cv2.putText(
                annotated, txt,
                (max(0, xi1), max(15, yi1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2, cv2.LINE_AA
            )

        # ---- Traffic light logic (your model outputs Red/Yellow/Green labels) ----
        for color in ("red", "yellow"):
            if color in best:
                conf, dist, bbox = best[color]
                if conf >= self.tl_conf and np.isfinite(dist) and (dist > self.tl_min_dist) and (dist < self.tl_stop_dist):
                    delay = max(delay, self.tl_hold)
                    detected = True
                    self.detection_cooldown = 0.0
                    self.t0 = time.time()
                    self.get_logger().info(f"Traffic Light {color.upper()} -> STOP @ {dist:.2f}m")

        # ---- Stop sign ----
        if "stop sign" in best:
            conf, dist, (x1, y1, x2, y2) = best["stop sign"]
            bh = int(y2 - y1)
            bw = int(x2 - x1)
            area = max(0, bh) * max(0, bw)

            size_ok = (bh >= self.sign_min_bbox_h) or (area >= self.sign_min_bbox_area)
            dist_ok = np.isfinite(dist) and (dist >= self.sign_min_valid_dist) and (dist <= self.stop_dist)

            if conf >= self.stop_conf and size_ok and dist_ok:
                self.get_logger().info(f"Stop Sign Detected at {dist:.2f}m! (h={bh}, area={area})")
                delay = max(delay, self.stop_hold)
                self.t0 = time.time()
                detected = True
                self.detection_cooldown = 10.0

        # ---- Yield sign ----
        if "yield sign" in best:
            conf, dist, (x1, y1, x2, y2) = best["yield sign"]
            bh = int(y2 - y1)
            bw = int(x2 - x1)
            area = max(0, bh) * max(0, bw)

            size_ok = (bh >= self.sign_min_bbox_h) or (area >= self.sign_min_bbox_area)
            dist_ok = np.isfinite(dist) and (dist >= self.sign_min_valid_dist) and (dist <= self.yield_dist)

            if conf >= self.yield_conf and size_ok and dist_ok:
                self.get_logger().info(f"Yield Sign Detected at {dist:.2f}m! (h={bh}, area={area})")
                delay = max(delay, self.yield_hold)
                self.t0 = time.time()
                detected = True
                self.detection_cooldown = 10.0

        # Roundabout sign: NO STOP (solo debug)
        # Crosswalk/Pedestrians: si luego quieres lógica, la añadimos

        # publish annotated overlay
        try:
            self.publish_rgb_yolo.publish(self.bridge.cv2_to_imgmsg(annotated, "bgr8"))
        except Exception as e:
            self.get_logger().warn(f"YOLO overlay publish failed: {e}")

        print("===============================")
        return delay, detected


def main():
    rclpy.init()
    node = ObjectDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        try:
            node.QCarImg.terminate()
        except Exception:
            pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()