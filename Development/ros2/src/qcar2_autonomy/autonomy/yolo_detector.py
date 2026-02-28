#! /usr/bin/env python3
import sys
sys.path.insert(0, "/workspaces/isaac_ros-dev/MDC_libraries/python")

from pit.YOLO.nets import YOLOv8
from pit.YOLO.utils import QCar2DepthAligned

import time
import numpy as np
import cv2
from pathlib import Path
import urllib.request
import tempfile
import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def ensure_model_exists(model_path: Path, url: str, logger=None) -> None:
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists() and model_path.stat().st_size > 1024 * 1024:
        if logger:
            logger.info(f"YOLO model found: {model_path} ({model_path.stat().st_size/1e6:.1f} MB)")
        return
    if logger:
        logger.warn(f"YOLO model not found, downloading to: {model_path}")
    with tempfile.NamedTemporaryFile(dir=str(model_path.parent), delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with urllib.request.urlopen(url) as r, open(tmp_path, "wb") as f:
            chunk = 1024 * 1024
            while True:
                data = r.read(chunk)
                if not data:
                    break
                f.write(data)
        size = tmp_path.stat().st_size
        if size < 1024 * 1024:
            raise RuntimeError(f"Downloaded file too small ({size} bytes)")
        tmp_path.replace(model_path)
        if logger:
            logger.info(f"YOLO model downloaded OK: {model_path}")
    finally:
        if tmp_path.exists() and (not model_path.exists() or tmp_path != model_path):
            try:
                tmp_path.unlink()
            except Exception:
                pass


class ObjectDetector(Node):

    def __init__(self):
        super().__init__('yolo_detector')

        imageWidth = 640
        imageHeight = 480
        self.QCarImg = QCar2DepthAligned()

        model_dir = Path("/workspaces/isaac_ros-dev/ros2/src/qcar2_autonomy/models")
        model_path = model_dir / "quanser_yolov8s-seg.pt"
        model_url = "https://quanserinc.box.com/shared/static/ce0gxomeg4b12wlcch9cmlh0376nditf.pt"
        ensure_model_exists(model_path, model_url, logger=self.get_logger())

        self.myYolo = YOLOv8(
            modelPath=str(model_path),
            imageHeight=imageHeight,
            imageWidth=imageWidth,
            convert_tensorrt=False,
        )

        self.dt = 1 / 30
        self.timer = self.create_timer(self.dt, self.on_timer)

        # /motion_enable — always True; nav_to_pose state machine owns all stopping
        self.motion_publisher = self.create_publisher(Bool, '/motion_enable', 1)
        self.flag_value = True

        # /intersection_rule — what we see at the current intersection node
        # Published values: "NONE" | "STOP" | "YIELD" | "RED" | "GREEN" | "YELLOW"
        from std_msgs.msg import String as StringMsg
        self.rule_pub = self.create_publisher(StringMsg, '/intersection_rule', 1)

        # Only run YOLO when nav_to_pose has stopped at an intersection node
        self.at_intersection = False
        self.create_subscription(Bool, '/at_intersection', self._at_intersection_cb, 1)

        # ── Tuning ───────────────────────────────────────────────────────────
        self.tl_conf      = 0.85
        self.tl_stop_dist = 3.0
        self.tl_min_dist  = 0.3
        self.stop_dist    = 0.40
        self.yield_dist   = 0.50
        self.depth_patch  = 9
        # ─────────────────────────────────────────────────────────────────────

        self.bridge = CvBridge()
        self.publish_rgb       = self.create_publisher(Image, '/qcar_camera/rgb',      10)
        self.publish_depth     = self.create_publisher(Image, '/qcar_camera/depth',    10)
        self.publish_rgb_yolo  = self.create_publisher(Image, '/qcar_camera/rgb_yolo', 10)

        self.timer2 = self.create_timer(1 / 500, self.flag_publisher)

    # ------------------------------------------------------------------

    def _at_intersection_cb(self, msg: Bool):
        prev = self.at_intersection
        self.at_intersection = bool(msg.data)
        if prev and not self.at_intersection:
            self._publish_rule("NONE")  # clear rule when leaving intersection

    def _publish_rule(self, rule: str):
        from std_msgs.msg import String as StringMsg
        m = StringMsg()
        m.data = rule
        self.rule_pub.publish(m)

    def flag_publisher(self):
        self.publish_motion_flag(self.flag_value)

    def on_timer(self):
        self.QCarImg.read()

        rgb   = self.QCarImg.rgb
        depth = self.QCarImg.depth

        if depth is not None:
            depth = np.asarray(depth)
            if depth.ndim == 3 and depth.shape[2] == 1:
                depth = depth[:, :, 0]
            depth = depth.astype(np.float32, copy=False)

        self.publish_rgb.publish(self.bridge.cv2_to_imgmsg(rgb, "bgr8"))
        self.publish_depth.publish(self.bridge.cv2_to_imgmsg(depth, "32FC1"))

        # motion_enable is always True — nav_to_pose owns all stopping
        self.flag_value = True

        # Only run YOLO when hard-stopped at an intersection node
        if self.at_intersection:
            self.yolo_detect()

    # ------------------------------------------------------------------

    def _stable_dist(self, obj) -> float:
        """
        Median depth over a self.depth_patch x self.depth_patch pixel square
        centred on the bounding box centre. Much more stable than the wrapper's
        per-mask average which jumps all over the place.
        Falls back to wrapper distance if bbox attribute not found.
        """
        depth = self.QCarImg.depth
        if depth is None:
            return float(obj.__dict__.get("distance", -1.0))

        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        h, w = depth.shape

        # Quanser wrapper may call it bbox / box / xyxy / rect — try all
        bbox = None
        for key in ("bbox", "box", "xyxy", "rect"):
            v = obj.__dict__.get(key)
            if v is not None:
                bbox = v
                break

        if bbox is None:
            return float(obj.__dict__.get("distance", -1.0))

        try:
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        except Exception:
            return float(obj.__dict__.get("distance", -1.0))

        cx   = int((x1 + x2) / 2)
        cy   = int((y1 + y2) / 2)
        half = self.depth_patch // 2

        r0 = max(0, cy - half);  r1 = min(h, cy + half + 1)
        c0 = max(0, cx - half);  c1 = min(w, cx + half + 1)

        vals  = depth[r0:r1, c0:c1].flatten()
        valid = vals[(vals > 0.05) & np.isfinite(vals)]

        if valid.size == 0:
            return float(obj.__dict__.get("distance", -1.0))

        return float(np.median(valid))

    # ------------------------------------------------------------------

    def _get_bbox(self, obj):
        for key in ("bbox", "box", "xyxy", "rect"):
            v = obj.__dict__.get(key)
            if v is not None:
                try:
                    return float(v[0]), float(v[1]), float(v[2]), float(v[3])
                except Exception:
                    continue
        return None

    def _classify_light_color(self, obj):
        """Crop YOLO bbox, classify traffic light via HSV. Returns (is_red, is_green, is_yellow)."""
        rgb = self.QCarImg.rgb
        if rgb is None:
            return False, False, False
        bbox = self._get_bbox(obj)
        if bbox is None:
            return False, False, False
        x1, y1, x2, y2 = bbox
        h, w = rgb.shape[:2]
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(w, int(x2)); y2 = min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return False, False, False
        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return False, False, False
        ch = crop.shape[0]
        top = crop[:max(1, ch // 3), :]
        bot = crop[max(0, 2 * ch // 3):, :]
        hsv_top = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
        hsv_bot = cv2.cvtColor(bot, cv2.COLOR_BGR2HSV)
        r1 = cv2.inRange(hsv_top, np.array([0,   200, 200]), np.array([10,  255, 255]))
        r2 = cv2.inRange(hsv_top, np.array([170, 120,  70]), np.array([180, 255, 255]))
        red_px    = cv2.countNonZero(r1) + cv2.countNonZero(r2)
        yellow_px = cv2.countNonZero(cv2.inRange(hsv_top, np.array([20, 100, 100]), np.array([30, 255, 255])))
        green_px  = cv2.countNonZero(cv2.inRange(hsv_bot, np.array([40, 100, 100]), np.array([90, 255, 255])))
        self.get_logger().info(f"  TL HSV: red={red_px} yellow={yellow_px} green={green_px}")
        return red_px >= 5, green_px > 30, yellow_px > 200

    def yolo_detect(self):
        """Called only when at an intersection node. Publishes /intersection_rule."""
        rgbProcessed = self.myYolo.pre_process(self.QCarImg.rgb)
        pred = self.myYolo.predict(
            inputImg=rgbProcessed,
            classes=[9, 11, 33],
            confidence=0.3,
            half=True,
            verbose=False
        )
        try:
            ann = None
            if isinstance(pred, (list, tuple)) and len(pred) > 0 and hasattr(pred[0], "plot"):
                ann = pred[0].plot()
            elif hasattr(pred, "plot"):
                ann = pred.plot()
            if ann is not None and isinstance(ann, np.ndarray) and ann.size:
                self.publish_rgb_yolo.publish(self.bridge.cv2_to_imgmsg(ann, "bgr8"))
        except Exception as e:
            self.get_logger().warn(f"YOLO overlay failed: {e}")

        processedResults = self.myYolo.post_processing(
            alignedDepth=self.QCarImg.depth,
            clippingDistance=5
        )

        rule = "NONE"  # default — nav_to_pose treats as yield after 1s grace

        for obj in processedResults:
            labelName  = obj.__dict__.get("name", "")
            labelConf  = float(obj.__dict__.get("conf", 0.0))
            objectDist = self._stable_dist(obj)

            self.get_logger().info(
                f"[INTERSECTION] {labelName} conf={labelConf:.2f} dist={objectDist:.3f}m")

            # ── TRAFFIC LIGHT — HSV color check ──────────────────────────────
            if str(labelName).startswith("traffic light"):
                valid_dist = self.tl_min_dist < objectDist < self.tl_stop_dist
                if labelConf >= self.tl_conf and valid_dist:
                    is_red, is_green, is_yellow = self._classify_light_color(obj)
                    if is_red and not is_green:
                        rule = "RED"
                        self.get_logger().info(f"RED LIGHT @ {objectDist:.2f}m")
                        break  # red is highest priority
                    elif is_green:
                        rule = "GREEN"
                        self.get_logger().info(f"GREEN LIGHT @ {objectDist:.2f}m")
                    elif is_yellow:
                        rule = "YELLOW"
                        self.get_logger().info(f"YELLOW LIGHT @ {objectDist:.2f}m")

            # ── STOP SIGN — 3s ────────────────────────────────────────────────
            elif labelName == "stop sign" and labelConf > 0.85 and 0 < objectDist < self.stop_dist:
                rule = "STOP"
                self.get_logger().info(f"STOP SIGN @ {objectDist:.2f}m conf={labelConf:.2f}")
                break

            # ── YIELD SIGN — 1.5s ────────────────────────────────────────────
            elif labelName == "yield sign" and labelConf > 0.85 and 0 < objectDist < self.yield_dist:
                rule = "YIELD"
                self.get_logger().info(f"YIELD SIGN @ {objectDist:.2f}m conf={labelConf:.2f}")

        self._publish_rule(rule)
        print("=" * 31)

    # ------------------------------------------------------------------

    def publish_motion_flag(self, enable: bool):
        msg = Bool()
        msg.data = enable
        self.motion_publisher.publish(msg)

    def terminate(self):
        self.QCarImg.terminate()


def main():
    rclpy.init()
    node = ObjectDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.terminate()
    rclpy.shutdown()


if __name__ == '__main__':
    main()