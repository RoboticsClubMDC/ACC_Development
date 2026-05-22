#! /usr/bin/env python3

import os
import sys

for library_root in (
    "/workspaces/isaac_ros-dev/MDC_libraries/python",
    "/home/nvidia/Documents/ACC_Development/Development/MDC_libraries/python",
    "/home/nvidia/Documents/Quanser/0_libraries/python",
):
    if os.path.isdir(library_root):
        sys.path.insert(0, library_root)
        break

from pit.YOLO.nets import YOLOv8
from pit.YOLO.utils import QCar2DepthAligned

import time
import numpy as np
import cv2
from pathlib import Path
import urllib.request
import tempfile

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from autonomy.cuda_utils import (
    clear_cuda_cache,
    is_cuda_runtime_error,
    select_yolo_device,
)


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

        if tmp_path.stat().st_size < 1024 * 1024:
            raise RuntimeError(f"Downloaded file too small, refusing to use it.")

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

        self.declare_parameter("device", 0)
        self.declare_parameter("use_cuda", True)
        self.declare_parameter("use_tensorrt", False)
        self.declare_parameter("inference_hz", 10.0)
        self.declare_parameter("motion_flag_hz", 30.0)
        self.declare_parameter("allow_cpu_fallback", False)

        imageWidth  = 640
        imageHeight = 480
        self.QCarImg = QCar2DepthAligned()

        device = int(self.get_parameter("device").get_parameter_value().integer_value)
        use_cuda = bool(self.get_parameter("use_cuda").get_parameter_value().bool_value)
        use_tensorrt = bool(self.get_parameter("use_tensorrt").get_parameter_value().bool_value)
        self.allow_cpu_fallback = bool(
            self.get_parameter("allow_cpu_fallback").get_parameter_value().bool_value)
        inference_hz = max(
            1.0,
            float(self.get_parameter("inference_hz").get_parameter_value().double_value),
        )
        motion_flag_hz = max(
            1.0,
            float(self.get_parameter("motion_flag_hz").get_parameter_value().double_value),
        )
        self.yolo_device = select_yolo_device(
            self.get_logger(),
            requested_device=device,
            use_cuda=use_cuda,
            context="Traffic YOLO",
            allow_cpu_fallback=self.allow_cpu_fallback,
        )

        model_dir = Path("/workspaces/isaac_ros-dev/ros2/src/qcar2_autonomy/models")
        if not model_dir.exists():
            model_dir = Path(__file__).resolve().parents[1] / "models"
        model_path = model_dir / "quanser_yolov8s-seg.pt"
        model_url  = "https://quanserinc.box.com/shared/static/ce0gxomeg4b12wlcch9cmlh0376nditf.pt"

        ensure_model_exists(model_path, model_url, logger=self.get_logger())

        self.myYolo = YOLOv8(
            modelPath=str(model_path),
            imageHeight=imageHeight,
            imageWidth=imageWidth,
            convert_tensorrt=use_tensorrt and self.yolo_device != "cpu",
            device=self.yolo_device,
        )
        self.get_logger().info(f"YOLO inference device: {self.yolo_device}")

        self.dt    = 1.0 / inference_hz
        self.timer = self.create_timer(self.dt, self.on_timer)

        self.motion_publisher = self.create_publisher(Bool, '/motion_enable', 1)
        self.flag_value       = False
        self.publish_motion_flag(True)

        self.sign_detected      = False
        self.disable_until      = 0.0
        self.detection_cooldown = 10.0
        self.t0                 = time.time()

        self.tl_conf       = 0.50
        self.tl_stop_dist  = 2.5
        self.tl_min_dist   = 0.5
        self.tl_hold       = 0.25
        self.tl_last_color = "idle"

        self.bridge          = CvBridge()
        self.publish_rgb     = self.create_publisher(Image, '/qcar_camera/rgb',       10)
        self.publish_depth   = self.create_publisher(Image, '/qcar_camera/depth',     10)
        self.publish_rgb_yolo = self.create_publisher(Image, '/qcar_camera/rgb_yolo', 10)
        self.timer2          = self.create_timer(1.0 / motion_flag_hz, self.flag_publisher)
        self.get_logger().info(
            f"YOLO timers: inference={inference_hz:.1f} Hz, motion_flag={motion_flag_hz:.1f} Hz"
        )

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

        self.publish_rgb.publish(self.bridge.cv2_to_imgmsg(rgb,   "bgr8"))
        self.publish_depth.publish(self.bridge.cv2_to_imgmsg(depth, "32FC1"))

        current_time = time.time() - self.t0

        if not self.sign_detected:
            delay, detected = self.yolo_detect()
            if detected and delay > 0.0:
                self.sign_detected = True
                self.disable_until = delay
                self.flag_value    = False
            else:
                self.flag_value = True
        else:
            if current_time >= self.disable_until:
                if current_time >= self.detection_cooldown:
                    self.sign_detected = False
                self.flag_value = True

    def yolo_detect(self):
        detected = False
        delay    = 0.0

        rgbProcessed = self.myYolo.pre_process(self.QCarImg.rgb)
        try:
            predicion, processedResults = self._run_yolo(rgbProcessed)
        except RuntimeError as e:
            if self.yolo_device == "cpu" or not is_cuda_runtime_error(e):
                raise
            if not self.allow_cpu_fallback:
                self.get_logger().error(
                    f"YOLO CUDA inference failed on device={self.yolo_device}: {e}. "
                    "CPU fallback is disabled for traffic detection.")
                raise
            self.get_logger().error(
                f"YOLO CUDA inference failed on device={self.yolo_device}: {e}. "
                "Falling back to CPU for traffic detection.")
            self.yolo_device = "cpu"
            clear_cuda_cache()
            if hasattr(self.myYolo, "set_device"):
                self.myYolo.set_device("cpu")
            predicion, processedResults = self._run_yolo(rgbProcessed)

        try:
            ann  = None
            pred = predicion
            if isinstance(pred, (list, tuple)) and len(pred) > 0 and hasattr(pred[0], "plot"):
                ann = pred[0].plot()
            elif hasattr(pred, "plot"):
                ann = pred.plot()
            if ann is not None and isinstance(ann, np.ndarray) and ann.size:
                self.publish_rgb_yolo.publish(self.bridge.cv2_to_imgmsg(ann, "bgr8"))
        except Exception as e:
            self.get_logger().warn(f"YOLO overlay publish failed: {e}")

        total_timer = 10.0

        for obj in processedResults:
            labelName  = obj.__dict__.get("name",     "")
            labelConf  = float(obj.__dict__.get("conf",     0.0))
            objectDist = float(obj.__dict__.get("distance", -1.0))

            self.get_logger().info(f"{labelName} @ {labelConf:.3f} conf. @ {objectDist:.3f}m")

            if str(labelName).startswith("traffic light"):
                color = str(obj.__dict__.get("lightColor", "")).strip().lower()
                self.tl_last_color = color if color else "idle"

                is_valid_dist = (objectDist > self.tl_min_dist) and (objectDist < self.tl_stop_dist)
                is_stop_color = ("red" in self.tl_last_color) or ("yellow" in self.tl_last_color)

                if (labelConf >= self.tl_conf) and is_valid_dist and is_stop_color:
                    delay = max(delay, self.tl_hold)
                    detected = True
                    self.detection_cooldown = 0.0
                    self.t0 = time.time()
                    self.get_logger().info(
                        f"Traffic Light {self.tl_last_color.upper()} @ {objectDist:.2f}m -> STOP")

            elif labelName == "stop sign" and labelConf > 0.9 and objectDist < 1.0:
                self.get_logger().info(f"Stop Sign Detected at {objectDist}m!")
                delay = max(delay, 3.0)
                self.t0 = time.time()
                detected = True
                self.detection_cooldown = total_timer

            elif labelName == "yield sign" and labelConf > 0.9 and objectDist < 1.0:
                self.get_logger().info(f"Yield Sign Detected at {objectDist}m!")
                delay = max(delay, 1.5)
                self.t0 = time.time()
                detected = True
                self.detection_cooldown = total_timer

        print("===============================")
        return delay, detected

    def _run_yolo(self, rgbProcessed):
        prediction = self.myYolo.predict(
            inputImg=rgbProcessed,
            classes=[9, 11, 33],
            confidence=0.3,
            half=self.yolo_device != "cpu",
            verbose=False
        )
        processedResults = self.myYolo.post_processing(
            alignedDepth=self.QCarImg.depth,
            clippingDistance=5
        )
        return prediction, processedResults

    def publish_motion_flag(self, enable: bool):
        msg      = Bool()
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
