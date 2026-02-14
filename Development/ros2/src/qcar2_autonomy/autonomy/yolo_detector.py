#! /usr/bin/env python3
import sys
sys.path.insert(0, "/workspaces/isaac_ros-dev/MDC_libraries/python")

# Quanser specific packages
from pit.YOLO.nets import YOLOv8
from pit.YOLO.utils import QCar2DepthAligned


# Generic python packages
import time  # Time library
import numpy as np
import cv2
from pathlib import Path
import urllib.request
import tempfile
import os

# ROS specific packages
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def ensure_model_exists(model_path: Path, url: str, logger=None) -> None:
    """
    Ensure model exists at model_path. If missing, download from url and save there atomically.
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # If file exists and looks non-empty, do nothing
    if model_path.exists() and model_path.stat().st_size > 1024 * 1024:
        if logger:
            logger.info(f"YOLO model found: {model_path} ({model_path.stat().st_size/1e6:.1f} MB)")
        return

    if logger:
        logger.warn(f"YOLO model not found, downloading to: {model_path}")
        logger.warn(f"Source: {url}")

    # Download to temp file first, then rename (atomic)
    with tempfile.NamedTemporaryFile(dir=str(model_path.parent), delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with urllib.request.urlopen(url) as r, open(tmp_path, "wb") as f:
            chunk = 1024 * 1024  # 1 MB
            while True:
                data = r.read(chunk)
                if not data:
                    break
                f.write(data)

        # basic sanity check
        size = tmp_path.stat().st_size
        if size < 1024 * 1024:
            raise RuntimeError(f"Downloaded file too small ({size} bytes), refusing to use it.")

        tmp_path.replace(model_path)

        if logger:
            logger.info(f"YOLO model downloaded OK: {model_path} ({model_path.stat().st_size/1e6:.1f} MB)")
    finally:
        # cleanup if something failed before replace()
        if tmp_path.exists() and (not model_path.exists() or tmp_path != model_path):
            try:
                tmp_path.unlink()
            except Exception:
                pass

'''
Description:

Node for detecting traffic light state and signs on the road. Provides flags
which define if a traffic signal has been detected and what action to take.
'''

class ObjectDetector(Node):

    def __init__(self):
        super().__init__('yolo_detector')
        # Additional parameters
        imageWidth  = 640
        imageHeight = 480
        self.QCarImg = QCar2DepthAligned()
        
        model_dir = Path("/workspaces/isaac_ros-dev/ros2/src/qcar2_autonomy/models")
        model_path = model_dir / "quanser_yolov8s-seg.pt"
        model_url = "https://quanserinc.box.com/shared/static/ce0gxomeg4b12wlcch9cmlh0376nditf.pt"

        ensure_model_exists(model_path, model_url, logger=self.get_logger())

        self.myYolo  = YOLOv8(
                    modelPath = str(model_path),
                    imageHeight= imageHeight,
                    imageWidth = imageWidth,
                    convert_tensorrt = False,
                )

        # Call on_timer function every second to receive pose info
        self.dt = 1/30
        self.timer = self.create_timer(self.dt, self.on_timer)

        self.motion_publisher = self.create_publisher(Bool,'/motion_enable',1)
        self.motion_enable = True
        self.detection_cooldown = 10.0
        self.disable_until = 0.0
        self.flag_value = False
        self.publish_motion_flag(True)
        self.t0 = time.time()
        
                # --- traffic light state ---
        self.tl_should_stop = False
        self.tl_last_seen = 0.0
        self.tl_timeout = 0.75          # seconds to trust last TL reading
        self.tl_conf = 0.50             # TL confidence threshold
        self.tl_stop_dist = 0.50       # meters (tune later)
        self.tl_color = "idle"

        self.sign_detected = False

        # publish image aligned information
        self.bridge = CvBridge()
        self.publish_rgb = self.create_publisher(Image,'/qcar_camera/rgb',10)
        self.publish_depth = self.create_publisher(Image,'/qcar_camera/depth',10)
        self.publish_rgb_yolo = self.create_publisher(Image, '/qcar_camera/rgb_yolo', 10)

        self.timer2 = self.create_timer(1/500, self.flag_publisher)

    def flag_publisher(self):
       self.publish_motion_flag(self.flag_value)

    def on_timer(self):
        # Get aligned RGB and Depth images and publish them
        self.QCarImg.read()

        rgb = self.QCarImg.rgb
        depth = self.QCarImg.depth

        # --- fix: force depth to float32 for 32FC1 ---
        if depth is not None:
            depth = np.asarray(depth)
            if depth.ndim == 3 and depth.shape[2] == 1:
                depth = depth[:, :, 0]
            depth = depth.astype(np.float32, copy=False)

        msg_rgb = self.bridge.cv2_to_imgmsg(rgb, "bgr8")
        msg_depth = self.bridge.cv2_to_imgmsg(depth, "32FC1")

        self.publish_rgb.publish(msg_rgb)
        self.publish_depth.publish(msg_depth)


        current_time = time.time()-self.t0
        delay = 0
        sign_delay = 0
        sign_detected = False
        if not self.sign_detected:
            # send image to the sign detector to check for a sign in the scene and return
            # a delay based on what's seen
            sign_delay, sign_detected = self.yolo_detect()

            if sign_detected:
                delay = sign_delay


            if delay > 0.0 and not self.sign_detected:
                self.sign_detected = True
                self.disable_until= delay
                self.flag_value = False
            else:
                self.flag_value = True
                # --- traffic light override (only when we are not currently in a sign stop window) ---
                if (time.time() - self.tl_last_seen) > self.tl_timeout:
                    self.tl_should_stop = False  # stale TL reading

                if self.tl_should_stop:
                    self.flag_value = False

        elif self.sign_detected:

          if current_time >= self.disable_until:
            if current_time >= self.detection_cooldown:
              self.sign_detected = False
            self.flag_value = True


    def yolo_detect(self):
        detected = False
        delay = 0.0

        rgbProcessed = self.myYolo.pre_process(self.QCarImg.rgb)
        predicion = self.myYolo.predict(inputImg = rgbProcessed,
                                    # classes = [2,9,11,33],
                                    classes = [9,11,33],
                                    confidence = 0.3,
                                    half = True,
                                    verbose = False
                                    )
                # ---- publish YOLO overlay (boxes/masks) for RViz ----
        try:
            ann = None
            pred = predicion

            # Ultralytics often returns a list of Results
            if isinstance(pred, (list, tuple)) and len(pred) > 0 and hasattr(pred[0], "plot"):
                ann = pred[0].plot()
            # or sometimes a single Results object
            elif hasattr(pred, "plot"):
                ann = pred.plot()

            if ann is not None and isinstance(ann, np.ndarray) and ann.size:
                self.publish_rgb_yolo.publish(self.bridge.cv2_to_imgmsg(ann, "bgr8"))
        except Exception as e:
            self.get_logger().warn(f"YOLO overlay publish failed: {e}")


        processedResults=self.myYolo.post_processing(alignedDepth = self.QCarImg.depth,
                                                clippingDistance = 5)
        labelName = []
        labelConf = []
        total_timer = 10
        for object in processedResults:

            labelName = object.__dict__["name"]
            labelConf = object.__dict__["conf"]
            objectDist = object.__dict__["distance"]
            self.get_logger().info(f"{labelName} @ {labelConf:.3f} @ {objectDist:.3f}")
                
            # --- traffic light handling (PIT sets name "traffic light (red)" and also lightColor) ---
            if labelName.startswith("traffic light"):
                color = str(object.__dict__.get("lightColor", "")).strip()  # "red", "yellow", "green", "idle"
                self.tl_color = color if color else "idle"
                self.tl_last_seen = time.time()

                if (labelConf > self.tl_conf) and (objectDist > 0.15) and (objectDist < self.tl_stop_dist):
                    if ("red" in self.tl_color) or ("yellow" in self.tl_color):
                        self.tl_should_stop = True
                        self.get_logger().info(f"Traffic Light {self.tl_color.upper()} @ {objectDist:.2f}m")
                else:
                    self.tl_should_stop = False

            # elif labelName == 'car' and labelConf > 0.9 and objectDist < 0.45 :
            #     self.get_logger().info(f"Car found at {objectDist}!")

            elif labelName == "stop sign" and labelConf > 0.80 and objectDist < 0.27:
            # elif labelName == "stop sign" and labelConf > 0.9:

                self.get_logger().info(f"Stop Sign Detected at {objectDist}!")
                delay = 3.0
                self.t0 = time.time()
                detected = True
                self.detection_cooldown = total_timer 

            elif labelName == "yield sign" and labelConf > 0.80 and objectDist < 0.27:
            # elif labelName == "yield sign" and labelConf > 0.9:
                self.get_logger().info(f"Yield Sign Detected at {objectDist}!")
                delay = 1.5
                self.t0 = time.time()
                detected = True
                self.detection_cooldown = total_timer 
                
            # print(object.__dict__)
        print("===============================")
        return delay, detected

    def publish_motion_flag(self, enable:bool):
       msg = Bool()
       msg.data = enable
       self.motion_publisher.publish(msg)

    def terminate(self):
       self.QCarImg.terminate()


def main():

  # Start the ROS 2 Python Client Library
  rclpy.init()

  node = ObjectDetector()
  try:
      rclpy.spin(node)
  except KeyboardInterrupt:
      node.terminate()
      pass

  rclpy.shutdown()

if __name__ == '__main__':
  main()