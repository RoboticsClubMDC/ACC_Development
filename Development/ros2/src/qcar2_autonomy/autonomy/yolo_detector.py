#! /usr/bin/env python3
import sys
import threading

sys.path.insert(0, "/workspaces/isaac_ros-dev/MDC_libraries/python")

# Quanser specific packages
from pit.YOLO.nets import YOLOv8
# --- 2026-05-14: camera ownership moved to qcar2_camera_bridge ---------
# This node previously instantiated QCar2DepthAligned itself, which made
# it a second owner of the RealSense alongside rgbd / the new bridge.
# Under the single-owner architecture the bridge is the sole PIT client;
# this node now subscribes to /camera/color_image and /camera/depth_image
# (aligned 32FC1 meters from the bridge, or MONO16 raw from legacy rgbd
# if camera_source:=rgbd is selected in the launch). YOLO inference,
# /motion_enable braking is unchanged. Trip LEDs are owned only by
# trip_planner -> Planner_server.
#
# To restore direct PIT-ownership behavior (not recommended outside
# diagnostic runs), uncomment the import + instantiation below and
# disable qcar2_camera_bridge in the launch file.
# from pit.YOLO.utils import QCar2DepthAligned
# -----------------------------------------------------------------------

# Generic python packages
import time  # Time library
import numpy as np
import cv2

# ROS specific packages
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

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
        # --- 2026-05-14: camera ownership replaced by ROS subscribers --
        # Original line (kept commented so the direct-PIT path can be
        # restored quickly if needed):
        #     self.QCarImg = QCar2DepthAligned()
        # The replacement is two camera subscribers + a small frame
        # buffer with thread-safe access. See _rgb_cb / _depth_cb below.
        self._frame_lock = threading.Lock()
        self._rgb_latest = None      # numpy.ndarray HxWx3 uint8 (bgr)
        self._depth_latest = None    # numpy.ndarray HxW float32 meters
        # ---------------------------------------------------------------
        self.myYolo  = YOLOv8(
                    modelPath = "./ros2/src/qcar2_autonomy/models/quanser_yolov8s-seg.pt",
                    imageHeight= imageHeight,
                    imageWidth = imageWidth,
                    convert_tensorrt = False,
                    device = "cpu",
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

        self.sign_detected = False

        # publish image aligned information
        self.bridge = CvBridge()
        self.publish_rgb = self.create_publisher(Image,'/qcar_camera/rgb',10)
        self.publish_depth = self.create_publisher(Image,'/qcar_camera/depth',10)

        # --- 2026-05-14: camera ownership replaced by ROS subscribers --
        # Subscribe to whichever publisher is supplying camera data:
        #   - qcar2_camera_bridge (default, physical) -> 32FC1 meters
        #   - rgbd.cpp            (fallback)          -> MONO16 raw counts
        # Encoding is detected per frame in _depth_cb.
        # 2026-05-15: sensor_data QoS (BEST_EFFORT) to match the bridge
        # publishers. Reliable QoS on Image topics half-dropped at the DDS
        # layer; BEST_EFFORT is the correct semantics for live sensor data.
        self.create_subscription(
            Image, '/camera/color_image', self._rgb_cb, qos_profile_sensor_data)
        self.create_subscription(
            Image, '/camera/depth_image', self._depth_cb, qos_profile_sensor_data)
        # ---------------------------------------------------------------

        # Debug window — annotated YOLO frame.
        self.debug_window_name = 'YOLO Debug'
        self.publish_debug = self.create_publisher(Image, '/qcar_camera/yolo_debug', 1)
        self._last_results = []
        try:
            cv2.namedWindow(self.debug_window_name, cv2.WINDOW_NORMAL)
        except Exception as exc:
            self.get_logger().warn(f'cv2 window unavailable (headless?): {exc}')

        self.timer2 = self.create_timer(1/500, self.flag_publisher)

    def flag_publisher(self):
        # keep existing behavior
        self.publish_motion_flag(self.flag_value)

    # --- 2026-05-14: ROS subscriber callbacks (replace direct PIT read) ---
    def _valid_image(self, frame):
        return (
            frame is not None and
            hasattr(frame, "shape") and
            len(frame.shape) >= 2 and
            frame.shape[0] > 0 and
            frame.shape[1] > 0
        )

    def _rgb_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:
            self.get_logger().warn(
                f"yolo_detector RGB decode error: {exc}",
                throttle_duration_sec=5.0)
            return
        if not self._valid_image(frame):
            self.get_logger().warn(
                "yolo_detector received empty RGB frame",
                throttle_duration_sec=5.0)
            return
        with self._frame_lock:
            self._rgb_latest = frame

    def _depth_cb(self, msg):
        """Accept either 32FC1 (aligned meters from bridge) or MONO16
        (raw counts from rgbd.cpp). Store as float32 meters internally."""
        enc = getattr(msg, "encoding", "") or ""
        try:
            if enc == "32FC1":
                d = self.bridge.imgmsg_to_cv2(msg, "32FC1")
                if d.ndim == 3:
                    d = d[:, :, 0]
                depth = d.astype(np.float32, copy=False)
            else:
                # Legacy MONO16: divide by RealSense depth-units default
                # (0.001 m/unit -> divisor 1000) to get meters.
                d = self.bridge.imgmsg_to_cv2(msg, "passthrough")
                if d.ndim == 3:
                    d = d[:, :, 0]
                depth = d.astype(np.float32) / 1000.0
        except Exception as exc:
            self.get_logger().warn(
                f"yolo_detector depth decode error: {exc}",
                throttle_duration_sec=5.0)
            return
        if not self._valid_image(depth):
            self.get_logger().warn(
                "yolo_detector received empty depth frame",
                throttle_duration_sec=5.0)
            return
        with self._frame_lock:
            self._depth_latest = depth
    # ----------------------------------------------------------------------

    def on_timer(self):
        # --- 2026-05-14: read frames from subscribers instead of PIT ---
        # Original (commented; lines that drove direct PIT ownership):
        #     self.QCarImg.read()
        #     rgb = self.QCarImg.rgb
        #     depth = self.QCarImg.depth
        with self._frame_lock:
            rgb = None if self._rgb_latest is None else self._rgb_latest.copy()
            depth = None if self._depth_latest is None else self._depth_latest.copy()
        if rgb is None or depth is None:
            # Bridge / rgbd has not delivered a frame yet; skip this tick.
            return
        if not self._valid_image(rgb) or not self._valid_image(depth):
            return
        # Cache the latest frames for yolo_detect() which still references
        # them by name (and previously accessed self.QCarImg.rgb / .depth).
        self._current_rgb = rgb
        self._current_depth = depth

        # depth is already float32 in meters (32FC1 path) or converted from
        # MONO16 to meters in _depth_cb; no extra processing required.
        msg_rgb = self.bridge.cv2_to_imgmsg(rgb, "bgr8")
        msg_depth = self.bridge.cv2_to_imgmsg(depth, "32FC1")

        self.publish_rgb.publish(msg_rgb)
        self.publish_depth.publish(msg_depth)

        current_time = time.time()-self.t0
        delay = 0
        sign_delay = 0
        sign_detected = False

        if not self.sign_detected:
            sign_delay, sign_detected = self.yolo_detect()

            if sign_detected:
                delay = sign_delay

            if delay > 0.0 and not self.sign_detected:
                self.sign_detected = True
                self.disable_until = delay
                self.flag_value = False
            else:
                self.flag_value = True

        elif self.sign_detected:
            if current_time >= self.disable_until:
                if current_time >= self.detection_cooldown:
                    self.sign_detected = False
                self.flag_value = True

        self._render_debug(rgb)

    def _render_debug(self, rgb):
        annotated = rgb.copy()
        for det in self._last_results or []:
            d = det.__dict__
            name = d.get('name', '?')
            conf = float(d.get('conf', 0.0))
            dist = float(d.get('distance', 0.0))
            box = d.get('box') or d.get('bbox') or d.get('xyxy')
            if box is not None:
                try:
                    x1, y1, x2, y2 = [int(v) for v in box[:4]]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'{name} {conf:.2f} {dist:.2f}m'
                    cv2.putText(annotated, label, (x1, max(0, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                except Exception:
                    pass

        status = 'RUN' if self.flag_value else 'WAIT'
        color = (0, 0, 255) if status == 'STOPPED' else (0, 255, 0)
        cv2.putText(annotated, f'YOLO: {status}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        try:
            self.publish_debug.publish(self.bridge.cv2_to_imgmsg(annotated, 'bgr8'))
        except Exception as exc:
            self.get_logger().warn(f'debug image publish failed: {exc}',
                                   throttle_duration_sec=5.0)
        try:
            cv2.imshow(self.debug_window_name, annotated)
            cv2.waitKey(1)
        except Exception:
            pass

    def yolo_detect(self):
        detected = False
        delay = 0.0

        # --- 2026-05-14: feed YOLO from the buffered frames instead of
        #     directly from QCar2DepthAligned. Equivalent data (aligned
        #     RGB + 32FC1 depth in meters) sourced from the bridge.
        # Original (commented):
        #     rgbProcessed = self.myYolo.pre_process(self.QCarImg.rgb)
        #     ...post_processing(alignedDepth=self.QCarImg.depth, ...)
        rgb_for_yolo = getattr(self, "_current_rgb", None)
        depth_for_yolo = getattr(self, "_current_depth", None)
        if rgb_for_yolo is None or depth_for_yolo is None:
            return delay, detected

        rgbProcessed = self.myYolo.pre_process(rgb_for_yolo)
        predecion = self.myYolo.predict(inputImg = rgbProcessed,
                                    classes = [2,9,11,33],
                                    confidence = 0.3,
                                    half = False,
                                    verbose = False
                                    )

        processedResults = self.myYolo.post_processing(alignedDepth = depth_for_yolo,
                                                clippingDistance = 5)
        self._last_results = processedResults
        labelName = []
        labelConf = []
        for object in processedResults:
            labelName = object.__dict__["name"]
            labelConf = object.__dict__["conf"]
            objectDist = object.__dict__["distance"]

            if labelName == 'car' and labelConf > 0.9 and objectDist < 0.45 :
                self.get_logger().info("Car found!")

            elif labelName == "stop sign" and labelConf > 0.9 and objectDist < 0.52:
                self.get_logger().info("Stop Sign Detected!")

                delay = 3.0
                self.t0 = time.time()
                detected = True
                self.detection_cooldown = 10.0

            elif labelName == "yield sign" and labelConf > 0.9 and objectDist < 0.52:
                self.get_logger().info("Yield Sign Detected!")
                delay = 1.5
                self.t0 = time.time()
                detected = True
                self.detection_cooldown = 10.0

        print("===============================")
        return delay, detected

    def publish_motion_flag(self, enable:bool):
       msg = Bool()
       msg.data = enable
       self.motion_publisher.publish(msg)

    def terminate(self):
       # --- 2026-05-14: nothing to terminate; this node no longer owns
       #     QCar2DepthAligned. The bridge (qcar2_camera_bridge) handles
       #     runtime shutdown via its destructor.
       # Original (commented):
       #     self.QCarImg.terminate()
       try:
           cv2.destroyWindow(self.debug_window_name)
       except Exception:
           pass


def main():
  rclpy.init()
  node = ObjectDetector()
  try:
      rclpy.spin(node)
  except KeyboardInterrupt:
      node.terminate()
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
