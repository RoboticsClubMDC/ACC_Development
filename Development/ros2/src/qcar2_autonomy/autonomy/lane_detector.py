#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import cv2
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from qcar2_interfaces.msg import MotorCommands
from std_srvs.srv import SetBool

import os
from ament_index_python.packages import get_package_share_directory

# Containers often have no controlling TTY, so os.getlogin() explodes.
# Patch it BEFORE importing pit.* (they call it at import-time).
try:
    os.getlogin()
except OSError:
    os.getlogin = lambda: os.environ.get("LOGNAME", os.environ.get("USER", "virtual"))

from pit.LaneNet.nets import LaneNet


class LaneDetector(Node):
    """
    SDCS-06 style lane following (LaneNet + BEV + errors + PD control)
    Exposed as the `lane_detector` executable for qcar2_autonomy.
    """

    def __init__(self):
        super().__init__("lane_detector")
        self.bridge = CvBridge()

        # ---------------- PARAMETERS ----------------
        self.declare_parameter("image_topic", "/camera/color_image")
        self.declare_parameter("cmd_topic", "/qcar2_motor_speed_cmd")

        self.declare_parameter("throttle", 0.18)
        self.declare_parameter("kp_lat", 1.2)
        self.declare_parameter("kp_head", 0.6)
        self.declare_parameter("kd", 0.05)
        self.declare_parameter("steer_limit", 0.55)

        self.declare_parameter("enabled", False)

        # BEV parameters
        self.declare_parameter("bev_width", 640)
        self.declare_parameter("bev_height", 480)

        # Allow overriding model path from params
        # If empty, we will auto-resolve to: <qcar2_autonomy share>/models/lanenet.pt
        self.declare_parameter("model_path", "")

        # ---------------- LOAD PARAMS ----------------
        self.image_topic = self.get_parameter("image_topic").value
        self.cmd_topic = self.get_parameter("cmd_topic").value

        self.throttle = float(self.get_parameter("throttle").value)
        self.kp_lat = float(self.get_parameter("kp_lat").value)
        self.kp_head = float(self.get_parameter("kp_head").value)
        self.kd = float(self.get_parameter("kd").value)
        self.steer_limit = float(self.get_parameter("steer_limit").value)

        self.enabled = bool(self.get_parameter("enabled").value)

        self.bev_w = int(self.get_parameter("bev_width").value)
        self.bev_h = int(self.get_parameter("bev_height").value)

        # ---------------- LaneNet model path ----------------
        param_model_path = str(self.get_parameter("model_path").value).strip()

        if param_model_path:
            model_path = os.path.normpath(param_model_path)
        else:
            # This is the correct ROS2 way: use the package share directory.
            # Your repo has qcar2_autonomy/models/lanenet.pt, so we look there.
            pkg_share = get_package_share_directory("qcar2_autonomy")
            model_path = os.path.join(pkg_share, "models", "lanenet.pt")
            model_path = os.path.normpath(model_path)

        if not os.path.exists(model_path):
            self.get_logger().error(f"LaneNet model not found at: {model_path}")
            self.get_logger().error(
                "Put it at: qcar2_autonomy/models/lanenet.pt "
                "or set ROS param `model_path`."
            )
            raise FileNotFoundError(model_path)

        # ---------------- LaneNet ----------------
        self.lanenet = LaneNet(
            imageWidth=640,
            imageHeight=480,
            rowUpperBound=240,
            modelPath=model_path
        )

        # ---------------- ROS IO ----------------
        self.sub = self.create_subscription(Image, self.image_topic, self.on_image, 10)
        self.pub_cmd = self.create_publisher(MotorCommands, self.cmd_topic, 10)

        # Service name will be: /lane_detector/enable
        self.enable_srv = self.create_service(SetBool, "enable", self.enable_cb)

        self.prev_error = 0.0
        self.prev_time = None

        self.get_logger().info(f"lane_detector READY (model={model_path})")

    # ---------------- ENABLE ----------------
    def enable_cb(self, req, resp):
        self.enabled = bool(req.data)
        resp.success = True
        resp.message = f"enabled={self.enabled}"
        return resp

    # ---------------- IMAGE CALLBACK ----------------
    def on_image(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # LaneNet
        inp = self.lanenet.pre_process(rgb)
        binary_mask, _ = self.lanenet.predict(inp)
        mask = binary_mask.astype(np.uint8)

        # BEV
        bev = self.compute_bev(mask)

        # Errors
        lat_err, head_err = self.compute_errors(bev)

        # Control
        steer = self.control(lat_err, head_err)

        if self.enabled:
            self.publish_cmd(steer, self.throttle)
        else:
            self.publish_cmd(0.0, 0.0)

    # ---------------- BEV ----------------
    def compute_bev(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape

        src = np.float32([
            [0.45 * w, 0.6 * h],
            [0.55 * w, 0.6 * h],
            [0.9 * w,  h],
            [0.1 * w,  h],
        ])

        dst = np.float32([
            [0, 0],
            [self.bev_w, 0],
            [self.bev_w, self.bev_h],
            [0, self.bev_h],
        ])

        H = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(mask, H, (self.bev_w, self.bev_h))

    # ---------------- ERROR COMPUTATION ----------------
    def compute_errors(self, bev: np.ndarray):
        h, w = bev.shape

        ys, xs = np.where(bev > 0)
        if xs.size < 300:
            return 0.0, 0.0

        # Fit x = a*y + b (centerline)
        a, b = np.polyfit(ys, xs, 1)

        y_eval = h - 1
        x_center = a * y_eval + b

        lat_err = (x_center - (w / 2.0)) / (w / 2.0)  # normalized [-1,1]
        head_err = float(np.arctan(a))                # radians

        return float(lat_err), float(head_err)

    # ---------------- CONTROLLER ----------------
    def control(self, lat_err: float, head_err: float) -> float:
        now = self.get_clock().now().nanoseconds / 1e9
        if self.prev_time is None:
            self.prev_time = now

        dt = max(1e-3, now - self.prev_time)
        d_err = (lat_err - self.prev_error) / dt

        steer = (self.kp_lat * lat_err) + (self.kp_head * head_err) + (self.kd * d_err)
        steer = float(np.clip(steer, -self.steer_limit, self.steer_limit))

        self.prev_error = lat_err
        self.prev_time = now
        return steer

    # ---------------- PUBLISH ----------------
    def publish_cmd(self, steering: float, throttle: float):
        cmd = MotorCommands()
        cmd.motor_names = ["steering_angle", "motor_throttle"]
        cmd.values = [steering, throttle]
        self.pub_cmd.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
