#!/usr/bin/env python3

import sys
sys.path.insert(0, "/workspaces/isaac_ros-dev/MDC_libraries/python")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Important for Docker/non-login shells:
# Quanser pit.YOLO.utils can call os.getlogin() during import.
os.getlogin = lambda: os.environ.get("USER", "admin")

import time
import math
import urllib.request
from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker, MarkerArray

from pit.YOLO.nets import YOLOv8
from pit.YOLO.utils import QCar2DepthAligned

print("### BP00 YOLO FILE IS RUNNING ###")

def ensure_model_exists(model_path: Path, model_url: str, logger=None):
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if model_path.exists():
        if logger:
            logger.info(f"YOLO model found: {model_path}")
        return

    if logger:
        logger.warn(f"YOLO model not found. Downloading to: {model_path}")

    urllib.request.urlretrieve(model_url, str(model_path))

    if logger:
        logger.info("YOLO model downloaded.")


class YoloDetector(Node):
    def __init__(self):
        super().__init__("yolo_detector")

        self.bridge = CvBridge()

        # -------------------------
        # Parameters
        # -------------------------
        self.declare_parameter("is_physical", False)
        self.declare_parameter("distance_scale", 0.1)
        self.declare_parameter("publish_rate", 5.0)
        self.declare_parameter("confidence", 0.30)

        # Use string so ROS parameter parsing is easy.
        # COCO: 2 car, 9 traffic light, 11 stop sign
        self.declare_parameter("class_filter", "2,9,11")

        self.declare_parameter("frame_id", "aligned_camera_optical_frame")

        # D435 RGB 640x480 intrinsics from your Quanser skill activity.
        self.declare_parameter("fx", 455.20)
        self.declare_parameter("fy", 459.43)
        self.declare_parameter("cx", 308.53)
        self.declare_parameter("cy", 213.56)

        self.declare_parameter("min_depth", 0.05)
        self.declare_parameter("max_depth", 2.0)
        self.declare_parameter("depth_crop_ratio", 0.5)

        # If true, only publish RGB/depth and skip YOLO.
        # Useful if camera stream freezes.
        self.declare_parameter("debug_camera_only", False)

        #change later to GPU, we are testing
        self.declare_parameter("device", "cpu")
        self.device = self.get_parameter("device").value
        #end of changeeeeeee

        self.is_physical = bool(self.get_parameter("is_physical").value)
        self.distance_scale = float(self.get_parameter("distance_scale").value)
        self.publish_rate = float(self.get_parameter("publish_rate").value)
        self.confidence = float(self.get_parameter("confidence").value)
        self.frame_id = str(self.get_parameter("frame_id").value)

        self.fx = float(self.get_parameter("fx").value)
        self.fy = float(self.get_parameter("fy").value)
        self.cx = float(self.get_parameter("cx").value)
        self.cy = float(self.get_parameter("cy").value)

        self.min_depth = float(self.get_parameter("min_depth").value)
        self.max_depth = float(self.get_parameter("max_depth").value)
        self.depth_crop_ratio = float(self.get_parameter("depth_crop_ratio").value)

        self.debug_camera_only = bool(self.get_parameter("debug_camera_only").value)

        self.class_filter = self.parse_class_filter(
            str(self.get_parameter("class_filter").value)
        )

        # -------------------------
        # Publishers
        # -------------------------
        self.publish_rgb = self.create_publisher(Image, "/qcar_camera/rgb", 10)
        self.publish_depth = self.create_publisher(Image, "/qcar_camera/depth", 10)
        self.publish_rgb_yolo = self.create_publisher(Image, "/qcar_camera/rgb_yolo", 10)

        self.motion_publisher = self.create_publisher(Bool, "/motion_enable", 1)

        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/semantic_yolo/markers",
            10
        )

        # Always allow motion for this semantic test node.
        # We are not using this node for stop logic right now.
        self.publish_motion_flag(True)

        self.get_logger().info("Initializing QCar2DepthAligned...")
        self.get_logger().info(f"is_physical={self.is_physical}")

        # Quanser typo is real: isPhyscial, not isPhysical.
        self.QCarImg = QCar2DepthAligned(isPhyscial=self.is_physical)

        # In virtual mode, avoid blocking behavior if supported by Camera3D.
        if not self.is_physical and hasattr(self.QCarImg, "camera"):
            try:
                self.QCarImg.camera.readMode = 0
            except Exception:
                pass

        self.get_logger().info("QCar2DepthAligned initialized.")

        # -------------------------
        # Quanser YOLO model
        # -------------------------
        image_height = 480
        image_width = 640

        model_dir = Path("/workspaces/isaac_ros-dev/ros2/src/qcar2_autonomy/models")
        model_path = model_dir / "quanser_yolov8s-seg.pt"

        model_url = (
            "https://quanserinc.box.com/shared/static/"
            "ce0gxomeg4b12wlcch9cmlh0376nditf.pt"
        )

        ensure_model_exists(model_path, model_url, logger=self.get_logger())

        self.get_logger().info("Loading Quanser YOLOv8 model...")

        self.myYolo = YOLOv8(
            modelPath=str(model_path),
            imageHeight=image_height,
            imageWidth=image_width,
            convert_tensorrt=False,
        )

        self.get_logger().info("YOLOv8 model loaded.")
        self.get_logger().info(f"class_filter={self.class_filter}")
        self.get_logger().info(f"distance_scale={self.distance_scale}")
        self.get_logger().info(
            f"K: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}"
        )

        self.frame_count = 0
        self.logged_first_detection_keys = False

        period = 1.0 / max(self.publish_rate, 0.5)
        self.timer = self.create_timer(period, self.on_timer)

        self.get_logger().info("yolo_detector timer started.")
        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/yolo_3d_markers",
            10
        )

        self.get_logger().info("YOLO 3D marker publisher ready: /yolo_3d_markers")
    
    def camera_to_base_link(self, x_c, y_c, z_c):
        """
        Convert RealSense camera optical-frame point to QCar2 base_link/body frame.

        Camera frame:
        x = left
        y = down
        z = forward/outward

        Body/base_link frame:
        x = forward
        y = left
        z = up
        """
        x_b = z_c + 0.095
        y_b = -x_c + 0.032
        z_b = -y_c + 0.172
        return x_b, y_b, z_b
    
    def publish_detection_markers(self, detections):
        """
        detections format:
        [
            {
                "class_name": "stop sign",
                "conf": 0.88,
                "camera_point": (x_c, y_c, z_c)
            }
        ]
        """
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        for i, det in enumerate(detections):
            class_name = det["class_name"]
            conf = det["conf"]
            x_c, y_c, z_c = det["camera_point"]

            x_b, y_b, z_b = self.camera_to_base_link(x_c, y_c, z_c)

            marker = Marker()
            marker.header.stamp = now
            marker.header.frame_id = "base_link"

            marker.ns = "yolo_3d_objects"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = float(x_b)
            marker.pose.position.y = float(y_b)
            marker.pose.position.z = float(z_b)

            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.08
            marker.scale.y = 0.08
            marker.scale.z = 0.08

            if "stop" in class_name.lower():
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            elif "car" in class_name.lower():
                marker.color.r = 0.0
                marker.color.g = 0.4
                marker.color.b = 1.0
            else:
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0

            marker.color.a = 1.0

            marker.lifetime.sec = 1
            marker_array.markers.append(marker)

            text_marker = Marker()
            text_marker.header.stamp = now
            text_marker.header.frame_id = "base_link"

            text_marker.ns = "yolo_3d_labels"
            text_marker.id = i + 1000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            text_marker.pose.position.x = float(x_b)
            text_marker.pose.position.y = float(y_b)
            text_marker.pose.position.z = float(z_b + 0.15)

            text_marker.pose.orientation.w = 1.0

            text_marker.scale.z = 0.12
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0

            text_marker.text = f"{class_name} {conf:.2f}"
            text_marker.lifetime.sec = 1

            marker_array.markers.append(text_marker)

        self.marker_pub.publish(marker_array)

    def parse_class_filter(self, text):
        text = text.strip()

        if text == "" or text.lower() in ["none", "all"]:
            return None

        output = []

        for item in text.split(","):
            item = item.strip()
            if item == "":
                continue
            output.append(int(item))

        return output

    def publish_motion_flag(self, enable: bool):
        msg = Bool()
        msg.data = bool(enable)
        self.motion_publisher.publish(msg)

    def on_timer(self):
        self.frame_count += 1

        if self.frame_count % 20 == 1:
            self.get_logger().info("Timer tick: reading QCar2DepthAligned...")

        try:
            new_frame = self.QCarImg.read()
        except Exception as exc:
            self.get_logger().error(f"QCarImg.read() failed: {exc}")
            return

        rgb = self.QCarImg.rgb
        depth = self.QCarImg.depth

        if rgb is None or depth is None:
            self.get_logger().warn("RGB or depth is None.")
            return

        rgb = np.asarray(rgb)

        depth = np.asarray(depth)
        if depth.ndim == 3 and depth.shape[2] == 1:
            depth = depth[:, :, 0]

        depth = depth.astype(np.float32, copy=False)

        # If depth looks like millimeters, convert to meters.
        if np.isfinite(depth).any() and np.nanmedian(depth) > 20.0:
            depth = depth / 1000.0

        # QLabs usually needs 0.1 because virtual distances are 10x.
        # Physical should usually be 1.0.
        depth = depth * self.distance_scale

        now = self.get_clock().now().to_msg()

        # From your test, QCarImg.rgb displays correctly as bgr8.
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb, encoding="bgr8")
        rgb_msg.header.stamp = now
        rgb_msg.header.frame_id = self.frame_id
        self.publish_rgb.publish(rgb_msg)

        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        depth_msg.header.stamp = now
        depth_msg.header.frame_id = self.frame_id
        self.publish_depth.publish(depth_msg)

        self.publish_motion_flag(True)

        if self.frame_count % 20 == 1:
            self.get_logger().info(
                "Published camera frames: "
                f"rgb={rgb.shape} depth={depth.shape} "
                f"depth_med={float(np.nanmedian(depth)):.3f}m "
                f"new_frame={new_frame}"
            )

        if self.debug_camera_only:
            return

        self.yolo_detect(rgb, depth)

    def yolo_detect(self, rgb, depth):
        try:
            rgb_processed = self.myYolo.pre_process(rgb)

            prediction = self.myYolo.predict(
                inputImg=rgb_processed,
                classes=self.class_filter,
                confidence=self.confidence,
                half=True,
                verbose=False,
            )

        except Exception as exc:
            self.get_logger().error(f"YOLO prediction failed: {exc}")
            return

        # -------------------------
        # Annotated image
        # -------------------------
        annotated = None

        try:
            # Quanser wrapper usually supports this.
            annotated = self.myYolo.post_process_render(showFPS=True)
        except Exception:
            annotated = None

        if annotated is None:
            try:
                pred0 = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
                if hasattr(pred0, "plot"):
                    annotated = pred0.plot()
            except Exception:
                annotated = None

        if annotated is not None and isinstance(annotated, np.ndarray):
            try:
                ann_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
                ann_msg.header.stamp = self.get_clock().now().to_msg()
                ann_msg.header.frame_id = self.frame_id
                self.publish_rgb_yolo.publish(ann_msg)
            except Exception as exc:
                self.get_logger().warn(f"Failed to publish annotated YOLO image: {exc}")

        # -------------------------
        # Quanser processed object info
        # -------------------------
        processed_results = []

        try:
            processed_results = self.myYolo.post_processing(
                alignedDepth=depth,
                clippingDistance=self.max_depth,
            )
        except Exception as exc:
            self.get_logger().warn(f"Quanser post_processing failed: {exc}")

        if processed_results and not self.logged_first_detection_keys:
            try:
                self.get_logger().info(
                    f"First Quanser object keys: {list(processed_results[0].__dict__.keys())}"
                )
                self.get_logger().info(
                    f"First Quanser object data: {processed_results[0].__dict__}"
                )
            except Exception:
                pass
            self.logged_first_detection_keys = True

        # -------------------------
        # Raw YOLO boxes
        # -------------------------
        boxes = None
        pred0 = None

        try:
            pred0 = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
            if hasattr(pred0, "boxes") and pred0.boxes is not None:
                boxes = pred0.boxes
        except Exception as exc:
            self.get_logger().warn(f"Could not access YOLO boxes: {exc}")

        marker_array = MarkerArray()

        if boxes is None or len(boxes) == 0:
            self.marker_pub.publish(marker_array)
            return

        marker_id = 0

        for i, box in enumerate(boxes):
            try:
                xyxy = box.xyxy[0].detach().cpu().numpy()
                x1, y1, x2, y2 = [int(round(v)) for v in xyxy]

                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

            except Exception as exc:
                self.get_logger().warn(f"Could not parse YOLO box: {exc}")
                continue

            class_name = self.get_class_name(pred0, cls_id)

            point = self.compute_object_3d(depth, x1, y1, x2, y2)

            if point is None:
                if self.frame_count % 20 == 1:
                    self.get_logger().info(
                        f"{class_name} conf={conf:.2f}: detected but no valid depth"
                    )
                continue

            X, Y, Z = point

            if self.frame_count % 10 == 1:
                self.get_logger().info(
                    f"{class_name} conf={conf:.2f} "
                    f"bbox=({x1},{y1},{x2},{y2}) "
                    f"camera_point=({X:.3f}, {Y:.3f}, {Z:.3f})"
                    detections_3d.append({
                        "class_name": class_name,
                        "conf": conf,
                        "camera_point": (X, Y, Z)
                    })
                )
                

            sphere = self.make_sphere_marker(
                marker_id,
                X,
                Y,
                Z,
                class_name,
                conf,
            )
            marker_array.markers.append(sphere)
            marker_id += 1

            text = self.make_text_marker(
                marker_id,
                X,
                Y,
                Z + 0.12,
                f"{class_name} {conf:.2f} Z={Z:.2f}m",
            )
            marker_array.markers.append(text)
            marker_id += 1

        self.marker_pub.publish(marker_array)

    def get_class_name(self, pred0, cls_id):
        try:
            if hasattr(pred0, "names") and pred0.names is not None:
                return str(pred0.names.get(cls_id, cls_id))
        except Exception:
            pass

        try:
            if hasattr(self.myYolo, "model") and hasattr(self.myYolo.model, "names"):
                return str(self.myYolo.model.names.get(cls_id, cls_id))
        except Exception:
            pass

        return str(cls_id)

    def compute_object_3d(self, depth, x1, y1, x2, y2):
        h, w = depth.shape[:2]

        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        ratio = max(0.1, min(1.0, self.depth_crop_ratio))

        cx_box = 0.5 * (x1 + x2)
        cy_box = 0.5 * (y1 + y2)
        bw = x2 - x1
        bh = y2 - y1

        crop_w = bw * ratio
        crop_h = bh * ratio

        rx1 = int(round(cx_box - crop_w / 2.0))
        rx2 = int(round(cx_box + crop_w / 2.0))
        ry1 = int(round(cy_box - crop_h / 2.0))
        ry2 = int(round(cy_box + crop_h / 2.0))

        rx1 = max(0, min(w - 1, rx1))
        rx2 = max(0, min(w, rx2))
        ry1 = max(0, min(h - 1, ry1))
        ry2 = max(0, min(h, ry2))

        if rx2 <= rx1 or ry2 <= ry1:
            return None

        patch = depth[ry1:ry2, rx1:rx2]
        vv, uu = np.mgrid[ry1:ry2, rx1:rx2]

        valid = (
            np.isfinite(patch)
            & (patch > self.min_depth)
            & (patch < self.max_depth)
        )

        if np.count_nonzero(valid) < 10:
            return None

        z_vals = patch[valid].astype(np.float32)
        u_vals = uu[valid].astype(np.float32)
        v_vals = vv[valid].astype(np.float32)

        x_vals = (u_vals - self.cx) * z_vals / self.fx
        y_vals = (v_vals - self.cy) * z_vals / self.fy

        X = float(np.nanmedian(x_vals))
        Y = float(np.nanmedian(y_vals))
        Z = float(np.nanmedian(z_vals))

        if not all(math.isfinite(v) for v in [X, Y, Z]):
            return None

        return X, Y, Z

    def make_sphere_marker(self, marker_id, x, y, z, label, conf):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.frame_id

        marker.ns = "semantic_yolo_objects"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.12
        marker.scale.y = 0.12
        marker.scale.z = 0.12

        marker.color.r = 1.0
        marker.color.g = 0.3
        marker.color.b = 0.1
        marker.color.a = 1.0

        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 500_000_000

        return marker

    def make_text_marker(self, marker_id, x, y, z, text):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.frame_id

        marker.ns = "semantic_yolo_labels"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.w = 1.0

        marker.scale.z = 0.12

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        marker.text = text

        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 500_000_000

        return marker

    def destroy_node(self):
        try:
            self.get_logger().info("Terminating QCar2DepthAligned.")
            self.QCarImg.terminate()
        except Exception as exc:
            self.get_logger().warn(f"Error while terminating QCar2DepthAligned: {exc}")

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = YoloDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()