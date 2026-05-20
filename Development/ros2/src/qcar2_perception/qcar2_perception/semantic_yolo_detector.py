#!/usr/bin/env python3
# ROS2 node for semantic detection using YOLOv8. This node subscribes to RGB images, 
# runs YOLOv8 inference, and publishes annotated images and detection results.
# This is the new designation for yolo v8, letting the motion_enable handling from another entity
# of behavior, this "behavior" will probably be delegated from a file for free use for
# nav_to_pose/path_follower, or another way of Arturo handling car node_following behavior as a helper.
# In case of changes please, date and sign change with a number and reason for better tracking
# I'm not a programmer but I want to keep this organized lol.
 
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.getlogin = lambda: os.environ.get("USER", "admin")


def add_mdc_paths():
    candidates = [
        "/workspaces/isaac_ros-dev/MDC_libraries/python",
        str(Path.home() / "Documents/ACC_Development/Development/MDC_libraries/python"),
        "/home/nvidia/Documents/ACC_Development/Development/MDC_libraries/python",
    ]

    env_path = os.getenv("MDC_PYTHON_PATH", "").strip()
    if env_path:
        candidates = env_path.split(":") + candidates

    for path in candidates:
        if path and Path(path).exists() and path not in sys.path:
            sys.path.insert(0, path)


add_mdc_paths()

from pit.YOLO.nets import YOLOv8

# ROS2 node responsible only for semantic 2D detection.
# It subscribes to aligned RGB and publishes YOLO detections + annotated image.
class SemanticYoloDetector(Node):

    # Initializes parameters, loads YOLO, creates subscribers and publishers.
    # This node subscribes to RGB only; depth and 3D estimation are handled elsewhere.
    def __init__(self):
        super().__init__("semantic_yolo_detector")

        self.bridge = CvBridge()

        self.declare_parameter(
            "model_path",
            "/workspaces/isaac_ros-dev/ros2/src/qcar2_autonomy/models/quanser_yolov8s-seg.pt",
        )
        self.declare_parameter("confidence", 0.30)
        self.declare_parameter("class_filter", "2,9,11")
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("input_topic", "/perception/d435/rgb/image_raw")
        self.declare_parameter("annotated_topic", "/perception/yolo/image_annotated")
        self.declare_parameter("detections_topic", "/perception/yolo/detections_2d")

        self.model_path = str(self.get_parameter("model_path").value)
        self.confidence = float(self.get_parameter("confidence").value)
        self.class_filter = self.parse_class_filter(
            str(self.get_parameter("class_filter").value)
        )
        self.image_width = int(self.get_parameter("image_width").value)
        self.image_height = int(self.get_parameter("image_height").value)

        input_topic = str(self.get_parameter("input_topic").value)
        annotated_topic = str(self.get_parameter("annotated_topic").value)
        detections_topic = str(self.get_parameter("detections_topic").value)

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")

        self.get_logger().info(f"Loading YOLO model: {self.model_path}")

        self.yolo = YOLOv8(
            modelPath=self.model_path,
            imageHeight=self.image_height,
            imageWidth=self.image_width,
            convert_tensorrt=False,
        )

        self.image_sub = self.create_subscription(
            Image,
            input_topic,
            self.image_cb,
            qos_profile_sensor_data,
        )

        self.annotated_pub = self.create_publisher(Image, annotated_topic, 10)
        self.detections_pub = self.create_publisher(String, detections_topic, 10)

        self.frame_count = 0

        self.get_logger().info(
            f"semantic_yolo_detector ready. input={input_topic}, "
            f"class_filter={self.class_filter}, confidence={self.confidence}"
        )

    # Converts a comma-separated class filter string into a list of YOLO class IDs.
    # Returns None when all classes should be allowed.
    def parse_class_filter(self, text):
        text = text.strip()
        if text == "" or text.lower() in ("none", "all"):
            return None

        output = []
        for item in text.split(","):
            item = item.strip()
            if item:
                output.append(int(item))
        return output

    def image_cb(self, msg):
    # Main image callback. Converts ROS image to OpenCV, runs YOLO,
    # publishes the annotated image, and publishes 2D detections as JSON.
    # This callback intentionally does not use depth or publish 3D markers.
    # The object_3d_estimator node handles depth-based object positioning.
    
        self.frame_count += 1

        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"Could not decode image: {exc}")
            return

        try:
            processed = self.yolo.pre_process(rgb)
            prediction = self.yolo.predict(
                inputImg=processed,
                classes=self.class_filter,
                confidence=self.confidence,
                half=False,
                verbose=False,
            )
        except Exception as exc:
            self.get_logger().error(f"YOLO prediction failed: {exc}")
            return

        self.publish_annotated_image(msg, prediction)
        detections = self.extract_detections(msg, prediction)

        out = String()
        out.data = json.dumps(detections)
        self.detections_pub.publish(out)

        if self.frame_count % 30 == 1:
            self.get_logger().info(
                f"Published {len(detections['detections'])} YOLO detections."
            )

    def publish_annotated_image(self, input_msg, prediction):
    # Builds a YOLO overlay image for debugging/RViz and publishes it.
    # The output keeps the same timestamp/frame_id as the input image.

        
        annotated = None

        try:
            annotated = self.yolo.post_process_render(showFPS=True)
        except Exception:
            annotated = None

        if annotated is None:
            try:
                pred0 = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
                if hasattr(pred0, "plot"):
                    annotated = pred0.plot()
            except Exception:
                annotated = None

        if annotated is None or not isinstance(annotated, np.ndarray):
            return

        try:
            ann_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            ann_msg.header.stamp = input_msg.header.stamp
            ann_msg.header.frame_id = input_msg.header.frame_id
            self.annotated_pub.publish(ann_msg)
        except Exception as exc:
            self.get_logger().warn(f"Could not publish annotated image: {exc}")

    def extract_detections(self, input_msg, prediction):
    # Converts YOLO prediction boxes into a structured detection dictionary.
    # This is a temporary JSON message format until custom Detection2D messages exist.
    
        pred0 = prediction[0] if isinstance(prediction, (list, tuple)) else prediction

        output = {
            "header": {
                "stamp": {
                    "sec": int(input_msg.header.stamp.sec),
                    "nanosec": int(input_msg.header.stamp.nanosec),
                },
                "frame_id": input_msg.header.frame_id,
            },
            "source_sensor": "d435_aligned_rgb",
            "detections": [],
        }

        boxes = None
        try:
            if hasattr(pred0, "boxes") and pred0.boxes is not None:
                boxes = pred0.boxes
        except Exception as exc:
            self.get_logger().warn(f"Could not access YOLO boxes: {exc}")
            return output

        if boxes is None or len(boxes) == 0:
            return output

        for i, box in enumerate(boxes):
            try:
                xyxy = box.xyxy[0].detach().cpu().numpy()
                x1, y1, x2, y2 = [float(v) for v in xyxy]

                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                class_name = self.get_class_name(pred0, cls_id)

                output["detections"].append({
                    "detection_id": f"{input_msg.header.stamp.sec}_{input_msg.header.stamp.nanosec}_{i}",
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    },
                    "source_sensor": "d435_aligned_rgb",
                })
            except Exception as exc:
                self.get_logger().warn(f"Could not parse YOLO box: {exc}")

        return output

    def get_class_name(self, pred0, cls_id):
    #Resolves a YOLO class id into readable class name.
    #Falls back to the numeric id if the model does not expose names.
        try:
            if hasattr(pred0, "names") and pred0.names is not None:
                return str(pred0.names.get(cls_id, cls_id))
        except Exception:
            pass

        try:
            if hasattr(self.yolo, "model") and hasattr(self.yolo.model, "names"):
                return str(self.yolo.model.names.get(cls_id, cls_id))
        except Exception:
            pass

        return str(cls_id)


def main(args=None):
    rclpy.init(args=args)
    node = SemanticYoloDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
