#!/usr/bin/env python3
# ROS2 node for semantic detection using YOLOv8. This node subscribes to RGB images, 
# runs YOLOv8 inference, and publishes annotated images and detection results.
# This is the new designation for yolo v8, letting the motion_enable handling from another entity
# of behavior, this "behavior" will probably be delegated from a file for free use for
# nav_to_pose/path_follower, or another way of Arturo handling car node_following behavior as a helper.
# In case of changes please, date and sign change with a number and reason for better tracking
# I'm not a programmer but I want to keep this organized lol.
"""
Semantic YOLO detector node.

Responsibilities:
- Subscribe to aligned D435 RGB image.
- Run YOLO object detection.
- Publish annotated image for visualization.
- Publish 2D detections as structured JSON.

Downstream supposed nodes:
- object_3d_estimator consumes detections + depth.
- semantic_landmark_mapper stores map-frame landmarks.
- perception_behavior_interface decides behavior.
""" 


import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

if os.environ.get("QCAR2_FORCE_CPU", "").strip() in ("1", "true", "True"):
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


def find_default_model_path():
    model_name = "quanser_yolov8s-seg.pt"
    candidates = []

    try:
        autonomy_share = Path(get_package_share_directory("qcar2_autonomy"))
        candidates.append(autonomy_share / "models" / model_name)
    except PackageNotFoundError:
        pass

    candidates.extend([
        Path("/workspaces/isaac_ros-dev/src/qcar2_autonomy/models") / model_name,
        Path("/workspaces/isaac_ros-dev/ros2/src/qcar2_autonomy/models") / model_name,
        Path("/workspaces/isaac_ros-dev/Development/ros2/src/qcar2_autonomy/models") / model_name,
        Path("/home/arturo-noble/Documents/GitHub/ACC_Development/Development/ros2/src/qcar2_autonomy/models") / model_name,
        Path.home() / "Documents/GitHub/ACC_Development/Development/ros2/src/qcar2_autonomy/models" / model_name,
        Path.cwd() / "src/qcar2_autonomy/models" / model_name,
        Path.cwd() / "Development/ros2/src/qcar2_autonomy/models" / model_name,
    ])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(candidates[0] if candidates else Path(model_name))


def resolve_model_path(model_path):
    path = Path(model_path).expanduser()
    if path.exists():
        return str(path)

    fallback = Path(find_default_model_path())
    if fallback.exists():
        return str(fallback)

    return str(path)


# ROS2 node responsible only for semantic 2D detection.
# It subscribes to aligned RGB and publishes YOLO detections + annotated image.
class SemanticYoloDetector(Node):

    # Initializes parameters, loads YOLO, creates subscribers and publishers.
    # This node subscribes to RGB only; depth and 3D estimation are handled elsewhere.
    def __init__(self):
        super().__init__("semantic_yolo_detector")

        self.bridge = CvBridge()

        # Updated 2026-05-22 23:03:54 EDT:
        # Resolve the model from qcar2_autonomy install/share or the active
        # Development/ros2 source tree. Avoid the stale /workspaces/.../ros2
        # absolute path because this workspace lives under Development/ros2.
        self.declare_parameter(
            "model_path",
            find_default_model_path(),
        )
        self.declare_parameter("confidence", 0.30)
        self.declare_parameter("class_filter", "2,9,11")
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        # Added 2026-05-20 15:43:50 EDT:
        # Ignore polygon for QCar body/dead image area.
        # YOLO should not use this image region for inference.
        self.declare_parameter("enable_ignore_mask", True)
        self.declare_parameter("input_topic", "/perception/d435/rgb/image_raw")
        self.declare_parameter("annotated_topic", "/perception/yolo/image_annotated")
        self.declare_parameter("detections_topic", "/perception/yolo/detections_2d")

        self.model_path = resolve_model_path(str(self.get_parameter("model_path").value))
        self.confidence = float(self.get_parameter("confidence").value)
        self.class_filter = self.parse_class_filter(
            str(self.get_parameter("class_filter").value)
        )
        self.image_width = int(self.get_parameter("image_width").value)
        self.image_height = int(self.get_parameter("image_height").value)
        self.enable_ignore_mask = bool(
            self.get_parameter("enable_ignore_mask").value
        )

        self.ignore_polygons = [
            {
                "name": "qcar_body_ignore",
                "points": [
                    {"x": 405, "y": 480},
                    {"x": 407, "y": 462},
                    {"x": 414, "y": 448},
                    {"x": 427, "y": 436},
                    {"x": 442, "y": 429},
                    {"x": 458, "y": 429},
                    {"x": 474, "y": 442},
                    {"x": 510, "y": 446},
                    {"x": 536, "y": 480},
                ],
            }
        ]

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
        try:
            names = getattr(self.yolo.net, "names", {})
            self.get_logger().info(f"YOLO class names: {names}")
        except Exception:
            pass

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
    
    def apply_ignore_mask(self, image):
        """
        Blacks out ignored image polygons before YOLO inference.

        This prevents the visible QCar body/dead image area from creating false
        detections or weak trash landmarks.
        """
        if not self.enable_ignore_mask:
            return image

        masked = image.copy()

        for polygon in self.ignore_polygons:
            points = polygon.get("points", [])
            if len(points) < 3:
                continue

            pts = np.array(
                [[int(p["x"]), int(p["y"])] for p in points],
                dtype=np.int32,
            )

            cv2.fillPoly(masked, [pts], (0, 0, 0))

        return masked
    
    def draw_ignore_mask_overlay(self, image):
        """
        Draws the ignored YOLO region on the debug annotated image.

        NA means this area is intentionally not available for YOLO inference.
        """
        if not self.enable_ignore_mask:
            return

        for polygon in self.ignore_polygons:
            points = polygon.get("points", [])
            if len(points) < 3:
                continue

            pts = np.array(
                [[int(p["x"]), int(p["y"])] for p in points],
                dtype=np.int32,
            )

            overlay = image.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 255))
            cv2.addWeighted(overlay, 0.25, image, 0.75, 0, dst=image)
            cv2.polylines(image, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

            center_x = int(np.mean(pts[:, 0]))
            center_y = int(np.mean(pts[:, 1]))

            cv2.putText(
                image,
                "NA",
                (center_x - 18, center_y + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

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
            rgb_for_yolo = self.apply_ignore_mask(rgb)
            processed = self.yolo.pre_process(rgb_for_yolo)
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

        self.draw_ignore_mask_overlay(annotated)

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

#This is just ROS2 console-script entry point.
#maintains it alive and kills gracefully on shutdown.
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
