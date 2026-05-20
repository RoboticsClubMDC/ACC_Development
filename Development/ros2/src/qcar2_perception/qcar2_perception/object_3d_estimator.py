#!/usr/bin/env python3

import json
import math

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


class Object3DEstimator(Node):
    """
    Converts YOLO 2D detections into 3D object observations.

    Responsibilities:
    - Subscribe to YOLO 2D detections.
    - Subscribe to aligned D435 depth.
    - Subscribe to D435 CameraInfo.
    - Estimate object position in aligned_camera_optical_frame.
    - Publish structured 3D object JSON.
    - Publish temporary RViz markers.

    Non-responsibilities:
    - Does not run YOLO.
    - Does not own the D435 camera.
    - Does not store semantic memory.
    - Does not decide vehicle behavior.
    """

    def __init__(self):
        super().__init__("object_3d_estimator")

        self.bridge = CvBridge()

        self.declare_parameter("detections_topic", "/perception/yolo/detections_2d")
        self.declare_parameter("depth_topic", "/perception/d435/depth/image_rect")
        self.declare_parameter("camera_info_topic", "/perception/d435/camera_info")
        self.declare_parameter("objects_topic", "/perception/objects_3d")
        self.declare_parameter("markers_topic", "/perception/object_markers")

        self.declare_parameter("min_depth", 0.05)
        self.declare_parameter("max_depth", 2.0)
        self.declare_parameter("depth_crop_ratio", 0.5)
        self.declare_parameter("min_valid_pixels", 10)
        self.declare_parameter("min_valid_ratio", 0.10)
        self.declare_parameter("min_quality", 0.05)

        self.min_depth = float(self.get_parameter("min_depth").value)
        self.max_depth = float(self.get_parameter("max_depth").value)
        self.depth_crop_ratio = float(self.get_parameter("depth_crop_ratio").value)
        self.min_valid_pixels = int(self.get_parameter("min_valid_pixels").value)
        self.min_valid_ratio = float(self.get_parameter("min_valid_ratio").value)
        self.min_quality = float(self.get_parameter("min_quality").value)

        detections_topic = str(self.get_parameter("detections_topic").value)
        depth_topic = str(self.get_parameter("depth_topic").value)
        camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        objects_topic = str(self.get_parameter("objects_topic").value)
        markers_topic = str(self.get_parameter("markers_topic").value)

        self.depth_img = None
        self.depth_stamp = None
        self.camera_info = None

        self.depth_sub = self.create_subscription(
            Image,
            depth_topic,
            self.depth_cb,
            qos_profile_sensor_data,
        )

        self.info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.info_cb,
            10,
        )

        self.detections_sub = self.create_subscription(
            String,
            detections_topic,
            self.detections_cb,
            10,
        )

        self.objects_pub = self.create_publisher(String, objects_topic, 10)
        self.markers_pub = self.create_publisher(MarkerArray, markers_topic, 10)

        self.frame_count = 0

        self.get_logger().info(
            f"object_3d_estimator ready. detections={detections_topic}, "
            f"depth={depth_topic}, camera_info={camera_info_topic}"
        )

    def depth_cb(self, msg):
        """
        Stores the latest aligned depth image.

        The depth source publishes 32FC1 depth in meters after distance_scale.
        """
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except Exception as exc:
            self.get_logger().warn(f"Could not decode depth image: {exc}")
            return

        if depth.ndim == 3:
            depth = depth[:, :, 0]

        self.depth_img = depth.astype(np.float32, copy=False)
        self.depth_stamp = msg.header.stamp

    def info_cb(self, msg):
        """
        Stores the latest camera intrinsics.

        We use K:
        fx = K[0], fy = K[4], cx = K[2], cy = K[5].
        """
        self.camera_info = msg

    def detections_cb(self, msg):
        """
        Main estimator callback.

        Receives YOLO 2D detections as JSON, combines them with latest
        aligned depth and camera intrinsics, then publishes 3D object estimates.
        """
        self.frame_count += 1

        if self.depth_img is None or self.camera_info is None:
            self.get_logger().warn(
                "Waiting for depth image and camera info.",
                throttle_duration_sec=2.0,
            )
            return

        try:
            detections_msg = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"Could not parse detections JSON: {exc}")
            return

        objects = []
        marker_array = MarkerArray()

        frame_id = detections_msg.get("header", {}).get(
            "frame_id",
            self.camera_info.header.frame_id,
        )

        detections = detections_msg.get("detections", [])

        marker_id = 0
        for det in detections:
            bbox = det.get("bbox", {})
            estimate = self.estimate_object_depth(bbox)

            if estimate is None:
                continue

            point = estimate["point_camera"]

            obj = {
                "detection_id": det.get("detection_id", ""),
                "class_id": det.get("class_id", -1),
                "class_name": det.get("class_name", "unknown"),
                "confidence": float(det.get("confidence", 0.0)),
                "source_sensor": det.get("source_sensor", "d435_aligned_rgb"),
                "frame_id": frame_id,
                "bbox": bbox,
                "pose_camera": {
                    "x": point[0],
                    "y": point[1],
                    "z": point[2],
                },
                "depth_median": estimate["median_depth"],
                "valid_depth_ratio": estimate["valid_ratio"],
                "depth_sigma": estimate["depth_sigma"],
                "depth_span": estimate["depth_span"],
                "quality_score": estimate["quality_score"],
                "uncertainty_radius": estimate["uncertainty_radius"],
            }

            objects.append(obj)

            marker_array.markers.append(
                self.make_sphere_marker(marker_id, frame_id, obj)
            )
            marker_id += 1

            marker_array.markers.append(
                self.make_text_marker(marker_id, frame_id, obj)
            )
            marker_id += 1

        out = String()
        out.data = json.dumps({
            "header": detections_msg.get("header", {}),
            "frame_id": frame_id,
            "objects": objects,
        })

        self.objects_pub.publish(out)
        self.markers_pub.publish(marker_array)

        if self.frame_count % 30 == 1:
            self.get_logger().info(
                f"Published {len(objects)} 3D object estimates."
            )

    def estimate_object_depth(self, bbox):
        """
        Estimates one object's 3D point from its bbox and aligned depth.

        This adapts the visual-odometry depth filtering idea for YOLO boxes:
        - center crop
        - valid depth ratio
        - median depth
        - MAD / robust sigma
        - p10-p90 span
        - quality score
        - uncertainty radius

        It is intentionally less aggressive than VO. YOLO boxes often contain
        edges, background, poles, signs, and floor, so weak estimates are kept
        with larger uncertainty instead of being dropped too early.
        """
        depth = self.depth_img
        h, w = depth.shape[:2]

        try:
            x1 = float(bbox["x1"])
            y1 = float(bbox["y1"])
            x2 = float(bbox["x2"])
            y2 = float(bbox["y2"])
        except Exception:
            return None

        x1 = max(0.0, min(float(w - 1), x1))
        x2 = max(0.0, min(float(w), x2))
        y1 = max(0.0, min(float(h - 1), y1))
        y2 = max(0.0, min(float(h), y2))

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
        patch_area = max(int(patch.size), 1)

        valid_mask = (
            np.isfinite(patch)
            & (patch > self.min_depth)
            & (patch < self.max_depth)
        )

        valid_count = int(np.count_nonzero(valid_mask))
        valid_ratio = float(valid_count) / float(patch_area)

        if valid_count < self.min_valid_pixels:
            return None

        if valid_ratio < self.min_valid_ratio:
            return None

        z_vals = patch[valid_mask].astype(np.float32)
        z_med = float(np.nanmedian(z_vals))

        if not math.isfinite(z_med) or z_med <= 0.0:
            return None

        mad = float(np.nanmedian(np.abs(z_vals - z_med)))
        depth_sigma = 1.4826 * mad

        p10 = float(np.nanpercentile(z_vals, 10))
        p90 = float(np.nanpercentile(z_vals, 90))
        depth_span = p90 - p10

        sigma_limit = max(0.015, 0.035 * z_med)
        span_limit = max(0.040, 0.120 * z_med)

        q_valid = min(1.0, valid_ratio / 0.50)
        q_noise = math.exp(-((depth_sigma / max(sigma_limit, 1e-6)) ** 2))
        q_span = math.exp(-((depth_span / max(span_limit, 1e-6)) ** 2))
        q_range = 1.0 / (1.0 + (z_med / 2.0) ** 2)

        quality = float(q_valid * q_noise * q_span * q_range)

        if quality < self.min_quality:
            return None

        fx = float(self.camera_info.k[0])
        fy = float(self.camera_info.k[4])
        cx = float(self.camera_info.k[2])
        cy = float(self.camera_info.k[5])

        u = cx_box
        v = cy_box

        x = (u - cx) * z_med / fx
        y = (v - cy) * z_med / fy
        z = z_med

        uncertainty_radius = self.estimate_uncertainty_radius(
            z_med,
            depth_sigma,
            depth_span,
            valid_ratio,
            quality,
        )

        return {
            "point_camera": [float(x), float(y), float(z)],
            "median_depth": float(z_med),
            "valid_ratio": float(valid_ratio),
            "depth_sigma": float(depth_sigma),
            "depth_span": float(depth_span),
            "quality_score": float(quality),
            "uncertainty_radius": float(uncertainty_radius),
        }

    def estimate_uncertainty_radius(
        self,
        z_med,
        depth_sigma,
        depth_span,
        valid_ratio,
        quality,
    ):
        """
        Converts depth quality metrics into a simple uncertainty radius.

        This is not a final calibrated covariance. It is a practical confidence
        bubble for RViz and landmark gating.
        """
        base = 0.04
        range_term = 0.03 * z_med
        noise_term = min(0.40, 2.0 * depth_sigma)
        span_term = min(0.40, 0.5 * depth_span)
        valid_penalty = 0.10 * (1.0 - min(1.0, valid_ratio))
        quality_penalty = 0.20 * (1.0 - min(1.0, quality))

        return base + range_term + noise_term + span_term + valid_penalty + quality_penalty

    def make_sphere_marker(self, marker_id, frame_id, obj):
        """
        Makes a sphere marker at the object camera-frame point.
        """
        p = obj["pose_camera"]
        radius = obj["uncertainty_radius"]

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = frame_id
        marker.ns = "perception_objects"
        marker.id = int(marker_id)
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = float(p["x"])
        marker.pose.position.y = float(p["y"])
        marker.pose.position.z = float(p["z"])
        marker.pose.orientation.w = 1.0

        marker.scale.x = max(0.05, float(radius))
        marker.scale.y = max(0.05, float(radius))
        marker.scale.z = max(0.05, float(radius))

        marker.color.r = 1.0
        marker.color.g = 0.35
        marker.color.b = 0.05
        marker.color.a = 0.85

        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 500_000_000
        return marker

    def make_text_marker(self, marker_id, frame_id, obj):
        """
        Makes a text label marker for the object estimate.
        """
        p = obj["pose_camera"]

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = frame_id
        marker.ns = "perception_object_labels"
        marker.id = int(marker_id)
        marker.type = Marker.TEXT_VIEW_FACING