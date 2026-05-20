#!/usr/bin/env python3

import json
import math

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node

import tf2_geometry_msgs  # noqa: F401  Registers geometry transforms with tf2.
import tf2_ros
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


class SemanticConsistencyMonitor(Node):
    """
    Compares live object observations against known semantic landmarks.

    This node is intentionally advisory. It publishes residuals and health,
    but it does not correct Cartographer and does not command the car.
    """

    def __init__(self):
        super().__init__("semantic_consistency_monitor")

        self.declare_parameter("objects_topic", "/perception/objects_3d")
        self.declare_parameter("landmarks_topic", "/perception/semantic_landmarks")
        self.declare_parameter(
            "residual_topic",
            "/perception/semantic_localization_residual",
        )
        self.declare_parameter("health_topic", "/perception/health")
        self.declare_parameter(
            "markers_topic",
            "/perception/semantic_residual_markers",
        )
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("tf_timeout_sec", 0.15)
        self.declare_parameter("match_radius", 0.75)
        self.declare_parameter("small_residual", 0.20)
        self.declare_parameter("medium_residual", 0.45)
        self.declare_parameter("min_drift_objects", 2)
        self.declare_parameter("large_residual_fraction", 0.60)
        self.declare_parameter("marker_lifetime_sec", 1.0)

        objects_topic = str(self.get_parameter("objects_topic").value)
        landmarks_topic = str(self.get_parameter("landmarks_topic").value)
        residual_topic = str(self.get_parameter("residual_topic").value)
        health_topic = str(self.get_parameter("health_topic").value)
        markers_topic = str(self.get_parameter("markers_topic").value)

        self.map_frame = str(self.get_parameter("map_frame").value)
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)
        self.match_radius = float(self.get_parameter("match_radius").value)
        self.small_residual = float(self.get_parameter("small_residual").value)
        self.medium_residual = float(self.get_parameter("medium_residual").value)
        self.min_drift_objects = int(self.get_parameter("min_drift_objects").value)
        self.large_residual_fraction = float(
            self.get_parameter("large_residual_fraction").value
        )
        self.marker_lifetime_sec = float(
            self.get_parameter("marker_lifetime_sec").value
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.landmarks = []
        self.latest_objects_msg = None

        self.objects_sub = self.create_subscription(
            String,
            objects_topic,
            self.objects_cb,
            10,
        )
        self.landmarks_sub = self.create_subscription(
            String,
            landmarks_topic,
            self.landmarks_cb,
            10,
        )

        self.residual_pub = self.create_publisher(String, residual_topic, 10)
        self.health_pub = self.create_publisher(String, health_topic, 10)
        self.markers_pub = self.create_publisher(MarkerArray, markers_topic, 10)

        self.get_logger().info(
            "semantic_consistency_monitor ready. "
            f"objects={objects_topic}, landmarks={landmarks_topic}"
        )

    def landmarks_cb(self, msg):
        """
        Stores the latest semantic landmark set from the mapper.
        """
        try:
            landmarks_msg = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"Could not parse landmarks JSON: {exc}")
            return

        self.landmarks = landmarks_msg.get("landmarks", [])

    def objects_cb(self, msg):
        """
        Computes residuals whenever a fresh Object3D message arrives.
        """
        try:
            objects_msg = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"Could not parse objects JSON: {exc}")
            return

        self.latest_objects_msg = objects_msg

        if not self.landmarks:
            self.publish_health([], "NO_LANDMARKS", "No semantic landmarks yet.")
            return

        residuals, marker_array = self.compute_residuals(objects_msg)
        self.publish_residuals(objects_msg, residuals)
        self.publish_health(residuals)
        self.markers_pub.publish(marker_array)

    def compute_residuals(self, objects_msg):
        """
        Compares expected camera observations with actual D435 object estimates.
        """
        source_frame = objects_msg.get("frame_id")
        if not source_frame:
            source_frame = objects_msg.get("header", {}).get("frame_id", "")

        if not source_frame:
            return [], MarkerArray()

        stamp_msg = self.stamp_from_header(objects_msg.get("header", {}))
        objects = objects_msg.get("objects", [])
        residuals = []
        marker_array = self.clear_marker_array()
        marker_id = 1

        for landmark in self.trackable_landmarks():
            expected_camera = self.landmark_to_camera(
                landmark,
                source_frame,
                stamp_msg,
            )
            if expected_camera is None:
                continue

            match = self.find_matching_object(landmark, expected_camera, objects)
            if match is None:
                continue

            actual_camera = match["object"]["pose_camera"]
            residual = self.residual_dict(
                landmark,
                match["object"],
                expected_camera,
                actual_camera,
                match["distance"],
            )
            residuals.append(residual)

            marker = self.make_residual_marker(
                marker_id,
                landmark,
                match["object"],
                source_frame,
                stamp_msg,
            )
            if marker is not None:
                marker_array.markers.append(marker)
                marker_id += 1

        return residuals, marker_array

    def landmark_to_camera(self, landmark, source_frame, stamp_msg):
        """
        Transforms a map-frame landmark into the current camera frame.
        """
        pose_map = landmark.get("pose_map", {})

        try:
            point = PointStamped()
            point.header.frame_id = self.map_frame
            point.header.stamp = stamp_msg
            point.point.x = float(pose_map["x"])
            point.point.y = float(pose_map["y"])
            point.point.z = float(pose_map["z"])

            return self.tf_buffer.transform(
                point,
                source_frame,
                timeout=Duration(seconds=self.tf_timeout_sec),
            )
        except Exception as exc:
            self.get_logger().warn(
                f"Could not transform {self.map_frame} -> {source_frame}: {exc}",
                throttle_duration_sec=2.0,
            )
            return None

    def find_matching_object(self, landmark, expected_camera, objects):
        """
        Finds nearest same-class live object to the expected camera observation.
        """
        best = None
        best_distance = float("inf")
        class_name = landmark.get("class_name", "unknown")

        for obj in objects:
            if obj.get("class_name") != class_name:
                continue

            pose_camera = obj.get("pose_camera", {})
            try:
                dx = float(pose_camera["x"]) - float(expected_camera.point.x)
                dy = float(pose_camera["y"]) - float(expected_camera.point.y)
                dz = float(pose_camera["z"]) - float(expected_camera.point.z)
            except Exception:
                continue

            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            gate = max(
                self.match_radius,
                float(landmark.get("uncertainty_radius", 0.20)),
                float(obj.get("uncertainty_radius", 0.20)),
            )

            if distance < gate and distance < best_distance:
                best = {
                    "object": obj,
                    "distance": distance,
                }
                best_distance = distance

        return best

    def residual_dict(
        self,
        landmark,
        obj,
        expected_camera,
        actual_camera,
        distance,
    ):
        dx = float(actual_camera["x"]) - float(expected_camera.point.x)
        dy = float(actual_camera["y"]) - float(expected_camera.point.y)
        dz = float(actual_camera["z"]) - float(expected_camera.point.z)
        norm = math.sqrt(dx * dx + dy * dy + dz * dz)

        if norm <= self.small_residual:
            level = "small"
        elif norm <= self.medium_residual:
            level = "medium"
        else:
            level = "large"

        return {
            "landmark_id": landmark.get("id", ""),
            "class_name": landmark.get("class_name", "unknown"),
            "landmark_status": landmark.get("status", "unknown"),
            "detection_id": obj.get("detection_id", ""),
            "residual": {
                "x": dx,
                "y": dy,
                "z": dz,
                "norm": norm,
            },
            "level": level,
            "expected_camera": {
                "x": float(expected_camera.point.x),
                "y": float(expected_camera.point.y),
                "z": float(expected_camera.point.z),
            },
            "actual_camera": {
                "x": float(actual_camera["x"]),
                "y": float(actual_camera["y"]),
                "z": float(actual_camera["z"]),
            },
            "association_distance": float(distance),
            "object_quality_score": float(obj.get("quality_score", 0.0)),
            "object_confidence": float(obj.get("confidence", 0.0)),
        }

    def publish_residuals(self, objects_msg, residuals):
        out = String()
        out.data = json.dumps({
            "header": {
                "stamp": self.stamp_dict(),
                "frame_id": objects_msg.get("frame_id", ""),
            },
            "map_frame": self.map_frame,
            "residuals": residuals,
        })
        self.residual_pub.publish(out)

    def publish_health(self, residuals, status_override=None, message_override=None):
        matched = len(residuals)
        large_count = sum(1 for item in residuals if item.get("level") == "large")

        if matched > 0:
            norms = [float(item["residual"]["norm"]) for item in residuals]
            residual_mean = sum(norms) / float(len(norms))
            residual_max = max(norms)
        else:
            residual_mean = 0.0
            residual_max = 0.0

        drift_warning = (
            matched >= self.min_drift_objects
            and large_count / float(max(matched, 1)) >= self.large_residual_fraction
        )

        if status_override is not None:
            status = status_override
            message = message_override or status_override
        elif matched == 0:
            status = "NO_MATCH"
            message = "No live objects matched known landmarks."
        elif drift_warning:
            status = "DRIFT_WARNING"
            message = "Many stable semantic landmarks have large residuals."
        elif large_count > 0:
            status = "WARN"
            message = "One or more landmark residuals are large."
        else:
            status = "OK"
            message = "Semantic observations are consistent."

        out = String()
        out.data = json.dumps({
            "header": {
                "stamp": self.stamp_dict(),
                "frame_id": self.map_frame,
            },
            "status": status,
            "residual_mean": residual_mean,
            "residual_max": residual_max,
            "matched_count": matched,
            "large_residual_count": large_count,
            "drift_warning": drift_warning,
            "active_sources": ["d435_aligned_rgbd"],
            "message": message,
        })
        self.health_pub.publish(out)

    def make_residual_marker(self, marker_id, landmark, obj, source_frame, stamp_msg):
        """
        Draws a map-frame line from stored landmark to live observed object.
        """
        pose_map = landmark.get("pose_map", {})
        pose_camera = obj.get("pose_camera", {})

        try:
            actual_camera = PointStamped()
            actual_camera.header.frame_id = source_frame
            actual_camera.header.stamp = stamp_msg
            actual_camera.point.x = float(pose_camera["x"])
            actual_camera.point.y = float(pose_camera["y"])
            actual_camera.point.z = float(pose_camera["z"])

            actual_map = self.tf_buffer.transform(
                actual_camera,
                self.map_frame,
                timeout=Duration(seconds=self.tf_timeout_sec),
            )

            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = self.map_frame
            marker.ns = "semantic_residuals"
            marker.id = int(marker_id)
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.scale.x = 0.025

            marker.points.append(Point(
                x=float(pose_map["x"]),
                y=float(pose_map["y"]),
                z=float(pose_map["z"]),
            ))
            marker.points.append(actual_map.point)

            marker.color.r = 1.0
            marker.color.g = 0.1
            marker.color.b = 0.1
            marker.color.a = 0.9

            self.set_marker_lifetime(marker)
            return marker
        except Exception:
            return None

    def clear_marker_array(self):
        marker_array = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        marker_array.markers.append(marker)
        return marker_array

    def trackable_landmarks(self):
        """
        Uses confirmed/stable landmarks first; candidates are too noisy.
        """
        return [
            landmark
            for landmark in self.landmarks
            if landmark.get("status") in ("confirmed", "stable")
        ]

    def stamp_from_header(self, header):
        stamp = header.get("stamp", {})
        msg = self.get_clock().now().to_msg()
        msg.sec = int(stamp.get("sec", msg.sec))
        msg.nanosec = int(stamp.get("nanosec", msg.nanosec))
        return msg

    def stamp_dict(self):
        stamp = self.get_clock().now().to_msg()
        return {
            "sec": int(stamp.sec),
            "nanosec": int(stamp.nanosec),
        }

    def set_marker_lifetime(self, marker):
        lifetime = max(0.1, float(self.marker_lifetime_sec))
        marker.lifetime.sec = int(lifetime)
        marker.lifetime.nanosec = int((lifetime - int(lifetime)) * 1_000_000_000)


def main(args=None):
    rclpy.init(args=args)
    node = SemanticConsistencyMonitor()

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
