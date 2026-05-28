#!/usr/bin/env python3
# Semantic map rule:
# The map is remembered evidence, not eternal truth.
# Same-class observations strengthen a landmark.
# Different static classes inside the same bubble create conflict evidence.
# Dynamic classes inside the bubble are treated as temporary occlusion.
import json
import math
from pathlib import Path

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node

import tf2_geometry_msgs  # noqa: F401  Registers geometry transforms with tf2.
import tf2_ros
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


class SemanticLandmarkMapper(Node):
    """
    Turns live 3D object observations into persistent semantic landmarks.

    Responsibilities:
    - Subscribe to object observations from object_3d_estimator.
    - Transform camera-frame points into the Cartographer map frame.
    - Match repeated observations into candidate/confirmed/stable landmarks.
    - Publish semantic landmark JSON.
    - Publish RViz/Foxglove markers in map.
    - Save a small JSON semantic memory file.

    Non-responsibilities:
    - Does not run YOLO.
    - Does not estimate depth.
    - Does not command vehicle motion.
    - Does not correct Cartographer.
    """

    def __init__(self):
        super().__init__("semantic_landmark_mapper")

        self.declare_parameter("objects_topic", "/perception/objects_3d")
        self.declare_parameter("landmarks_topic", "/perception/semantic_landmarks")
        self.declare_parameter(
            "markers_topic",
            "/perception/semantic_landmark_markers",
        )

        # Added 2026-05-20 16:08:30 EDT:
        # Split semantic visualization into stable landmarks, hypotheses, and
        # currently visible semantic observations so Foxglove is not ambiguous.
        self.declare_parameter(
            "hypothesis_markers_topic",
            "/perception/semantic_hypothesis_markers",
        )
        self.declare_parameter(
            "current_markers_topic",
            "/perception/semantic_current_markers",
        )
        self.declare_parameter("map_frame", "map")
        self.declare_parameter(
            "semantic_map_path",
            "/workspaces/isaac_ros-dev/ros2/src/qcar2_perception/maps/semantic_map.json",
        )
        self.declare_parameter("association_radius", 0.45)
        self.declare_parameter("confirmed_seen_count", 3)
        self.declare_parameter("stable_seen_count", 8)
        # Semantic conflict handling:
        # Same-place different-class observations should not overwrite immediately.
        # They first become conflict evidence and only replace after repeated proof.
        self.declare_parameter("conflict_evidence_count", 5)
        self.declare_parameter("dynamic_class_names", [
            "person",
            "car",
            "truck",
            "bus",
            "bicycle",
            "motorcycle",
            "robot",
            "cow",
        ])
        self.declare_parameter("max_history", 20)
        self.declare_parameter("enable_json_save", True)
        self.declare_parameter("publish_rate_hz", 2.0)
        self.declare_parameter("tf_timeout_sec", 0.15)

        # Added 2026-05-20 15:42:04 EDT:
        # Competition semantic memory is a run-local overlay on Cartographer.
        # Start clean each perception launch unless persistent reload is enabled.
        self.declare_parameter("reset_map_on_start", True)

        # Added 2026-05-20 13:58:39 EDT:
        # Candidates live in RAM for matching, but only permanent statuses are
        # written to semantic_map.json. XY gating is less fragile for signs.
        self.declare_parameter("permanent_statuses", ["stable"])
        self.declare_parameter("load_only_permanent", True)
        self.declare_parameter("association_use_xy_only", True)

        # Added 2026-05-20 13:47:02 EDT:
        # Separate long-term landmark confidence from whether YOLO sees it now.
        self.declare_parameter("visible_timeout_sec", 2.0)
        self.declare_parameter("max_marker_radius", 0.50)

        objects_topic = str(self.get_parameter("objects_topic").value)
        landmarks_topic = str(self.get_parameter("landmarks_topic").value)
        markers_topic = str(self.get_parameter("markers_topic").value)
        hypothesis_markers_topic = str(
            self.get_parameter("hypothesis_markers_topic").value
        )
        current_markers_topic = str(
            self.get_parameter("current_markers_topic").value
        )

        self.map_frame = str(self.get_parameter("map_frame").value)
        self.semantic_map_path = Path(
            str(self.get_parameter("semantic_map_path").value)
        )
        self.association_radius = float(
            self.get_parameter("association_radius").value
        )
        self.confirmed_seen_count = int(
            self.get_parameter("confirmed_seen_count").value
        )
        self.stable_seen_count = int(self.get_parameter("stable_seen_count").value)
        self.conflict_evidence_count = int(
            self.get_parameter("conflict_evidence_count").value
        )
        self.dynamic_class_names = set(
            self.normalize_class_name(name)
            for name in self.get_parameter("dynamic_class_names").value
        )
        self.max_history = int(self.get_parameter("max_history").value)
        self.enable_json_save = bool(self.get_parameter("enable_json_save").value)
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)
        self.reset_map_on_start = bool(
            self.get_parameter("reset_map_on_start").value
        )
        self.permanent_statuses = set(
            str(status)
            for status in self.get_parameter("permanent_statuses").value
        )
        self.load_only_permanent = bool(
            self.get_parameter("load_only_permanent").value
        )
        self.association_use_xy_only = bool(
            self.get_parameter("association_use_xy_only").value
        )
        self.visible_timeout_sec = float(
            self.get_parameter("visible_timeout_sec").value
        )
        self.max_marker_radius = float(self.get_parameter("max_marker_radius").value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.landmarks = []
        self.next_landmark_index = 1
        self.dirty = False

        if self.reset_map_on_start:
            self.get_logger().info(
                "reset_map_on_start=True, starting with empty semantic session map."
            )
            self.dirty = True
            if self.enable_json_save:
                self.save_semantic_map()
        else:
            self.load_semantic_map()

        self.objects_sub = self.create_subscription(
            String,
            objects_topic,
            self.objects_cb,
            10,
        )
        self.landmarks_pub = self.create_publisher(String, landmarks_topic, 10)
        self.markers_pub = self.create_publisher(MarkerArray, markers_topic, 10)
        self.hypothesis_markers_pub = self.create_publisher(
            MarkerArray,
            hypothesis_markers_topic,
            10,
        )
        self.current_markers_pub = self.create_publisher(
            MarkerArray,
            current_markers_topic,
            10,
        )

        publish_period = 1.0 / max(
            0.1,
            float(self.get_parameter("publish_rate_hz").value),
        )
        self.publish_timer = self.create_timer(publish_period, self.publish_state)

        self.get_logger().info(
            "semantic_landmark_mapper ready. "
            f"objects={objects_topic}, map_frame={self.map_frame}, "
            f"memory={self.semantic_map_path}, stable_markers={markers_topic}, "
            f"hypothesis_markers={hypothesis_markers_topic}, "
            f"current_markers={current_markers_topic}"
        )

    def objects_cb(self, msg):
        """
        Main mapper callback.

        Receives live Object3D JSON, transforms each object into map, then
        either updates a nearby landmark or creates a new candidate landmark.
        """
        try:
            objects_msg = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"Could not parse Object3D JSON: {exc}")
            return

        source_frame = objects_msg.get("frame_id")
        if not source_frame:
            source_frame = objects_msg.get("header", {}).get("frame_id", "")

        if not source_frame:
            self.get_logger().warn("Object message has no source frame.")
            return

        stamp_msg = self.stamp_from_header(objects_msg.get("header", {}))
        updated = 0

        for obj in objects_msg.get("objects", []):
            point_map = self.transform_object_to_map(obj, source_frame, stamp_msg)
            if point_map is None:
                continue

            same_class_landmark = self.find_matching_landmark(obj, point_map)
            if same_class_landmark is not None:
                self.update_landmark(same_class_landmark, obj, point_map)
            else:
                spatial_landmark = self.find_spatial_landmark(point_map)

                if spatial_landmark is not None:
                    self.handle_semantic_conflict(spatial_landmark, obj, point_map)
                else:
                    landmark = self.create_landmark(obj, point_map)
                    self.landmarks.append(landmark)

            self.dirty = True
            updated += 1

        if updated > 0:
            self.publish_state()
            if self.enable_json_save:
                self.save_semantic_map()

    def transform_object_to_map(self, obj, source_frame, stamp_msg):
        """
        Uses tf2 to transform one object point from camera frame into map.
        """
        pose_camera = obj.get("pose_camera", {})

        try:
            point = PointStamped()
            point.header.frame_id = source_frame
            point.header.stamp = stamp_msg
            point.point.x = float(pose_camera["x"])
            point.point.y = float(pose_camera["y"])
            point.point.z = float(pose_camera["z"])

            return self.tf_buffer.transform(
                point,
                self.map_frame,
                timeout=Duration(seconds=self.tf_timeout_sec),
            )
        except Exception as exc:
            self.get_logger().warn(
                f"Could not transform {source_frame} -> {self.map_frame}: {exc}",
                throttle_duration_sec=2.0,
            )
            return None

    def find_matching_landmark(self, obj, point_map):
        """
        Finds the nearest existing landmark with the same class inside the gate.
        """
        class_name = obj.get("class_name", "unknown")

        best_landmark = None
        best_distance = float("inf")

        for landmark in self.landmarks:
            if landmark.get("class_name") != class_name:
                continue

            distance = self.distance_to_landmark(landmark, point_map)
            gate = max(
                self.association_radius,
                float(landmark.get("uncertainty_radius", 0.20)),
                float(obj.get("uncertainty_radius", 0.20)),
            )

            if distance < gate and distance < best_distance:
                best_landmark = landmark
                best_distance = distance

        return best_landmark
    
    def find_spatial_landmark(self, point_map):
        """
        Finds the nearest landmark bubble by position only.

        This is used for semantic conflict detection:
        same bubble + different class = conflict, not instant replacement.
        """
        best_landmark = None
        best_distance = float("inf")

        for landmark in self.landmarks:
            distance = self.distance_to_landmark(landmark, point_map)
            gate = max(
                self.association_radius,
                float(landmark.get("uncertainty_radius", 0.20)),
            )

            if distance < gate and distance < best_distance:
                best_landmark = landmark
                best_distance = distance

        return best_landmark
    


    def create_landmark(self, obj, point_map):
        """
        Creates a new candidate landmark from the first observation.
        """
        now_sec = self.now_sec()
        landmark_id = self.make_landmark_id(obj)

        return {
            "id": landmark_id,
            "class_id": int(obj.get("class_id", -1)),
            "class_name": obj.get("class_name", "unknown"),
            "pose_map": self.point_dict(point_map),
            "uncertainty_radius": float(obj.get("uncertainty_radius", 0.30)),
            "confidence": self.observation_weight(obj),
            "seen_count": 1,
            "first_seen": now_sec,
            "last_seen": now_sec,
            "source_sensor": obj.get("source_sensor", "unknown"),
            "status": "candidate",

            # Memory status starts as candidate. Visibility is separate and
            # only says whether the object was observed recently.
            "visibility": "visible",
            "last_detection_id": obj.get("detection_id", ""),
            "history": [self.history_entry(obj, point_map, now_sec)],
        }

    def handle_semantic_conflict(self, landmark, obj, point_map):
        """
        Handles same-place different-class observations.

        Dynamic objects are treated as temporary occlusion.
        Static different-class objects become conflict evidence.
        Repeated conflict evidence can eventually replace the old landmark.
        """
        observed_class = obj.get("class_name", "unknown")

        if self.is_dynamic_class(observed_class):
            landmark["visibility"] = "possibly_occluded"
            landmark["last_observed_class"] = observed_class
            return

        conflicts = landmark.setdefault("semantic_conflicts", [])
        normalized_observed = self.normalize_class_name(observed_class)

        conflict = None
        for item in conflicts:
            if self.normalize_class_name(item.get("new_class", "")) == normalized_observed:
                conflict = item
                break

        now_sec = self.now_sec()

        if conflict is None:
            conflict = {
                "old_class": landmark.get("class_name", "unknown"),
                "new_class": observed_class,
                "first_seen": now_sec,
                "last_seen": now_sec,
                "evidence_count": 1,
                "status": "conflict_candidate",
            }
            conflicts.append(conflict)
        else:
            conflict["last_seen"] = now_sec
            conflict["evidence_count"] = int(conflict.get("evidence_count", 0)) + 1

        landmark["last_observed_class"] = observed_class

        if conflict["evidence_count"] >= self.conflict_evidence_count:
            self.replace_landmark_class(landmark, obj, point_map, conflict)

    def replace_landmark_class(self, landmark, obj, point_map, conflict):
        """
        Replaces a stable semantic identity only after repeated conflict evidence.
        """
        old_class = landmark.get("class_name", "unknown")
        previous_classes = landmark.setdefault("previous_classes", [])
        previous_classes.append(old_class)

        landmark["class_id"] = int(obj.get("class_id", -1))
        landmark["class_name"] = obj.get("class_name", "unknown")
        landmark["pose_map"] = self.point_dict(point_map)
        landmark["confidence"] = self.observation_weight(obj)
        landmark["seen_count"] = 1
        landmark["status"] = "candidate"
        landmark["visibility"] = "visible"
        landmark["replacement_reason"] = "semantic_conflict"
        landmark["last_replaced"] = self.now_sec()

        conflict["status"] = "replaced"



    def update_landmark(self, landmark, obj, point_map):
        """
        Updates an existing landmark with a weighted running average.
        """
        now_sec = self.now_sec()
        obs_weight = self.observation_weight(obj)
        blend = max(0.10, min(0.65, obs_weight))

        old_pose = landmark["pose_map"]
        new_pose = self.point_dict(point_map)

        old_pose["x"] = self.blend(old_pose["x"], new_pose["x"], blend)
        old_pose["y"] = self.blend(old_pose["y"], new_pose["y"], blend)
        old_pose["z"] = self.blend(old_pose["z"], new_pose["z"], blend)

        landmark["confidence"] = self.blend(
            float(landmark.get("confidence", 0.0)),
            obs_weight,
            0.35,
        )
        landmark["uncertainty_radius"] = self.blend(
            float(landmark.get("uncertainty_radius", 0.30)),
            float(obj.get("uncertainty_radius", 0.30)),
            0.25,
        )
        landmark["seen_count"] = int(landmark.get("seen_count", 0)) + 1
        landmark["last_seen"] = now_sec
        landmark["source_sensor"] = obj.get("source_sensor", "unknown")
        landmark["last_detection_id"] = obj.get("detection_id", "")
        landmark["status"] = self.status_for_seen_count(landmark["seen_count"])
        landmark["visibility"] = "visible"

        history = landmark.setdefault("history", [])
        history.append(self.history_entry(obj, point_map, now_sec))
        if len(history) > self.max_history:
            del history[:-self.max_history]

    def publish_state(self):
        """
        Publishes the current semantic map as JSON and visualization markers.
        """
        out = String()
        out.data = json.dumps({
            "header": {
                "stamp": self.stamp_dict(),
                "frame_id": self.map_frame,
            },
            "landmarks": self.public_landmarks(),
        })
        self.landmarks_pub.publish(out)

        # Added 2026-05-20 16:08:30 EDT:
        # Publish separate Foxglove layers so stable map landmarks are not
        # confused with candidate/confirmed hypotheses or current sightings.
        self.markers_pub.publish(
            self.make_marker_array(
                statuses=("stable",),
                only_visible=None,
                namespace_prefix="semantic_landmarks",
            )
        )
        self.hypothesis_markers_pub.publish(
            self.make_marker_array(
                statuses=("candidate", "confirmed"),
                only_visible=None,
                namespace_prefix="semantic_hypotheses",
            )
        )
        self.current_markers_pub.publish(
            self.make_marker_array(
                statuses=("candidate", "confirmed", "stable"),
                only_visible=True,
                namespace_prefix="semantic_current",
            )
        )

    def load_semantic_map(self):
        """
        Loads semantic memory from JSON if it exists.
        """
        if not self.semantic_map_path.exists():
            return

        try:
            data = json.loads(self.semantic_map_path.read_text())
            self.landmarks = data.get("landmarks", [])

            if self.load_only_permanent:
                self.landmarks = [
                    landmark
                    for landmark in self.landmarks
                    if landmark.get("status") in self.permanent_statuses
                ]

            for landmark in self.landmarks:
                landmark.setdefault("visibility", "not_recently_seen")

            self.next_landmark_index = self.next_index_from_landmarks()
            self.get_logger().info(
                f"Loaded {len(self.landmarks)} semantic landmarks."
            )
        except Exception as exc:
            self.get_logger().warn(
                f"Could not load semantic map {self.semantic_map_path}: {exc}"
            )
            self.landmarks = []
            self.next_landmark_index = 1

    def save_semantic_map(self):
        """
        Saves semantic memory to JSON.
        """
        if not self.dirty:
            return

        try:
            self.semantic_map_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": 1,
                "frame_id": self.map_frame,
                "saved_at": self.now_sec(),
                "permanent_statuses": sorted(self.permanent_statuses),
                "live_landmark_count": len(self.landmarks),
                "permanent_landmark_count": len(self.permanent_landmarks()),
                "landmarks": self.permanent_landmarks(),
            }
            self.semantic_map_path.write_text(json.dumps(data, indent=2) + "\n")
            self.dirty = False
        except Exception as exc:
            self.get_logger().warn(
                f"Could not save semantic map {self.semantic_map_path}: {exc}",
                throttle_duration_sec=2.0,
            )

    def make_marker_array(
        self,
        statuses=None,
        only_visible=None,
        namespace_prefix="semantic_landmarks",
    ):
        """
        Builds markers for one semantic visualization layer.

        /perception/semantic_landmark_markers uses stable landmarks only.
        /perception/semantic_hypothesis_markers uses candidate/confirmed memory.
        /perception/semantic_current_markers uses objects visible right now.
        """
        marker_array = MarkerArray()

        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        marker_id = 1

        for landmark in self.public_landmarks():
            if statuses is not None and landmark.get("status") not in statuses:
                continue

            if only_visible is True and landmark.get("visibility") != "visible":
                continue

            if only_visible is False and landmark.get("visibility") == "visible":
                continue

            marker_array.markers.append(
                self.make_sphere_marker(marker_id, landmark, namespace_prefix)
            )
            marker_id += 1

            marker_array.markers.append(
                self.make_text_marker(marker_id, landmark, namespace_prefix)
            )
            marker_id += 1

        return marker_array

    def make_sphere_marker(self, marker_id, landmark, namespace_prefix):
        pose = landmark["pose_map"]
        radius = max(
            0.08,
            min(
                float(self.max_marker_radius),
                float(landmark.get("uncertainty_radius", 0.20)),
            ),
        )

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.map_frame
        marker.ns = namespace_prefix
        marker.id = int(marker_id)
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = float(pose["x"])
        marker.pose.position.y = float(pose["y"])
        marker.pose.position.z = float(pose["z"])
        marker.pose.orientation.w = 1.0

        marker.scale.x = radius
        marker.scale.y = radius
        marker.scale.z = radius

        r, g, b = self.color_for_status(landmark.get("status", "candidate"))
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b

        # Fade old-but-trusted landmarks instead of demoting stable memory.
        if landmark.get("visibility") == "visible":
            marker.color.a = 0.85
        else:
            marker.color.a = 0.35

        return marker

    def make_text_marker(self, marker_id, landmark, namespace_prefix):
        pose = landmark["pose_map"]

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.map_frame
        marker.ns = f"{namespace_prefix}_labels"
        marker.id = int(marker_id)
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = float(pose["x"])
        marker.pose.position.y = float(pose["y"])
        marker.pose.position.z = float(pose["z"]) + 0.20
        marker.pose.orientation.w = 1.0

        marker.scale.z = 0.12

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        marker.text = (
            f"{landmark['class_name']} "
            f"{landmark['status']} "
            f"{landmark.get('visibility', 'unknown')} "
            f"n={landmark['seen_count']} "
            f"c={landmark['confidence']:.2f}"
        )

        return marker

    def make_landmark_id(self, obj):
        class_name = str(obj.get("class_name", "object")).lower()
        clean_name = "".join(
            ch if ch.isalnum() else "_"
            for ch in class_name
        ).strip("_")
        if not clean_name:
            clean_name = "object"

        index = self.next_landmark_index
        self.next_landmark_index += 1
        return f"{clean_name}_{index:03d}"

    def next_index_from_landmarks(self):
        max_index = 0
        for landmark in self.landmarks:
            suffix = str(landmark.get("id", "")).rsplit("_", 1)[-1]
            if suffix.isdigit():
                max_index = max(max_index, int(suffix))
        return max_index + 1

    def public_landmarks(self):
        self.refresh_landmark_visibility()
        return [
            landmark
            for landmark in self.landmarks
            if landmark.get("status") != "rejected"
        ]

    def permanent_landmarks(self):
        """
        Returns landmarks trusted enough to save across runs.

        Candidates are useful live hypotheses, but they are not permanent map
        truth. By default only stable landmarks are written to semantic_map.json.
        """
        self.refresh_landmark_visibility()
        return [
            landmark
            for landmark in self.landmarks
            if landmark.get("status") in self.permanent_statuses
        ]

    def refresh_landmark_visibility(self):
        """
        Updates live visibility without changing long-term memory confidence.

        `status` means memory confidence: candidate / confirmed / stable.
        `visibility` means current observation state: visible / not_recently_seen.
        A stable sign should not become a candidate again just because it leaves
        the camera view.
        """
        now_sec = self.now_sec()
        for landmark in self.landmarks:
            last_seen = float(landmark.get("last_seen", 0.0))
            if now_sec - last_seen <= self.visible_timeout_sec:
                landmark["visibility"] = "visible"
            else:
                landmark["visibility"] = "not_recently_seen"
    
    def normalize_class_name(self, class_name):
        """
        Normalizes labels so stop_sign and stop sign compare as the same class.
        """
        return " ".join(
            str(class_name).lower().replace("_", " ").strip().split()
        )

    def is_dynamic_class(self, class_name):
        """
        Dynamic classes can pass through a landmark bubble without replacing it.
        """
        return self.normalize_class_name(class_name) in self.dynamic_class_names

    def status_for_seen_count(self, seen_count):
        if seen_count >= self.stable_seen_count:
            return "stable"
        if seen_count >= self.confirmed_seen_count:
            return "confirmed"
        return "candidate"

    def observation_weight(self, obj):
        confidence = float(obj.get("confidence", 0.0))
        quality = float(obj.get("quality_score", 0.0))
        valid_ratio = float(obj.get("valid_depth_ratio", 0.0))
        return max(0.05, min(1.0, confidence * (0.40 + 0.40 * quality + 0.20 * valid_ratio)))

    def history_entry(self, obj, point_map, now_sec):
        return {
            "stamp": now_sec,
            "detection_id": obj.get("detection_id", ""),
            "confidence": float(obj.get("confidence", 0.0)),
            "quality_score": float(obj.get("quality_score", 0.0)),
            "valid_depth_ratio": float(obj.get("valid_depth_ratio", 0.0)),
            "uncertainty_radius": float(obj.get("uncertainty_radius", 0.30)),
            "pose_map": self.point_dict(point_map),
        }

    def distance_to_landmark(self, landmark, point_map):
        pose = landmark.get("pose_map", {})
        dx = float(pose.get("x", 0.0)) - float(point_map.point.x)
        dy = float(pose.get("y", 0.0)) - float(point_map.point.y)
        dz = float(pose.get("z", 0.0)) - float(point_map.point.z)

        if self.association_use_xy_only:
            return math.sqrt(dx * dx + dy * dy)

        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def point_dict(self, point_map):
        return {
            "x": float(point_map.point.x),
            "y": float(point_map.point.y),
            "z": float(point_map.point.z),
        }

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

    def now_sec(self):
        now = self.get_clock().now().to_msg()
        return float(now.sec) + 1e-9 * float(now.nanosec)

    @staticmethod
    def blend(old_value, new_value, alpha):
        return (1.0 - alpha) * float(old_value) + alpha * float(new_value)

    @staticmethod
    def color_for_status(status):
        if status == "stable":
            return 0.10, 0.90, 0.25
        if status == "confirmed":
            return 0.15, 0.45, 1.0
        if status == "stale":
            return 0.55, 0.55, 0.55
        return 1.0, 0.62, 0.08


def main(args=None):
    rclpy.init(args=args)
    node = SemanticLandmarkMapper()

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
