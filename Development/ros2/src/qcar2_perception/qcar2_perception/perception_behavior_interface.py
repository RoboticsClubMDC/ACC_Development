#!/usr/bin/env python3

import json
import math
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from std_msgs.msg import Bool, String


class PerceptionBehaviorInterface(Node):
    """
    Converts semantic landmarks into lightweight behavior advisories.

    This node is intentionally advisory only. It does not publish drive commands
    or change path_follower mode. Autonomy can subscribe to these outputs when we
    are ready to make signs/lights affect driving.
    """

    def __init__(self):
        super().__init__("perception_behavior_interface")

        self.declare_parameter("landmarks_topic", "/perception/semantic_landmarks")
        self.declare_parameter("pose_topic", "/qcar2_pose_fused")
        self.declare_parameter("objects_topic", "/perception/objects_3d")
        self.declare_parameter("events_topic", "/perception/behavior_events")
        self.declare_parameter("stop_required_topic", "/perception/stop_required")
        self.declare_parameter("motion_enable_topic", "/motion_enable")
        self.declare_parameter("track_classes", ["stop sign", "traffic light"])
        self.declare_parameter("active_statuses", ["confirmed", "stable"])
        self.declare_parameter("max_event_distance_m", 1.50)
        self.declare_parameter("forward_fov_deg", 100.0)
        self.declare_parameter("publish_rate_hz", 5.0)
        self.declare_parameter("enable_landmark_distance_stop", True)
        self.declare_parameter("stop_trigger_distance_m", 0.20)
        self.declare_parameter("stop_hold_seconds", 3.0)
        self.declare_parameter("stop_cooldown_seconds", 10.0)
        self.declare_parameter("stop_min_confidence", 0.30)

        landmarks_topic = str(self.get_parameter("landmarks_topic").value)
        pose_topic = str(self.get_parameter("pose_topic").value)
        objects_topic = str(self.get_parameter("objects_topic").value)
        events_topic = str(self.get_parameter("events_topic").value)
        stop_required_topic = str(
            self.get_parameter("stop_required_topic").value
        )
        motion_enable_topic = str(
            self.get_parameter("motion_enable_topic").value
        )

        self.track_classes = {
            self.normalize_class_name(item)
            for item in self.get_parameter("track_classes").value
        }
        self.active_statuses = {
            str(item)
            for item in self.get_parameter("active_statuses").value
        }
        self.max_event_distance_m = float(
            self.get_parameter("max_event_distance_m").value
        )
        self.forward_fov_rad = math.radians(
            float(self.get_parameter("forward_fov_deg").value)
        )
        self.enable_landmark_distance_stop = bool(
            self.get_parameter("enable_landmark_distance_stop").value
        )
        self.stop_trigger_distance_m = float(
            self.get_parameter("stop_trigger_distance_m").value
        )
        self.stop_hold_seconds = float(
            self.get_parameter("stop_hold_seconds").value
        )
        self.stop_cooldown_seconds = float(
            self.get_parameter("stop_cooldown_seconds").value
        )
        self.stop_min_confidence = float(
            self.get_parameter("stop_min_confidence").value
        )
        self.add_on_set_parameters_callback(self.parameter_update_callback)

        self.landmarks = []
        self.robot_pose = None
        self.last_event_key = None
        self.stop_hold_until = 0.0
        self.stop_cooldown_until = 0.0
        self.last_stop_landmark_id = ""

        self.create_subscription(String, landmarks_topic, self.landmarks_cb, 10)
        self.create_subscription(
            PoseWithCovarianceStamped,
            pose_topic,
            self.pose_cb,
            10,
        )

        self.events_pub = self.create_publisher(String, events_topic, 10)
        self.stop_required_pub = self.create_publisher(
            Bool,
            stop_required_topic,
            10,
        )
        self.motion_enable_pub = self.create_publisher(
            Bool,
            motion_enable_topic,
            10,
        )

        publish_period = 1.0 / max(
            0.5,
            float(self.get_parameter("publish_rate_hz").value),
        )
        self.create_timer(publish_period, self.publish_behavior_event)

        self.get_logger().info(
            "perception_behavior_interface ready. "
            f"landmarks={landmarks_topic}, objects={objects_topic}, pose={pose_topic}, "
            f"events={events_topic}, stop_required={stop_required_topic}, "
            f"motion_enable={motion_enable_topic}, "
            f"stop_trigger_distance={self.stop_trigger_distance_m:.2f}m"
        )

    def landmarks_cb(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"Could not parse semantic landmarks: {exc}")
            return

        self.landmarks = data.get("landmarks", [])

    def pose_cb(self, msg):
        self.robot_pose = msg

    def parameter_update_callback(self, params):
        for param in params:
            if param.name in ("enable_landmark_distance_stop", "enable_direct_object_stop"):
                self.enable_landmark_distance_stop = bool(param.value)
            elif param.name == "stop_trigger_distance_m":
                self.stop_trigger_distance_m = max(0.0, float(param.value))
            elif param.name == "stop_hold_seconds":
                self.stop_hold_seconds = max(0.0, float(param.value))
            elif param.name == "stop_cooldown_seconds":
                self.stop_cooldown_seconds = max(0.0, float(param.value))
            elif param.name == "stop_min_confidence":
                self.stop_min_confidence = max(0.0, min(1.0, float(param.value)))

        return SetParametersResult(successful=True)

    def publish_behavior_event(self):
        self.update_stop_hold_from_landmarks()
        event = self.nearest_relevant_landmark()
        landmark_stop_active = time.monotonic() < self.stop_hold_until
        stop_required = landmark_stop_active

        if event is not None:
            self.publish_event(event)

        stop_msg = Bool()
        stop_msg.data = bool(stop_required)
        self.stop_required_pub.publish(stop_msg)

        motion_msg = Bool()
        motion_msg.data = not bool(landmark_stop_active)
        self.motion_enable_pub.publish(motion_msg)

        if self.last_stop_landmark_id and time.monotonic() >= self.stop_hold_until:
            self.get_logger().info(
                f"Stop-sign landmark hold complete; motion re-enabled. "
                f"cooldown={max(0.0, self.stop_cooldown_until - time.monotonic()):.1f}s"
            )
            self.last_stop_landmark_id = ""

    def update_stop_hold_from_landmarks(self):
        if not self.enable_landmark_distance_stop:
            return

        now = time.monotonic()
        if now < self.stop_hold_until or now < self.stop_cooldown_until:
            return

        for landmark in self.landmarks:
            class_name = self.normalize_class_name(
                landmark.get("class_name", "unknown")
            )
            if class_name != "stop sign":
                continue

            if landmark.get("status") not in self.active_statuses:
                continue

            if landmark.get("visibility") != "visible":
                continue

            confidence = float(landmark.get("confidence", 0.0))
            if confidence < self.stop_min_confidence:
                continue

            distance = self.landmark_observation_distance_m(landmark)
            if distance is None or distance > self.stop_trigger_distance_m:
                continue

            self.stop_hold_until = now + self.stop_hold_seconds
            self.stop_cooldown_until = self.stop_hold_until + self.stop_cooldown_seconds
            self.last_stop_landmark_id = str(landmark.get("id", ""))
            self.get_logger().warn(
                f"STOP SIGN LANDMARK HOLD: id={self.last_stop_landmark_id} "
                f"distance={distance:.3f}m conf={confidence:.2f}; "
                f"publishing stop for {self.stop_hold_seconds:.1f}s"
            )
            return

    def nearest_relevant_landmark(self):
        if self.robot_pose is None:
            return None

        rx = float(self.robot_pose.pose.pose.position.x)
        ry = float(self.robot_pose.pose.pose.position.y)
        yaw = self.yaw_from_pose(self.robot_pose)

        best = None
        best_distance = float("inf")

        for landmark in self.landmarks:
            class_name = self.normalize_class_name(
                landmark.get("class_name", "unknown")
            )
            if class_name not in self.track_classes:
                continue

            if landmark.get("status") not in self.active_statuses:
                continue

            pose = landmark.get("pose_map", {})
            try:
                lx = float(pose["x"])
                ly = float(pose["y"])
            except Exception:
                continue

            dx = lx - rx
            dy = ly - ry
            distance = math.hypot(dx, dy)
            if distance > self.max_event_distance_m:
                continue

            bearing = self.wrap_angle(math.atan2(dy, dx) - yaw)
            if abs(bearing) > 0.5 * self.forward_fov_rad:
                continue

            if distance < best_distance:
                best_distance = distance
                best = {
                    "event": self.event_name_for_class(class_name),
                    "landmark_id": landmark.get("id", ""),
                    "class_name": landmark.get("class_name", "unknown"),
                    "status": landmark.get("status", "unknown"),
                    "visibility": landmark.get("visibility", "unknown"),
                    "distance_m": distance,
                    "bearing_rad": bearing,
                    "confidence": float(landmark.get("confidence", 0.0)),
                    "pose_map": pose,
                }

        return best

    def publish_event(self, event):
        event_key = (
            event["event"],
            event["landmark_id"],
            round(event["distance_m"], 1),
        )
        if event_key != self.last_event_key:
            self.get_logger().info(
                f"{event['event']}: {event['class_name']} "
                f"{event['landmark_id']} distance={event['distance_m']:.2f}m "
                f"bearing={math.degrees(event['bearing_rad']):.1f}deg"
            )
            self.last_event_key = event_key

        msg = String()
        msg.data = json.dumps({
            "header": {
                "stamp": self.stamp_dict(),
                "frame_id": "map",
            },
            **event,
        })
        self.events_pub.publish(msg)

    @staticmethod
    def event_name_for_class(class_name):
        if class_name == "stop sign":
            return "STOP_SIGN_AHEAD"
        if class_name == "traffic light":
            return "TRAFFIC_LIGHT_AHEAD"
        return "SEMANTIC_LANDMARK_AHEAD"

    @staticmethod
    def normalize_class_name(class_name):
        return " ".join(
            str(class_name).lower().replace("_", " ").strip().split()
        )

    @staticmethod
    def landmark_observation_distance_m(landmark):
        for key in ("last_observation_distance_m", "last_observation_depth_m"):
            try:
                value = float(landmark.get(key, 0.0))
                if math.isfinite(value) and value > 0.0:
                    return value
            except Exception:
                pass
        return None

    @staticmethod
    def yaw_from_pose(msg):
        q = msg.pose.pose.orientation
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    @staticmethod
    def wrap_angle(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def stamp_dict(self):
        stamp = self.get_clock().now().to_msg()
        return {
            "sec": int(stamp.sec),
            "nanosec": int(stamp.nanosec),
        }


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionBehaviorInterface()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
