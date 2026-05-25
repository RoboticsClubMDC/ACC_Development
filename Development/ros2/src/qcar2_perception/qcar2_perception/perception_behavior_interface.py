#!/usr/bin/env python3

import json
import math

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
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
        self.declare_parameter("events_topic", "/perception/behavior_events")
        self.declare_parameter("stop_required_topic", "/perception/stop_required")
        self.declare_parameter("track_classes", ["stop sign", "traffic light"])
        self.declare_parameter("active_statuses", ["confirmed", "stable"])
        self.declare_parameter("max_event_distance_m", 1.50)
        self.declare_parameter("forward_fov_deg", 100.0)
        self.declare_parameter("publish_rate_hz", 5.0)

        landmarks_topic = str(self.get_parameter("landmarks_topic").value)
        pose_topic = str(self.get_parameter("pose_topic").value)
        events_topic = str(self.get_parameter("events_topic").value)
        stop_required_topic = str(
            self.get_parameter("stop_required_topic").value
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

        self.landmarks = []
        self.robot_pose = None
        self.last_event_key = None

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

        publish_period = 1.0 / max(
            0.5,
            float(self.get_parameter("publish_rate_hz").value),
        )
        self.create_timer(publish_period, self.publish_behavior_event)

        self.get_logger().info(
            "perception_behavior_interface ready. "
            f"landmarks={landmarks_topic}, pose={pose_topic}, "
            f"events={events_topic}, stop_required={stop_required_topic}"
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

    def publish_behavior_event(self):
        event = self.nearest_relevant_landmark()
        stop_required = False

        if event is not None:
            stop_required = (
                self.normalize_class_name(event["class_name"]) == "stop sign"
            )
            self.publish_event(event)

        stop_msg = Bool()
        stop_msg.data = bool(stop_required)
        self.stop_required_pub.publish(stop_msg)

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
