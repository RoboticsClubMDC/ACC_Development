#!/usr/bin/env python3

import json
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String


DEFAULT_SETTINGS = {
    "H_min": 0,
    "H_max": 174,
    "S_min": 0,
    "S_max": 202,
    "V_min": 0,
    "V_max": 101,
    "ROI_top_percent": 67,
    "Morph_kernel": 1,
    "Close_kernel": 1,
    "Min_area_percent": 1,
    "SCANLINE_COUNT": 12,
    "MIN_SEGMENT_WIDTH_PX": 20,
    "MAX_CENTER_JUMP_PX": 95,
    "ALLOW_FIRST_ANCHOR_JUMP": True,
    "USE_EGO_CONNECTED_MASK": True,
    "EGO_SEED_X_RATIO": 0.50,
    "EGO_SEED_Y_RATIO": 0.95,
    "EGO_SEED_SEARCH_RADIUS_PX": 120,
    "EGO_BOTTOM_BAND_PERCENT": 18,
    "EGO_MIN_COMPONENT_AREA_PERCENT": 1.0,
    "CENTERLINE_SMOOTHING_ALPHA": 0.45,
    "LAST_CENTER_HOLD_FRAMES": 12,
    "SAFE_ERROR_DEADBAND_MM": 5.0,
    "SAFE_STEERING_GAIN": 0.01,
    "SAFE_MAX_STEERING_CORRECTION": 0.20,
    "YELLOW_H_MIN": 18,
    "YELLOW_H_MAX": 45,
    "YELLOW_S_MIN": 80,
    "YELLOW_S_MAX": 255,
    "YELLOW_V_MIN": 80,
    "YELLOW_V_MAX": 255,
    "YELLOW_BOUNDARY_DILATE_PX": 7,
    "CAMERA_CENTER_X_RATIO": 0.50,
    "CAMERA_CENTER_OFFSET_PX": 0,
}


@dataclass
class LaneResult:
    road_mask: np.ndarray
    raw_mask: np.ndarray
    yellow_mask: np.ndarray
    road_detected: bool
    road_confidence: float
    road_center_x: Optional[float]
    road_center_error_px: Optional[float]
    steering_correction: float
    valid_scanlines: int
    rejected_scanlines: int


def _cfg(settings, name):
    return settings.get(name, DEFAULT_SETTINGS[name])


def _odd_kernel_size(value):
    size = max(1, int(value))
    return size if size % 2 == 1 else size + 1


def _find_segments(row_mask, min_width_px):
    segments = []
    start_x = None
    for x, value in enumerate(row_mask):
        if value > 0 and start_x is None:
            start_x = x
        elif value == 0 and start_x is not None:
            _add_segment(segments, start_x, x - 1, min_width_px)
            start_x = None
    if start_x is not None:
        _add_segment(segments, start_x, len(row_mask) - 1, min_width_px)
    return segments


def _add_segment(segments, left_x, right_x, min_width_px):
    if right_x - left_x + 1 >= min_width_px:
        segments.append((int(left_x), int(right_x), float((left_x + right_x) / 2.0)))


def _connected_to_ego(mask, settings):
    if not bool(_cfg(settings, "USE_EGO_CONNECTED_MASK")):
        return mask

    height, width = mask.shape[:2]
    seed_x = int(np.clip(float(_cfg(settings, "EGO_SEED_X_RATIO")) * width, 0, width - 1))
    seed_y = int(np.clip(float(_cfg(settings, "EGO_SEED_Y_RATIO")) * height, 0, height - 1))
    radius = int(_cfg(settings, "EGO_SEED_SEARCH_RADIUS_PX"))

    if mask[seed_y, seed_x] > 0:
        anchor_x, anchor_y = seed_x, seed_y
    else:
        x0 = max(0, seed_x - radius)
        x1 = min(width - 1, seed_x + radius)
        y0 = max(0, seed_y - radius)
        y1 = min(height - 1, seed_y + radius)
        points = cv2.findNonZero(mask[y0:y1 + 1, x0:x1 + 1])
        if points is None:
            return _largest_bottom_component(mask, settings)
        pts = points.reshape(-1, 2)
        distances = (pts[:, 0] + x0 - seed_x) ** 2 + (pts[:, 1] + y0 - seed_y) ** 2
        best = pts[int(np.argmin(distances))]
        anchor_x = int(best[0] + x0)
        anchor_y = int(best[1] + y0)

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if labels_count <= 1:
        return np.zeros_like(mask)

    label = labels[anchor_y, anchor_x]
    min_area = mask.size * float(_cfg(settings, "EGO_MIN_COMPONENT_AREA_PERCENT")) / 100.0
    if label <= 0 or stats[label, cv2.CC_STAT_AREA] < min_area:
        return _largest_bottom_component(mask, settings)

    out = np.zeros_like(mask)
    out[labels == label] = 255
    return out


def _largest_bottom_component(mask, settings):
    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if labels_count <= 1:
        return np.zeros_like(mask)

    height, width = mask.shape[:2]
    image_center_x = width / 2.0
    bottom_y = height * (1.0 - float(_cfg(settings, "EGO_BOTTOM_BAND_PERCENT")) / 100.0)
    best_label = 0
    best_score = -1.0

    for label in range(1, labels_count):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        center_distance = abs(float(centroids[label][0]) - image_center_x) / max(1.0, image_center_x)
        bottom_bonus = 1.5 if y + h >= bottom_y and x <= image_center_x <= x + w else 0.0
        score = area / 1000.0 + bottom_bonus - center_distance
        if score > best_score:
            best_label = label
            best_score = score

    out = np.zeros_like(mask)
    if best_label > 0:
        out[labels == best_label] = 255
    return out


def _build_yellow_mask(hsv, settings):
    lower = np.array([
        int(_cfg(settings, "YELLOW_H_MIN")),
        int(_cfg(settings, "YELLOW_S_MIN")),
        int(_cfg(settings, "YELLOW_V_MIN")),
    ], dtype=np.uint8)
    upper = np.array([
        int(_cfg(settings, "YELLOW_H_MAX")),
        int(_cfg(settings, "YELLOW_S_MAX")),
        int(_cfg(settings, "YELLOW_V_MAX")),
    ], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    dilate_px = int(_cfg(settings, "YELLOW_BOUNDARY_DILATE_PX"))
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_px, dilate_px))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _estimate_scanline_centers(mask, roi_top, settings, last_center_x, frames_since_valid):
    height, width = mask.shape[:2]
    count = max(2, int(_cfg(settings, "SCANLINE_COUNT")))
    min_width = int(_cfg(settings, "MIN_SEGMENT_WIDTH_PX"))
    max_jump = float(_cfg(settings, "MAX_CENTER_JUMP_PX"))
    hold_frames = int(_cfg(settings, "LAST_CENTER_HOLD_FRAMES"))
    allow_first_jump = bool(_cfg(settings, "ALLOW_FIRST_ANCHOR_JUMP"))

    y_values = np.linspace(height - 1, roi_top, count, dtype=int)
    seed_center_x = last_center_x if last_center_x is not None and frames_since_valid <= hold_frames else width / 2.0
    previous_center_x = float(seed_center_x)
    points = []
    rejected = 0

    for index, y in enumerate(y_values):
        segments = _find_segments(mask[y, :], min_width)
        if not segments:
            rejected += 1
            continue

        left_x, right_x, center_x = min(segments, key=lambda segment: abs(segment[2] - previous_center_x))
        center_jump = center_x - previous_center_x
        if index == 0 and allow_first_jump:
            points.append((center_x, left_x, right_x, int(y)))
            previous_center_x = center_x
            continue
        if abs(center_jump) > max_jump:
            rejected += 1
            continue
        points.append((center_x, left_x, right_x, int(y)))
        previous_center_x = center_x

    if not points:
        return [], rejected

    alpha = float(np.clip(_cfg(settings, "CENTERLINE_SMOOTHING_ALPHA"), 0.0, 1.0))
    smoothed_center_x = points[0][0]
    smoothed = []
    for center_x, left_x, right_x, y in points:
        smoothed_center_x = (1.0 - alpha) * smoothed_center_x + alpha * center_x
        smoothed.append((float(smoothed_center_x), left_x, right_x, y))
    return smoothed, rejected


def detect_lane(frame_bgr, settings, last_center_x=None, frames_since_valid=999):
    height, width = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([
        int(_cfg(settings, "H_min")),
        int(_cfg(settings, "S_min")),
        int(_cfg(settings, "V_min")),
    ], dtype=np.uint8)
    upper = np.array([
        int(_cfg(settings, "H_max")),
        int(_cfg(settings, "S_max")),
        int(_cfg(settings, "V_max")),
    ], dtype=np.uint8)

    roi_top = int(np.clip(float(_cfg(settings, "ROI_top_percent")) / 100.0, 0.0, 0.95) * height)
    yellow_mask = _build_yellow_mask(hsv, settings)
    raw_mask = cv2.inRange(hsv, lower, upper)
    raw_mask[:roi_top, :] = 0
    yellow_mask[:roi_top, :] = 0
    raw_mask[yellow_mask > 0] = 0

    morph_kernel = _odd_kernel_size(_cfg(settings, "Morph_kernel"))
    if morph_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)

    close_kernel = _odd_kernel_size(_cfg(settings, "Close_kernel"))
    if close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)

    road_mask = _connected_to_ego(raw_mask, settings)
    roi_area = max(1, (height - roi_top) * width)
    min_area = roi_area * float(_cfg(settings, "Min_area_percent")) / 100.0
    area = int(cv2.countNonZero(road_mask))
    scan_points, rejected_scanlines = _estimate_scanline_centers(
        road_mask, roi_top, settings, last_center_x, frames_since_valid
    )
    road_detected = area >= min_area and len(scan_points) >= 2
    confidence = float(np.clip((area / roi_area) / 0.20, 0.0, 1.0))

    center_x = None
    error_px = None
    steering = 0.0
    if road_detected:
        centers = np.array([point[0] for point in scan_points], dtype=np.float32)
        weights = np.linspace(2.0, 1.0, len(scan_points), dtype=np.float32)
        center_x = float(np.average(centers, weights=weights))
        camera_center = (
            float(_cfg(settings, "CAMERA_CENTER_X_RATIO")) * width
            + float(_cfg(settings, "CAMERA_CENTER_OFFSET_PX"))
        )
        error_px = center_x - camera_center
        normalized_error_mm = error_px / max(1.0, width / 2.0) * 127.0
        deadband = float(_cfg(settings, "SAFE_ERROR_DEADBAND_MM"))
        if abs(normalized_error_mm) > deadband:
            steering = float(_cfg(settings, "SAFE_STEERING_GAIN")) * normalized_error_mm
            max_steering = float(_cfg(settings, "SAFE_MAX_STEERING_CORRECTION"))
            steering = float(np.clip(steering, -max_steering, max_steering))

    return LaneResult(
        road_mask=road_mask,
        raw_mask=raw_mask,
        yellow_mask=yellow_mask,
        road_detected=road_detected,
        road_confidence=confidence if road_detected else 0.0,
        road_center_x=center_x,
        road_center_error_px=error_px,
        steering_correction=steering,
        valid_scanlines=len(scan_points),
        rejected_scanlines=rejected_scanlines,
    )


class CsiFrontLaneDetection(Node):
    def __init__(self):
        super().__init__("lane_detection")

        self.declare_parameter("image_topic", "/camera/csi_image")
        self.declare_parameter("config_file", "csi_front_config.json")
        self.declare_parameter("publish_raw_mask", False)

        self.bridge = CvBridge()
        self.settings = self._load_settings()
        self.last_center_x = None
        self.frames_since_valid = int(_cfg(self.settings, "LAST_CENTER_HOLD_FRAMES")) + 1

        image_topic = self.get_parameter("image_topic").value
        self.pub_road_mask = self.create_publisher(Image, "/lane_detection/road_mask", qos_profile_sensor_data)
        self.pub_yellow_mask = self.create_publisher(Image, "/lane_detection/yellow_boundary_mask", qos_profile_sensor_data)
        self.pub_status = self.create_publisher(String, "/lane_detection/status", 10)
        self.pub_steering = self.create_publisher(Float32, "/lane_detection/steering_correction", 10)
        self.pub_raw_mask = None
        if bool(self.get_parameter("publish_raw_mask").value):
            self.pub_raw_mask = self.create_publisher(Image, "/lane_detection/raw_mask", qos_profile_sensor_data)

        self.create_subscription(Image, image_topic, self._image_cb, qos_profile_sensor_data)
        self.get_logger().info(f"CSI front lane detection subscribed to {image_topic}")

    def _load_settings(self):
        config_file = self.get_parameter("config_file").value
        candidates = []
        if os.path.isabs(config_file):
            candidates.append(config_file)
        else:
            candidates.append(os.path.join(get_package_share_directory("qcar2_autonomy"), "config", config_file))
            candidates.append(os.path.join(os.getcwd(), config_file))

        settings = DEFAULT_SETTINGS.copy()
        for path in candidates:
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            settings.update(payload.get("settings", payload))
            self.get_logger().info(f"Loaded lane config: {path}")
            return settings

        self.get_logger().warn(f"Lane config {config_file!r} not found; using built-in defaults.")
        return settings

    def _image_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"cv_bridge failed for CSI image: {exc}")
            return

        result = detect_lane(frame, self.settings, self.last_center_x, self.frames_since_valid)
        if result.road_detected and result.road_center_x is not None:
            self.last_center_x = result.road_center_x
            self.frames_since_valid = 0
        else:
            self.frames_since_valid += 1

        road_msg = self.bridge.cv2_to_imgmsg(result.road_mask, encoding="mono8")
        road_msg.header = msg.header
        self.pub_road_mask.publish(road_msg)

        yellow_msg = self.bridge.cv2_to_imgmsg(result.yellow_mask, encoding="mono8")
        yellow_msg.header = msg.header
        self.pub_yellow_mask.publish(yellow_msg)

        if self.pub_raw_mask is not None:
            raw_msg = self.bridge.cv2_to_imgmsg(result.raw_mask, encoding="mono8")
            raw_msg.header = msg.header
            self.pub_raw_mask.publish(raw_msg)

        steering_msg = Float32()
        steering_msg.data = float(result.steering_correction)
        self.pub_steering.publish(steering_msg)

        status_msg = String()
        status_msg.data = json.dumps({
            "road_detected": result.road_detected,
            "road_confidence": round(result.road_confidence, 3),
            "road_center_x": result.road_center_x,
            "road_center_error_px": result.road_center_error_px,
            "steering_correction": result.steering_correction,
            "valid_scanlines": result.valid_scanlines,
            "rejected_scanlines": result.rejected_scanlines,
        })
        self.pub_status.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CsiFrontLaneDetection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
