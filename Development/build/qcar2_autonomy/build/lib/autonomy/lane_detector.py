#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge

import cv2
import numpy as np

from qcar2_interfaces.msg import MotorCommands


class LaneDetector(Node):
    def __init__(self):
        super().__init__('lane_detector')

        # ---------------- Parameters ----------------
        # NOTE: using your exact topic string
        self.declare_parameter('image_topic', '/camera/color_image/compressed')

        self.declare_parameter('bev_width', 400)
        self.declare_parameter('bev_height', 600)
        self.declare_parameter('lookahead_distance', 1.0)  # meters
        self.declare_parameter('meters_per_pixel', 0.02)

        # Hough tuning
        self.declare_parameter('canny_low', 50)
        self.declare_parameter('canny_high', 150)
        self.declare_parameter('hough_threshold', 50)
        self.declare_parameter('min_line_length', 50)
        self.declare_parameter('max_line_gap', 100)
        self.declare_parameter('slope_min', 0.5)

        # Binary processing parameters (NEW - inspired by lab guide SECTION C)
        self.declare_parameter('binary_threshold', 127)
        self.declare_parameter('dilate_iterations', 2)
        self.declare_parameter('morph_kernel_size', 5)

        # Just move the car (test)
        self.declare_parameter('enable_drive', False)
        self.declare_parameter('drive_throttle', 0.12)
        self.declare_parameter('drive_steering', 0.0)
        self.declare_parameter('cmd_topic', '/qcar2_motor_speed_cmd')

        # ---------------- Load params ----------------
        self.image_topic = str(self.get_parameter('image_topic').value)

        self.bev_width = int(self.get_parameter('bev_width').value)
        self.bev_height = int(self.get_parameter('bev_height').value)
        self.lookahead_distance = float(self.get_parameter('lookahead_distance').value)
        self.meters_per_pixel = float(self.get_parameter('meters_per_pixel').value)

        self.canny_low = int(self.get_parameter('canny_low').value)
        self.canny_high = int(self.get_parameter('canny_high').value)
        self.hough_threshold = int(self.get_parameter('hough_threshold').value)
        self.min_line_length = int(self.get_parameter('min_line_length').value)
        self.max_line_gap = int(self.get_parameter('max_line_gap').value)
        self.slope_min = float(self.get_parameter('slope_min').value)

        self.binary_threshold = int(self.get_parameter('binary_threshold').value)
        self.dilate_iterations = int(self.get_parameter('dilate_iterations').value)
        self.morph_kernel_size = int(self.get_parameter('morph_kernel_size').value)

        self.enable_drive = bool(self.get_parameter('enable_drive').value)
        self.drive_throttle = float(self.get_parameter('drive_throttle').value)
        self.drive_steering = float(self.get_parameter('drive_steering').value)
        self.cmd_topic = str(self.get_parameter('cmd_topic').value)

        self.bridge = CvBridge()

        # Perspective transform (computed lazily once we know image size)
        self.M = None
        self.Minv = None
        self._last_img_shape = None

        # ---------------- Subscribers ----------------
        self.image_sub = self.create_subscription(
            CompressedImage,
            self.image_topic,
            self.image_callback,
            10
        )

        # ---------------- Publishers ----------------
        self.cte_pub = self.create_publisher(Float32, '/lane_detector/cte', 10)
        self.heading_error_pub = self.create_publisher(Float32, '/lane_detector/heading_error', 10)

        self.bev_pub = self.create_publisher(Image, '/lane_detector/BEV', 10)
        self.lane_marking_pub = self.create_publisher(Image, '/lane_detector/lane_marking', 10)
        self.lane_marking_bev_pub = self.create_publisher(Image, '/lane_detector/lane_marking_BEV', 10)

        self.cmd_pub = self.create_publisher(MotorCommands, self.cmd_topic, 10)

        self.get_logger().info(f'Lane Detector initialized. Subscribing to {self.image_topic}')

    def _ensure_perspective_transform(self, img_h: int, img_w: int):
        """Compute perspective transform for BEV based on the incoming image size."""
        if self.M is not None and self._last_img_shape == (img_h, img_w):
            return

        self._last_img_shape = (img_h, img_w)

        src_points = np.float32([
            [img_w * 0.45, img_h * 0.65],
            [img_w * 0.55, img_h * 0.65],
            [img_w * 0.90, img_h * 1.00],
            [img_w * 0.10, img_h * 1.00]
        ])

        dst_points = np.float32([
            [self.bev_width * 0.30, 0],
            [self.bev_width * 0.70, 0],
            [self.bev_width * 0.70, self.bev_height - 1],
            [self.bev_width * 0.30, self.bev_height - 1]
        ])

        self.M = cv2.getPerspectiveTransform(src_points, dst_points)
        self.Minv = cv2.getPerspectiveTransform(dst_points, src_points)

        self.get_logger().info(f'Updated BEV homography for image size {img_w}x{img_h}')

    def _decode_compressed(self, msg: CompressedImage):
        """CompressedImage -> BGR OpenCV image."""
        if not msg.data:
            return None
        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # BGR
        return img

    def image_callback(self, msg: CompressedImage):
        try:
            cv_image = self._decode_compressed(msg)
            if cv_image is None:
                self.get_logger().warn("Compressed image decode returned None/empty.")
                return

            h, w = cv_image.shape[:2]
            self._ensure_perspective_transform(h, w)

            # Detect lanes + BINARY marking image (IMPROVED)
            left_lane, right_lane, marking_binary = self.detect_lanes(cv_image)

            # Publish BINARY lane marking (this is the key improvement!)
            self.lane_marking_pub.publish(self.bridge.cv2_to_imgmsg(marking_binary, 'mono8'))

            # Transform binary lane marking to BEV
            bev_lane_marking = cv2.warpPerspective(
                marking_binary, 
                self.M, 
                (self.bev_width, self.bev_height)
            )

            # Process BEV lane marking to ensure binary and clean (like lab SECTION C)
            bev_lane_marking_processed = self.preprocess_lane_marking_bev(bev_lane_marking)

            # Publish BINARY BEV lane marking
            self.lane_marking_bev_pub.publish(
                self.bridge.cv2_to_imgmsg(bev_lane_marking_processed, 'mono8')
            )

            # Transform to BEV (visualization)
            bev_image = cv2.warpPerspective(cv_image, self.M, (self.bev_width, self.bev_height))

            # Calculate errors
            cte, heading_error = self.calculate_errors(left_lane, right_lane)

            # Publish errors
            cte_msg = Float32()
            cte_msg.data = float(cte)
            self.cte_pub.publish(cte_msg)

            heading_msg = Float32()
            heading_msg.data = float(heading_error)
            self.heading_error_pub.publish(heading_msg)

            # Draw BEV overlay and publish
            display_img = self.draw_bev(bev_image, left_lane, right_lane, cte, heading_error)
            self.bev_pub.publish(self.bridge.cv2_to_imgmsg(display_img, 'bgr8'))

            self.get_logger().info(f'CTE: {cte:+.4f} m | Heading: {np.degrees(heading_error):+.2f} deg')

            # Optional: just move forward for testing
            if self.enable_drive:
                cmd = MotorCommands()
                cmd.motor_names = ["steering_angle", "motor_throttle"]
                cmd.values = [float(self.drive_steering), float(self.drive_throttle)]
                self.cmd_pub.publish(cmd)

        except Exception as e:
            self.get_logger().error(f'Error in callback: {e}')

    def detect_lanes(self, image):
        """
        Detect lane lines using classical CV and return BINARY image.
        IMPROVED: Now follows lab guide approach with binary output.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)

        h, w = edges.shape
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (w * 0.1, h),
            (w * 0.45, h * 0.6),
            (w * 0.55, h * 0.6),
            (w * 0.9, h)
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(
            masked_edges, 2, np.pi / 180,
            self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        # ==== KEY IMPROVEMENT: Create BINARY lane marking ====
        # Initialize as black (0)
        lane_marking = np.zeros((h, w), dtype=np.uint8)
        
        left_lines, right_lines = [], []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = x2 - x1
                if dx == 0:
                    continue
                slope = (y2 - y1) / dx

                # Draw line on binary canvas as WHITE (255)
                cv2.line(lane_marking, (x1, y1), (x2, y2), 255, 2)

                if slope < -self.slope_min and x1 < w / 2:
                    left_lines.append(line[0])
                elif slope > self.slope_min and x1 > w / 2:
                    right_lines.append(line[0])

        # ==== CRITICAL: Ensure PURE BINARY (no gray pixels) ====
        # Apply threshold to eliminate any gray values
        _, lane_marking_binary = cv2.threshold(
            lane_marking, 
            self.binary_threshold, 
            255, 
            cv2.THRESH_BINARY
        )

        # ==== Apply morphological processing (like lab SECTION C) ====
        # This merges the two edges of the same lane marking
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (self.morph_kernel_size, self.morph_kernel_size)
        )
        
        # Dilate to merge nearby edges
        dilated = cv2.dilate(lane_marking_binary, kernel, iterations=self.dilate_iterations)
        
        # Erode slightly to restore approximately original thickness
        lane_marking_processed = cv2.erode(dilated, kernel, iterations=1)

        # ==== Fit lanes in IMAGE space (before BEV transform) ====
        left_lane = self.fit_lane(left_lines, h)
        right_lane = self.fit_lane(right_lines, h)

        if left_lane is None or right_lane is None:
            self.get_logger().warn(
                f"Lane missing: left={'OK' if left_lane is not None else 'None'} "
                f"right={'OK' if right_lane is not None else 'None'}"
            )

        # Return the BINARY processed marking instead of debug image
        return left_lane, right_lane, lane_marking_processed

    def fit_lane(self, lines, h):
        """Fit polynomial to lane lines in IMAGE space."""
        if not lines:
            return None

        x_points, y_points = [], []
        for x1, y1, x2, y2 in lines:
            x_points.extend([x1, x2])
            y_points.extend([y1, y2])

        if len(x_points) < 6:
            return None

        coeffs = np.polyfit(y_points, x_points, 2)
        y_vals = np.linspace(h * 0.6, h - 1, 50)
        x_vals = np.polyval(coeffs, y_vals)

        return np.column_stack((x_vals, y_vals)).astype(np.float32)

    def calculate_errors(self, left_lane, right_lane):
        """Calculate CTE and heading error in BEV space."""
        if self.M is None:
            return 0.0, 0.0

        vehicle_x = self.bev_width / 2.0
        vehicle_y = self.bev_height - 1.0

        cte = 0.0
        heading_error = 0.0

        if left_lane is None or right_lane is None:
            return cte, heading_error

        left_bev = cv2.perspectiveTransform(left_lane.reshape(-1, 1, 2), self.M).reshape(-1, 2)
        right_bev = cv2.perspectiveTransform(right_lane.reshape(-1, 1, 2), self.M).reshape(-1, 2)

        left_closest = self.find_closest_point(left_bev, vehicle_x, vehicle_y)
        right_closest = self.find_closest_point(right_bev, vehicle_x, vehicle_y)

        center_x = (left_closest[0] + right_closest[0]) / 2.0
        cte = (vehicle_x - center_x) * self.meters_per_pixel

        lookahead_pixels = self.lookahead_distance / self.meters_per_pixel
        lookahead_y = vehicle_y - lookahead_pixels

        if lookahead_y > 0:
            left_ahead = self.interpolate_x_at_y(left_bev, lookahead_y)
            right_ahead = self.interpolate_x_at_y(right_bev, lookahead_y)
            if left_ahead is not None and right_ahead is not None:
                center_ahead_x = (left_ahead + right_ahead) / 2.0
                dx = center_ahead_x - vehicle_x
                dy = vehicle_y - lookahead_y
                heading_error = float(np.arctan2(dx, dy))

        return float(cte), float(heading_error)

    def preprocess_lane_marking_bev(self, lane_marking_bev):
        """
        Clean up the Lane Marking BEV - similar to SECTION C in lab guide.
        Convert to binary (no grey pixels) and merge edges of same lane marking.
        """
        # Ensure binary (no gray pixels) - important after perspective transform
        _, binary = cv2.threshold(lane_marking_bev, 127, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to merge edges and clean up
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (self.morph_kernel_size, self.morph_kernel_size)
        )
        
        # Dilate to merge nearby edges (edges of same lane marking)
        dilated = cv2.dilate(binary, kernel, iterations=self.dilate_iterations)
        
        # Erode slightly to restore approximately original thickness
        processed = cv2.erode(dilated, kernel, iterations=1)
        
        return processed

    def find_closest_point(self, points, x, y):
        distances = np.sqrt((points[:, 0] - x) ** 2 + (points[:, 1] - y) ** 2)
        return points[np.argmin(distances)]

    def interpolate_x_at_y(self, points, y_target):
        """Interpolate x at y_target (sorted by y)."""
        if points is None or len(points) < 2:
            return None

        idx = np.argsort(points[:, 1])
        y_sorted = points[idx, 1]
        x_sorted = points[idx, 0]

        if y_target < y_sorted[0] or y_target > y_sorted[-1]:
            return None

        return float(np.interp(y_target, y_sorted, x_sorted))

    def draw_bev(self, bev_image, left_lane, right_lane, cte, heading_error):
        display = bev_image.copy()

        if self.M is not None:
            if left_lane is not None:
                left_bev = cv2.perspectiveTransform(left_lane.reshape(-1, 1, 2), self.M).reshape(-1, 2).astype(np.int32)
                cv2.polylines(display, [left_bev], False, (0, 255, 0), 3)

            if right_lane is not None:
                right_bev = cv2.perspectiveTransform(right_lane.reshape(-1, 1, 2), self.M).reshape(-1, 2).astype(np.int32)
                cv2.polylines(display, [right_bev], False, (0, 255, 0), 3)

        vehicle_x = int(self.bev_width / 2)
        vehicle_y = int(self.bev_height - 20)
        cv2.circle(display, (vehicle_x, vehicle_y), 8, (0, 0, 255), -1)
        cv2.line(display, (vehicle_x, 0), (vehicle_x, self.bev_height), (255, 0, 0), 2)

        cv2.putText(display, f'CTE: {cte:+.4f} m', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, f'Heading: {np.degrees(heading_error):+.2f} deg', (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, f'DRIVE: {self.enable_drive}', (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return display


def main():
    rclpy.init()
    node = LaneDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()