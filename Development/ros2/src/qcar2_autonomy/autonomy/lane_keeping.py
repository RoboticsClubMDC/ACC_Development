#!/usr/bin/env python3
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class LaneKeeping(Node):
    """
    Subscribes:
      - /cmd_vel_raw (Twist): raw command from path_follower
      - /lane_seg/no_go_margin (mono8): 0/255 dilated no-go mask

    Publishes:
      - /cmd_vel_nav (Twist): safe command that tries to avoid entering red soon

    Strategy (image-space, no calibration assumptions):
      - Define a wedge/corridor ROI in front of the car in image coordinates.
      - If red pixels are present in ROI:
          * steer away from red centroid (simple proportional rule)
          * reduce speed as red occupancy increases
      - Else: pass-through.
    """

    def __init__(self):
        super().__init__("lane_keeping")

        # ---- Parameters (all adjustable; defaults are just starting points) ----
        self.declare_parameter("input_cmd_topic", "/cmd_vel_raw")
        self.declare_parameter("output_cmd_topic", "/cmd_vel_nav")
        self.declare_parameter("mask_topic", "/sidewalk_detection/no_go_margin")
        self.declare_parameter("lane_mask_topic", "/lane_detection/lane_selected")

        # Vehicle anchor in the image (you confirmed ~55% is good)
        self.declare_parameter("seed_x_frac", 0.55)

        # ROI vertical span (fractions of image height)
        # y0_frac = start of ROI (higher up), y1_frac = bottom (near car)
        self.declare_parameter("roi_y0_frac", 0.65)
        self.declare_parameter("roi_y1_frac", 0.98)

        # ROI half-width as fraction of image width (corridor around the car)
        self.declare_parameter("roi_half_width_frac", 0.22)

        # Control gains / limits
        self.declare_parameter("steer_gain", 2.0)           # steering away from red centroid
        self.declare_parameter("max_steer_adjust", 0.40)    # max rad to add/subtract from incoming cmd
        self.declare_parameter("min_speed_scale", 0.60)     # never scale speed below this (unless you later want full stop)
        self.declare_parameter("red_occupancy_for_max_brake", 0.20)  # ROI red fraction that triggers strongest slowdown

        self.declare_parameter("lane_y0_frac", 0.80)          # bottom band start
        self.declare_parameter("lane_y1_frac", 0.98)          # bottom band end
        self.declare_parameter("lane_center_gain", 0.8)       # steering bias strength
        self.declare_parameter("lane_center_max", 0.25)       # max rad bias
        
        # Internal
        self.bridge = CvBridge()
        self.last_cmd = Twist()
        self.have_cmd = False
        self.have_mask = False
        self.last_mask_u8 = None
        self.last_lane_u8 = None
        self.have_lane = False

        in_cmd = self.get_parameter("input_cmd_topic").value
        out_cmd = self.get_parameter("output_cmd_topic").value
        mask_topic = self.get_parameter("mask_topic").value
        lane_topic = self.get_parameter("lane_mask_topic").value

        self.pub = self.create_publisher(Twist, out_cmd, 10)
        self.sub_lane = self.create_subscription(Image, lane_topic, self.lane_cb, qos_profile_sensor_data)
        self.sub_cmd = self.create_subscription(Twist, in_cmd, self.cmd_cb, 10)
        self.sub_mask = self.create_subscription(Image, mask_topic, self.mask_cb, qos_profile_sensor_data)

        self.get_logger().info(f"Input cmd:  {in_cmd}")
        self.get_logger().info(f"Output cmd: {out_cmd}")
        self.get_logger().info(f"Mask:       {mask_topic}")

    def cmd_cb(self, msg: Twist):
        self.last_cmd = msg
        self.have_cmd = True
        self.maybe_publish()

    def mask_cb(self, msg: Image):
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge mask failed: {e}")
            return

        if mask is None:
            return

        self.last_mask_u8 = mask
        self.have_mask = True
        self.maybe_publish()

    def maybe_publish(self):
        if not (self.have_cmd and self.have_mask):
            return

        cmd_in = self.last_cmd
        # If we're not moving, pass-through exactly (prevents steering chatter at standstill)
        if abs(cmd_in.linear.x) < 0.02:
            self.pub.publish(cmd_in)
            return
        mask_u8 = self.last_mask_u8
        h, w = mask_u8.shape[:2]

        seed_x = int(float(self.get_parameter("seed_x_frac").value) * w)

        y0 = int(float(self.get_parameter("roi_y0_frac").value) * h)
        y1 = int(float(self.get_parameter("roi_y1_frac").value) * h)
        y0 = max(0, min(h - 1, y0))
        y1 = max(0, min(h - 1, y1))
        if y1 <= y0:
            y0, y1 = min(y0, y1), max(y0, y1) + 1

        half_w = int(float(self.get_parameter("roi_half_width_frac").value) * w)
        x0 = max(0, seed_x - half_w)
        x1 = min(w, seed_x + half_w)

        roi = mask_u8[y0:y1, x0:x1]
        red = (roi > 127)

        safe = Twist()
        safe.linear.x = cmd_in.linear.x
        safe.angular.z = cmd_in.angular.z
        
        # ----------------- Lane centering (near-field only; ignores curves) -----------------
        if self.have_lane and self.last_lane_u8 is not None:
            lane_u8 = self.last_lane_u8
            if lane_u8.shape[:2] == (h, w):
                ly0 = int(float(self.get_parameter("lane_y0_frac").value) * h)
                ly1 = int(float(self.get_parameter("lane_y1_frac").value) * h)
                ly0 = max(0, min(h - 1, ly0))
                ly1 = max(0, min(h, ly1))
                if ly1 <= ly0:
                    ly1 = min(h, ly0 + 1)

                band = (lane_u8[ly0:ly1, :] > 127)
                if band.any():
                    ys, xs = np.where(band)
                    cx = float(xs.mean())

                    desired_cx = float(seed_x)  # keep lane centered at seed_x
                    err_px = (cx - desired_cx)
                    err_norm = err_px / max(1.0, 0.5 * w)  # normalize ~[-1..1]

                    k = float(self.get_parameter("lane_center_gain").value)
                    max_bias = float(self.get_parameter("lane_center_max").value)

                    # If lane centroid is to the right (+err), steer left (-bias)
                    bias = float(np.clip(-k * err_norm, -max_bias, max_bias))

                    safe.angular.z = float(safe.angular.z + bias)
        # -------------------------------------------------------------------

        if red.any():
            # Red occupancy (how much red in ROI)
            occ = float(red.mean())  # 0..1

            # Centroid of red pixels within ROI (x only matters for steering)
            ys, xs = np.where(red)
            cx = float(xs.mean())  # 0..(roi_w-1)

            # Convert centroid into normalized offset from corridor center (-1..+1)
            roi_w = max(1, (x1 - x0))
            center = (roi_w - 1) / 2.0
            offset = (cx - center) / max(1.0, center)  # -1 (left) to +1 (right)

            # Steer away: if red is to the right (+), steer left (-)
            steer_gain = float(self.get_parameter("steer_gain").value)
            max_adj = float(self.get_parameter("max_steer_adjust").value)
            steer_adj = np.clip(-steer_gain * offset, -max_adj, max_adj)

            safe.angular.z = float(safe.angular.z + steer_adj)

            # Speed scaling based on occupancy
            occ_max = float(self.get_parameter("red_occupancy_for_max_brake").value)
            min_scale = float(self.get_parameter("min_speed_scale").value)

            # occ=0 -> scale 1.0 ; occ>=occ_max -> scale min_scale
            t = np.clip(occ / max(1e-6, occ_max), 0.0, 1.0)
            scale = (1.0 - t) * 1.0 + t * min_scale

            safe.linear.x = float(cmd_in.linear.x * scale)
            
            safe.angular.z = float(np.clip(safe.angular.z, -0.5, 0.5))

        safe.angular.z = float(np.clip(safe.angular.z, -0.5, 0.5))
        
        self.pub.publish(safe)

    def lane_cb(self, msg: Image):
        try:
            lane = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception:
            return
        self.last_lane_u8 = lane
        self.have_lane = True
        
        
def main():
    rclpy.init()
    node = LaneKeeping()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()