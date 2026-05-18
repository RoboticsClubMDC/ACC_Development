#!/usr/bin/env python3
"""Overlay VO diagnostics text on the live RGB camera image.

Publishes:
  /vo/overlay_image (sensor_msgs/Image, bgr8)

Inputs:
  /camera/color_image (sensor_msgs/Image, bgr8)
  /vo/fault_status    (std_msgs/String, compact status string)
"""

import math
import re
import threading

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


class VOImageOverlay(Node):
    _STATUS_RE = re.compile(
        r"^(?P<state>\w+)\s+rho=(?P<rho>[-0-9.]+)\s+w=(?P<w>[-0-9.]+)\s+\|\s+"
        r"vo\((?P<vo_x>[-0-9.]+),(?P<vo_y>[-0-9.]+),(?P<vo_psi_deg>[-0-9.]+)\)\s+"
        r"ct\((?P<ct_x>[-0-9.]+),(?P<ct_y>[-0-9.]+),(?P<ct_psi_deg>[-0-9.]+)\)\s+\|\s+"
        r"dx=(?P<dx>[-0-9.]+)\s+dy=(?P<dy>[-0-9.]+)\s+dpsi=(?P<dpsi_deg>[-0-9.]+)\s+"
        r"inl=(?P<inl>\d+)\s+sp=(?P<sp>[-0-9.]+)"
        r"(?:\s+psi_raw=(?P<psi_raw_deg>[-0-9.]+))?$"
    )

    def __init__(self):
        super().__init__("vo_image_overlay")
        self.declare_parameter("image_topic", "/camera/color_image")
        self.declare_parameter("status_topic", "/vo/fault_status")
        self.declare_parameter("overlay_topic", "/vo/overlay_image")
        self.declare_parameter("font_scale", 0.62)
        self.declare_parameter("line_height", 24)
        self.declare_parameter("margin_px", 12)

        image_topic = str(self.get_parameter("image_topic").value)
        status_topic = str(self.get_parameter("status_topic").value)
        overlay_topic = str(self.get_parameter("overlay_topic").value)
        self.font_scale = float(self.get_parameter("font_scale").value)
        self.line_height = int(self.get_parameter("line_height").value)
        self.margin = int(self.get_parameter("margin_px").value)

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, overlay_topic, 10)
        self.create_subscription(Image, image_topic, self._image_cb, 10)
        self.create_subscription(String, status_topic, self._status_cb, 10)

        self._lock = threading.Lock()
        self._status_raw = ""
        self._status = {}

        self.get_logger().info(
            f"VO overlay started: image={image_topic}, status={status_topic}, "
            f"out={overlay_topic}"
        )

    def _status_cb(self, msg: String) -> None:
        text = msg.data.strip()
        parsed = {}
        m = self._STATUS_RE.match(text)
        if m:
            g = m.groupdict()
            parsed = {
                "state": g["state"],
                "rho": float(g["rho"]),
                "w": float(g["w"]),
                "vo_x": float(g["vo_x"]),
                "vo_y": float(g["vo_y"]),
                "vo_psi_deg": float(g["vo_psi_deg"]),
                "ct_x": float(g["ct_x"]),
                "ct_y": float(g["ct_y"]),
                "ct_psi_deg": float(g["ct_psi_deg"]),
                "dx": float(g["dx"]),
                "dy": float(g["dy"]),
                "dpsi_deg": float(g["dpsi_deg"]),
                "inl": int(g["inl"]),
                "sp": float(g["sp"]),
                "psi_raw_deg": (float(g["psi_raw_deg"])
                                if g.get("psi_raw_deg") is not None else None),
            }
        with self._lock:
            self._status_raw = text
            self._status = parsed

    def _state_color(self, state: str):
        if state == "agree":
            return (0, 220, 0)
        if state == "vo_suspect":
            return (0, 165, 255)
        if state == "odom_suspect":
            return (0, 0, 255)
        if state == "warming":
            return (0, 220, 220)
        return (220, 220, 220)

    def _draw_text(self, img, lines, state):
        h, w = img.shape[:2]
        line_h = self.line_height
        pad = 10
        box_w = min(w - 2 * self.margin, 620)
        box_h = min(h - 2 * self.margin, pad * 2 + line_h * len(lines))

        x0 = self.margin
        y0 = self.margin
        x1 = x0 + box_w
        y1 = y0 + box_h

        cv2.rectangle(img, (x0, y0), (x1, y1), (12, 12, 12), thickness=-1)
        cv2.rectangle(img, (x0, y0), (x1, y1), self._state_color(state), thickness=2)

        y = y0 + pad + line_h - 8
        for i, line in enumerate(lines):
            color = (245, 245, 245) if i > 0 else self._state_color(state)
            cv2.putText(
                img,
                line,
                (x0 + pad, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                color,
                2,
                cv2.LINE_AA,
            )
            y += line_h

    def _image_cb(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:
            self.get_logger().warn(f"image decode error: {exc}", throttle_duration_sec=5.0)
            return

        # Some camera backends occasionally emit an empty frame while the
        # stream is starting/recovering. Skip gracefully instead of crashing.
        if frame is None or getattr(frame, "size", 0) == 0:
            self.get_logger().warn(
                "received empty camera frame", throttle_duration_sec=5.0
            )
            return

        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        with self._lock:
            parsed = dict(self._status)
            raw = self._status_raw

        if parsed:
            drift = math.sqrt(
                (parsed["vo_x"] - parsed["ct_x"]) ** 2 + (parsed["vo_y"] - parsed["ct_y"]) ** 2
            )
            lines = [
                f"STATE={parsed['state']}  rho={parsed['rho']:.3f}  w={parsed['w']:.2f}",
                (
                    f"VO  x={parsed['vo_x']:+.2f} y={parsed['vo_y']:+.2f} "
                    f"psi={parsed['vo_psi_deg']:+.0f}deg"
                ),
                (
                    f"CT  x={parsed['ct_x']:+.2f} y={parsed['ct_y']:+.2f} "
                    f"psi={parsed['ct_psi_deg']:+.0f}deg"
                ),
                (
                    f"dxy dx={parsed['dx']:+.4f} dy={parsed['dy']:+.4f} "
                    f"dpsi={parsed['dpsi_deg']:+.1f}deg"
                ),
                f"inliers={parsed['inl']} spread={parsed['sp']:.1f} drift={drift:.3f}m",
            ]
            if parsed.get("psi_raw_deg") is not None:
                psi_err_raw = parsed["psi_raw_deg"] - parsed["ct_psi_deg"]
                # wrap to [-180, 180] for readability
                psi_err_raw = (psi_err_raw + 180.0) % 360.0 - 180.0
                lines.append(
                    f"psi_raw={parsed['psi_raw_deg']:+.0f}deg "
                    f"(camera-only, err={psi_err_raw:+.0f}deg)"
                )
            state = parsed["state"]
        else:
            lines = [
                "STATE=waiting",
                "No parsed /vo/fault_status yet",
                raw if raw else "waiting for /vo/fault_status ...",
            ]
            state = "init"

        self._draw_text(frame, lines, state)

        out = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        out.header = msg.header
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = VOImageOverlay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
