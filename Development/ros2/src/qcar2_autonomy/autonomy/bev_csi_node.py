#!/usr/bin/env python3
"""
bev_csi_node.py
===============
ROS 2 (Humble) BEV node for QCar2 front CSI camera — VIRTUAL environment.

Uses Quanser's InversePerspectiveMapping approach (cv2.getPerspectiveTransform
+ cv2.warpPerspective) ported directly from their codebase. No TF tree needed.

Key parameters to tune:
  bev_x_min/max  : forward range in meters (virtual = 10x physical)
  bev_y_min/max  : lateral range in meters
  cam_height     : camera height above ground (virtual = 1.10 m)
  cam_pitch_deg  : camera tilt down from horizontal (tune this for flat BEV)

Run
---
  ros2 run <your_pkg> bev_csi_node

  ros2 run <your_pkg> bev_csi_node --ros-args \
      -p cam_height:=1.10 \
      -p cam_pitch_deg:=20.0 \
      -p bev_x_min:=2.0 -p bev_x_max:=20.0 \
      -p bev_y_min:=-6.0 -p bev_y_max:=6.0 \
      -p bev_width_px:=600 -p bev_height_px:=600 \
      -p debug_grid:=true
"""

import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


# ---------------------------------------------------------------------------
# InversePerspectiveMapping — ported directly from Quanser's codebase
# Adapted for QCar2 front CSI camera parameters
# ---------------------------------------------------------------------------

class InversePerspectiveMapping:
    """
    Builds a homography from camera image to Bird's-Eye View using
    camera intrinsics and extrinsics.

    Ported from Quanser's hal/utilities/qcar.py InversePerspectiveMapping.
    """

    def __init__(self, bev_shape, bev_world_dims, cam_height, cam_pitch_deg,
                 K, img_w, img_h):
        """
        Args:
            bev_shape      : (width, height) of output BEV image in pixels
            bev_world_dims : [x_min, x_max, y_min, y_max] in meters
            cam_height     : camera height above ground plane (meters)
            cam_pitch_deg  : camera tilt downward from horizontal (degrees, positive = down)
            K              : 3x3 camera intrinsic matrix
            img_w, img_h   : input image resolution
        """
        self.bev_shape   = bev_shape
        self.world_dims  = bev_world_dims
        self.cam_height  = cam_height
        self.cam_pitch   = math.radians(cam_pitch_deg)
        self.K           = K
        self.img_w       = img_w
        self.img_h       = img_h

        x_range = bev_world_dims[1] - bev_world_dims[0]
        y_range = bev_world_dims[3] - bev_world_dims[2]
        self.m_per_pix_x = x_range / bev_shape[1]  # height axis = X (forward)
        self.m_per_pix_y = y_range / bev_shape[0]  # width axis  = Y (lateral)

        self._build_extrinsics()
        self._build_homography()

    def _build_extrinsics(self):
        """
        Build camera extrinsic matrix (vehicle -> camera).
        Camera is mounted at height h, pitched down by cam_pitch.
        Vehicle frame: X forward, Y left, Z up.
        Camera frame: Z out (forward), X right, Y down.

        Matches Quanser convention from their get_extrinsics():
          phi   = pi/2   (roll:  rotate Y axis down)
          psi   = pi/2   (yaw:   align X with lateral)
          theta = -pitch (pitch: nose down)
        """
        phi   = math.pi / 2.0
        theta = -self.cam_pitch   # negative = nose down
        psi   = math.pi / 2.0

        cx, sx = math.cos(phi),   math.sin(phi)
        cy, sy = math.cos(theta), math.sin(theta)
        cz, sz = math.cos(psi),   math.sin(psi)

        Rx = np.array([[1,  0,   0 ],
                       [0,  cx, -sx],
                       [0,  sx,  cx]])
        Ry = np.array([[ cy, 0, sy],
                       [  0, 1,  0],
                       [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0],
                       [sz,  cz, 0],
                       [ 0,   0, 1]])

        self.R_v2cam = Rx @ Ry @ Rz
        self.t_v2cam = np.array([[0, self.cam_height, 0]]).T
        self.T_v2cam = np.vstack([
            np.hstack([self.R_v2cam, self.t_v2cam]),
            np.array([[0, 0, 0, 1]])
        ])

    def v2img(self, XYZ):
        """Project Nx3 vehicle-frame points to (u,v) image pixels."""
        # Build 3x4 projection matrix: K * [R | t]  (drop last row of T)
        P = self.K @ self.T_v2cam[:3, :]
        XYZ1 = np.hstack([XYZ, np.ones((XYZ.shape[0], 1))])
        img_h = P @ XYZ1.T
        img_h /= img_h[2]
        return img_h[:2].T.astype(np.float32)

    def _build_homography(self):
        """
        Pick 4 ground-plane corners in vehicle frame, project to image,
        map to BEV pixel coords, compute homography H.
        """
        x_min, x_max = self.world_dims[0], self.world_dims[1]
        y_min, y_max = self.world_dims[2], self.world_dims[3]

        # 4 corners of the BEV world window on the ground plane (Z=0)
        world_corners = np.array([
            [x_max, y_max, 0],   # far-left
            [x_max, y_min, 0],   # far-right
            [x_min, y_max, 0],   # near-left
            [x_min, y_min, 0],   # near-right
        ], dtype=np.float64)

        # Project to image
        img_corners = self.v2img(world_corners)

        # Corresponding BEV pixel coords
        bev_w, bev_h = self.bev_shape
        bev_corners = np.array([
            [0,      0     ],   # far-left   -> top-left
            [bev_w-1, 0    ],   # far-right  -> top-right
            [0,      bev_h-1],  # near-left  -> bottom-left
            [bev_w-1, bev_h-1], # near-right -> bottom-right
        ], dtype=np.float32)

        self.M = cv2.getPerspectiveTransform(img_corners, bev_corners)

    def warp(self, img):
        """Apply BEV warp to input image."""
        return cv2.warpPerspective(
            img, self.M, self.bev_shape,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(50, 50, 50))


# ---------------------------------------------------------------------------
# Debug grid overlay
# ---------------------------------------------------------------------------

def draw_debug_grid(img, x_min, x_max, y_min, y_max,
                    bev_w, bev_h, x_step=2.0, y_step=1.0):
    out = img.copy()

    def to_px(X, Y):
        # BEV: col = (Y - y_min)/(y_max-y_min)*w,  row = (x_max-X)/(x_max-x_min)*h
        c = int(np.clip((Y - y_min) / (y_max - y_min) * (bev_w - 1), 0, bev_w - 1))
        r = int(np.clip((x_max - X) / (x_max - x_min) * (bev_h - 1), 0, bev_h - 1))
        return c, r

    GR, AX, TX = (0, 200, 0), (0, 80, 255), (255, 255, 0)

    y = y_min
    while y <= y_max + 1e-9:
        c = AX if abs(y) < 1e-9 else GR
        cv2.line(out, to_px(x_max, y), to_px(x_min, y), c, 1)
        cv2.putText(out, f'Y={y:.1f}', to_px((x_min + x_max) / 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, TX, 1, cv2.LINE_AA)
        y = round(y + y_step, 9)

    x = x_min
    while x <= x_max + 1e-9:
        c = AX if abs(x) < 1e-9 else GR
        cv2.line(out, to_px(x, y_min), to_px(x, y_max), c, 1)
        cv2.putText(out, f'X={x:.1f}', to_px(x, (y_min + y_max) / 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, TX, 1, cv2.LINE_AA)
        x = round(x + x_step, 9)

    org = to_px(0.0, 0.0)
    if 0 <= org[0] < bev_w and 0 <= org[1] < bev_h:
        cv2.circle(out, org, 5, (0, 0, 255), -1)
        cv2.putText(out, 'car', (org[0] + 5, org[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    return out


# ---------------------------------------------------------------------------
# ROS 2 Node
# ---------------------------------------------------------------------------

# Nominal intrinsics for 820x410 — scaled if actual frame differs
K_NOMINAL = np.array([
    [318.86,   0.00, 401.34],
    [  0.00, 312.14, 201.50],
    [  0.00,   0.00,   1.00]
], dtype=np.float64)
NOMINAL_W, NOMINAL_H = 820.0, 410.0


def scale_K(K, w, h):
    Ks = K.copy()
    Ks[0, 0] *= w / NOMINAL_W;  Ks[0, 2] *= w / NOMINAL_W
    Ks[1, 1] *= h / NOMINAL_H;  Ks[1, 2] *= h / NOMINAL_H
    return Ks


class BevCsiNode(Node):

    def __init__(self):
        super().__init__('bev_csi_node')

        # ---- Parameters ----
        self.declare_parameter('image_topic',   'camera/csi_image')
        # Camera mounting (virtual = 10x physical heights)
        self.declare_parameter('cam_height',    1.10)   # meters above ground
        self.declare_parameter('cam_pitch_deg', 20.0)   # degrees nose-down (tune this)
        # BEV world window (virtual scale)
        self.declare_parameter('bev_x_min',    2.0)
        self.declare_parameter('bev_x_max',   20.0)
        self.declare_parameter('bev_y_min',   -6.0)
        self.declare_parameter('bev_y_max',    6.0)
        self.declare_parameter('bev_width_px',  600)
        self.declare_parameter('bev_height_px', 600)
        self.declare_parameter('debug_grid',    True)

        self.img_topic    = self.get_parameter('image_topic').value
        self.cam_height   = self.get_parameter('cam_height').value
        self.cam_pitch    = self.get_parameter('cam_pitch_deg').value
        self.x_min        = self.get_parameter('bev_x_min').value
        self.x_max        = self.get_parameter('bev_x_max').value
        self.y_min        = self.get_parameter('bev_y_min').value
        self.y_max        = self.get_parameter('bev_y_max').value
        self.bev_w        = self.get_parameter('bev_width_px').value
        self.bev_h        = self.get_parameter('bev_height_px').value
        self.dbg          = self.get_parameter('debug_grid').value

        self.bridge = CvBridge()
        self.ipm    = None
        self._last_fw = None
        self._last_fh = None

        self.pub_bev = self.create_publisher(
            Image, '/csi/front/image_bev', QoSProfile(depth=2))

        sensor_qos = QoSProfile(
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(Image, self.img_topic, self._cb, sensor_qos)

        self.get_logger().info(
            f"BEV node ready | subscribing '{self.img_topic}' | "
            f"height={self.cam_height}m pitch={self.cam_pitch}deg | "
            f"world X[{self.x_min},{self.x_max}] Y[{self.y_min},{self.y_max}] m")

    def _build_ipm(self, fw, fh):
        K = scale_K(K_NOMINAL, fw, fh)
        self.ipm = InversePerspectiveMapping(
            bev_shape   = (self.bev_w, self.bev_h),
            bev_world_dims = [self.x_min, self.x_max, self.y_min, self.y_max],
            cam_height  = self.cam_height,
            cam_pitch_deg = self.cam_pitch,
            K           = K,
            img_w       = fw,
            img_h       = fh)
        self._last_fw = fw
        self._last_fh = fh
        self.get_logger().info(f'IPM built for {fw}x{fh} | K:\n{np.round(K,2)}')

    def _cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge: {e}')
            return

        fh, fw = frame.shape[:2]

        if self.ipm is None or fw != self._last_fw or fh != self._last_fh:
            self._build_ipm(fw, fh)

        bev = self.ipm.warp(frame)

        if self.dbg:
            bev = draw_debug_grid(bev,
                                  self.x_min, self.x_max,
                                  self.y_min, self.y_max,
                                  self.bev_w, self.bev_h)

        out = self.bridge.cv2_to_imgmsg(bev, encoding='bgr8')
        out.header = msg.header
        self.pub_bev.publish(out)


# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = BevCsiNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()