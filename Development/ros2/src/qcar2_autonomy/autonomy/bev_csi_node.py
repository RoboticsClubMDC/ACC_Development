#!/usr/bin/env python3
"""
bev_csi_node.py
===============
ROS 2 (Humble) BEV node for QCar2 front CSI camera — VIRTUAL environment.

Broadcasts TWO static transforms (and only these):
  base_link -> csi_front_mount   (physical position + pitch)
  csi_front_mount -> csi_front_optical  (zero translation, optical axes)

BEV projection uses csi_front_optical as the camera frame via TF lookup.

Run
---
  ros2 run <your_pkg> bev_csi_node

  ros2 run <your_pkg> bev_csi_node --ros-args \
      -p pitch_deg:=-20.0 \
      -p bev_x_min:=0.0 -p bev_x_max:=6.0 \
      -p bev_y_min:=-2.0 -p bev_y_max:=2.0 \
      -p debug_grid:=true

Validate TF after running:
  ros2 run tf2_ros tf2_echo base_link csi_front_optical
  # Expect: translation ≈ (0.183, 0.0, 0.110)  — close to car, NOT huge

  ros2 topic info /tf_static -v
  # Should show ONLY ONE publisher (this node). Kill any others.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import tf2_ros
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image

import cv2
import numpy as np
from cv_bridge import CvBridge


# ---------------------------------------------------------------------------
# Nominal intrinsics (820 x 410) — auto-scaled to actual frame size
# ---------------------------------------------------------------------------
K_NOMINAL = np.array([
    [318.86,   0.00, 401.34],
    [  0.00, 312.14, 201.50],
    [  0.00,   0.00,   1.00]
], dtype=np.float64)
NOMINAL_W, NOMINAL_H = 820.0, 410.0


def scale_K(K, actual_w, actual_h):
    sx, sy = actual_w / NOMINAL_W, actual_h / NOMINAL_H
    Ks = K.copy()
    Ks[0, 0] *= sx;  Ks[0, 2] *= sx
    Ks[1, 1] *= sy;  Ks[1, 2] *= sy
    return Ks


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def rotation_matrix_to_quaternion(Rm):
    """3x3 rotation matrix -> (x, y, z, w) quaternion. No scipy needed."""
    trace = Rm[0,0] + Rm[1,1] + Rm[2,2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (Rm[2,1] - Rm[1,2]) * s
        y = (Rm[0,2] - Rm[2,0]) * s
        z = (Rm[1,0] - Rm[0,1]) * s
    elif Rm[0,0] > Rm[1,1] and Rm[0,0] > Rm[2,2]:
        s = 2.0 * math.sqrt(1.0 + Rm[0,0] - Rm[1,1] - Rm[2,2])
        w = (Rm[2,1] - Rm[1,2]) / s
        x = 0.25 * s
        y = (Rm[0,1] + Rm[1,0]) / s
        z = (Rm[0,2] + Rm[2,0]) / s
    elif Rm[1,1] > Rm[2,2]:
        s = 2.0 * math.sqrt(1.0 + Rm[1,1] - Rm[0,0] - Rm[2,2])
        w = (Rm[0,2] - Rm[2,0]) / s
        x = (Rm[0,1] + Rm[1,0]) / s
        y = 0.25 * s
        z = (Rm[1,2] + Rm[2,1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + Rm[2,2] - Rm[0,0] - Rm[1,1])
        w = (Rm[1,0] - Rm[0,1]) / s
        x = (Rm[0,2] + Rm[2,0]) / s
        y = (Rm[1,2] + Rm[2,1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


def transform_stamped_to_matrix(ts: TransformStamped) -> np.ndarray:
    """TransformStamped -> 4x4 homogeneous matrix. No scipy needed."""
    q = ts.transform.rotation
    t = ts.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)]
    ], dtype=np.float64)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3,  3] = [t.x, t.y, t.z]
    return M


# ---------------------------------------------------------------------------
# BEV helpers
# ---------------------------------------------------------------------------

def body_to_image_single(body_point, C_T_B, K):
    bp     = np.array([[body_point[0]], [body_point[1]], [body_point[2]]])
    cam_pt = C_T_B @ np.vstack([bp, [[1.0]]])
    z_cam  = cam_pt[2, 0]
    if z_cam <= 0:
        return None, None
    ih = K @ cam_pt[:3]
    return (
        (cam_pt[0,0], cam_pt[1,0], z_cam),
        (ih[0,0] / ih[2,0], ih[1,0] / ih[2,0])
    )


def build_remap_tables(C_T_B, K, bev_w, bev_h,
                        x_min, x_max, y_min, y_max, z0):
    """
    Vectorised BEV remap tables.

    BEV layout:
      row 0   -> X = x_max  (far)
      row H-1 -> X = x_min  (near)
      col 0   -> Y = y_max  (left)
      col W-1 -> Y = y_min  (right)

    z_cam <= 0 -> coord = -1 (border fill, prevents wedge artifacts)
    """
    col_g, row_g = np.meshgrid(
        np.arange(bev_w, dtype=np.float64),
        np.arange(bev_h, dtype=np.float64))

    X_w = x_max - row_g * (x_max - x_min) / max(bev_h - 1, 1)
    Y_w = y_max - col_g * (y_max - y_min) / max(bev_w - 1, 1)

    body_pts = np.stack([
        X_w.ravel(), Y_w.ravel(),
        np.full(bev_h * bev_w, z0),
        np.ones(bev_h * bev_w)
    ])

    cam_pts = C_T_B @ body_pts
    z_cam   = cam_pts[2]
    img_h   = K @ cam_pts[:3]

    u = img_h[0] / img_h[2]
    v = img_h[1] / img_h[2]

    invalid    = z_cam <= 1e-6
    u[invalid] = -1.0
    v[invalid] = -1.0

    map_x = u.reshape(bev_h, bev_w).astype(np.float32)
    map_y = v.reshape(bev_h, bev_w).astype(np.float32)
    return map_x, map_y, int(invalid.sum()), bev_h * bev_w


def log_verification(C_T_B, K, z0, logger):
    logger.info('=== Projection verification ===')
    for X, Y in [(0.5,0.), (1.0,0.), (2.0,0.5), (2.0,-0.5), (3.0,0.)]:
        cam, px = body_to_image_single((X, Y, z0), C_T_B, K)
        if px is None:
            logger.warn(f'  body({X:.1f},{Y:.1f}) -> BEHIND CAMERA')
        else:
            logger.info(f'  body({X:.1f},{Y:.1f}) -> '
                        f'z_cam={cam[2]:.3f} -> px({px[0]:.1f},{px[1]:.1f})')
    logger.info('===============================')


def draw_debug_grid(img, x_min, x_max, y_min, y_max,
                    bev_w, bev_h, x_step=1.0, y_step=0.5):
    out = img.copy()

    def to_px(X, Y):
        c = int(np.clip((y_max-Y)/(y_max-y_min)*(bev_w-1), 0, bev_w-1))
        r = int(np.clip((x_max-X)/(x_max-x_min)*(bev_h-1), 0, bev_h-1))
        return c, r

    GR, AX, TX = (0,200,0), (0,80,255), (255,255,0)

    y = y_min
    while y <= y_max + 1e-9:
        c = AX if abs(y) < 1e-9 else GR
        cv2.line(out, to_px(x_max, y), to_px(x_min, y), c, 1)
        cv2.putText(out, f'Y={y:.1f}', to_px((x_min+x_max)/2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, TX, 1, cv2.LINE_AA)
        y = round(y + y_step, 9)

    x = x_min
    while x <= x_max + 1e-9:
        c = AX if abs(x) < 1e-9 else GR
        cv2.line(out, to_px(x, y_min), to_px(x, y_max), c, 1)
        cv2.putText(out, f'X={x:.1f}', to_px(x, (y_min+y_max)/2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, TX, 1, cv2.LINE_AA)
        x = round(x + x_step, 9)

    org = to_px(0., 0.)
    cv2.circle(out, org, 5, (0,0,255), -1)
    cv2.putText(out, 'car', (org[0]+5, org[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255), 1)
    return out


# ---------------------------------------------------------------------------
# ROS 2 Node
# ---------------------------------------------------------------------------

class BevCsiNode(Node):

    def __init__(self):
        super().__init__('bev_csi_node')

        # ---- Parameters ----
        self.declare_parameter('image_topic',    'camera/csi_image')
        self.declare_parameter('base_frame',     'base_link')
        self.declare_parameter('camera_frame',   'csi_front_optical')
        self.declare_parameter('pitch_deg',      -20.0)
        self.declare_parameter('ground_plane_z',  0.0)
        self.declare_parameter('bev_width_px',    600)
        self.declare_parameter('bev_height_px',   600)
        self.declare_parameter('bev_x_min',    0.0)
        self.declare_parameter('bev_x_max',    6.0)
        self.declare_parameter('bev_y_min',   -2.0)
        self.declare_parameter('bev_y_max',    2.0)
        self.declare_parameter('debug_grid',   True)

        self.img_topic  = self.get_parameter('image_topic').value
        self.base_frame = self.get_parameter('base_frame').value
        self.cam_frame  = self.get_parameter('camera_frame').value
        self.pitch_deg  = self.get_parameter('pitch_deg').value
        self.z0         = self.get_parameter('ground_plane_z').value
        self.bev_w      = self.get_parameter('bev_width_px').value
        self.bev_h      = self.get_parameter('bev_height_px').value
        self.x_min      = self.get_parameter('bev_x_min').value
        self.x_max      = self.get_parameter('bev_x_max').value
        self.y_min      = self.get_parameter('bev_y_min').value
        self.y_max      = self.get_parameter('bev_y_max').value
        self.dbg        = self.get_parameter('debug_grid').value

        self.bridge      = CvBridge()
        self.bev_map_x   = None
        self.bev_map_y   = None
        self._maps_ready = False
        self._last_fw    = None
        self._last_fh    = None

        # ---- Broadcast static TF chain FIRST ----
        self._static_broadcaster = StaticTransformBroadcaster(self)
        self._broadcast_static_tf()

        # ---- TF listener ----
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- Publisher ----
        self.pub_bev = self.create_publisher(
            Image, '/csi/front/image_bev', QoSProfile(depth=2))

        # ---- Subscriber — BEST_EFFORT to match simulator ----
        sensor_qos = QoSProfile(
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(Image, self.img_topic, self._cb, sensor_qos)

        self.get_logger().info(f"Subscribing to '{self.img_topic}'")
        self.get_logger().info(
            'Validate TF: ros2 run tf2_ros tf2_echo '
            f'{self.base_frame} {self.cam_frame}')
        self.get_logger().info(
            'Check for stale publishers: ros2 topic info /tf_static -v')

    # ------------------------------------------------------------------
    def _broadcast_static_tf(self):
        """
        Publish exactly TWO transforms:

        1) base_link -> csi_front_mount
           Physical camera position + pitch (nose down toward ground).
           Translation: x=0.183m forward, y=0.0, z=0.110m up.

        2) csi_front_mount -> csi_front_optical
           Zero translation — pure rotation to ROS optical axes:
             Z_opt = +X_mount  (forward becomes optical Z, into scene)
             X_opt = -Y_mount  (left body becomes optical right = -X)
             Y_opt = -Z_mount  (up body becomes optical down = +Y)
        """
        stamp = self.get_clock().now().to_msg()
        pitch_rad = math.radians(self.pitch_deg)
        qy_mount = math.sin(pitch_rad / 2.0)
        qw_mount = math.cos(pitch_rad / 2.0)

        # --- Transform 1: base_link -> csi_front_mount ---
        ts_mount = TransformStamped()
        ts_mount.header.stamp    = stamp
        ts_mount.header.frame_id = self.base_frame
        ts_mount.child_frame_id  = 'csi_front_mount'
        ts_mount.transform.translation.x = 0.183
        ts_mount.transform.translation.y = 0.000
        ts_mount.transform.translation.z = 0.110
        ts_mount.transform.rotation.x    = 0.0
        ts_mount.transform.rotation.y    = qy_mount
        ts_mount.transform.rotation.z    = 0.0
        ts_mount.transform.rotation.w    = qw_mount

        # --- Transform 2: csi_front_mount -> csi_front_optical ---
        # R_om: each row = where that optical axis points IN mount frame
        #   X_opt (right)   = -Y_mount
        #   Y_opt (down)    = -Z_mount
        #   Z_opt (forward) = +X_mount
        R_om = np.array([
            [ 0, -1,  0],
            [ 0,  0, -1],
            [ 1,  0,  0],
        ], dtype=np.float64)
        R_mo = R_om.T   # mount -> optical  (what TF actually wants)
        qx, qy, qz, qw = rotation_matrix_to_quaternion(R_mo)

        ts_opt = TransformStamped()
        ts_opt.header.stamp    = stamp
        ts_opt.header.frame_id = 'csi_front_mount'
        ts_opt.child_frame_id  = 'csi_front_optical'
        ts_opt.transform.translation.x = 0.0
        ts_opt.transform.translation.y = 0.0
        ts_opt.transform.translation.z = 0.0
        ts_opt.transform.rotation.x    = qx
        ts_opt.transform.rotation.y    = qy
        ts_opt.transform.rotation.z    = qz
        ts_opt.transform.rotation.w    = qw

        self._static_broadcaster.sendTransform([ts_mount, ts_opt])

        self.get_logger().info(
            f'{self.base_frame}->csi_front_mount | '
            f't=(0.183, 0.0, 0.110) | pitch={self.pitch_deg:.1f} deg')
        self.get_logger().info(
            'csi_front_mount->csi_front_optical | t=(0,0,0) optical axes')

    # ------------------------------------------------------------------
    def _build_maps(self, fw, fh):
        try:
            # Lookup: csi_front_optical FROM base_link  =>  C_T_B
            ts = self.tf_buffer.lookup_transform(
                self.cam_frame,     # target = camera (optical)
                self.base_frame,    # source = body
                rclpy.time.Time())
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'TF not ready: {e}')
            return False

        C_T_B = transform_stamped_to_matrix(ts)
        tr = ts.transform.translation
        self.get_logger().info(
            f'TF lookup OK: t=[{tr.x:.3f},{tr.y:.3f},{tr.z:.3f}]')
        self.get_logger().info(f'C_T_B:\n{np.round(C_T_B, 4)}')

        K = scale_K(K_NOMINAL, fw, fh)
        self.get_logger().info(f'K (scaled {fw}x{fh}):\n{np.round(K, 2)}')
        self.get_logger().info(
            f'Building BEV | X[{self.x_min},{self.x_max}] '
            f'Y[{self.y_min},{self.y_max}] | {self.bev_w}x{self.bev_h}px')

        self.bev_map_x, self.bev_map_y, n_inv, n_tot = build_remap_tables(
            C_T_B, K,
            self.bev_w, self.bev_h,
            self.x_min, self.x_max,
            self.y_min, self.y_max,
            self.z0)

        self.get_logger().info(
            f'Maps built | {n_inv}/{n_tot} masked '
            f'({100*n_inv/n_tot:.1f}% behind camera)')

        log_verification(C_T_B, K, self.z0, self.get_logger())

        self._last_fw = fw
        self._last_fh = fh
        return True

    # ------------------------------------------------------------------
    def _cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge: {e}')
            return

        fh, fw = frame.shape[:2]

        if not self._maps_ready or fw != self._last_fw or fh != self._last_fh:
            if not self._build_maps(fw, fh):
                return
            self._maps_ready = True

        bev = cv2.remap(frame,
                        self.bev_map_x, self.bev_map_y,
                        cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(50, 50, 50))

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