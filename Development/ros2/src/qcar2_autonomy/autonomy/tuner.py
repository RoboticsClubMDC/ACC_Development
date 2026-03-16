#!/usr/bin/env python3
"""
tune_bev.py
-----------
Live BEV tuner — matches bev_csi_node.py (OVERALL FIXED) exactly.
Same extrinsics, same homography, same K scaling.

Topic: camera/color_image (change IMAGE_TOPIC below)

Keys:
    W / S   — pitch +-0.5 deg
    E / D   — pitch +-0.1 deg (fine)
    R / F   — height +-0.01 m
    G       — print values to paste into bev_csi_node.py
    Q       — quit
"""

import sys, math, threading
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

# ── Tune these starting values ────────────────────────────────────────────────
CAM_HEIGHT    = 0.960
CAM_PITCH_DEG = 0.1

X_MIN, X_MAX =  0.05, 2.0
Y_MIN, Y_MAX = -0.6,  0.6
BEV_W = BEV_H = 600

IMAGE_TOPIC = 'camera/color_image'  # change to .../compressed if needed
# ─────────────────────────────────────────────────────────────────────────────

# D435 intrinsics at 640x480
K_NOMINAL = np.array([
    [455.20,   0.00, 308.53],
    [  0.00, 459.43, 213.56],
    [  0.00,   0.00,   1.00]
], dtype=np.float64)
NOMINAL_W, NOMINAL_H = 640.0, 480.0


def scale_K(K, w, h):
    Ks = K.copy()
    Ks[0,0] *= w/NOMINAL_W; Ks[0,2] *= w/NOMINAL_W
    Ks[1,1] *= h/NOMINAL_H; Ks[1,2] *= h/NOMINAL_H
    return Ks


def build_homography(K, cam_height, cam_pitch_deg):
    """Exactly matches OVERALL FIXED bev_csi_node.py _build_extrinsics."""
    pitch = math.radians(cam_pitch_deg)
    phi, psi = math.pi/2, math.pi/2
    theta = -pitch  # negative = nose down (Quanser convention)

    cx, sx = math.cos(phi),   math.sin(phi)
    cy, sy = math.cos(theta), math.sin(theta)
    cz, sz = math.cos(psi),   math.sin(psi)

    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    R  = Rx @ Ry @ Rz

    # ✅ FIXED: height on Z (up), matches OVERALL FIXED node
    t = np.array([[0.0, 0.0, cam_height]]).T
    T = np.vstack([np.hstack([R, t]), [0,0,0,1]])
    P = K @ T[:3, :]

    def v2img(pts):
        XYZ1 = np.hstack([pts, np.ones((len(pts),1))])
        h = P @ XYZ1.T
        h /= h[2]
        return h[:2].T.astype(np.float32)

    world = np.array([
        [X_MAX, Y_MAX, 0], [X_MAX, Y_MIN, 0],
        [X_MIN, Y_MAX, 0], [X_MIN, Y_MIN, 0],
    ], dtype=np.float64)
    img_pts = v2img(world)
    bev_pts = np.array([
        [0,       0      ],
        [BEV_W-1, 0      ],
        [0,       BEV_H-1],
        [BEV_W-1, BEV_H-1],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(img_pts, bev_pts)


def draw_grid(img):
    out = img.copy()
    def px(X, Y):
        c = int(np.clip((Y-Y_MIN)/(Y_MAX-Y_MIN)*(BEV_W-1), 0, BEV_W-1))
        r = int(np.clip((X_MAX-X)/(X_MAX-X_MIN)*(BEV_H-1), 0, BEV_H-1))
        return c, r
    for y in np.arange(Y_MIN, Y_MAX+1e-9, (Y_MAX-Y_MIN)/4):
        cv2.line(out, px(X_MAX,y), px(X_MIN,y), (0,180,0), 1)
    for x in np.arange(X_MIN, X_MAX+1e-9, (X_MAX-X_MIN)/5):
        cv2.line(out, px(x,Y_MIN), px(x,Y_MAX), (0,180,0), 1)
    o = px(0,0)
    if 0<=o[0]<BEV_W and 0<=o[1]<BEV_H:
        cv2.circle(out, o, 6, (0,0,255), -1)
        cv2.putText(out,'car',(o[0]+8,o[1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
    return out


class TuneNode(Node):
    def __init__(self):
        super().__init__('tune_bev')
        self.bridge = CvBridge()
        self.frame  = None
        self.lock   = threading.Lock()
        qos = QoSProfile(depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE)
        if IMAGE_TOPIC.endswith('/compressed'):
            self.create_subscription(CompressedImage, IMAGE_TOPIC, self._cb_compressed, qos)
        else:
            self.create_subscription(Image, IMAGE_TOPIC, self._cb_raw, qos)
        self.get_logger().info(f"Listening: {IMAGE_TOPIC}")

    def _cb_raw(self, msg):
        try:
            f = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            if f is not None and f.size > 0:
                with self.lock: self.frame = f
        except: pass

    def _cb_compressed(self, msg):
        try:
            arr = np.frombuffer(msg.data, np.uint8)
            f = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if f is not None and f.size > 0:
                with self.lock: self.frame = f
        except: pass


def main():
    global CAM_HEIGHT, CAM_PITCH_DEG

    rclpy.init()
    node = TuneNode()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    print(f"Waiting for frames on '{IMAGE_TOPIC}'...")
    cv2.namedWindow('BEV Tuner', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('BEV Tuner', BEV_W + 280, BEV_H)

    fw = fh = None
    M  = None

    while True:
        with node.lock:
            frame = node.frame.copy() if node.frame is not None else None

        if frame is not None:
            h, w = frame.shape[:2]
            if w != fw or h != fh or M is None:
                fw, fh = w, h
                K = scale_K(K_NOMINAL, fw, fh)
                M = build_homography(K, CAM_HEIGHT, CAM_PITCH_DEG)

            bev = cv2.warpPerspective(frame, M, (BEV_W, BEV_H))
            bev = draw_grid(bev)

            info = np.zeros((BEV_H, 280, 3), dtype=np.uint8)
            for i, l in enumerate([
                "BEV TUNER  (D435)", "",
                f"pitch : {CAM_PITCH_DEG:+.1f} deg",
                f"height: {CAM_HEIGHT:.3f} m", "",
                "W/S  pitch +-0.5",
                "E/D  pitch +-0.1",
                "R/F  height+-0.01", "",
                "G  print values",
                "Q  quit",
            ]):
                cv2.putText(info, l, (10, 28+i*26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1)

            cv2.imshow('BEV Tuner', np.hstack([bev, info]))

        key = cv2.waitKey(30) & 0xFF
        if   key == ord('q'): break
        elif key == ord('w'): CAM_PITCH_DEG += 0.5;  M = None
        elif key == ord('s'): CAM_PITCH_DEG -= 0.5;  M = None
        elif key == ord('e'): CAM_PITCH_DEG += 0.1;  M = None
        elif key == ord('d'): CAM_PITCH_DEG -= 0.1;  M = None
        elif key == ord('r'): CAM_HEIGHT    += 0.01; M = None
        elif key == ord('f'): CAM_HEIGHT    -= 0.01; M = None
        elif key == ord('g'):
            print("\n── Paste into bev_csi_node.py ─────────────────────────")
            print(f"  self.declare_parameter('cam_height',    {CAM_HEIGHT:.3f})")
            print(f"  self.declare_parameter('cam_pitch_deg', {CAM_PITCH_DEG:.1f})")
            print("────────────────────────────────────────────────────────\n")

        if M is None and fw:
            K = scale_K(K_NOMINAL, fw, fh)
            M = build_homography(K, CAM_HEIGHT, CAM_PITCH_DEG)

    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()