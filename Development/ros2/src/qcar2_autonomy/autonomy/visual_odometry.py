"""Visual Odometry module for QCar2.

Produces an independent pose estimate (x, y, psi) from camera images
that can be compared against Cartographer odometry for redundancy.

FILE LOCATION:
    Development/ros2/src/qcar2_autonomy/autonomy/visual_odometry.py

CAMERA INTRINSICS (from qcar_functions.py InversePerspectiveMapping):
    Calibrated at 640x480: fx=483.671, fy=483.579, cx=321.188, cy=238.462

    About resolution vs intrinsics:
    The PHYSICAL focal length of the lens doesn't change with resolution.
    But fx/fy/cx/cy are measured in PIXELS, so they scale with resolution.
    At 640x480 (virtual default): the stored values match directly.
    At 1280x720 (physical max):  fx*2, cx*2, fy*1.5, cy*1.5.
    Since qcar2_virtual_launch.py sets 640x480, no scaling needed now.

EXTRINSICS (from qcar_functions.py InversePerspectiveMapping):
    phi=pi/2, theta=0, psi=pi/2, height=1.72
    These are Quanser's QLabs coordinates, not meters. We trust them.
"""

import numpy as np
import cv2


# =====================================================================
# GROUND PLANE PROJECTOR — the INVERSE of IPM.v2img()
# =====================================================================

class GroundPlaneProjector:
    """Converts pixel (u,v) to ground-plane (X,Y) or 3D (X,Y,Z) points.

    IPM.v2img() goes:  vehicle 3D -> pixel      [exists in library]
    This class goes:   pixel      -> ground/3D   [NEW - needed for VO]

    Math :
        H = K * [r0 | r1 | t]     (forward homography, ground->pixel)
        G = H^{-1}                 (inverse, pixel->ground)
        For pixel p = [u,v,1]^T:   q = G*p,  X=q[0]/q[2], Y=q[1]/q[2]
    """

    def __init__(self, img_width=640, img_height=480,
                 calib_width=640, calib_height=480):

        # --- Intrinsics from IPM Section B.1 ---
        K_calib = np.array([
            [483.671,       0, 321.188],
            [      0, 483.579, 238.462],
            [      0,       0,       1]
        ])

        sx = img_width / calib_width
        sy = img_height / calib_height
        self.K = np.array([
            [K_calib[0, 0] * sx,                0, K_calib[0, 2] * sx],
            [                 0, K_calib[1, 1] * sy, K_calib[1, 2] * sy],
            [                 0,                  0,                  1]
        ], dtype=np.float64)
        self.K_inv = np.linalg.inv(self.K)

        # --- Extrinsics from IPM Section B.2 ---
        phi, theta, psi = np.pi / 2, 0.0, np.pi / 2
        height = 1.72

        cx_, sx_ = np.cos(phi), np.sin(phi)
        cy_, sy_ = np.cos(theta), np.sin(theta)
        cz_, sz_ = np.cos(psi), np.sin(psi)

        Rx = np.array([[1,    0,     0],
                        [0, cx_,  -sx_],
                        [0, sx_,   cx_]])
        Ry = np.array([[ cy_, 0, sy_],
                        [   0, 1,   0],
                        [-sy_, 0, cy_]])
        Rz = np.array([[cz_, -sz_, 0],
                        [sz_,  cz_, 0],
                        [  0,    0, 1]])

        self.R = Rx @ Ry @ Rz
        self.t = np.array([[0, height, 0]], dtype=np.float64).T

        # Ground-plane homography: H = K * [r0 | r1 | t]
        A = np.column_stack([self.R[:, 0], self.R[:, 1], self.t.flatten()])
        self.H = self.K @ A

        det_H = np.linalg.det(self.H)
        if abs(det_H) < 1e-12:
            raise ValueError(
                f"Homography singular (det={det_H:.2e}). Check extrinsics.")
        self.G = np.linalg.inv(self.H)

    def pixels_to_ground(self, pixels):
        """Nx2 pixels -> Nx2 ground (X,Y). Returns (ground, valid_mask)."""
        N = pixels.shape[0]
        p_hom = np.hstack([pixels.astype(np.float64), np.ones((N, 1))])
        q = self.G @ p_hom.T  # 3xN

        valid = np.abs(q[2, :]) > 1e-10
        ground = np.zeros((N, 2))
        ground[valid, 0] = q[0, valid] / q[2, valid]
        ground[valid, 1] = q[1, valid] / q[2, valid]

        # Gate: reject points way outside feasible road region
        valid &= (ground[:, 0] > -10.0) & (ground[:, 0] < 30.0)
        valid &= (ground[:, 1] > -10.0) & (ground[:, 1] < 10.0)
        return ground, valid

    def pixels_to_3d(self, pixels, depths):
        """Nx2 pixels + N depths (meters) -> Nx3 vehicle-frame points."""
        N = pixels.shape[0]
        valid = depths > 0.01
        points_3d = np.zeros((N, 3))

        if np.any(valid):
            p_hom = np.hstack([pixels[valid].astype(np.float64),
                               np.ones((np.sum(valid), 1))])
            P_cam = depths[valid].reshape(-1, 1) * (self.K_inv @ p_hom.T).T
            R_inv = self.R.T
            P_veh = (R_inv @ (P_cam.T - self.t)).T
            points_3d[valid] = P_veh

        return points_3d, valid


# =====================================================================
# VISUAL ODOMETRY PIPELINE
# =====================================================================

class VisualOdometry:
    """Frame-to-frame VO using ORB features, RANSAC, SVD Procrustes.

    Per frame:
        1. Detect ORB features
        2. Match to previous frame (Lowe's ratio test)
        3. Convert matched pixels to ground/3D points
        4. RANSAC outlier rejection + SVD rigid motion estimation
        5. Accumulate pose [x, y, psi], compute velocity
    """

    def __init__(self, img_width=640, img_height=480, use_depth=True,
                 n_features=500, match_ratio=0.75,
                 ransac_threshold=0.05, min_inliers=8):

        self.projector = GroundPlaneProjector(img_width, img_height)
        self.use_depth = use_depth

        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.match_ratio = match_ratio
        self.ransac_threshold = ransac_threshold
        self.ransac_iterations = 200
        self.min_inliers = min_inliers

        self.pose = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0])
        self.is_initialized = False
        self.inlier_count = 0
        self.confidence = 0.0

        self._prev_kp = None
        self._prev_desc = None
        self._prev_depth = None
        self._prev_time = None

    def update(self, image, timestamp, depth_image=None):
        """Process one frame. Returns dict with pose/velocity/confidence."""
        gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if len(image.shape) == 3 else image.copy())

        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        result = {
            'pose': self.pose.copy(),
            'velocity': self.velocity.copy(),
            'delta_pose': None,
            'inlier_count': 0,
            'confidence': 0.0,
            'valid': False
        }

        if not self.is_initialized:
            self._store(keypoints, descriptors, depth_image, timestamp)
            self.is_initialized = True
            return result

        if (descriptors is None or self._prev_desc is None
                or len(descriptors) < 2 or len(self._prev_desc) < 2):
            self._store(keypoints, descriptors, depth_image, timestamp)
            return result

        # Match
        matches = self._match(self._prev_desc, descriptors)
        if len(matches) < self.min_inliers:
            self._store(keypoints, descriptors, depth_image, timestamp)
            return result

        prev_pts = np.array([self._prev_kp[m.queryIdx].pt for m in matches])
        curr_pts = np.array([keypoints[m.trainIdx].pt for m in matches])

        # Project to ground/3D
        if (self.use_depth and depth_image is not None
                and self._prev_depth is not None):
            prev_g, curr_g, valid = self._to_3d(
                prev_pts, curr_pts, self._prev_depth, depth_image)
        else:
            prev_g, curr_g, valid = self._to_ground(prev_pts, curr_pts)

        if np.sum(valid) < self.min_inliers:
            self._store(keypoints, descriptors, depth_image, timestamp)
            return result

        prev_g, curr_g = prev_g[valid], curr_g[valid]

        # RANSAC + SVD
        dx, dy, dpsi, inlier_mask = self._ransac_motion(prev_g, curr_g)

        self.inlier_count = int(np.sum(inlier_mask))
        self.confidence = (self.inlier_count / len(matches)
                           if len(matches) > 0 else 0.0)

        if self.inlier_count < self.min_inliers:
            self._store(keypoints, descriptors, depth_image, timestamp)
            result['inlier_count'] = self.inlier_count
            result['confidence'] = self.confidence
            return result

        # Accumulate in map frame
        cos_p, sin_p = np.cos(self.pose[2]), np.sin(self.pose[2])
        dx_map = cos_p * dx - sin_p * dy
        dy_map = sin_p * dx + cos_p * dy
        self.pose[0] += dx_map
        self.pose[1] += dy_map
        self.pose[2] = np.arctan2(
            np.sin(self.pose[2] + dpsi),
            np.cos(self.pose[2] + dpsi))

        dt = timestamp - self._prev_time
        if dt > 1e-6:
            self.velocity = np.array([dx_map / dt, dy_map / dt])

        self._store(keypoints, descriptors, depth_image, timestamp)
        result.update({
            'pose': self.pose.copy(),
            'velocity': self.velocity.copy(),
            'delta_pose': np.array([dx, dy, dpsi]),
            'inlier_count': self.inlier_count,
            'confidence': self.confidence,
            'valid': True
        })
        return result

    # ---- internal methods ----

    def _match(self, d1, d2):
        raw = self.matcher.knnMatch(d1, d2, k=2)
        return [m for m, n in raw if len([m, n]) == 2
                and m.distance < self.match_ratio * n.distance]

    def _to_ground(self, prev_pts, curr_pts):
        pg, pv = self.projector.pixels_to_ground(prev_pts)
        cg, cv_ = self.projector.pixels_to_ground(curr_pts)
        return pg, cg, pv & cv_

    def _to_3d(self, prev_pts, curr_pts, prev_depth, curr_depth):
        N = prev_pts.shape[0]
        prev_d, curr_d = np.zeros(N), np.zeros(N)
        for i in range(N):
            pu, pv = int(round(prev_pts[i, 0])), int(round(prev_pts[i, 1]))
            cu, cv_ = int(round(curr_pts[i, 0])), int(round(curr_pts[i, 1]))
            if 0 <= pv < prev_depth.shape[0] and 0 <= pu < prev_depth.shape[1]:
                prev_d[i] = prev_depth[pv, pu] / 1000.0
            if 0 <= cv_ < curr_depth.shape[0] and 0 <= cu < curr_depth.shape[1]:
                curr_d[i] = curr_depth[cv_, cu] / 1000.0

        p3, pv_ = self.projector.pixels_to_3d(prev_pts, prev_d)
        c3, cv__ = self.projector.pixels_to_3d(curr_pts, curr_d)
        both = pv_ & cv__
        return p3[:, :2], c3[:, :2], both

    def _ransac_motion(self, pts_prev, pts_curr):
        M = pts_prev.shape[0]
        best_inliers = np.zeros(M, dtype=bool)
        best_count, best_R, best_t = 0, np.eye(2), np.zeros(2)

        for _ in range(self.ransac_iterations):
            idx = np.random.choice(M, 2, replace=False)
            if np.linalg.norm(pts_prev[idx[0]] - pts_prev[idx[1]]) < 1e-8:
                continue
            R_e, t_e = self._svd_rigid_2d(pts_prev[idx], pts_curr[idx])
            res = np.linalg.norm(pts_curr - (R_e @ pts_prev.T).T - t_e, axis=1)
            inl = res < self.ransac_threshold
            c = np.sum(inl)
            if c > best_count:
                best_count, best_inliers, best_R, best_t = c, inl, R_e, t_e

        if best_count >= 2:
            best_R, best_t = self._svd_rigid_2d(
                pts_prev[best_inliers], pts_curr[best_inliers])

        dpsi = np.arctan2(best_R[1, 0], best_R[0, 0])
        return best_t[0], best_t[1], dpsi, best_inliers

    @staticmethod
    def _svd_rigid_2d(pts_a, pts_b):
        ca, cb = np.mean(pts_a, axis=0), np.mean(pts_b, axis=0)
        H = (pts_a - ca).T @ (pts_b - cb)
        U, _, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        R = Vt.T @ np.diag([1.0, np.sign(d)]) @ U.T
        return R, cb - R @ ca

    def _store(self, kp, desc, depth, ts):
        self._prev_kp, self._prev_desc = kp, desc
        self._prev_depth, self._prev_time = depth, ts

    def reset(self, x=0.0, y=0.0, psi=0.0):
        self.pose = np.array([x, y, psi])
        self.velocity = np.array([0.0, 0.0])
        self.is_initialized = False
        self._prev_kp = self._prev_desc = None
        self._prev_depth = self._prev_time = None
        self.inlier_count = 0
        self.confidence = 0.0