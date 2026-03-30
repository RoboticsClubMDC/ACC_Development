"""Depth-based visual odometry for QCar2.

This module estimates planar frame-to-frame motion `(dx, dy, dpsi)` from
color features and depth backprojection.

Calibration sources intentionally referenced in this file:
- Virtual D435 intrinsics: `Virtual_Stage_ROS_FAQ.md`
- Physical post-calibration intrinsics: `Camera Intrinsics Post.txt`
- Camera-to-body extrinsics: `user_manual_system_hardware.pdf`
- Virtual depth-alignment homography: legacy Quanser/PIT utility code

Several legacy fallback constants remain in the file for compatibility. The
mode-specific virtual and physical calibration tables are the intended source
of truth for current development.
"""

import numpy as np
import cv2


# =====================================================================
# DEPTH-BASED 3D PROJECTOR
# =====================================================================

class DepthProjector:
    """Converts pixel (u,v) + depth to 3D vehicle-frame coordinates.

    Pipeline:
        1. Warp depth image using alignment matrix M (depth → color pixel space)
        2. Look up aligned depth at color pixel (u,v)
        3. Backproject using the active intrinsics:
           - RGB K_inv when depth has been aligned to color space
           - Depth K_inv when alignment is disabled as a raw-depth fallback
           P_cam = depth * K_inv @ [u, v, 1]^T
        4. Transform camera → vehicle body frame:
           P_body = T_cam2body @ [P_cam; 1]
        5. Use (P_body.x, P_body.y) for 2D motion estimation

    This produces correct 3D positions for features at ANY height,
    not just Z=0 ground plane.
    """

    # =================================================================
    # MODE-SPECIFIC CAMERA PARAMETERS (640x480 resolution)
    # =================================================================
    #
    # Mode-specific calibration tables at 640x480.
    # Virtual D435 values are taken from Virtual_Stage_ROS_FAQ.md.
    # Physical D435 values are taken from Camera Intrinsics Post.txt.
    # =================================================================

    # --- VIRTUAL MODE (Virtual_Stage_ROS_FAQ.md) ---
    # D435 RGB camera intrinsic matrix at resolution [640, 480]:
    # [[455.2    0.00  308.53]
    #  [  0.00  459.43  213.56]
    #  [  0.00    0.00    1.00]]
    VIRTUAL_FX_RGB = 455.2
    VIRTUAL_FY_RGB = 459.43
    VIRTUAL_CX_RGB = 308.53
    VIRTUAL_CY_RGB = 213.56

    # D435 Depth camera intrinsic matrix at resolution [640, 480]:
    # [[385.6    0.00  321.9]
    #  [  0.0   385.6  237.3]
    #  [  0.0     0.0    1.0]]
    VIRTUAL_FX_DEPTH = 385.6
    VIRTUAL_FY_DEPTH = 385.6
    VIRTUAL_CX_DEPTH = 321.9
    VIRTUAL_CY_DEPTH = 237.3

    # Virtual depth scale: UINT16 → QLabs units
    # Empirically verified: raw=21760 at 1.3855 QLabs units → 21760/1.3855 = 15707
    # This is QLabs-specific. Physical mode uses different scale.
    VIRTUAL_DEPTH_SCALE_UINT16 = 15707.0

    # Virtual extrinsics (QLabs 10x physical meters):
    # Physical camera height: 0.172m → QLabs: 1.72
    # t = [0.95, 0.32, 1.72] (10x physical)
    VIRTUAL_T_CAM2BODY = np.array([
        [ 0,  0,  1,  0.95],
        [-1,  0,  0,  0.32],
        [ 0, -1,  0,  1.72],
        [ 0,  0,  0,  1.00]
    ], dtype=np.float64)

    # Virtual alignment matrix M (from PIT/YOLO utils.py lines 155-157)
    # Warps depth → color pixel space for virtual mode
    # Only used in virtual mode; physical has different alignment
    VIRTUAL_ALIGNMENT_M = np.array([
        [1.440749898, -6.45417e-16, -126.2303115],
        [-0.000294167,  1.445138872, -106.3509378],
        [-2.24559e-06, -1.62662e-18,  1.0        ]
    ], dtype=np.float64)

    # --- PHYSICAL MODE (Camera Intrinsics Post.txt - calibrated) ---
    # Color 640x480 (post-calibration, no distortion):
    # fx = 607.327270507812, fy = 607.345092773438
    # cx = 324.950225830078, cy = 249.868499755859
    PHYSICAL_FX_RGB = 607.327270507812
    PHYSICAL_FY_RGB = 607.345092773438
    PHYSICAL_CX_RGB = 324.950225830078
    PHYSICAL_CY_RGB = 249.868499755859

    # Depth 640x480 (post-calibration, no distortion):
    # fx = 384.758270263672, fy = 384.758270263672
    # cx = 324.023345947266, cy = 237.567321777344
    PHYSICAL_FX_DEPTH = 384.758270263672
    PHYSICAL_FY_DEPTH = 384.758270263672
    PHYSICAL_CX_DEPTH = 324.023345947266
    PHYSICAL_CY_DEPTH = 237.567321777344

    # Placeholder retained in code until the physical MONO16 path is
    # measured directly in this ROS pipeline.
    PHYSICAL_DEPTH_SCALE_UINT16 = 1000.0

    # Physical extrinsics (actual meters from hardware manual):
    # t = [0.095, 0.032, 0.172] meters
    PHYSICAL_T_CAM2BODY = np.array([
        [ 0,  0,  1,  0.095],
        [-1,  0,  0,  0.032],
        [ 0, -1,  0,  0.172],
        [ 0,  0,  0,  1.00]
    ], dtype=np.float64)

    # Placeholder retained in code until the physical RGB-depth alignment
    # is implemented and validated for this ROS pipeline.
    PHYSICAL_ALIGNMENT_M = np.eye(3, dtype=np.float64)

    # Backwards-compatibility aliases still point to the virtual table.
    # Runtime mode selection below now chooses the full physical/virtual stack
    # instead of letting physical mode silently inherit these virtual defaults.
    DEFAULT_FX_RGB = VIRTUAL_FX_RGB
    DEFAULT_FY_RGB = VIRTUAL_FY_RGB
    DEFAULT_CX_RGB = VIRTUAL_CX_RGB
    DEFAULT_CY_RGB = VIRTUAL_CY_RGB
    DEFAULT_FX_DEPTH = VIRTUAL_FX_DEPTH
    DEFAULT_FY_DEPTH = VIRTUAL_FY_DEPTH
    DEFAULT_CX_DEPTH = VIRTUAL_CX_DEPTH
    DEFAULT_CY_DEPTH = VIRTUAL_CY_DEPTH
    DEFAULT_DEPTH_SCALE_UINT16 = VIRTUAL_DEPTH_SCALE_UINT16
    DEFAULT_T_CAM2BODY = VIRTUAL_T_CAM2BODY
    DEFAULT_ALIGNMENT_M = VIRTUAL_ALIGNMENT_M

    # For UINT8 data from PAL Camera3D (if someone reads depth directly):
    #   depth_qlabs = raw_uint8 / DEPTH_SCALE_UINT8
    # Source: utils.py line 162 (Quanser code, validated)
    DEFAULT_DEPTH_SCALE_UINT8 = 5.5

    def __init__(self, img_width=640, img_height=480,
                 fx_rgb=None, fy_rgb=None, cx_rgb=None, cy_rgb=None,
                 depth_scale=None,
                 alignment_M=None,
                 T_cam2body=None,
                 use_alignment=None,
                 camera_mode='virtual',
                 # Distortion: disabled for QLabs (virtual pinhole camera)
                 # Enable + provide coefficients for physical D435 after
                 # running checkerboard calibration with image_processing.py
                 use_undistortion=False,
                 dist_coeffs=None):
        """Initialize the depth projector.

        Args:
            img_width, img_height: Image resolution (default 640x480 for QLabs)
            fx_rgb, fy_rgb, cx_rgb, cy_rgb: RGB intrinsics override
            depth_scale: Raw UINT16 divisor override. If omitted, uses the
                mode-specific default for virtual or physical runs.
            alignment_M: 3x3 depth-to-color alignment homography
            T_cam2body: 4x4 camera-to-body extrinsic transform
            use_alignment: If True, warp depth before lookup. If omitted, uses
                the mode-specific default (`True` virtual, `False` physical).
            camera_mode: 'virtual' or 'physical' - selects intrinsics defaults
                - 'virtual': Uses FAQ.md intrinsics (fx=455.2, fy=459.43, cx=308.53, cy=213.56)
                - 'physical': Uses calibrated intrinsics from Camera Intrinsics Post.txt
                              (fx=607.327, fy=607.345, cx=324.95, cy=249.868)
            use_undistortion: If True, undistort color image before processing
            dist_coeffs: np.array([k1, k2, p1, p2, k3]) for cv2.undistort
                For QLabs: not needed (no distortion)
                For physical D435: calibrate with checkerboard first
                    Tool: image_processing.py calibrate_camera()
        """
        self.camera_mode = 'physical' if camera_mode == 'physical' else 'virtual'

        # Keep the geometry tables together so a mode switch updates the full
        # stack: RGB intrinsics, depth intrinsics, depth units, alignment, and
        # camera-to-body extrinsics.
        if self.camera_mode == 'physical':
            mode_defaults = {
                'fx_rgb': self.PHYSICAL_FX_RGB,
                'fy_rgb': self.PHYSICAL_FY_RGB,
                'cx_rgb': self.PHYSICAL_CX_RGB,
                'cy_rgb': self.PHYSICAL_CY_RGB,
                'fx_depth': self.PHYSICAL_FX_DEPTH,
                'fy_depth': self.PHYSICAL_FY_DEPTH,
                'cx_depth': self.PHYSICAL_CX_DEPTH,
                'cy_depth': self.PHYSICAL_CY_DEPTH,
                'depth_scale': self.PHYSICAL_DEPTH_SCALE_UINT16,
                'alignment_M': self.PHYSICAL_ALIGNMENT_M,
                'T_cam2body': self.PHYSICAL_T_CAM2BODY,
                'depth_min': 0.01,
                'depth_max': 8.0,
                'depth_units_label': 'meters',
                'physical_m_per_unit': 1.0,
                # Until physical RGB-depth alignment is validated, default to
                # the raw depth grid instead of pretending the placeholder
                # identity homography is a real registration.
                'use_alignment': False,
            }
        else:
            mode_defaults = {
                'fx_rgb': self.VIRTUAL_FX_RGB,
                'fy_rgb': self.VIRTUAL_FY_RGB,
                'cx_rgb': self.VIRTUAL_CX_RGB,
                'cy_rgb': self.VIRTUAL_CY_RGB,
                'fx_depth': self.VIRTUAL_FX_DEPTH,
                'fy_depth': self.VIRTUAL_FY_DEPTH,
                'cx_depth': self.VIRTUAL_CX_DEPTH,
                'cy_depth': self.VIRTUAL_CY_DEPTH,
                'depth_scale': self.VIRTUAL_DEPTH_SCALE_UINT16,
                'alignment_M': self.VIRTUAL_ALIGNMENT_M,
                'T_cam2body': self.VIRTUAL_T_CAM2BODY,
                'depth_min': 0.1,
                'depth_max': 80.0,
                'depth_units_label': 'QLabs units',
                'physical_m_per_unit': 0.1,
                'use_alignment': True,
            }

        # RGB intrinsics: mode-specific defaults unless explicitly overridden.
        _fx = fx_rgb if fx_rgb is not None else mode_defaults['fx_rgb']
        _fy = fy_rgb if fy_rgb is not None else mode_defaults['fy_rgb']
        _cx = cx_rgb if cx_rgb is not None else mode_defaults['cx_rgb']
        _cy = cy_rgb if cy_rgb is not None else mode_defaults['cy_rgb']

        self.K_rgb = np.array([
            [_fx,   0, _cx],
            [  0, _fy, _cy],
            [  0,   0,   1]
        ], dtype=np.float64)
        self.K_rgb_inv = np.linalg.inv(self.K_rgb)

        # Store the raw depth camera intrinsics as well. They become the
        # fallback projection model when alignment is disabled.
        self.K_depth = np.array([
            [mode_defaults['fx_depth'], 0, mode_defaults['cx_depth']],
            [0, mode_defaults['fy_depth'], mode_defaults['cy_depth']],
            [0, 0, 1]
        ], dtype=np.float64)
        self.K_depth_inv = np.linalg.inv(self.K_depth)

        # Depth scale
        self.depth_scale = depth_scale if depth_scale is not None \
            else mode_defaults['depth_scale']

        # Alignment matrix
        self.use_alignment = (mode_defaults['use_alignment']
                              if use_alignment is None
                              else bool(use_alignment))
        self.alignment_M = alignment_M if alignment_M is not None \
            else mode_defaults['alignment_M'].copy()

        # Extrinsics: camera → body
        self.T_cam2body = T_cam2body if T_cam2body is not None \
            else mode_defaults['T_cam2body'].copy()
        # Extract R and t for efficient computation
        self.R_cam2body = self.T_cam2body[:3, :3]
        self.t_cam2body = self.T_cam2body[:3, 3]

        # Expose the active unit system for diagnostics in vo_node.py.
        self.depth_units_label = mode_defaults['depth_units_label']
        self.physical_m_per_unit = mode_defaults['physical_m_per_unit']

        # Distortion (disabled by default for QLabs)
        self.use_undistortion = use_undistortion
        self.dist_coeffs = dist_coeffs
        # Pre-compute undistortion maps if needed
        self._undistort_map1 = None
        self._undistort_map2 = None
        if self.use_undistortion and self.dist_coeffs is not None:
            self._undistort_map1, self._undistort_map2 = cv2.initUndistortRectifyMap(
                self.K_rgb, self.dist_coeffs, None, self.K_rgb,
                (img_width, img_height), cv2.CV_32FC1
            )

        # Depth range gates in the active mode's distance units.
        # Virtual mode uses QLabs units; physical mode uses meters.
        self.depth_min = mode_defaults['depth_min']
        self.depth_max = mode_defaults['depth_max']

    def undistort_image(self, image):
        """Undistort a color image using pre-computed maps.

        For QLabs: returns image unchanged (no distortion in simulation).
        For physical: applies lens distortion correction.
        Calibrate first with image_processing.calibrate_camera().
        """
        if not self.use_undistortion or self._undistort_map1 is None:
            return image
        return cv2.remap(image, self._undistort_map1, self._undistort_map2,
                         cv2.INTER_LINEAR)

    def align_depth(self, depth_raw):
        """Warp raw depth image to align with color image pixel grid.

        Input: depth_raw — (H, W) uint16 array from /camera/depth_image
        Output: depth_aligned — (H, W) float64, in active mode units

        The alignment matrix M accounts for:
        - Different FOVs (depth 87° vs RGB 69.4°) → ~1.44x zoom
        - Physical sensor offset (~50mm baseline in D435 housing)

        After alignment, depth_aligned[v, u] corresponds to color[v, u].
        Source of M: utils.py (PIT/YOLO library) lines 163-165
        """
        
        # Guard against empty or None depth frames
        if depth_raw is None or depth_raw.size == 0:
            return np.zeros((480, 640), dtype=np.float64)

        # Ensure 2D array (H, W), not (H, W, 1)
        if len(depth_raw.shape) == 3:
            depth_raw = depth_raw[:, :, 0]

        # Ensure contiguous memory layout for OpenCV warpPerspective
        depth_raw = np.ascontiguousarray(depth_raw)
        
        # Convert raw counts to the active mode units FIRST, then warp. This
        # keeps interpolation in distance space instead of raw integer space.
        depth_units = depth_raw.astype(np.float64) / self.depth_scale

        if self.use_alignment:
            depth_aligned = cv2.warpPerspective(
                depth_units, self.alignment_M, (depth_units.shape[1], depth_units.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0
            )
            return depth_aligned
        else:
            return depth_units

    def pixels_to_3d_body(self, pixels, aligned_depth):
        """Convert Nx2 pixel coords + aligned depth image → Nx3 body-frame points.

        Args:
            pixels: Nx2 array of (u, v) pixel coordinates in COLOR image
            aligned_depth: (H, W) depth image in the active mode units.
                           If alignment is enabled it is on the color grid.
                           If alignment is disabled it remains on the raw depth
                           grid and is treated as a fallback approximation.

        Returns:
            points_body: Nx3 array of (X, Y, Z) in vehicle body frame
                X = forward, Y = left, Z = up
            valid: N boolean mask (True where depth is usable)

        Math:
            1. Look up depth d at pixel (u, v) in aligned depth image
            2. Backproject to camera 3D:
               P_cam = d * K_inv @ [u, v, 1]^T
               (K_rgb when aligned, K_depth when using raw depth fallback)
            3. Transform to body frame:
               P_body = R_cam2body @ P_cam + t_cam2body
        """
        N = pixels.shape[0]
        points_body = np.zeros((N, 3), dtype=np.float64)

        # Sample depth at each pixel location
        depths = np.zeros(N, dtype=np.float64)
        H, W = aligned_depth.shape[:2]
        for i in range(N):
            u, v = int(round(pixels[i, 0])), int(round(pixels[i, 1]))
            if 0 <= v < H and 0 <= u < W:
                depths[i] = aligned_depth[v, u]

        # Valid depth mask
        valid = (depths > self.depth_min) & (depths < self.depth_max)

        if not np.any(valid):
            return points_body, valid

        # Backproject valid pixels to camera 3D
        pix_valid = pixels[valid].astype(np.float64)
        d_valid = depths[valid]

        # Homogeneous pixel coords: [u, v, 1]
        p_hom = np.hstack([pix_valid, np.ones((pix_valid.shape[0], 1))])

        # Camera-frame 3D: P_cam = depth * K_inv @ [u, v, 1]^T
        # If depth has been aligned to the color grid, use RGB intrinsics.
        # If alignment is disabled, fall back to the depth intrinsics so the
        # raw depth grid is interpreted with the correct pinhole model.
        K_inv = self.K_rgb_inv if self.use_alignment else self.K_depth_inv
        rays = (K_inv @ p_hom.T).T  # Nx3 direction vectors
        P_cam = d_valid.reshape(-1, 1) * rays   # Nx3 camera-frame points

        # Transform to body frame: P_body = R @ P_cam + t
        P_body = (self.R_cam2body @ P_cam.T).T + self.t_cam2body

        points_body[valid] = P_body
        return points_body, valid


# =====================================================================
# VISUAL ODOMETRY PIPELINE — depth-based
# =====================================================================

class VisualOdometryDepth:
    """Frame-to-frame VO using ORB features, depth backprojection,
    RANSAC outlier rejection, and SVD Procrustes rigid-body fitting.

    Per frame:
        1. Detect ORB features in grayscale color image
        2. Match to previous frame (brute-force + Lowe's ratio test)
        3. Align depth image to color space via alignment matrix M
        4. Backproject matched pixels to 3D body-frame coordinates using depth
        5. Extract (X_body, Y_body) for 2D ground-plane motion
        6. 2-point RANSAC + SVD Procrustes to find (dx, dy, dpsi)
        7. Negate (feature motion → vehicle motion)
        8. Accumulate pose [x, y, psi] in map frame

    Key difference from homography-only:
        - Features on walls/signs contribute rotation signal (not Z=0 only)
        - Depth provides true 3D geometry, improving both translation and heading
        - Requires depth_scale and alignment_M to be correct
    """

    def __init__(self, img_width=640, img_height=480,
                 fx_rgb=None, fy_rgb=None, cx_rgb=None, cy_rgb=None,
                 depth_scale=None,
                 alignment_M=None,
                 T_cam2body=None,
                 use_alignment=None,
                 camera_mode='virtual',
                 use_undistortion=False,
                 dist_coeffs=None,
                 n_features=800, match_ratio=0.75,
                 ransac_threshold=0.05, ransac_iterations=300,
                 min_inliers=8,
                 max_translation=0.20, max_rotation_deg=15.0,
                 max_dt=0.20,
                 negate_deltas=True):
        """Initialize Visual Odometry with depth-based 3D backprojection.

        Args:
            camera_mode: 'virtual' or 'physical' - selects intrinsics defaults
                - 'virtual': Uses FAQ.md intrinsics for QLabs simulation
                - 'physical': Uses calibrated intrinsics from physical QCar2
        """

        self.projector = DepthProjector(
            img_width, img_height,
            fx_rgb=fx_rgb, fy_rgb=fy_rgb,
            cx_rgb=cx_rgb, cy_rgb=cy_rgb,
            depth_scale=depth_scale,
            alignment_M=alignment_M,
            T_cam2body=T_cam2body,
            use_alignment=use_alignment,
            camera_mode=camera_mode,
            use_undistortion=use_undistortion,
            dist_coeffs=dist_coeffs,
        )

        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.match_ratio = match_ratio
        self.ransac_threshold = ransac_threshold
        self.ransac_iterations = ransac_iterations
        self.min_inliers = min_inliers
        self.max_translation = max_translation
        self.max_rotation = np.radians(max_rotation_deg)
        self.max_dt = max_dt
        self.negate_deltas = negate_deltas

        # State
        self.pose = np.array([0.0, 0.0, 0.0])  # [x, y, psi] in map frame
        self.velocity = np.array([0.0, 0.0])
        self.is_initialized = False
        self.inlier_count = 0
        self.confidence = 0.0
        self.inlier_spread = 0.0

        # Previous frame storage
        self._prev_kp = None
        self._prev_desc = None
        self._prev_depth = None  # aligned depth from previous frame
        self._prev_time = None

    def update(self, color_image, depth_raw, timestamp):
        """Process one frame with color + depth.

        Args:
            color_image: BGR uint8 image from /camera/color_image
            depth_raw: uint16 depth image from /camera/depth_image (MONO16)
                       This is the RAW image — alignment and scaling are
                       applied internally.
            timestamp: float, seconds

        Returns:
            dict with pose, velocity, delta_pose, inlier_count,
            confidence, inlier_spread, valid
        """
        # Optional undistortion (disabled for QLabs, enable for physical)
        color_image = self.projector.undistort_image(color_image)

        gray = (cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                if len(color_image.shape) == 3 else color_image.copy())

        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        # Align depth to color pixel space and convert to QLabs units
        aligned_depth = self.projector.align_depth(depth_raw)

        result = {
            'pose': self.pose.copy(),
            'velocity': self.velocity.copy(),
            'delta_pose': None,
            'inlier_count': 0,
            'confidence': 0.0,
            'inlier_spread': 0.0,
            'valid': False
        }

        # First frame: store and return
        if not self.is_initialized:
            self._store(keypoints, descriptors, aligned_depth, timestamp)
            self.is_initialized = True
            return result

        # Need descriptors in both frames
        if (descriptors is None or self._prev_desc is None
                or len(descriptors) < 2 or len(self._prev_desc) < 2):
            self._store(keypoints, descriptors, aligned_depth, timestamp)
            return result

        # --- Match features ---
        matches = self._match(self._prev_desc, descriptors)
        if len(matches) < self.min_inliers:
            self._store(keypoints, descriptors, aligned_depth, timestamp)
            return result

        prev_pts = np.array([self._prev_kp[m.queryIdx].pt for m in matches])
        curr_pts = np.array([keypoints[m.trainIdx].pt for m in matches])

        # --- Backproject matched pixels to 3D body-frame ---
        # Previous frame features use previous aligned depth
        prev_3d, prev_valid = self.projector.pixels_to_3d_body(
            prev_pts, self._prev_depth)
        # Current frame features use current aligned depth
        curr_3d, curr_valid = self.projector.pixels_to_3d_body(
            curr_pts, aligned_depth)

        # Both must have valid depth
        both_valid = prev_valid & curr_valid

        if np.sum(both_valid) < self.min_inliers:
            self._store(keypoints, descriptors, aligned_depth, timestamp)
            return result

        # Extract 2D body-frame coordinates (X=forward, Y=left)
        # Features at any height contribute — this is the key advantage
        # over the homography path which constrains everything to Z=0.
        prev_2d = prev_3d[both_valid, :2]  # (X, Y) in body frame
        curr_2d = curr_3d[both_valid, :2]
        valid_prev_px = prev_pts[both_valid]  # pixel coords for spread

        # --- RANSAC + SVD ---
        dx, dy, dpsi, inlier_mask = self._ransac_motion(prev_2d, curr_2d)

        self.inlier_count = int(np.sum(inlier_mask))
        self.confidence = (self.inlier_count / len(matches)
                           if len(matches) > 0 else 0.0)

        # Inlier pixel spread (quality metric)
        if self.inlier_count >= 2:
            inlier_px = valid_prev_px[inlier_mask]
            std_x = np.std(inlier_px[:, 0])
            std_y = np.std(inlier_px[:, 1])
            self.inlier_spread = np.sqrt(max(std_x, 1e-6) * max(std_y, 1e-6))
        else:
            self.inlier_spread = 0.0

        if self.inlier_count < self.min_inliers:
            self._store(keypoints, descriptors, aligned_depth, timestamp)
            result['inlier_count'] = self.inlier_count
            result['confidence'] = self.confidence
            result['inlier_spread'] = self.inlier_spread
            return result

        # --- Safety gates ---
        dt = timestamp - self._prev_time if self._prev_time else 0.0
        trans = np.sqrt(dx**2 + dy**2)
        if (trans > self.max_translation
                or abs(dpsi) > self.max_rotation
                or dt > self.max_dt):
            self._store(keypoints, descriptors, aligned_depth, timestamp)
            result['inlier_count'] = self.inlier_count
            result['confidence'] = self.confidence
            result['inlier_spread'] = self.inlier_spread
            return result

        # --- Negate: SVD gives feature motion, we need vehicle motion ---
        # When the car moves forward, features move backward in body frame.
        # SVD finds the transform that maps previous features to current,
        # which is the NEGATIVE of vehicle displacement.
        if self.negate_deltas:
            dx, dy, dpsi = -dx, -dy, -dpsi

        # --- Accumulate pose in map frame ---
        cos_p, sin_p = np.cos(self.pose[2]), np.sin(self.pose[2])
        dx_map = cos_p * dx - sin_p * dy
        dy_map = sin_p * dx + cos_p * dy
        self.pose[0] += dx_map
        self.pose[1] += dy_map
        self.pose[2] = np.arctan2(
            np.sin(self.pose[2] + dpsi),
            np.cos(self.pose[2] + dpsi))

        if dt > 1e-6:
            self.velocity = np.array([dx_map / dt, dy_map / dt])

        self._store(keypoints, descriptors, aligned_depth, timestamp)
        result.update({
            'pose': self.pose.copy(),
            'velocity': self.velocity.copy(),
            'delta_pose': np.array([dx, dy, dpsi]),
            'inlier_count': self.inlier_count,
            'confidence': self.confidence,
            'inlier_spread': self.inlier_spread,
            'valid': True
        })
        return result

    # ---- Internal methods ----

    def _match(self, d1, d2):
        """Brute-force k=2 matching + Lowe's ratio test."""
        raw = self.matcher.knnMatch(d1, d2, k=2)
        return [m for m, n in raw if len([m, n]) == 2
                and m.distance < self.match_ratio * n.distance]

    def _ransac_motion(self, pts_prev, pts_curr):
        """2-point RANSAC with SVD Procrustes on body-frame 2D points."""
        M = pts_prev.shape[0]
        best_inliers = np.zeros(M, dtype=bool)
        best_count, best_R, best_t = 0, np.eye(2), np.zeros(2)

        for _ in range(self.ransac_iterations):
            idx = np.random.choice(M, 2, replace=False)
            if np.linalg.norm(pts_prev[idx[0]] - pts_prev[idx[1]]) < 1e-8:
                continue
            R_e, t_e = self._svd_rigid_2d(pts_prev[idx], pts_curr[idx])
            res = np.linalg.norm(
                pts_curr - (R_e @ pts_prev.T).T - t_e, axis=1)
            inl = res < self.ransac_threshold
            c = np.sum(inl)
            if c > best_count:
                best_count, best_inliers, best_R, best_t = c, inl, R_e, t_e

        # Refit on all inliers
        if best_count >= 2:
            best_R, best_t = self._svd_rigid_2d(
                pts_prev[best_inliers], pts_curr[best_inliers])

        dpsi = np.arctan2(best_R[1, 0], best_R[0, 0])
        return best_t[0], best_t[1], dpsi, best_inliers

    @staticmethod
    def _svd_rigid_2d(pts_a, pts_b):
        """SVD Procrustes: find R, t such that pts_b ≈ R @ pts_a + t."""
        ca = np.mean(pts_a, axis=0)
        cb = np.mean(pts_b, axis=0)
        S = (pts_a - ca).T @ (pts_b - cb)
        U, _, Vt = np.linalg.svd(S)
        d = np.linalg.det(Vt.T @ U.T)
        R = Vt.T @ np.diag([1.0, np.sign(d)]) @ U.T
        t = cb - R @ ca
        return R, t

    def _store(self, kp, desc, aligned_depth, ts):
        self._prev_kp = kp
        self._prev_desc = desc
        self._prev_depth = aligned_depth
        self._prev_time = ts

    def reset(self, x=0.0, y=0.0, psi=0.0):
        """Full reset: clear pose and feature history."""
        self.pose = np.array([x, y, psi])
        self.velocity = np.array([0.0, 0.0])
        self.is_initialized = False
        self._prev_kp = self._prev_desc = self._prev_depth = None
        self._prev_time = None
        self.inlier_count = 0
        self.confidence = 0.0
        self.inlier_spread = 0.0

    def soft_reset(self, x, y, psi):
        """Re-anchor pose without clearing feature history."""
        self.pose = np.array([x, y, psi])
