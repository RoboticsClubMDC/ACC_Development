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
        1. Align depth to color pixels
           - virtual: homography warp
           - physical: projective depth->color reprojection
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

    # Physical depth->color extrinsics from RealSense calibration snapshot
    # (Depth -> Color, 640x480 profile family):
    # Rotation:
    #   0.999925   0.0122108   1.18136e-05
    #  -0.0122108  0.999925    0.000544984
    #  -5.15798e-06 -0.000545088 1
    # Translation (meters):
    #   [0.0148963882, -0.0002293478, 0.0004778]
    PHYSICAL_T_DEPTH_TO_COLOR = np.array([
        [0.999925,   0.0122108,   1.18136e-05,   0.0148963881656528],
        [-0.0122108, 0.999925,    0.000544984, -0.000229347773711197],
        [-5.15798e-06, -0.000545088, 1.0,       0.000477800000226125],
        [0.0,        0.0,         0.0,          1.0]
    ], dtype=np.float64)

    # Legacy homography placeholder kept only for backwards compatibility.
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
                 depth_on_color_grid=None,
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
                (used by virtual mode)
            T_cam2body: 4x4 camera-to-body extrinsic transform
            use_alignment: If True, warp depth before lookup. If omitted, uses
                the mode-specific default (`True` virtual, `False` physical).
            depth_on_color_grid: True if depth pixels correspond to color
                pixels (after software warp OR when the upstream stream is
                already aligned, e.g. PIT QCar2DepthAligned). False if depth
                is on the raw depth-sensor grid. Defaults to `use_alignment`.
                This flag governs which intrinsics K_inv is used in
                `pixels_to_3d_body` — RGB K when on color grid, depth K
                otherwise.
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
                # Physical mode defaults to alignment OFF until projective
                # depth->color registration is explicitly requested at runtime.
                'use_alignment': False,
                'alignment_model': 'projective',
                'T_depth_to_color': self.PHYSICAL_T_DEPTH_TO_COLOR,
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
                'alignment_model': 'homography',
                'T_depth_to_color': np.eye(4, dtype=np.float64),
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
        # Whether depth values currently live on the color pixel grid.
        # Default mirrors use_alignment (legacy behavior). PIT/aligned inputs
        # set this True even though no software warp is performed.
        self.depth_on_color_grid = (self.use_alignment
                                    if depth_on_color_grid is None
                                    else bool(depth_on_color_grid))
        self.alignment_M = alignment_M if alignment_M is not None \
            else mode_defaults['alignment_M'].copy()
        self.alignment_model = mode_defaults.get('alignment_model', 'homography')
        self.T_depth_to_color = mode_defaults.get(
            'T_depth_to_color', np.eye(4, dtype=np.float64)).copy()
        self.R_depth_to_color = self.T_depth_to_color[:3, :3]
        self.t_depth_to_color = self.T_depth_to_color[:3, 3]

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

        # Cached depth pixel grid for projective alignment.
        self._grid_shape = None
        self._grid_u = None
        self._grid_v = None

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
        """Align raw depth image to color image pixel grid.

        Input:
            depth_raw — depth image from camera pipeline:
                - uint16 raw counts (e.g., ROS MONO16), or
                - float depth already in active mode units (e.g., meters)
        Output: depth_aligned — (H, W) float64, in active mode units

        Alignment behavior:
        - virtual mode: homography warp (`alignment_M`)
        - physical mode: projective depth->color reprojection using
          depth intrinsics + depth->color extrinsics

        After alignment, `depth_aligned[v, u]` corresponds to color `(u, v)`.
        """
        
        # Guard against empty or None depth frames
        if depth_raw is None or depth_raw.size == 0:
            return np.zeros((480, 640), dtype=np.float64)

        # Ensure 2D array (H, W), not (H, W, 1)
        if len(depth_raw.shape) == 3:
            depth_raw = depth_raw[:, :, 0]

        # Ensure contiguous memory layout for OpenCV warpPerspective
        depth_raw = np.ascontiguousarray(depth_raw)
        
        # Convert raw counts to active units for integer inputs.
        # If the input is already floating-point depth (meters/units), keep it
        # as-is to avoid a second scaling pass.
        if np.issubdtype(depth_raw.dtype, np.floating):
            depth_units = depth_raw.astype(np.float64)
        else:
            # Raw integer path (MONO16, etc.): divide by configured scale.
            depth_units = depth_raw.astype(np.float64) / self.depth_scale

        if not self.use_alignment:
            return depth_units

        if self.alignment_model == 'projective':
            return self._align_depth_projective(depth_units)

        depth_aligned = cv2.warpPerspective(
            depth_units, self.alignment_M, (depth_units.shape[1], depth_units.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0
        )
        return depth_aligned

    def _ensure_depth_grid(self, H, W):
        """Cache flattened (u, v) depth pixel coordinates."""
        if self._grid_shape == (H, W):
            return
        u = np.arange(W, dtype=np.float64)
        v = np.arange(H, dtype=np.float64)
        uu, vv = np.meshgrid(u, v)
        self._grid_u = uu.ravel()
        self._grid_v = vv.ravel()
        self._grid_shape = (H, W)

    def _align_depth_projective(self, depth_units):
        """Align depth to color pixels via depth->color 3D reprojection."""
        H, W = depth_units.shape[:2]
        aligned = np.zeros((H, W), dtype=np.float64)
        valid = (depth_units > self.depth_min) & (depth_units < self.depth_max)
        if not np.any(valid):
            return aligned

        self._ensure_depth_grid(H, W)
        z_flat = depth_units.ravel()
        valid_flat = valid.ravel()

        u_d = self._grid_u[valid_flat]
        v_d = self._grid_v[valid_flat]
        z_d = z_flat[valid_flat]

        fx_d = self.K_depth[0, 0]
        fy_d = self.K_depth[1, 1]
        cx_d = self.K_depth[0, 2]
        cy_d = self.K_depth[1, 2]

        # Backproject depth pixels into depth-camera 3D.
        x_d = (u_d - cx_d) * z_d / fx_d
        y_d = (v_d - cy_d) * z_d / fy_d
        pts_d = np.stack([x_d, y_d, z_d], axis=1)

        # Transform depth-camera points into color-camera coordinates.
        pts_c = pts_d @ self.R_depth_to_color.T + self.t_depth_to_color
        z_c = pts_c[:, 2]
        valid_z = (z_c > self.depth_min) & (z_c < self.depth_max)
        if not np.any(valid_z):
            return aligned

        pts_c = pts_c[valid_z]
        z_c = z_c[valid_z]

        fx_c = self.K_rgb[0, 0]
        fy_c = self.K_rgb[1, 1]
        cx_c = self.K_rgb[0, 2]
        cy_c = self.K_rgb[1, 2]

        u_c = fx_c * pts_c[:, 0] / z_c + cx_c
        v_c = fy_c * pts_c[:, 1] / z_c + cy_c

        ui = np.rint(u_c).astype(np.int32)
        vi = np.rint(v_c).astype(np.int32)
        inside = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        if not np.any(inside):
            return aligned

        ui = ui[inside]
        vi = vi[inside]
        z_c = z_c[inside]

        # Z-buffer rasterization: keep nearest depth for each color pixel.
        flat_idx = vi * W + ui
        order = np.argsort(z_c)  # nearest first
        idx_sorted = flat_idx[order]
        z_sorted = z_c[order]
        unique_idx, first_pos = np.unique(idx_sorted, return_index=True)

        aligned_flat = aligned.ravel()
        aligned_flat[unique_idx] = z_sorted[first_pos]
        return aligned

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
            depths: N array of the sampled camera-frame depth (range
                along the optical axis) in the active mode units, 0
                where the pixel was out of bounds. Step 1 uses this to
                weight each correspondence by its depth noise — the
                body-frame Z column of points_body is height, NOT the
                camera range, so the range must be returned separately.

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
            return points_body, valid, depths

        # Backproject valid pixels to camera 3D
        pix_valid = pixels[valid].astype(np.float64)
        d_valid = depths[valid]

        # Homogeneous pixel coords: [u, v, 1]
        p_hom = np.hstack([pix_valid, np.ones((pix_valid.shape[0], 1))])

        # Camera-frame 3D: P_cam = depth * K_inv @ [u, v, 1]^T
        # If depth lives on the color pixel grid (software-aligned OR upstream
        # pre-aligned, e.g. PIT QCar2DepthAligned), use RGB intrinsics.
        # If depth lives on the raw depth grid, use depth intrinsics.
        K_inv = self.K_rgb_inv if self.depth_on_color_grid else self.K_depth_inv
        rays = (K_inv @ p_hom.T).T  # Nx3 direction vectors
        P_cam = d_valid.reshape(-1, 1) * rays   # Nx3 camera-frame points

        # Transform to body frame: P_body = R @ P_cam + t
        P_body = (self.R_cam2body @ P_cam.T).T + self.t_cam2body

        points_body[valid] = P_body
        return points_body, valid, depths


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
        6. s-point RANSAC + SVD Procrustes to find (dx, dy, dpsi)
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
                 depth_on_color_grid=None,
                 camera_mode='virtual',
                 use_undistortion=False,
                 dist_coeffs=None,
                 n_features=800, match_ratio=0.75,
                 ransac_threshold=0.05, ransac_iterations=300,
                 min_inliers=8,
                 max_translation=0.20, max_rotation_deg=15.0,
                 max_dt=0.20,
                 negate_deltas=True,
                 roi_top_fraction=0.0,
                 ransac_sample_size=2,
                 depth_weight_power=0.0,
                 max_vo_feature_depth_m=0.0):
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
            depth_on_color_grid=depth_on_color_grid,
            camera_mode=camera_mode,
            use_undistortion=use_undistortion,
            dist_coeffs=dist_coeffs,
        )

        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.match_ratio = match_ratio
        self.ransac_threshold = ransac_threshold
        self.ransac_iterations = ransac_iterations
        # 2026-05-15: RANSAC minimum-sample size. The 2D rigid transform
        # has 3 DOF (theta, tx, ty); s=2 is the minimum geometric sample,
        # s=3 is the over-constrained robust choice that resists locking
        # onto a degenerate pair on low-texture / bare-wall scenes. Clamped
        # to >= 2. Default 2 preserves the historical behavior exactly.
        self.ransac_sample_size = max(2, int(ransac_sample_size))
        self.min_inliers = min_inliers
        self.max_translation = max_translation
        self.max_rotation = np.radians(max_rotation_deg)
        self.max_dt = max_dt
        self.negate_deltas = negate_deltas

        # 2026-05-15: ORB region-of-interest mask. The top fraction of the
        # image is excluded from feature detection. Default 0.0 = full
        # frame (legacy behavior). Typical productive values are 0.30-0.40
        # to exclude sky / ceiling / horizon and keep ORB focused on
        # ground-level scene features that actually carry useful depth.
        # Implementation note: this uses cv2.ORB.detectAndCompute's mask
        # argument, so image dimensions and camera intrinsics are
        # unchanged. Pixel coords reported back by ORB stay in the full
        # (640x480) frame, so DepthProjector.pixels_to_3d_body sees the
        # same coordinate system it did before and no intrinsics need to
        # be rescaled.
        self.roi_top_fraction = float(roi_top_fraction)
        if not (0.0 <= self.roi_top_fraction < 1.0):
            self.roi_top_fraction = 0.0
        self._orb_mask = None  # built lazily on first frame

        # 2026-05-18: Step 1 — depth-weighted Procrustes. Each matched
        # correspondence is weighted by 1 / range**depth_weight_power
        # before the SVD rigid fit, where `range` is the camera-frame
        # depth (m, physical mode) of the noisier of the two endpoints.
        # Rationale: RealSense stereo depth noise grows ~ sigma_Z ∝ Z^2,
        # so a far bare-wall point is far noisier than a crisp near-mat
        # point. Unweighted Procrustes weights them equally, letting the
        # noisy far point drag x,y. Weighting de-emphasizes far points.
        #   0.0 = OFF -> every weight 1.0 -> bit-identical to the prior
        #         unweighted behavior (Test 6). This is the default and
        #         the A/B control.
        #   2.0 = gentle (treats sigma ∝ Z, weight ∝ 1/Z^2)
        #   4.0 = full RealSense model (sigma_Z ∝ Z^2 -> weight ∝ 1/Z^4)
        # Exposed as a default-safe ROS param so it can be A/B'd on the
        # mat against Test 6 without a rebuild, mirroring the
        # roi_top_fraction / ransac_sample_size pattern. Clamped to a
        # sane [0, 8] band (8 is already extreme).
        self.depth_weight_power = float(depth_weight_power)
        if not (0.0 <= self.depth_weight_power <= 8.0):
            self.depth_weight_power = 0.0

        # 2026-05-18: Step 1b — HARD max-depth cutoff (operator's idea;
        # direct Quanser precedent: pit/YOLO/nets.py post_processing
        # zeroes any depth pixel past `clippingDistance`). A matched
        # feature whose camera range (EITHER endpoint) exceeds this is
        # dropped from VO motion entirely — past a few metres RealSense
        # depth is too noisy to contribute useful geometry, so it is
        # excluded rather than merely down-weighted. This is the hard
        # complement to depth_weight_power's soft trust; the two compose
        # (cutoff filters first, weighting then trusts the survivors).
        # Value is in the ACTIVE mode's depth units (metres in physical
        # mode). 0.0 = OFF -> no extra masking -> prior behavior
        # (default-safe A/B control). Negative -> treated as OFF.
        self.max_vo_feature_depth = float(max_vo_feature_depth_m)
        if not (self.max_vo_feature_depth > 0.0):
            self.max_vo_feature_depth = 0.0

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
            depth_raw: depth image for the current frame:
                       - uint16 MONO16 raw depth (scaled internally), or
                       - float depth already in active mode units.
                       Alignment/registration is applied internally based on
                       the configured projector path.
            timestamp: float, seconds

        Returns:
            dict with pose, velocity, delta_pose, inlier_count,
            confidence, inlier_spread, valid
        """
        # Optional undistortion (disabled for QLabs, enable for physical)
        color_image = self.projector.undistort_image(color_image)

        gray = (cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                if len(color_image.shape) == 3 else color_image.copy())

        # ROI mask for ORB feature detection. Built lazily and reused
        # across frames as long as the image dimensions stay the same.
        mask = None
        if self.roi_top_fraction > 0.0:
            H, W = gray.shape[:2]
            if (self._orb_mask is None
                    or self._orb_mask.shape[0] != H
                    or self._orb_mask.shape[1] != W):
                self._orb_mask = np.zeros((H, W), dtype=np.uint8)
                y_start = int(H * self.roi_top_fraction)
                self._orb_mask[y_start:, :] = 255
            mask = self._orb_mask

        keypoints, descriptors = self.orb.detectAndCompute(gray, mask)

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
        prev_3d, prev_valid, prev_depths = self.projector.pixels_to_3d_body(
            prev_pts, self._prev_depth)
        # Current frame features use current aligned depth
        curr_3d, curr_valid, curr_depths = self.projector.pixels_to_3d_body(
            curr_pts, aligned_depth)

        # Both must have valid depth
        both_valid = prev_valid & curr_valid

        # Step 1b: hard max-depth cutoff. Drop any correspondence whose
        # camera range exceeds the cutoff on EITHER frame. Applied before
        # the min_inliers gate so an over-aggressive cutoff cleanly falls
        # through to "not enough points" (VO abstains) instead of fitting
        # on a near-empty set. 0.0 = off -> both_valid unchanged.
        if self.max_vo_feature_depth > 0.0:
            within = ((prev_depths <= self.max_vo_feature_depth)
                      & (curr_depths <= self.max_vo_feature_depth))
            both_valid = both_valid & within

        if np.sum(both_valid) < self.min_inliers:
            self._store(keypoints, descriptors, aligned_depth, timestamp)
            return result

        # Extract 2D body-frame coordinates (X=forward, Y=left)
        # Features at any height contribute — this is the key advantage
        # over the homography path which constrains everything to Z=0.
        prev_2d = prev_3d[both_valid, :2]  # (X, Y) in body frame
        curr_2d = curr_3d[both_valid, :2]
        valid_prev_px = prev_pts[both_valid]  # pixel coords for spread

        # Step 1: per-correspondence depth-noise weight. Use the noisier
        # (farther) of the two endpoints so a pair is only as trustworthy
        # as its worse observation. weight = 1 / range**power; power 0
        # -> all-ones -> identical to the prior unweighted path. Pass
        # weights=None in that case so the code path is literally the old
        # one (default-safe A/B control).
        weights = None
        if self.depth_weight_power > 0.0:
            pair_range = np.maximum(prev_depths[both_valid],
                                    curr_depths[both_valid])
            # both_valid guarantees each endpoint passed depth>depth_min,
            # so pair_range is strictly positive; clamp anyway for safety.
            pair_range = np.maximum(pair_range, 1e-6)
            weights = pair_range ** (-self.depth_weight_power)
            # Normalize so the largest weight is 1.0 (Procrustes only
            # uses weights up to a common scale; this just keeps the
            # numbers well-conditioned across the wide range span).
            wmax = np.max(weights)
            if wmax > 0.0:
                weights = weights / wmax

        # --- RANSAC + SVD ---
        dx, dy, dpsi, inlier_mask = self._ransac_motion(
            prev_2d, curr_2d, weights)

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
        """Brute-force k=2 matching + Lowe's ratio test.

        cv2.BFMatcher.knnMatch can return a row with fewer than 2
        neighbors when the train set has < 2 descriptors. The old
        ``for m, n in raw`` unpack raised ValueError on such a row;
        the only reason it never crashed in production is the
        descriptor-count guard in update() (>= 2 descriptors required
        before calling _match), so the short-row case is excluded
        upstream. This guard makes _match correct on its own and
        keeps the live path bit-for-bit identical (every row has
        exactly 2 entries there).
        """
        raw = self.matcher.knnMatch(d1, d2, k=2)
        good = []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair[0], pair[1]
            if m.distance < self.match_ratio * n.distance:
                good.append(m)
        return good

    def _ransac_motion(self, pts_prev, pts_curr, weights=None):
        """s-point RANSAC with SVD Procrustes on body-frame 2D points.

        Sample size s = self.ransac_sample_size (>=2). s=2 is the minimum
        for the 3-DOF 2D rigid transform; s=3+ produces more robust
        hypotheses on low-texture scenes at the cost of needing more
        iterations for the same confidence (iterations are fixed here, so
        higher s simply trades a few weak hypotheses for stronger ones).

        Step 1: `weights` (M,) is the per-correspondence depth-noise
        weight. It is applied ONLY inside the SVD Procrustes fits (the
        minimal-sample hypothesis fit and the final inlier refit) — the
        RANSAC inlier test stays an UNWEIGHTED geometric distance vs the
        locked ransac_threshold, so the consensus search and its tuned
        threshold are unchanged; only the pose recovered FROM the chosen
        consensus set is depth-weighted. weights=None reproduces the old
        unweighted estimator exactly.
        """
        M = pts_prev.shape[0]
        best_inliers = np.zeros(M, dtype=bool)
        best_count, best_R, best_t = 0, np.eye(2), np.zeros(2)

        s = self.ransac_sample_size
        if M < s:
            return best_t[0], best_t[1], 0.0, best_inliers

        for _ in range(self.ransac_iterations):
            idx = np.random.choice(M, s, replace=False)
            # Degenerate sample guard (generalized from the old 2-pt
            # distance check): skip if the sampled previous-frame points
            # are effectively coincident (no spread to fit a transform).
            if np.max(np.std(pts_prev[idx], axis=0)) < 1e-8:
                continue
            w_idx = None if weights is None else weights[idx]
            R_e, t_e = self._svd_rigid_2d(
                pts_prev[idx], pts_curr[idx], w_idx)
            res = np.linalg.norm(
                pts_curr - (R_e @ pts_prev.T).T - t_e, axis=1)
            inl = res < self.ransac_threshold
            c = np.sum(inl)
            if c > best_count:
                best_count, best_inliers, best_R, best_t = c, inl, R_e, t_e

        # Refit on all inliers (depth-weighted when weights are given)
        if best_count >= s:
            w_in = None if weights is None else weights[best_inliers]
            best_R, best_t = self._svd_rigid_2d(
                pts_prev[best_inliers], pts_curr[best_inliers], w_in)

        dpsi = np.arctan2(best_R[1, 0], best_R[0, 0])
        return best_t[0], best_t[1], dpsi, best_inliers

    @staticmethod
    def _svd_rigid_2d(pts_a, pts_b, w=None):
        """SVD Procrustes: find R, t such that pts_b ≈ R @ pts_a + t.

        w (N,) is an optional per-point weight (Step 1 depth weighting).
        With w=None — or any uniform weight — this is the original
        unweighted Kabsch solve: uniform weights cancel out of the
        normalized centroid and only scale S by a positive constant,
        which leaves the SVD rotation/translation unchanged. So the
        depth_weight_power=0 path is bit-identical to the prior code.
        """
        if w is None:
            ca = np.mean(pts_a, axis=0)
            cb = np.mean(pts_b, axis=0)
            S = (pts_a - ca).T @ (pts_b - cb)
        else:
            w = np.asarray(w, dtype=np.float64)
            wsum = w.sum()
            if wsum <= 0.0 or not np.isfinite(wsum):
                # Degenerate weights — fall back to the unweighted fit
                # rather than producing NaNs.
                ca = np.mean(pts_a, axis=0)
                cb = np.mean(pts_b, axis=0)
                S = (pts_a - ca).T @ (pts_b - cb)
            else:
                ca = (w[:, None] * pts_a).sum(axis=0) / wsum
                cb = (w[:, None] * pts_b).sum(axis=0) / wsum
                A = pts_a - ca
                B = pts_b - cb
                S = (w[:, None] * A).T @ B
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
