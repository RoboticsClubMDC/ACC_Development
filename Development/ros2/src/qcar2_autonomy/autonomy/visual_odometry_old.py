# """Visual Odometry module for QCar2 - v4.1.

# v4.1 CHANGE from v4:
#   - inlier_spread metric: after RANSAC, computes geometric mean of
#     std-dev(x) and std-dev(y) of inlier pixel locations. Returned in
#     result dict as 'inlier_spread'. Measures geometric degeneracy:
#     high inlier count + low spread = clustered features (turn problem).
#     Used by vo_node v4.1+ for Part 2 weighting decisions.

# v4 CHANGES from v3:
#   - negate_deltas parameter (default False):
#       When True, negates dx/dy/dpsi after RANSAC.
#       Theory: SVD Procrustes finds FEATURE motion (opposite of vehicle).
#       Evidence is inconclusive — use for A/B testing only.
#       Default False = same sign convention as v1/v2/v3.

# v3 changes preserved:
#   - soft_reset(): re-anchor pose WITHOUT destroying tracking state
#   - Dual depth decoding: HIGH_BYTE shift (QLabs) or scale (D435)
#   - n_features 800, ransac_iterations 300

# FILE: autonomy/visual_odometry.py

# INTRINSICS: fx=483.671 fy=483.579 cx=321.188 cy=238.462 at 640x480
# EXTRINSICS: phi=pi/2 theta=0 psi=pi/2 height=1.72 (QLabs)
# """

# import numpy as np
# import cv2


# class GroundPlaneProjector:
#     def __init__(self, img_width=640, img_height=480,
#                  calib_width=640, calib_height=480,
#                  fx_override=None, fy_override=None):
#         K_calib = np.array([
#             [483.671,       0, 321.188],
#             [      0, 483.579, 238.462],
#             [      0,       0,       1]])
#         sx = img_width / calib_width
#         sy = img_height / calib_height
#         self.K = np.array([
#             [K_calib[0,0]*sx,            0, K_calib[0,2]*sx],
#             [              0, K_calib[1,1]*sy, K_calib[1,2]*sy],
#             [              0,              0,              1]],
#             dtype=np.float64)

#         # Apply focal length overrides AFTER resolution scaling.
#         # This lets us calibrate the virtual camera's actual FOV
#         # independently of the resolution-matching logic.
#         if fx_override is not None:
#             self.K[0, 0] = float(fx_override)
#         if fy_override is not None:
#             self.K[1, 1] = float(fy_override)

#         self.K_inv = np.linalg.inv(self.K)

#         phi, theta, psi = np.pi/2, 0.0, np.pi/2
#         height = 1.72
#         cx_, sx_ = np.cos(phi), np.sin(phi)
#         cy_, sy_ = np.cos(theta), np.sin(theta)
#         cz_, sz_ = np.cos(psi), np.sin(psi)
#         Rx = np.array([[1,0,0],[0,cx_,-sx_],[0,sx_,cx_]])
#         Ry = np.array([[cy_,0,sy_],[0,1,0],[-sy_,0,cy_]])
#         Rz = np.array([[cz_,-sz_,0],[sz_,cz_,0],[0,0,1]])
#         self.R = Rx @ Ry @ Rz
#         self.t = np.array([[0, height, 0]], dtype=np.float64).T
#         A = np.column_stack([self.R[:,0], self.R[:,1], self.t.flatten()])
#         self.H = self.K @ A
#         if abs(np.linalg.det(self.H)) < 1e-12:
#             raise ValueError("Homography singular")
#         self.G = np.linalg.inv(self.H)

#     def pixels_to_ground(self, pixels):
#         N = pixels.shape[0]
#         p = np.hstack([pixels.astype(np.float64), np.ones((N,1))])
#         q = self.G @ p.T
#         v = np.abs(q[2,:]) > 1e-10
#         g = np.zeros((N,2))
#         g[v,0] = q[0,v]/q[2,v]; g[v,1] = q[1,v]/q[2,v]
#         v &= (g[:,0]>-10)&(g[:,0]<30)&(g[:,1]>-10)&(g[:,1]<10)
#         return g, v

#     def pixels_to_3d(self, pixels, depths):
#         N = pixels.shape[0]
#         v = depths > 0.01
#         pts = np.zeros((N,3))
#         if np.any(v):
#             p = np.hstack([pixels[v].astype(np.float64),
#                            np.ones((np.sum(v),1))])
#             Pc = depths[v].reshape(-1,1) * (self.K_inv @ p.T).T
#             pts[v] = (self.R.T @ (Pc.T - self.t)).T
#         return pts, v


# class VisualOdometry:
#     """ORB + RANSAC + SVD Procrustes visual odometry, v4."""

#     MIN_INLIERS         = 20
#     MAX_TRANS_PER_FRAME  = 0.20
#     MAX_ROT_PER_FRAME    = np.deg2rad(15.0)
#     MAX_TIMESTAMP_GAP    = 0.20
#     DEPTH_MIN            = 0.10
#     DEPTH_MAX            = 8.0
#     DEPTH_RATIO_MAX      = 3.0
#     CONFIDENCE_FLOOR     = 30

#     def __init__(self, img_width=640, img_height=480, use_depth=True,
#                  n_features=800, match_ratio=0.75,
#                  ransac_threshold=0.05, min_inliers=None,
#                  depth_scale=5.5,
#                  depth_shift_bits=0, depth_unit_m=0.001,
#                  depth_ratio_max=3.0,
#                  negate_deltas=False,
#                  fx_override=None, fy_override=None):
#         self.projector = GroundPlaneProjector(
#             img_width, img_height,
#             fx_override=fx_override, fy_override=fy_override)
#         self.use_depth = use_depth
#         self.img_height = img_height
#         self.img_width  = img_width
#         self.depth_shift_bits = int(depth_shift_bits)
#         self.depth_unit_m = float(depth_unit_m)
#         self.depth_scale = float(depth_scale)
#         self._depth_divisor = depth_scale * 1000.0
#         self._depth_diag_done = False
#         self.negate_deltas = bool(negate_deltas)
#         self.DEPTH_RATIO_MAX = float(depth_ratio_max)

#         self.orb = cv2.ORB_create(nfeatures=n_features)
#         self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
#         self.match_ratio = match_ratio
#         self.ransac_threshold = ransac_threshold
#         self.ransac_iterations = 300
#         self.min_inliers = min_inliers or self.MIN_INLIERS

#         self.pose = np.array([0.0, 0.0, 0.0])
#         self.velocity = np.array([0.0, 0.0])
#         self.is_initialized = False
#         self.inlier_count = 0
#         self.confidence = 0.0
#         self.rejected_reason = ""
#         self._prev_kp = self._prev_desc = None
#         self._prev_depth = self._prev_time = None

#     def update(self, image, timestamp, depth_image=None):
#         gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#                 if len(image.shape) == 3 else image.copy())
#         kp, desc = self.orb.detectAndCompute(gray, None)
#         r = {'pose': self.pose.copy(), 'velocity': self.velocity.copy(),
#              'delta_pose': None, 'inlier_count': 0, 'confidence': 0.0,
#              'valid': False, 'rejected_reason': '',
#              'inlier_spread': 0.0}  # v4.1: Part 2 weighting signal

#         if not self.is_initialized:
#             self._store(kp, desc, depth_image, timestamp)
#             self.is_initialized = True
#             r['rejected_reason'] = 'init'
#             return r

#         if self._prev_time is not None and timestamp is not None:
#             dt = timestamp - self._prev_time
#             if dt > self.MAX_TIMESTAMP_GAP or dt < 0:
#                 self._store(kp, desc, depth_image, timestamp)
#                 r['rejected_reason'] = 'frame_skip dt=%.3fs' % dt
#                 return r

#         if (desc is None or self._prev_desc is None
#                 or len(desc) < 2 or len(self._prev_desc) < 2):
#             self._store(kp, desc, depth_image, timestamp)
#             r['rejected_reason'] = 'no_descriptors'
#             return r

#         matches = self._match(self._prev_desc, desc)
#         if len(matches) < self.min_inliers:
#             self._store(kp, desc, depth_image, timestamp)
#             r['rejected_reason'] = 'few_matches (%d<%d)' % (len(matches), self.min_inliers)
#             r['inlier_count'] = len(matches)
#             return r

#         pp = np.array([self._prev_kp[m.queryIdx].pt for m in matches])
#         cp = np.array([kp[m.trainIdx].pt for m in matches])

#         depth_attempted = False
#         if self.use_depth and depth_image is not None and self._prev_depth is not None:
#             pg, cg, ok = self._to_3d(pp, cp, self._prev_depth, depth_image)
#             depth_attempted = True
#             # v4.1: graceful degradation — if depth is present but
#             # patchy (not enough pixels have valid depth readings),
#             # fall back to ground-plane projection instead of
#             # rejecting the frame outright. This improves uptime
#             # in mixed-quality depth conditions.
#             if np.sum(ok) < self.min_inliers:
#                 pg, cg, ok = self._to_ground(pp, cp)
#         else:
#             pg, cg, ok = self._to_ground(pp, cp)

#         if np.sum(ok) < self.min_inliers:
#             self._store(kp, desc, depth_image, timestamp)
#             label = ('few_valid_ground_after_depth' if depth_attempted
#                      else 'few_valid_ground')
#             r['rejected_reason'] = '%s (%d<%d)' % (label, np.sum(ok), self.min_inliers)
#             return r

#         pg, cg = pg[ok], cg[ok]
#         dx, dy, dpsi, inl = self._ransac_motion(pg, cg)

#         # Optional sign flip for A/B testing (Phase 2)
#         if self.negate_deltas:
#             dx  = -dx
#             dy  = -dy
#             dpsi = -dpsi

#         self.inlier_count = int(np.sum(inl))
#         self.confidence = self._conf(self.inlier_count, len(matches))

#         # v4.1: Compute inlier pixel spread for Part 2 weighting.
#         # cp[ok] = pixel coords that passed depth/ground validation.
#         # [inl]  = further filtered to RANSAC survivors.
#         # Geometric mean of std-dev in x and y measures how spread out
#         # inliers are across the image. Low spread + high count means
#         # features are clustered (classic turn degeneracy: the rigid
#         # transform looks well-supported but is geometrically fragile).
#         # Typical values at 640x480:
#         #   Well-distributed (straight): spread ≈ 100-170
#         #   Clustered (mid-turn):        spread ≈ 30-60
#         inlier_pixels = cp[ok][inl]
#         if len(inlier_pixels) >= 4:
#             sx = np.std(inlier_pixels[:, 0])  # horizontal spread
#             sy = np.std(inlier_pixels[:, 1])  # vertical spread
#             r['inlier_spread'] = float(np.sqrt(max(sx * sy, 0.0)))
#         else:
#             r['inlier_spread'] = 0.0

#         if self.inlier_count < self.min_inliers:
#             self._store(kp, desc, depth_image, timestamp)
#             r['rejected_reason'] = 'few_inliers (%d<%d)' % (self.inlier_count, self.min_inliers)
#             r['inlier_count'] = self.inlier_count
#             r['confidence'] = self.confidence
#             return r

#         tmag = np.sqrt(dx**2 + dy**2)
#         if tmag > self.MAX_TRANS_PER_FRAME:
#             self._store(kp, desc, depth_image, timestamp)
#             r['rejected_reason'] = 'motion_too_large (t=%.3fm)' % tmag
#             r['inlier_count'] = self.inlier_count
#             r['confidence'] = self.confidence
#             return r

#         if abs(dpsi) > self.MAX_ROT_PER_FRAME:
#             self._store(kp, desc, depth_image, timestamp)
#             r['rejected_reason'] = 'rotation_too_large (r=%.1fdeg)' % np.rad2deg(dpsi)
#             r['inlier_count'] = self.inlier_count
#             r['confidence'] = self.confidence
#             return r

#         c, s = np.cos(self.pose[2]), np.sin(self.pose[2])
#         dx_m, dy_m = c*dx - s*dy, s*dx + c*dy
#         self.pose[0] += dx_m
#         self.pose[1] += dy_m
#         # v4.1: VO yaw integration disabled. pose[2] is driven by
#         # cartographer yaw — vo_node sets self.vo.pose[2] = cart_psi
#         # before every update() call. Integrating dpsi here would be
#         # immediately overwritten next tick, wasting compute and
#         # risking subtle inconsistencies during turns where VO dpsi
#         # is unreliable (Part 1 conclusion: turn drift is geometric).
#         # dpsi is still computed and returned in delta_pose for
#         # diagnostic/logging purposes.

#         dt2 = timestamp - self._prev_time
#         if dt2 > 1e-6:
#             self.velocity = np.array([dx_m/dt2, dy_m/dt2])

#         self.rejected_reason = ""
#         self._store(kp, desc, depth_image, timestamp)
#         r.update({'pose': self.pose.copy(), 'velocity': self.velocity.copy(),
#                   'delta_pose': np.array([dx, dy, dpsi]),
#                   'inlier_count': self.inlier_count,
#                   'confidence': self.confidence,
#                   'inlier_spread': r['inlier_spread'],  # v4.1: pass through
#                   'valid': True, 'rejected_reason': ''})
#         return r

#     def reset(self, x=0.0, y=0.0, psi=0.0):
#         """Hard reset: clears ALL state. Use for first anchor only."""
#         self.pose = np.array([x, y, psi])
#         self.velocity = np.zeros(2)
#         self.is_initialized = False
#         self._prev_kp = self._prev_desc = None
#         self._prev_depth = self._prev_time = None
#         self.inlier_count = 0
#         self.confidence = 0.0
#         self.rejected_reason = ""

#     def soft_reset(self, x=0.0, y=0.0, psi=0.0):
#         """Re-anchor pose but KEEP tracking state.

#         prev keypoints/descriptors/depth/time stay intact.
#         Next update() matches immediately — zero blind frames.
#         """
#         self.pose = np.array([float(x), float(y), float(psi)])
#         self.velocity = np.zeros(2)

#     # ── INTERNAL ──

#     def _match(self, d1, d2):
#         raw = self.matcher.knnMatch(d1, d2, k=2)
#         return [m for p in raw if len(p)==2
#                 for m in [p[0]] if m.distance < self.match_ratio*p[1].distance]

#     def _to_ground(self, pp, cp):
#         pg, pv = self.projector.pixels_to_ground(pp)
#         cg, cv = self.projector.pixels_to_ground(cp)
#         return pg, cg, pv & cv

#     def _decode_raw(self, val):
#         if self.depth_shift_bits > 0:
#             return float(int(val) >> self.depth_shift_bits) * self.depth_unit_m
#         return float(val) / self._depth_divisor

#     def _sample_depth(self, img, u, v, hw=1):
#         h, w = img.shape[:2]
#         ui, vi = int(round(u)), int(round(v))
#         patch = img[max(0,vi-hw):min(h,vi+hw+1),
#                     max(0,ui-hw):min(w,ui+hw+1)].astype(np.float64)
#         vp = patch[patch > 0]
#         if len(vp) == 0:
#             return 0.0
#         raw_med = float(np.median(vp))
#         if self.depth_shift_bits > 0:
#             d = float(int(raw_med) >> self.depth_shift_bits) * self.depth_unit_m
#         else:
#             d = raw_med / self._depth_divisor

#         if not self._depth_diag_done:
#             self._depth_diag_done = True
#             mode = 'HIGH_BYTE_U8' if self.depth_shift_bits > 0 else 'SCALE'
#             rc = img[h//2, w//2]
#             cc = self._decode_raw(rc)
#             ok = self.DEPTH_MIN <= cc <= self.DEPTH_MAX
#             print('[VO DEPTH DIAGNOSTIC]')
#             print('  mode=%s shift_bits=%d' % (mode, self.depth_shift_bits))
#             print('  depth_scale=%s divisor=%s' % (self.depth_scale, self._depth_divisor))
#             print('  depth_unit_m=%s' % self.depth_unit_m)
#             print('  center raw=%s -> %.4f m  passes=%s' % (rc, cc, ok))
#             print('  sample raw_med=%.0f -> %.4f m at (%d,%d)' % (raw_med, d, ui, vi))

#         return d if self.DEPTH_MIN <= d <= self.DEPTH_MAX else 0.0

#     def _to_3d(self, pp, cp, pd, cd):
#         N = pp.shape[0]
#         prev_d = np.array([self._sample_depth(pd, pp[i,0], pp[i,1]) for i in range(N)])
#         curr_d = np.array([self._sample_depth(cd, cp[i,0], cp[i,1]) for i in range(N)])
#         bv = (prev_d > 0) & (curr_d > 0)
#         dc = np.ones(N, dtype=bool)
#         if np.any(bv):
#             rat = np.ones(N)
#             rat[bv] = np.maximum(prev_d[bv]/curr_d[bv], curr_d[bv]/prev_d[bv])
#             dc = rat <= self.DEPTH_RATIO_MAX
#         p3, pv = self.projector.pixels_to_3d(pp, prev_d)
#         c3, cv = self.projector.pixels_to_3d(cp, curr_d)
#         return p3[:,:2], c3[:,:2], pv & cv & dc

#     def _ransac_motion(self, pp, cp):
#         M = pp.shape[0]
#         bi = np.zeros(M, dtype=bool)
#         bc, bR, bt = 0, np.eye(2), np.zeros(2)
#         for _ in range(self.ransac_iterations):
#             idx = np.random.choice(M, 2, replace=False)
#             if np.linalg.norm(pp[idx[0]]-pp[idx[1]]) < 1e-8:
#                 continue
#             Re, te = self._svd2(pp[idx], cp[idx])
#             res = np.linalg.norm(cp - (Re @ pp.T).T - te, axis=1)
#             inl = res < self.ransac_threshold
#             c = np.sum(inl)
#             if c > bc:
#                 bc, bi, bR, bt = c, inl, Re, te
#         if bc >= 2:
#             bR, bt = self._svd2(pp[bi], cp[bi])
#         return bt[0], bt[1], np.arctan2(bR[1,0], bR[0,0]), bi

#     @staticmethod
#     def _svd2(a, b):
#         ca, cb = np.mean(a,0), np.mean(b,0)
#         H = (a-ca).T @ (b-cb)
#         U, _, Vt = np.linalg.svd(H)
#         d = np.linalg.det(Vt.T @ U.T)
#         R = Vt.T @ np.diag([1.0, np.sign(d)]) @ U.T
#         return R, cb - R @ ca

#     def _conf(self, inl, tot):
#         if tot == 0: return 0.0
#         return (inl/tot) * min(1.0, inl/self.CONFIDENCE_FLOOR)

#     def _store(self, kp, desc, depth, ts):
#         self._prev_kp, self._prev_desc = kp, desc
#         self._prev_depth, self._prev_time = depth, ts