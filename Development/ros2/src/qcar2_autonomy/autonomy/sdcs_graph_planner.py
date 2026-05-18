"""
sdcs_graph_planner.py
Self-contained directed graph planner for the SDCS right-hand traffic road map.
Requires only numpy — no hal/pal dependency.
"""

import heapq
import math

import numpy as np

# ---------------------------------------------------------------------------
# SDCS node geometry  (right-hand traffic, large map)
# Converted from mats.py pixel data:  x = SCALE*(px-1134),  y = SCALE*(2363-py)
# ---------------------------------------------------------------------------
_S  = 0.002035          # pixel-to-metre scale
_PI = math.pi
_HP = _PI / 2

def _n(px, py, yaw):
    return (_S * (px - 1134), _S * (2363 - py), yaw)

SDCS_NODES = [
    _n(1134, 2299, -_HP),           #  0
    _n(1266, 2323,  _HP),           #  1
    _n(1688, 2896,  0.0),           #  2
    _n(1688, 2763,  _PI),           #  3
    _n(2242, 2323,  _HP),           #  4
    _n(2109, 2323, -_HP),           #  5
    _n(1632, 1822,  _PI),           #  6
    _n(1741, 1955,  0.0),           #  7
    _n( 766, 1822,  _PI),           #  8
    _n( 766, 1955,  0.0),           #  9
    _n( 504, 2589, -42*_PI/180),    # 10  taxi hub
    _n(1134, 1300, -_HP),           # 11
    _n(1134, 1454, -_HP),           # 12
    _n(1266, 1454,  _HP),           # 13
    _n(2242,  905,  _HP),           # 14
    _n(2109, 1454, -_HP),           # 15
    _n(1580,  540, -80.6*_PI/180),  # 16
    _n(1854.4, 814.5, -9.4*_PI/180),# 17
    _n(1440,  856, -138*_PI/180),   # 18
    _n(1523,  958,  42*_PI/180),    # 19
    _n(1134,  153,  _PI),           # 20
    _n(1134,  286,  0.0),           # 21
    _n( 159,  905, -_HP),           # 22
    _n( 291,  905,  _HP),           # 23
]

_INNER  = 305.5 * _S   # 0.6217 m
_OUTER  = 438.0 * _S   # 0.8913 m
_CIRCLE = 333.0 * _S   # 0.6777 m
_ONEWAY = 350.0 * _S   # 0.7123 m
_KINK   = 375.0 * _S   # 0.7631 m

# (from_idx, to_idx, turn_radius_m)  — right-hand traffic, large map
SDCS_EDGES = [
    ( 0,  2, _OUTER),
    ( 1,  7, _INNER),
    ( 1,  8, _OUTER),
    ( 2,  4, _OUTER),
    ( 3,  1, _INNER),
    ( 4,  6, _OUTER),
    ( 5,  3, _INNER),
    ( 6,  0, _OUTER),
    ( 6,  8,    0.0),
    ( 7,  5, _INNER),
    ( 8, 10, _ONEWAY),
    ( 9,  0, _INNER),
    ( 9,  7,    0.0),
    (10,  1, _INNER),
    (10,  2, _INNER),
    ( 1, 13,    0.0),
    ( 4, 14,    0.0),
    ( 6, 13, _INNER),
    ( 7, 14, _OUTER),
    ( 8, 23, _INNER),
    ( 9, 13, _OUTER),
    (11, 12,    0.0),
    (12,  0,    0.0),
    (12,  7, _OUTER),
    (12,  8, _INNER),
    (13, 19, _INNER),
    (14, 16, _CIRCLE),
    (14, 20, _CIRCLE),
    (15,  5, _OUTER),
    (15,  6, _INNER),
    (16, 17, _CIRCLE),
    (16, 18, _INNER),
    (17, 15, _INNER),
    (17, 16, _CIRCLE),
    (17, 20, _CIRCLE),
    (18, 11, _KINK),
    (19, 17, _INNER),
    (20, 22, _OUTER),
    (21, 16, _INNER),
    (22,  9, _OUTER),
    (22, 10, _OUTER),
    (23, 21, _INNER),
]

# ---------------------------------------------------------------------------
# Math helpers (replaces pal.utilities.math)
# ---------------------------------------------------------------------------

def _signed_angle(v1, v2):
    """Signed angle from 2-D vector v1 to v2 (CCW positive)."""
    v1 = np.asarray(v1, dtype=float).ravel()
    v2 = np.asarray(v2, dtype=float).ravel()
    return math.atan2(v1[0]*v2[1] - v1[1]*v2[0],
                      v1[0]*v2[0] + v1[1]*v2[1])

def _wrap_to_pi(a):
    return math.atan2(math.sin(a), math.cos(a))

def _wrap_to_2pi(a):
    return a % (2 * math.pi)

# ---------------------------------------------------------------------------
# SCSPath  (Straight-Curve-Straight path between two poses)
# Ported from hal.utilities.path_planning — no pal import needed.
# ---------------------------------------------------------------------------

def _scs_path(p1, th1, p2, th2, radius, step):
    """
    p1, p2  : 2×1 numpy arrays in SDCS metres
    th1, th2: float headings (radians)
    radius  : turn radius > 0
    step    : waypoint spacing (metres)
    Returns (waypoints_2xN, length_m) — waypoints exclude p1/p2 themselves.
    Returns (None, None) on geometry failure.
    """
    TOL = 0.01
    t1 = np.array([[math.cos(th1)], [math.sin(th1)]])
    t2 = np.array([[math.cos(th2)], [math.sin(th2)]])

    direction = 1 if _signed_angle(t1, p2 - p1) > 0 else -1

    n1 = radius * np.array([[-t1[1, 0]], [t1[0, 0]]]) * direction
    n2 = radius * np.array([[-t2[1, 0]], [t2[0, 0]]]) * direction

    th_diff     = abs(_wrap_to_pi(th2 - th1))
    th_diff_opp = abs(_wrap_to_pi(th2 - th1 + math.pi))

    if th_diff < TOL:
        v = p2 - p1
        vn = float(np.linalg.norm(v))
        if vn < 1e-9:
            return None, None
        if 1.0 - abs(float(t1.ravel() @ (v / vn).ravel())) < TOL:
            c = p2 + n1
        else:
            return None, None
    elif th_diff_opp < TOL:
        v = (p2 + 2*n2) - p1
        vn = float(np.linalg.norm(v))
        if vn < 1e-9:
            return None, None
        if 1.0 - abs(float(t1.ravel() @ (v / vn).ravel())) < TOL:
            sv = float(t1.ravel() @ v.ravel())
            c = p1 + n1 if sv < TOL else p2 + n2
        else:
            return None, None
    else:
        d1 = p1 + n1
        d2 = p2 + n2
        A  = np.hstack((t1, -t2))
        b  = d2 - d1
        try:
            params = np.linalg.solve(A, b).ravel()
        except np.linalg.LinAlgError:
            return None, None
        alpha, beta = float(params[0]), float(params[1])
        if alpha >= -TOL and beta <= TOL:
            c = d1 + alpha * t1
        else:
            return None, None

    b1 = c - n1
    b2 = c - n2

    def _seg(pa, pb):
        seg = float(np.linalg.norm(pb - pa))
        pts = []
        if seg > step:
            ds = step / seg
            s  = ds
            while s < 1.0:
                pts.append(pa + s * (pb - pa))
                s += ds
        return pts, seg

    line1_pts, line1_len = _seg(p1, b1)

    ang_dist = _wrap_to_2pi(direction * _signed_angle(b1 - c, b2 - c))
    arc_len  = abs(ang_dist * radius)
    arc_pts  = []
    if arc_len > step:
        sa  = math.atan2(float(b1[1, 0] - c[1, 0]), float(b1[0, 0] - c[0, 0]))
        dth = step / radius
        s   = dth
        while s < ang_dist:
            th = sa + s * direction
            arc_pts.append(c + radius * np.array([[math.cos(th)], [math.sin(th)]]))
            s += dth

    line2_pts, line2_len = _seg(b2, p2)

    all_pts   = line1_pts + arc_pts + line2_pts
    total_len = line1_len + arc_len + line2_len

    wp = np.hstack([p.reshape(2, 1) for p in all_pts]) if all_pts else np.empty((2, 0))
    return wp, total_len


def _straight_path(p1, p2, step):
    """Dense straight-line intermediate waypoints (endpoints excluded)."""
    diff   = p2 - p1
    length = float(np.linalg.norm(diff))
    if length < 1e-9:
        return np.empty((2, 0)), 0.0
    n   = max(1, int(math.ceil(length / step)))
    pts = [p1 + (i / n) * diff for i in range(1, n)]
    wp  = np.hstack([p.reshape(2, 1) for p in pts]) if pts else np.empty((2, 0))
    return wp, length

# ---------------------------------------------------------------------------
# Coordinate transforms  (SDCS ↔ ROS map frame)
# Matches trip_planner._goal_to_route_frame row-vector convention:
#   map_point = scale * (sdcs_point @ R) + T
#   where R = [[cos θ, -sin θ], [sin θ, cos θ]]
# For column-vector batches this becomes: map_col = scale * R_c @ sdcs_col + T_col
#   where R_c = [[cos θ, sin θ], [-sin θ, cos θ]]
# ---------------------------------------------------------------------------

def sdcs_to_map_pt(xy_sdcs, scale, rot_rad, translation):
    x_s, y_s = float(xy_sdcs[0]), float(xy_sdcs[1])
    c, s = math.cos(rot_rad), math.sin(rot_rad)
    return np.array([
        scale * (x_s * c + y_s * s) + translation[0],
        scale * (-x_s * s + y_s * c) + translation[1],
    ])


def sdcs_to_map_batch(pts_2xn, scale, rot_rad, translation):
    """Transform 2×N SDCS array to map frame."""
    c, s = math.cos(rot_rad), math.sin(rot_rad)
    R_c  = np.array([[c, s], [-s, c]])
    T    = np.array([[translation[0]], [translation[1]]])
    return scale * (R_c @ pts_2xn) + T


def sdcs_heading_to_map(th_sdcs, rot_rad):
    return _wrap_to_pi(th_sdcs - rot_rad)

# ---------------------------------------------------------------------------
# Graph planner
# ---------------------------------------------------------------------------

class SDCSGraphPlanner:
    """
    Directed graph A* planner for the SDCS right-hand traffic road map.

    The SDCS→map transform is applied ONCE at construction time.  After that
    all internal data (_node_map, _wp) lives in map frame and every query
    method works purely in map frame — no transform parameters anywhere.
    """

    def __init__(
        self,
        waypoint_spacing: float = 0.03,
        scale: float = 1.0,
        rot_rad: float = 0.0,
        translation=(0.0, 0.0),
    ):
        self._step        = waypoint_spacing
        self._scale       = scale
        self._rot_rad     = rot_rad
        self._translation = list(translation)
        self._n           = len(SDCS_NODES)
        self._adj: list   = [[] for _ in range(self._n)]
        self._wp:  dict   = {}   # (fi, ti) → 2×N ndarray in MAP frame
        self._len: dict   = {}   # (fi, ti) → float metres
        self._node_map: list = []  # (mx, my, mth) in map frame, one per node
        self._build()

    def _build(self):
        # Pre-bake SDCS→map for every node
        for (nx, ny, nth) in SDCS_NODES:
            mp  = sdcs_to_map_pt((nx, ny), self._scale, self._rot_rad, self._translation)
            mth = sdcs_heading_to_map(nth, self._rot_rad)
            self._node_map.append((float(mp[0]), float(mp[1]), float(mth)))

        failed = []
        for (fi, ti, radius) in SDCS_EDGES:
            fx, fy, fth = SDCS_NODES[fi]
            tx, ty, tth = SDCS_NODES[ti]
            p1 = np.array([[fx], [fy]])
            p2 = np.array([[tx], [ty]])

            if radius < 1e-9:
                wp_sdcs, length = _straight_path(p1, p2, self._step)
            else:
                wp_sdcs, length = _scs_path(p1, fth, p2, tth, radius, self._step)

            if length is None or length < 1e-6:
                failed.append((fi, ti))
                continue

            if wp_sdcs.shape[1] > 0:
                wp_map = sdcs_to_map_batch(wp_sdcs, self._scale, self._rot_rad, self._translation)
            else:
                wp_map = np.empty((2, 0))

            self._wp[(fi, ti)]  = wp_map   # stored in map frame
            self._len[(fi, ti)] = float(length)
            self._adj[fi].append(ti)

        if failed:
            import warnings
            warnings.warn(f'SDCSGraphPlanner: {len(failed)} edges failed geometry: {failed}')

    # ------------------------------------------------------------------

    def node_position(self, idx: int) -> np.ndarray:
        """Return (x, y) of node idx in map frame."""
        mx, my, _ = self._node_map[idx]
        return np.array([mx, my], dtype=float)

    def node_positions_map(self):
        return [np.array([mx, my]) for mx, my, _ in self._node_map]

    # ------------------------------------------------------------------

    def find_nearest_node(
        self,
        xy_map, yaw_map,
        heading_weight=0.30,
        max_heading_error=1.57,
    ):
        """Nearest graph node to a map-frame pose (heading-aware). Returns (idx, score)."""
        best_idx   = None
        best_score = float('inf')

        for i, (mx, my, mth) in enumerate(self._node_map):
            dist = math.hypot(mx - float(xy_map[0]), my - float(xy_map[1]))
            if yaw_map is not None:
                yaw_err = abs(_wrap_to_pi(mth - float(yaw_map)))
                if yaw_err > max_heading_error:
                    continue
                score = dist + heading_weight * yaw_err
            else:
                score = dist
            if score < best_score:
                best_score = score
                best_idx   = i

        if best_idx is None:
            best_idx, best_score = self.find_nearest_node_position_only(xy_map)
        return best_idx, best_score

    def find_nearest_node_position_only(self, xy_map):
        """Nearest graph node by Euclidean distance only. Returns (idx, dist)."""
        best_idx  = 0
        best_dist = float('inf')
        for i, (mx, my, _) in enumerate(self._node_map):
            d = math.hypot(mx - float(xy_map[0]), my - float(xy_map[1]))
            if d < best_dist:
                best_dist = d
                best_idx  = i
        return best_idx, best_dist

    def find_start_on_edge(
        self,
        xy_map, yaw_map,
        heading_weight=0.30,
        max_heading_error=1.57,
    ):
        """
        Project robot pose onto the nearest compatible directed edge polyline.

        Returns (fi, ti, proj_pt_2x1, remaining_wp_2xN, proj_dist_m, heading_err_rad)
        where remaining_wp contains the map-frame edge waypoints after the projection
        up to (but NOT including) the ti node itself.

        Returns None if every edge is rejected by the heading filter.
        """
        robot = np.array([float(xy_map[0]), float(xy_map[1])], dtype=float)
        yaw   = float(yaw_map)

        best       = None
        best_score = float('inf')

        for (fi, ti, _radius) in SDCS_EDGES:
            key = (fi, ti)
            if key not in self._wp:
                continue

            mx_fi, my_fi, _ = self._node_map[fi]
            mx_ti, my_ti, _ = self._node_map[ti]
            from_pt = np.array([[mx_fi], [my_fi]])
            to_pt   = np.array([[mx_ti], [my_ti]])

            wp_map = self._wp[key]
            poly = np.hstack([from_pt, wp_map, to_pt]) if wp_map.shape[1] > 0 \
                   else np.hstack([from_pt, to_pt])

            best_k = 0
            best_t = 0.0
            best_d = float('inf')
            for k in range(poly.shape[1] - 1):
                a    = poly[:, k]
                b    = poly[:, k + 1]
                ab   = b - a
                ab_sq = float(ab @ ab)
                if ab_sq < 1e-12:
                    continue
                t    = float(np.clip((robot - a) @ ab / ab_sq, 0.0, 1.0))
                d    = float(np.linalg.norm(a + t * ab - robot))
                if d < best_d:
                    best_d = d
                    best_k = k
                    best_t = t

            a  = poly[:, best_k]
            b  = poly[:, best_k + 1]
            ab = b - a
            if float(np.linalg.norm(ab)) < 1e-9:
                continue
            heading_err = abs(_wrap_to_pi(math.atan2(float(ab[1]), float(ab[0])) - yaw))
            if heading_err > max_heading_error:
                continue

            score = best_d + heading_weight * heading_err
            if score >= best_score:
                continue

            proj_pt      = (a + best_t * ab).reshape(2, 1)
            remaining_wp = poly[:, best_k + 1:-1]   # excl. ti node itself
            best_score   = score
            best = (fi, ti, proj_pt, remaining_wp, best_d, heading_err)

        return best

    # ------------------------------------------------------------------

    def astar(self, start_idx: int, goal_idx: int):
        """A* on the directed graph. Returns node-index list or None."""
        if start_idx == goal_idx:
            return [start_idx]

        gx, gy, _ = self._node_map[goal_idx]

        def h(i):
            ix, iy, _ = self._node_map[i]
            return math.hypot(gx - ix, gy - iy)

        counter = 0
        heap    = []
        heapq.heappush(heap, (h(start_idx), counter, start_idx))
        came    = {}
        g       = [float('inf')] * self._n
        g[start_idx] = 0.0
        visited = set()

        while heap:
            _, _, cur = heapq.heappop(heap)
            if cur in visited:
                continue
            visited.add(cur)

            if cur == goal_idx:
                path, node = [], cur
                while node in came:
                    path.append(node)
                    node = came[node]
                path.append(start_idx)
                path.reverse()
                return path

            for nb in self._adj[cur]:
                if nb in visited:
                    continue
                tg = g[cur] + self._len[(cur, nb)]
                if tg < g[nb]:
                    came[nb] = cur
                    g[nb]    = tg
                    counter += 1
                    heapq.heappush(heap, (tg + h(nb), counter, nb))

        return None

    # ------------------------------------------------------------------

    def build_path(self, node_seq) -> np.ndarray:
        """Assemble dense 2×N waypoint array in map frame from A* node sequence."""
        if not node_seq:
            return np.empty((2, 0))

        cols = []
        for i in range(len(node_seq) - 1):
            fi = node_seq[i]
            ti = node_seq[i + 1]
            mx, my, _ = self._node_map[fi]
            cols.append(np.array([[mx], [my]]))
            wp_map = self._wp.get((fi, ti))
            if wp_map is not None and wp_map.shape[1] > 0:
                cols.append(wp_map)

        mx, my, _ = self._node_map[node_seq[-1]]
        cols.append(np.array([[mx], [my]]))
        return np.hstack(cols)

    def route_length_m(self, node_seq):
        return sum(self._len.get((node_seq[i], node_seq[i + 1]), 0.0)
                   for i in range(len(node_seq) - 1))
