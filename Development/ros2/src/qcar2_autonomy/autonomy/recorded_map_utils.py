from pathlib import Path
import json
import math
import numpy as np


PACKAGE_NAME = 'qcar2_autonomy'


def get_source_package_root():
    here = Path(__file__).resolve()

    for parent in here.parents:
        if parent.name == PACKAGE_NAME and (parent / 'package.xml').exists():
            return parent

    for parent in here.parents:
        if parent.name == 'install':
            workspace_root = parent.parent
            for candidate in (
                workspace_root / 'ros2' / 'src' / PACKAGE_NAME,
                workspace_root / 'src' / PACKAGE_NAME,
            ):
                if (candidate / 'package.xml').exists():
                    return candidate

    cwd = Path.cwd().resolve()
    search_roots = [cwd] + list(cwd.parents)
    for root in search_roots:
        for candidate in (
            root / 'src' / PACKAGE_NAME,
            root / 'Development' / 'ros2' / 'src' / PACKAGE_NAME,
        ):
            if (candidate / 'package.xml').exists():
                return candidate

    return here.parent.parent


def get_recording_maps_dir():
    maps_dir = get_source_package_root() / 'recording_maps'
    maps_dir.mkdir(parents=True, exist_ok=True)
    return maps_dir


def find_latest_recording_map():
    maps_dir = get_recording_maps_dir()
    json_files = sorted(maps_dir.glob('*.json'), key=lambda p: p.stat().st_mtime)
    return json_files[-1] if json_files else None


def load_recording_map(path):
    return json.loads(Path(path).read_text())


def recorded_nodes_to_array(nodes):
    if not nodes:
        return np.zeros((2, 0), dtype=float)
    pts = [(float(node['x']), float(node['y'])) for node in nodes]
    return np.array(pts, dtype=float).T


def _wrap_to_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def filter_recorded_nodes(nodes, min_distance=0.08, min_yaw_change=0.20):
    if not nodes:
        return []

    filtered = [nodes[0]]
    last_kept = nodes[0]
    last_index = len(nodes) - 1

    for idx, node in enumerate(nodes[1:], start=1):
        dx = float(node['x']) - float(last_kept['x'])
        dy = float(node['y']) - float(last_kept['y'])
        dist = math.hypot(dx, dy)

        yaw = float(node.get('yaw', 0.0))
        last_yaw = float(last_kept.get('yaw', 0.0))
        dyaw = abs(_wrap_to_pi(yaw - last_yaw))

        if dist >= min_distance or dyaw >= min_yaw_change or idx == last_index:
            filtered.append(node)
            last_kept = node

    return filtered


def densify_polyline(points_2xn, spacing=0.03):
    if points_2xn.size == 0:
        return np.zeros((2, 0), dtype=float)

    spacing = max(float(spacing), 1e-3)
    dense_points = [points_2xn[:, 0].astype(float)]

    for idx in range(points_2xn.shape[1] - 1):
        p0 = points_2xn[:, idx].astype(float)
        p1 = points_2xn[:, idx + 1].astype(float)
        seg = p1 - p0
        seg_len = float(np.linalg.norm(seg))

        if seg_len < 1e-9:
            continue

        steps = max(1, int(math.ceil(seg_len / spacing)))
        for step in range(1, steps + 1):
            alpha = step / steps
            dense_points.append(p0 + alpha * seg)

    return np.array(dense_points, dtype=float).T


def build_waypoint_path_from_nodes(
    nodes,
    min_node_spacing=0.08,
    min_yaw_change=0.20,
    waypoint_spacing=0.03,
):
    filtered_nodes = filter_recorded_nodes(
        nodes,
        min_distance=min_node_spacing,
        min_yaw_change=min_yaw_change,
    )
    control_points = recorded_nodes_to_array(filtered_nodes)
    dense_waypoints = densify_polyline(control_points, spacing=waypoint_spacing)
    return filtered_nodes, control_points, dense_waypoints


def closest_recorded_node_index(points_2xn, xy):
    if points_2xn.size == 0:
        return None
    q = np.array(xy, dtype=float).reshape(2, 1)
    d = np.linalg.norm(points_2xn - q, axis=0)
    return int(np.argmin(d))


def path_length(points_2xn):
    if points_2xn.shape[1] < 2:
        return 0.0
    diffs = np.diff(points_2xn, axis=1)
    return float(np.sum(np.linalg.norm(diffs, axis=0)))


def build_recorded_segment(points_2xn, start_idx, goal_idx, closed_loop=False):
    if points_2xn.size == 0:
        return np.zeros((2, 0), dtype=float)

    n = points_2xn.shape[1]
    start_idx = int(np.clip(start_idx, 0, n - 1))
    goal_idx = int(np.clip(goal_idx, 0, n - 1))

    if start_idx <= goal_idx:
        direct = points_2xn[:, start_idx:goal_idx + 1]
    else:
        direct = np.flip(points_2xn[:, goal_idx:start_idx + 1], axis=1)

    if not closed_loop or n < 3:
        return direct

    if start_idx <= goal_idx:
        wrap = np.hstack([points_2xn[:, goal_idx:], points_2xn[:, :start_idx + 1]])
        wrap = np.flip(wrap, axis=1)
    else:
        wrap = np.hstack([points_2xn[:, start_idx:], points_2xn[:, :goal_idx + 1]])

    return wrap if path_length(wrap) < path_length(direct) else direct


def build_dense_recorded_segment(
    points_2xn,
    start_idx,
    goal_idx,
    spacing=0.03,
    closed_loop=False,
):
    segment = build_recorded_segment(
        points_2xn,
        start_idx,
        goal_idx,
        closed_loop=closed_loop,
    )
    return densify_polyline(segment, spacing=spacing)
