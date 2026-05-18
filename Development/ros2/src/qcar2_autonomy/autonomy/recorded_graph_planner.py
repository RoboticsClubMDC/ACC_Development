#!/usr/bin/env python3

import heapq
import math

import numpy as np

from autonomy.sdcs_graph_planner import SDCS_EDGES, _wrap_to_pi


class RecordedMapGraphPlanner:
    """
    Directed SDCS topology whose edge geometry comes from the taught ROS map.

    The hardcoded SDCS graph is still used for topology and rough edge length,
    but every published waypoint is sliced out of recorded_waypoints.  This keeps
    the car in the taught map frame and avoids applying the SDCS transform to the
    path it actually follows.
    """

    def __init__(
        self,
        recorded_waypoints,
        rough_graph,
        max_anchor_dist=0.25,
        max_node_candidates=8,
        min_candidate_separation=20,
        min_edge_length_ratio=0.35,
        max_edge_length_ratio=2.20,
        max_edge_length_slack_m=0.90,
    ):
        self._recorded = np.asarray(recorded_waypoints, dtype=float)
        self._rough_graph = rough_graph
        self._n = len(rough_graph._adj)
        self._adj = [[] for _ in range(self._n)]
        self._wp = {}
        self._len = {}
        self._node_map = []
        self._node_candidates = []
        self._missing_edges = []

        self._max_anchor_dist = float(max_anchor_dist)
        self._max_node_candidates = int(max_node_candidates)
        self._min_candidate_separation = int(min_candidate_separation)
        self._min_edge_length_ratio = float(min_edge_length_ratio)
        self._max_edge_length_ratio = float(max_edge_length_ratio)
        self._max_edge_length_slack_m = float(max_edge_length_slack_m)

        if self._recorded.ndim != 2 or self._recorded.shape[0] != 2 or self._recorded.shape[1] < 2:
            raise ValueError('recorded_waypoints must be a 2xN array with N >= 2')

        diffs = np.linalg.norm(np.diff(self._recorded, axis=1), axis=0)
        self._cumlen = np.concatenate(([0.0], np.cumsum(diffs)))

        self._build_node_candidates()
        self._build_edges()
        self._refresh_node_headings()

    def _build_node_candidates(self):
        for node_idx in range(self._n):
            rough_xy = self._rough_graph.node_position(node_idx).reshape(2, 1)
            dists = np.linalg.norm(self._recorded - rough_xy, axis=0)
            nearest = np.argsort(dists)

            kept = []
            for idx in nearest:
                idx = int(idx)
                dist = float(dists[idx])
                if dist > self._max_anchor_dist and kept:
                    continue
                if all(abs(idx - old_idx) > self._min_candidate_separation for old_idx, _ in kept):
                    kept.append((idx, dist))
                if len(kept) >= self._max_node_candidates:
                    break

            if not kept:
                idx = int(nearest[0])
                kept.append((idx, float(dists[idx])))

            self._node_candidates.append(kept)

            anchor_idx = kept[0][0]
            mx = float(self._recorded[0, anchor_idx])
            my = float(self._recorded[1, anchor_idx])
            rough_heading = float(self._rough_graph._node_map[node_idx][2])
            self._node_map.append((mx, my, rough_heading))

    def _segment_length(self, start_idx, end_idx):
        if end_idx <= start_idx:
            return None
        return float(self._cumlen[end_idx] - self._cumlen[start_idx])

    def _segment(self, start_idx, end_idx):
        return self._recorded[:, start_idx:end_idx + 1]

    def _choose_recorded_edge(self, fi, ti):
        rough_len = float(self._rough_graph._len.get((fi, ti), 0.0))
        best = None

        for start_idx, start_dist in self._node_candidates[fi]:
            for end_idx, end_dist in self._node_candidates[ti]:
                seg_len = self._segment_length(start_idx, end_idx)
                if seg_len is None or seg_len < 0.05:
                    continue

                if rough_len > 1e-6:
                    min_len = rough_len * self._min_edge_length_ratio
                    max_len = max(self._max_edge_length_slack_m,
                                  rough_len * self._max_edge_length_ratio)
                    if seg_len < min_len or seg_len > max_len:
                        continue

                score = abs(seg_len - rough_len) + 0.20 * (start_dist + end_dist)
                if best is None or score < best[0]:
                    best = (score, start_idx, end_idx, seg_len)

        return best

    def _build_edges(self):
        for fi, ti, _radius in SDCS_EDGES:
            chosen = self._choose_recorded_edge(fi, ti)
            if chosen is None:
                self._missing_edges.append((fi, ti))
                continue

            _score, start_idx, end_idx, seg_len = chosen
            self._wp[(fi, ti)] = self._segment(start_idx, end_idx)
            self._len[(fi, ti)] = seg_len
            self._adj[fi].append(ti)

    def _refresh_node_headings(self):
        refreshed = []
        for idx, (mx, my, rough_heading) in enumerate(self._node_map):
            heading = rough_heading
            if self._adj[idx]:
                edge = self._wp.get((idx, self._adj[idx][0]))
                if edge is not None and edge.shape[1] >= 2:
                    delta = edge[:, 1] - edge[:, 0]
                    if float(np.linalg.norm(delta)) > 1e-9:
                        heading = math.atan2(float(delta[1]), float(delta[0]))
            refreshed.append((mx, my, float(heading)))
        self._node_map = refreshed

    def node_position(self, idx):
        mx, my, _ = self._node_map[int(idx)]
        return np.array([mx, my], dtype=float)

    def node_positions_map(self):
        return [np.array([mx, my], dtype=float) for mx, my, _ in self._node_map]

    def find_nearest_node(self, xy_map, yaw_map, heading_weight=0.30, max_heading_error=1.57):
        best_idx = None
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
                best_idx = i
                best_score = score

        if best_idx is None:
            return self.find_nearest_node_position_only(xy_map)
        return best_idx, best_score

    def find_nearest_node_position_only(self, xy_map):
        best_idx = 0
        best_dist = float('inf')
        for i, (mx, my, _mth) in enumerate(self._node_map):
            dist = math.hypot(mx - float(xy_map[0]), my - float(xy_map[1]))
            if dist < best_dist:
                best_idx = i
                best_dist = dist
        return best_idx, best_dist

    def find_start_on_edge(self, xy_map, yaw_map, heading_weight=0.30, max_heading_error=1.57):
        robot = np.array([float(xy_map[0]), float(xy_map[1])], dtype=float)
        yaw = float(yaw_map)

        best = None
        best_score = float('inf')

        for (fi, ti), poly in self._wp.items():
            if poly.shape[1] < 2:
                continue

            best_k = 0
            best_t = 0.0
            best_d = float('inf')
            for k in range(poly.shape[1] - 1):
                a = poly[:, k]
                b = poly[:, k + 1]
                ab = b - a
                ab_sq = float(ab @ ab)
                if ab_sq < 1e-12:
                    continue
                t = float(np.clip((robot - a) @ ab / ab_sq, 0.0, 1.0))
                d = float(np.linalg.norm(a + t * ab - robot))
                if d < best_d:
                    best_k = k
                    best_t = t
                    best_d = d

            a = poly[:, best_k]
            b = poly[:, best_k + 1]
            ab = b - a
            if float(np.linalg.norm(ab)) < 1e-9:
                continue

            heading_err = abs(_wrap_to_pi(math.atan2(float(ab[1]), float(ab[0])) - yaw))
            if heading_err > max_heading_error:
                continue

            score = best_d + heading_weight * heading_err
            if score >= best_score:
                continue

            proj_pt = (a + best_t * ab).reshape(2, 1)
            remaining_wp = poly[:, best_k + 1:]
            best_score = score
            best = (fi, ti, proj_pt, remaining_wp, best_d, heading_err)

        return best

    def astar(self, start_idx, goal_idx):
        start_idx = int(start_idx)
        goal_idx = int(goal_idx)
        if start_idx == goal_idx:
            return [start_idx]

        gx, gy, _ = self._node_map[goal_idx]

        def h(node_idx):
            ix, iy, _ = self._node_map[node_idx]
            return math.hypot(gx - ix, gy - iy)

        heap = []
        counter = 0
        heapq.heappush(heap, (h(start_idx), counter, start_idx))
        came = {}
        g_score = [float('inf')] * self._n
        g_score[start_idx] = 0.0
        visited = set()

        while heap:
            _priority, _counter, cur = heapq.heappop(heap)
            if cur in visited:
                continue
            visited.add(cur)

            if cur == goal_idx:
                path = []
                node = cur
                while node in came:
                    path.append(node)
                    node = came[node]
                path.append(start_idx)
                path.reverse()
                return path

            for nb in self._adj[cur]:
                if nb in visited:
                    continue
                tentative = g_score[cur] + self._len[(cur, nb)]
                if tentative < g_score[nb]:
                    came[nb] = cur
                    g_score[nb] = tentative
                    counter += 1
                    heapq.heappush(heap, (tentative + h(nb), counter, nb))

        return None

    def build_path(self, node_seq):
        if not node_seq:
            return np.empty((2, 0))
        if len(node_seq) == 1:
            return self.node_position(node_seq[0]).reshape(2, 1)

        cols = []
        for i in range(len(node_seq) - 1):
            key = (int(node_seq[i]), int(node_seq[i + 1]))
            edge = self._wp.get(key)
            if edge is None or edge.shape[1] == 0:
                return np.empty((2, 0))
            cols.append(edge if not cols else edge[:, 1:])
        return np.hstack(cols) if cols else np.empty((2, 0))

    def route_length_m(self, node_seq):
        return sum(self._len.get((int(node_seq[i]), int(node_seq[i + 1])), 0.0)
                   for i in range(len(node_seq) - 1))
