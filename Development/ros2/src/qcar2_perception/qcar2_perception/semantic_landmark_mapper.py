#!/usr/bin/env python3
# Semantic map rule:
# The map is remembered evidence, not eternal truth.
# Same-class observations strengthen a landmark.
# Different static classes inside the same bubble create conflict evidence.
# Dynamic classes inside the bubble are treated as temporary occlusion.
import json
import math
from pathlib import Path

import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node

import tf2_geometry_msgs  # noqa: F401  Registers geometry transforms with tf2.
import tf2_ros
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


# χ² critical values for the 3D Mahalanobis association gate (spec §14).
CHI2_3_95 = 7.815
CHI2_3_99 = 11.345


def yaw_from_quat(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_to_rot(qx, qy, qz, qw):
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1.0 - (qy * qy + qz * qz) * s, (qx * qy - qz * qw) * s,       (qx * qz + qy * qw) * s],
        [(qx * qy + qz * qw) * s,       1.0 - (qx * qx + qz * qz) * s, (qy * qz - qx * qw) * s],
        [(qx * qz - qy * qw) * s,       (qy * qz + qx * qw) * s,       1.0 - (qx * qx + qy * qy) * s],
    ])


class SemanticLandmarkMapper(Node):
    """
    Turns live 3D object observations into persistent semantic landmarks.

    Responsibilities:
    - Subscribe to object observations from object_3d_estimator.
    - Transform camera-frame points into the Cartographer map frame.
    - Match repeated observations into candidate/confirmed/stable landmarks.
    - Publish semantic landmark JSON.
    - Publish RViz/Foxglove markers in map.
    - Save a small JSON semantic memory file.

    Non-responsibilities:
    - Does not run YOLO.
    - Does not estimate depth.
    - Does not command vehicle motion.
    - Does not correct Cartographer.
    """

    def __init__(self):
        super().__init__("semantic_landmark_mapper")

        self.declare_parameter("objects_topic", "/perception/objects_3d")
        self.declare_parameter("landmarks_topic", "/perception/semantic_landmarks")
        self.declare_parameter(
            "markers_topic",
            "/perception/semantic_landmark_markers",
        )

        # Added 2026-05-20 16:08:30 EDT:
        # Split semantic visualization into stable landmarks, hypotheses, and
        # currently visible semantic observations so Foxglove is not ambiguous.
        self.declare_parameter(
            "hypothesis_markers_topic",
            "/perception/semantic_hypothesis_markers",
        )
        self.declare_parameter(
            "current_markers_topic",
            "/perception/semantic_current_markers",
        )
        # Phase-3 (added 2026-05-25): covariance ellipsoids for each Kalman
        # landmark, so Foxglove shows "how certain is this landmark?".
        # Spec §20-A. The gate ellipsoid (S = P_l + R_obs) is per-detection
        # and is not visualized in v1.
        self.declare_parameter(
            "covariance_markers_topic",
            "/perception/semantic_landmark_cov_markers",
        )
        # Mahalanobis "sigma units" used to scale the visualized ellipsoid.
        # 2.0 ≈ 95% in 1D / 86% in 3D for Gaussian. 2.795 ≈ 95% in 3D Chi-square.
        self.declare_parameter("covariance_sigma_scale", 2.0)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter(
            "semantic_map_path",
            "/workspaces/isaac_ros-dev/ros2/src/qcar2_perception/maps/semantic_map.json",
        )
        self.declare_parameter("association_radius", 0.45)
        self.declare_parameter("confirmed_seen_count", 3)
        self.declare_parameter("stable_seen_count", 8)
        # Semantic conflict handling:
        # Same-place different-class observations should not overwrite immediately.
        # They first become conflict evidence and only replace after repeated proof.
        self.declare_parameter("conflict_evidence_count", 5)
        self.declare_parameter("dynamic_class_names", [
            "person",
            "car",
            "truck",
            "bus",
            "bicycle",
            "motorcycle",
            "robot",
            "cow",
        ])
        self.declare_parameter("max_history", 20)
        self.declare_parameter("enable_json_save", True)
        self.declare_parameter("publish_rate_hz", 2.0)
        self.declare_parameter("tf_timeout_sec", 0.15)

        # Added 2026-05-20 15:42:04 EDT:
        # Competition semantic memory is a run-local overlay on Cartographer.
        # Start clean each perception launch unless persistent reload is enabled.
        self.declare_parameter("reset_map_on_start", True)

        # Added 2026-05-20 13:58:39 EDT:
        # Candidates live in RAM for matching, but only permanent statuses are
        # written to semantic_map.json. XY gating is less fragile for signs.
        self.declare_parameter("permanent_statuses", ["stable"])
        self.declare_parameter("load_only_permanent", True)
        self.declare_parameter("association_use_xy_only", True)

        # Added 2026-05-20 13:47:02 EDT:
        # Separate long-term landmark confidence from whether YOLO sees it now.
        self.declare_parameter("visible_timeout_sec", 2.0)
        self.declare_parameter("max_marker_radius", 0.50)

        # ---- Phase-2 Kalman landmark filter params (added 2026-05-25) ----
        # When an observation arrives with position_map + R_obs (from
        # object_3d_estimator Phase-1), we run a proper 3D Kalman update
        # instead of the legacy distance gate + blended average. See spec §14-§18.
        self.declare_parameter("use_kalman_path", True)
        self.declare_parameter("mahalanobis_chi2", CHI2_3_99)        # spec §14, 11.345
        self.declare_parameter("ambiguity_ratio", 0.6)              # spec §15
        # Process noise per tick (static landmark drift); spec §13
        self.declare_parameter("q_landmark_xy", 1.0e-5)
        self.declare_parameter("q_landmark_z", 1.0e-5)
        # Stable-promotion thresholds (spec §18)
        self.declare_parameter("stable_sigma_xy_max", 0.08)
        self.declare_parameter("stable_sigma_z_max", 0.12)
        self.declare_parameter("stable_confidence_min", 0.70)
        # Visibility / miss-count thresholds (this is on top of spec — it answers
        # "I saw a stop sign here; if I drive past again, is it still there?")
        self.declare_parameter("enable_visibility_check", True)
        self.declare_parameter("pose_topic", "/qcar2_pose_fused")
        self.declare_parameter("camera_frame", "aligned_camera_optical_frame")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("fov_horizontal_deg", 87.0)           # D435 RGB ~87°
        self.declare_parameter("fov_vertical_deg", 58.0)
        # Raised 2026-05-25 from 3.0 to 6.0 to match object_3d_estimator's
        # max_depth bump for traffic-light / stop-sign approaches.
        self.declare_parameter("max_visibility_range_m", 6.0)
        # Per-state miss thresholds before demotion/removal
        self.declare_parameter("stable_miss_demote", 10)
        self.declare_parameter("confirmed_miss_demote", 6)
        self.declare_parameter("candidate_miss_remove", 4)

        # ---- Phase-4 landmark→EKF pose correction (added 2026-05-25) ----
        # OFF BY DEFAULT. Only flip to true once stable landmarks have been
        # validated against ground truth across multiple drives. See
        # CLAUDE.md §7 roadmap item 9 for the gates we want cleared first.
        self.declare_parameter("enable_landmark_correction", False)
        self.declare_parameter("landmark_correction_topic", "/perception/landmark_pose_correction")
        # Only emit corrections from landmarks that exceed this hit count, on
        # top of the stable promotion. Extra safety margin.
        self.declare_parameter("min_stable_hits_for_correction", 12)
        # Yaw covariance to publish — set huge so ekf_fusor's measurement
        # update treats this as an x/y-only fix and ignores yaw.
        self.declare_parameter("correction_yaw_variance", 1.0e6)

        objects_topic = str(self.get_parameter("objects_topic").value)
        landmarks_topic = str(self.get_parameter("landmarks_topic").value)
        markers_topic = str(self.get_parameter("markers_topic").value)
        hypothesis_markers_topic = str(
            self.get_parameter("hypothesis_markers_topic").value
        )
        current_markers_topic = str(
            self.get_parameter("current_markers_topic").value
        )

        self.map_frame = str(self.get_parameter("map_frame").value)
        self.semantic_map_path = Path(
            str(self.get_parameter("semantic_map_path").value)
        )
        self.association_radius = float(
            self.get_parameter("association_radius").value
        )
        self.confirmed_seen_count = int(
            self.get_parameter("confirmed_seen_count").value
        )
        self.stable_seen_count = int(self.get_parameter("stable_seen_count").value)
        self.conflict_evidence_count = int(
            self.get_parameter("conflict_evidence_count").value
        )
        self.dynamic_class_names = set(
            self.normalize_class_name(name)
            for name in self.get_parameter("dynamic_class_names").value
        )
        self.max_history = int(self.get_parameter("max_history").value)
        self.enable_json_save = bool(self.get_parameter("enable_json_save").value)
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)
        self.reset_map_on_start = bool(
            self.get_parameter("reset_map_on_start").value
        )
        self.permanent_statuses = set(
            str(status)
            for status in self.get_parameter("permanent_statuses").value
        )
        self.load_only_permanent = bool(
            self.get_parameter("load_only_permanent").value
        )
        self.association_use_xy_only = bool(
            self.get_parameter("association_use_xy_only").value
        )
        self.visible_timeout_sec = float(
            self.get_parameter("visible_timeout_sec").value
        )
        self.max_marker_radius = float(self.get_parameter("max_marker_radius").value)

        # Phase-2 Kalman params
        self.use_kalman_path = bool(self.get_parameter("use_kalman_path").value)
        self.mahalanobis_chi2 = float(self.get_parameter("mahalanobis_chi2").value)
        self.ambiguity_ratio = float(self.get_parameter("ambiguity_ratio").value)
        q_xy = float(self.get_parameter("q_landmark_xy").value)
        q_z = float(self.get_parameter("q_landmark_z").value)
        self.Q_landmark = np.diag([q_xy, q_xy, q_z])
        self.stable_sigma_xy_max = float(self.get_parameter("stable_sigma_xy_max").value)
        self.stable_sigma_z_max = float(self.get_parameter("stable_sigma_z_max").value)
        self.stable_confidence_min = float(self.get_parameter("stable_confidence_min").value)
        self.enable_visibility_check = bool(self.get_parameter("enable_visibility_check").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.fov_h_rad = math.radians(float(self.get_parameter("fov_horizontal_deg").value))
        self.fov_v_rad = math.radians(float(self.get_parameter("fov_vertical_deg").value))
        self.max_visibility_range_m = float(self.get_parameter("max_visibility_range_m").value)
        self.stable_miss_demote = int(self.get_parameter("stable_miss_demote").value)
        self.confirmed_miss_demote = int(self.get_parameter("confirmed_miss_demote").value)
        self.candidate_miss_remove = int(self.get_parameter("candidate_miss_remove").value)
        # Phase-4 correction params
        self.enable_landmark_correction = bool(self.get_parameter("enable_landmark_correction").value)
        landmark_correction_topic = str(self.get_parameter("landmark_correction_topic").value)
        self.min_stable_hits_for_correction = int(self.get_parameter("min_stable_hits_for_correction").value)
        self.correction_yaw_variance = float(self.get_parameter("correction_yaw_variance").value)
        pose_topic = str(self.get_parameter("pose_topic").value)

        # Cache latest robot pose (map frame) for visibility check
        self.latest_robot_xy_theta = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            pose_topic,
            self.pose_cb,
            10,
        )

        self.landmarks = []
        self.next_landmark_index = 1
        self.dirty = False

        if self.reset_map_on_start:
            self.get_logger().info(
                "reset_map_on_start=True, starting with empty semantic session map."
            )
            self.dirty = True
            if self.enable_json_save:
                self.save_semantic_map()
        else:
            self.load_semantic_map()

        self.objects_sub = self.create_subscription(
            String,
            objects_topic,
            self.objects_cb,
            10,
        )
        self.landmarks_pub = self.create_publisher(String, landmarks_topic, 10)
        self.markers_pub = self.create_publisher(MarkerArray, markers_topic, 10)
        self.hypothesis_markers_pub = self.create_publisher(
            MarkerArray,
            hypothesis_markers_topic,
            10,
        )
        self.current_markers_pub = self.create_publisher(
            MarkerArray,
            current_markers_topic,
            10,
        )
        cov_markers_topic = str(self.get_parameter("covariance_markers_topic").value)
        self.covariance_sigma_scale = float(self.get_parameter("covariance_sigma_scale").value)
        self.cov_markers_pub = self.create_publisher(MarkerArray, cov_markers_topic, 10)
        self.landmark_correction_pub = self.create_publisher(
            PoseWithCovarianceStamped, landmark_correction_topic, 10,
        )

        publish_period = 1.0 / max(
            0.1,
            float(self.get_parameter("publish_rate_hz").value),
        )
        self.publish_timer = self.create_timer(publish_period, self.publish_state)

        self.get_logger().info(
            "semantic_landmark_mapper ready. "
            f"objects={objects_topic}, map_frame={self.map_frame}, "
            f"memory={self.semantic_map_path}, stable_markers={markers_topic}, "
            f"hypothesis_markers={hypothesis_markers_topic}, "
            f"current_markers={current_markers_topic}"
        )

    def pose_cb(self, msg):
        """Cache /qcar2_pose_fused yaw + xy for visibility prediction."""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        theta = yaw_from_quat(q.x, q.y, q.z, q.w)
        if not all(map(math.isfinite, [p.x, p.y, theta])):
            return
        self.latest_robot_xy_theta = (float(p.x), float(p.y), float(theta))

    def objects_cb(self, msg):
        """
        Main mapper callback.

        For each detection:
          * If position_map + R_obs are present (Phase-1 estimator output and
            use_kalman_path=True): run the proper Mahalanobis 3D association
            + Joseph-form Kalman update (spec §14-§18).
          * Otherwise: fall back to the legacy distance-gate + blended average
            path. Keeps things working if anyone runs the old estimator.

        After the batch: run the visibility check — for confirmed/stable
        landmarks that should have been in the camera FOV but were NOT seen
        this tick, bump miss_count and demote/remove on threshold.
        """
        try:
            objects_msg = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"Could not parse Object3D JSON: {exc}")
            return

        source_frame = objects_msg.get("frame_id")
        if not source_frame:
            source_frame = objects_msg.get("header", {}).get("frame_id", "")

        if not source_frame:
            self.get_logger().warn("Object message has no source frame.")
            return

        stamp_msg = self.stamp_from_header(objects_msg.get("header", {}))
        updated = 0
        observed_landmark_ids = set()

        # Predict step (spec §13): bump every Kalman landmark's covariance once
        # per detection batch. Static-landmark prediction is a no-op on mean.
        if self.use_kalman_path:
            for landmark in self.landmarks:
                if landmark.get("kf"):
                    cov = np.array(landmark["cov_flat"], dtype=np.float64).reshape(3, 3)
                    cov = cov + self.Q_landmark
                    landmark["cov_flat"] = cov.flatten().tolist()

        for obj in objects_msg.get("objects", []):
            use_kf = (
                self.use_kalman_path
                and "position_map" in obj
                and "R_obs" in obj
            )

            if use_kf:
                landmark = self.kf_process_observation(obj)
                if landmark is not None:
                    observed_landmark_ids.add(landmark["id"])
                    self.dirty = True
                    updated += 1
                continue

            # Legacy path (no R_obs in this obj) — pre-Phase-1 estimator.
            point_map = self.transform_object_to_map(obj, source_frame, stamp_msg)
            if point_map is None:
                continue

            same_class_landmark = self.find_matching_landmark(obj, point_map)
            if same_class_landmark is not None:
                self.update_landmark(same_class_landmark, obj, point_map)
                observed_landmark_ids.add(same_class_landmark["id"])
            else:
                spatial_landmark = self.find_spatial_landmark(point_map)
                if spatial_landmark is not None:
                    self.handle_semantic_conflict(spatial_landmark, obj, point_map)
                else:
                    landmark = self.create_landmark(obj, point_map)
                    self.landmarks.append(landmark)
                    observed_landmark_ids.add(landmark["id"])

            self.dirty = True
            updated += 1

        # Visibility tick — bump miss_count for landmarks that SHOULD have been
        # seen but weren't this cycle. Only runs when we actually received a
        # detections message (so quiet YOLO ticks still count as "no detect
        # of an expected landmark"). Skipped if robot pose / camera TF unknown.
        if self.enable_visibility_check and self.use_kalman_path:
            self.visibility_tick(observed_landmark_ids)

        if updated > 0 or self.enable_visibility_check:
            self.publish_state()
            if self.enable_json_save and self.dirty:
                self.save_semantic_map()

    # ============= Kalman landmark filter (spec §13-§18) =============

    def kf_process_observation(self, obj):
        """
        Run one Kalman association+update cycle for a single detection.
        Returns the affected landmark (existing or newly created), or None.
        """
        z = np.array(obj["position_map"], dtype=np.float64)
        R_obs = np.array(obj["R_obs"], dtype=np.float64)
        if z.shape != (3,) or R_obs.shape != (3, 3) or not (np.all(np.isfinite(z)) and np.all(np.isfinite(R_obs))):
            return None

        class_name = obj.get("class_name", "unknown")
        same_class = [l for l in self.landmarks if l.get("kf") and l.get("class_name") == class_name]

        # Compute d² against each same-class landmark
        scored = []
        for lm in same_class:
            mean = np.array(lm["mean"], dtype=np.float64)
            cov = np.array(lm["cov_flat"], dtype=np.float64).reshape(3, 3)
            y = z - mean
            S = cov + R_obs                       # spec §14, H = I_3
            try:
                Sinv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                continue
            d2 = float(y @ Sinv @ y)
            if math.isfinite(d2):
                scored.append((d2, lm, S, Sinv, y))

        scored.sort(key=lambda x: x[0])

        accept = None
        if scored:
            d2_best, lm_best, S_best, Sinv_best, y_best = scored[0]
            if d2_best <= self.mahalanobis_chi2:
                # Spec §15 ambiguity rule: only accept if clearly the best.
                if len(scored) == 1 or d2_best < self.ambiguity_ratio * scored[1][0]:
                    accept = (lm_best, S_best, Sinv_best, y_best, d2_best)
                # If ambiguous, skip update; we'll still treat the observation
                # as a hint by not creating a new landmark (could fight existing).

        if accept is not None:
            lm, S, Sinv, y, d2 = accept
            self.kf_joseph_update(lm, z, R_obs, S, Sinv, y, d2, obj)
            # Phase-4: after the Kalman update, if the landmark is stable AND
            # passes the extra hit-count safety gate, publish the implied
            # robot-pose correction so ekf_fusor can consume it. Gated by
            # `enable_landmark_correction` parameter (default OFF).
            if (
                self.enable_landmark_correction
                and lm.get("status") == "stable"
                and int(lm.get("hit_count", 0)) >= self.min_stable_hits_for_correction
                and "position_base" in obj
            ):
                self.publish_landmark_pose_correction(lm, z, R_obs, obj)
            return lm

        # Spec §17: no association → new candidate landmark.
        new_lm = self.kf_create_landmark(obj, z, R_obs)
        self.landmarks.append(new_lm)
        return new_lm

    def publish_landmark_pose_correction(self, lm, z, R_obs, obj):
        """
        Phase-4: back-solve the implied robot pose from a stable-landmark match
        and publish as PoseWithCovarianceStamped on /perception/landmark_pose_correction.

        Math:
          z (observed in map)  = robot_xy + R(theta) * p_b
          L (known stable lm)  = TRUE_robot_xy + R(theta) * p_b
          ==> implied_robot_xy = current_robot_xy + (L - z)
                              = current_robot_xy + (lm.mean - z)
        Covariance(implied_robot_xy) ≈ P_l + R_obs
        Yaw is intentionally NOT corrected from a single point landmark — we
        publish a huge yaw variance so ekf_fusor's measurement update only
        adjusts x/y.
        """
        if self.latest_robot_xy_theta is None:
            return

        x_r, y_r, theta = self.latest_robot_xy_theta
        L = np.array(lm["mean"], dtype=np.float64)
        P_l = np.array(lm["cov_flat"], dtype=np.float64).reshape(3, 3)

        delta = L - z   # if z drifted right, robot is actually to the right by the same delta
        implied_xy = np.array([x_r, y_r]) + delta[:2]
        if not np.all(np.isfinite(implied_xy)):
            return

        # x/y covariance is innovation covariance of the residual
        cov_xy = P_l[:2, :2] + R_obs[:2, :2]

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.pose.pose.position.x = float(implied_xy[0])
        msg.pose.pose.position.y = float(implied_xy[1])
        msg.pose.pose.position.z = 0.0
        # Quaternion = current yaw (we're not correcting yaw, but the field
        # must be filled so subscribers don't trip on identity-zero quats).
        half = 0.5 * theta
        msg.pose.pose.orientation.z = math.sin(half)
        msg.pose.pose.orientation.w = math.cos(half)

        # 6x6 covariance over (x, y, z, roll, pitch, yaw). Fill x/y from
        # innovation; rest are huge so ekf_fusor's measurement update treats
        # this as an x/y-only fix.
        cov = [1.0e6] * 36
        cov[0] = float(cov_xy[0, 0])
        cov[1] = float(cov_xy[0, 1])
        cov[6] = float(cov_xy[1, 0])
        cov[7] = float(cov_xy[1, 1])
        cov[35] = float(self.correction_yaw_variance)
        msg.pose.covariance = cov
        self.landmark_correction_pub.publish(msg)

    def kf_joseph_update(self, lm, z, R_obs, S, Sinv, y, d2, obj):
        """Joseph-form Kalman update. H = I, so K = P · S^{-1}, P = (I-K)P(I-K)^T + K R K^T."""
        P = np.array(lm["cov_flat"], dtype=np.float64).reshape(3, 3)
        mean = np.array(lm["mean"], dtype=np.float64)

        K = P @ Sinv                                # spec §16
        new_mean = mean + K @ y
        IKH = np.eye(3) - K                         # H = I_3
        new_P = IKH @ P @ IKH.T + K @ R_obs @ K.T
        new_P = 0.5 * (new_P + new_P.T)             # symmetry hygiene

        now_sec = self.now_sec()
        prev_hits = int(lm.get("hit_count", 0))
        prev_conf = float(lm.get("confidence_avg", float(obj.get("confidence", 0.0))))
        new_hits = prev_hits + 1
        new_conf = prev_conf + (float(obj.get("confidence", 0.0)) - prev_conf) / new_hits

        lm["mean"] = [float(new_mean[0]), float(new_mean[1]), float(new_mean[2])]
        lm["cov_flat"] = new_P.flatten().tolist()
        # Mirror to legacy 'pose_map' so existing visualization works untouched.
        lm["pose_map"] = {"x": float(new_mean[0]), "y": float(new_mean[1]), "z": float(new_mean[2])}
        lm["hit_count"] = new_hits
        lm["seen_count"] = new_hits  # legacy alias
        lm["miss_count"] = 0
        lm["last_seen"] = now_sec
        lm["last_d2"] = float(d2)
        lm["confidence_avg"] = float(new_conf)
        lm["confidence"] = float(new_conf)  # legacy alias
        lm["visibility"] = "visible"
        lm["status"] = self.kf_status(lm)
        lm["source_sensor"] = obj.get("source_sensor", "unknown")
        lm["last_detection_id"] = obj.get("detection_id", "")

    def kf_create_landmark(self, obj, z, R_obs):
        """Spec §17: new candidate with P_l_0 = R_obs + P_birth (small)."""
        P_birth = np.diag([0.02 ** 2, 0.02 ** 2, 0.03 ** 2])
        P0 = R_obs + P_birth
        now_sec = self.now_sec()
        return {
            "id": self.make_landmark_id(obj),
            "kf": True,                              # Kalman-managed
            "class_id": int(obj.get("class_id", -1)),
            "class_name": obj.get("class_name", "unknown"),
            "mean": [float(z[0]), float(z[1]), float(z[2])],
            "cov_flat": P0.flatten().tolist(),
            "pose_map": {"x": float(z[0]), "y": float(z[1]), "z": float(z[2])},
            "uncertainty_radius": float(np.sqrt(P0[0, 0] + P0[1, 1] + P0[2, 2])),
            "hit_count": 1,
            "seen_count": 1,
            "miss_count": 0,
            "confidence_avg": float(obj.get("confidence", 0.0)),
            "confidence": float(obj.get("confidence", 0.0)),
            "first_seen": now_sec,
            "last_seen": now_sec,
            "last_d2": 0.0,
            "source_sensor": obj.get("source_sensor", "unknown"),
            "status": "candidate",
            "visibility": "visible",
            "last_detection_id": obj.get("detection_id", ""),
            "history": [],
        }

    def kf_status(self, lm):
        """
        Spec §18 promotion: status from hit_count + sqrt(P_diag) + confidence_avg.
        """
        hits = int(lm.get("hit_count", 0))
        if hits < self.confirmed_seen_count:
            return "candidate"
        if hits < self.stable_seen_count:
            return "confirmed"
        P = np.array(lm["cov_flat"], dtype=np.float64).reshape(3, 3)
        sigma_xx = math.sqrt(max(P[0, 0], 0.0))
        sigma_yy = math.sqrt(max(P[1, 1], 0.0))
        sigma_zz = math.sqrt(max(P[2, 2], 0.0))
        if (
            sigma_xx < self.stable_sigma_xy_max
            and sigma_yy < self.stable_sigma_xy_max
            and sigma_zz < self.stable_sigma_z_max
            and float(lm.get("confidence_avg", 0.0)) > self.stable_confidence_min
        ):
            return "stable"
        return "confirmed"

    # ============= Visibility check (the "is my sign still here?" answer) =============

    def visibility_tick(self, observed_ids):
        """
        For each Kalman-managed landmark that should be in camera FOV but
        wasn't observed this cycle: bump miss_count. Demote/remove on threshold.
        """
        if self.latest_robot_xy_theta is None:
            return

        # Need T_base<-cam (static, set up by semantic_tf_launch.py)
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_frame, rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
        except Exception:
            return

        t = tf.transform.translation
        q = tf.transform.rotation
        t_bc = np.array([t.x, t.y, t.z])
        R_bc = quat_to_rot(q.x, q.y, q.z, q.w)
        R_cb = R_bc.T                   # cam <- base
        t_cb = -R_cb @ t_bc

        x_r, y_r, theta = self.latest_robot_xy_theta
        st, ct = math.sin(theta), math.cos(theta)
        R_bm = np.array([[ct, st, 0.0], [-st, ct, 0.0], [0.0, 0.0, 1.0]])  # base <- map

        survivors = []
        for lm in self.landmarks:
            if not lm.get("kf"):
                survivors.append(lm)
                continue
            if lm["id"] in observed_ids:
                survivors.append(lm)
                continue
            if lm.get("status") == "candidate" and int(lm.get("hit_count", 0)) < 2:
                # Don't let very fresh hypotheses miss-count themselves out.
                survivors.append(lm)
                continue

            mean = np.array(lm["mean"], dtype=np.float64)
            # map -> base
            p_b = R_bm @ (mean - np.array([x_r, y_r, 0.0]))
            # base -> cam
            p_c = R_cb @ p_b + t_cb
            x_c, y_c, z_c = p_c[0], p_c[1], p_c[2]
            # Standard OpenCV/ROS optical frame: +Z forward, +X right, +Y down.
            if z_c <= 0.0 or z_c > self.max_visibility_range_m:
                survivors.append(lm)
                continue
            ang_h = math.atan2(abs(x_c), z_c)
            ang_v = math.atan2(abs(y_c), z_c)
            if ang_h > 0.5 * self.fov_h_rad or ang_v > 0.5 * self.fov_v_rad:
                survivors.append(lm)
                continue

            # In FOV but not observed → miss
            misses = int(lm.get("miss_count", 0)) + 1
            lm["miss_count"] = misses
            status = lm.get("status", "candidate")
            # Demotion / removal
            if status == "stable" and misses >= self.stable_miss_demote:
                lm["status"] = "confirmed"
                lm["miss_count"] = 0
                self.get_logger().info(
                    f"landmark {lm['id']} ({lm.get('class_name')}) demoted stable->confirmed (missed {misses} expected views)"
                )
                survivors.append(lm)
            elif status == "confirmed" and misses >= self.confirmed_miss_demote:
                lm["status"] = "candidate"
                lm["miss_count"] = 0
                self.get_logger().info(
                    f"landmark {lm['id']} ({lm.get('class_name')}) demoted confirmed->candidate (missed {misses} expected views)"
                )
                survivors.append(lm)
            elif status == "candidate" and misses >= self.candidate_miss_remove:
                self.get_logger().info(
                    f"landmark {lm['id']} ({lm.get('class_name')}) removed (missed {misses} expected views as candidate)"
                )
                self.dirty = True
                # do NOT add to survivors → it's gone
            else:
                survivors.append(lm)

        if len(survivors) != len(self.landmarks):
            self.landmarks = survivors
            self.dirty = True

    def transform_object_to_map(self, obj, source_frame, stamp_msg):
        """
        Uses tf2 to transform one object point from camera frame into map.
        """
        pose_camera = obj.get("pose_camera", {})

        try:
            point = PointStamped()
            point.header.frame_id = source_frame
            point.header.stamp = stamp_msg
            point.point.x = float(pose_camera["x"])
            point.point.y = float(pose_camera["y"])
            point.point.z = float(pose_camera["z"])

            return self.tf_buffer.transform(
                point,
                self.map_frame,
                timeout=Duration(seconds=self.tf_timeout_sec),
            )
        except Exception as exc:
            self.get_logger().warn(
                f"Could not transform {source_frame} -> {self.map_frame}: {exc}",
                throttle_duration_sec=2.0,
            )
            return None

    def find_matching_landmark(self, obj, point_map):
        """
        Finds the nearest existing landmark with the same class inside the gate.
        """
        class_name = obj.get("class_name", "unknown")

        best_landmark = None
        best_distance = float("inf")

        for landmark in self.landmarks:
            if landmark.get("class_name") != class_name:
                continue

            distance = self.distance_to_landmark(landmark, point_map)
            gate = max(
                self.association_radius,
                float(landmark.get("uncertainty_radius", 0.20)),
                float(obj.get("uncertainty_radius", 0.20)),
            )

            if distance < gate and distance < best_distance:
                best_landmark = landmark
                best_distance = distance

        return best_landmark
    
    def find_spatial_landmark(self, point_map):
        """
        Finds the nearest landmark bubble by position only.

        This is used for semantic conflict detection:
        same bubble + different class = conflict, not instant replacement.
        """
        best_landmark = None
        best_distance = float("inf")

        for landmark in self.landmarks:
            distance = self.distance_to_landmark(landmark, point_map)
            gate = max(
                self.association_radius,
                float(landmark.get("uncertainty_radius", 0.20)),
            )

            if distance < gate and distance < best_distance:
                best_landmark = landmark
                best_distance = distance

        return best_landmark
    


    def create_landmark(self, obj, point_map):
        """
        Creates a new candidate landmark from the first observation.
        """
        now_sec = self.now_sec()
        landmark_id = self.make_landmark_id(obj)

        return {
            "id": landmark_id,
            "class_id": int(obj.get("class_id", -1)),
            "class_name": obj.get("class_name", "unknown"),
            "pose_map": self.point_dict(point_map),
            "uncertainty_radius": float(obj.get("uncertainty_radius", 0.30)),
            "confidence": self.observation_weight(obj),
            "seen_count": 1,
            "first_seen": now_sec,
            "last_seen": now_sec,
            "source_sensor": obj.get("source_sensor", "unknown"),
            "status": "candidate",

            # Memory status starts as candidate. Visibility is separate and
            # only says whether the object was observed recently.
            "visibility": "visible",
            "last_detection_id": obj.get("detection_id", ""),
            "history": [self.history_entry(obj, point_map, now_sec)],
        }

    def handle_semantic_conflict(self, landmark, obj, point_map):
        """
        Handles same-place different-class observations.

        Dynamic objects are treated as temporary occlusion.
        Static different-class objects become conflict evidence.
        Repeated conflict evidence can eventually replace the old landmark.
        """
        observed_class = obj.get("class_name", "unknown")

        if self.is_dynamic_class(observed_class):
            landmark["visibility"] = "possibly_occluded"
            landmark["last_observed_class"] = observed_class
            return

        conflicts = landmark.setdefault("semantic_conflicts", [])
        normalized_observed = self.normalize_class_name(observed_class)

        conflict = None
        for item in conflicts:
            if self.normalize_class_name(item.get("new_class", "")) == normalized_observed:
                conflict = item
                break

        now_sec = self.now_sec()

        if conflict is None:
            conflict = {
                "old_class": landmark.get("class_name", "unknown"),
                "new_class": observed_class,
                "first_seen": now_sec,
                "last_seen": now_sec,
                "evidence_count": 1,
                "status": "conflict_candidate",
            }
            conflicts.append(conflict)
        else:
            conflict["last_seen"] = now_sec
            conflict["evidence_count"] = int(conflict.get("evidence_count", 0)) + 1

        landmark["last_observed_class"] = observed_class

        if conflict["evidence_count"] >= self.conflict_evidence_count:
            self.replace_landmark_class(landmark, obj, point_map, conflict)

    def replace_landmark_class(self, landmark, obj, point_map, conflict):
        """
        Replaces a stable semantic identity only after repeated conflict evidence.
        """
        old_class = landmark.get("class_name", "unknown")
        previous_classes = landmark.setdefault("previous_classes", [])
        previous_classes.append(old_class)

        landmark["class_id"] = int(obj.get("class_id", -1))
        landmark["class_name"] = obj.get("class_name", "unknown")
        landmark["pose_map"] = self.point_dict(point_map)
        landmark["confidence"] = self.observation_weight(obj)
        landmark["seen_count"] = 1
        landmark["status"] = "candidate"
        landmark["visibility"] = "visible"
        landmark["replacement_reason"] = "semantic_conflict"
        landmark["last_replaced"] = self.now_sec()

        conflict["status"] = "replaced"



    def update_landmark(self, landmark, obj, point_map):
        """
        Updates an existing landmark with a weighted running average.
        """
        now_sec = self.now_sec()
        obs_weight = self.observation_weight(obj)
        blend = max(0.10, min(0.65, obs_weight))

        old_pose = landmark["pose_map"]
        new_pose = self.point_dict(point_map)

        old_pose["x"] = self.blend(old_pose["x"], new_pose["x"], blend)
        old_pose["y"] = self.blend(old_pose["y"], new_pose["y"], blend)
        old_pose["z"] = self.blend(old_pose["z"], new_pose["z"], blend)

        landmark["confidence"] = self.blend(
            float(landmark.get("confidence", 0.0)),
            obs_weight,
            0.35,
        )
        landmark["uncertainty_radius"] = self.blend(
            float(landmark.get("uncertainty_radius", 0.30)),
            float(obj.get("uncertainty_radius", 0.30)),
            0.25,
        )
        landmark["seen_count"] = int(landmark.get("seen_count", 0)) + 1
        landmark["last_seen"] = now_sec
        landmark["source_sensor"] = obj.get("source_sensor", "unknown")
        landmark["last_detection_id"] = obj.get("detection_id", "")
        landmark["status"] = self.status_for_seen_count(landmark["seen_count"])
        landmark["visibility"] = "visible"

        history = landmark.setdefault("history", [])
        history.append(self.history_entry(obj, point_map, now_sec))
        if len(history) > self.max_history:
            del history[:-self.max_history]

    def publish_state(self):
        """
        Publishes the current semantic map as JSON and visualization markers.
        """
        out = String()
        out.data = json.dumps({
            "header": {
                "stamp": self.stamp_dict(),
                "frame_id": self.map_frame,
            },
            "landmarks": self.public_landmarks(),
        })
        self.landmarks_pub.publish(out)

        # Added 2026-05-20 16:08:30 EDT:
        # Publish separate Foxglove layers so stable map landmarks are not
        # confused with candidate/confirmed hypotheses or current sightings.
        self.markers_pub.publish(
            self.make_marker_array(
                statuses=("stable",),
                only_visible=None,
                namespace_prefix="semantic_landmarks",
            )
        )
        self.hypothesis_markers_pub.publish(
            self.make_marker_array(
                statuses=("candidate", "confirmed"),
                only_visible=None,
                namespace_prefix="semantic_hypotheses",
            )
        )
        self.current_markers_pub.publish(
            self.make_marker_array(
                statuses=("candidate", "confirmed", "stable"),
                only_visible=True,
                namespace_prefix="semantic_current",
            )
        )
        # Phase-3 covariance ellipsoid markers (spec §20-A).
        self.cov_markers_pub.publish(self.make_covariance_marker_array())

    def make_covariance_marker_array(self):
        """
        One ellipsoid marker per Kalman landmark, scaled by the eigenvalues of P_l.
        Color encodes state (candidate/confirmed/stable). Useful in Foxglove
        for "how certain is this landmark right now" at a glance.
        """
        ma = MarkerArray()
        marker_id = 0
        for lm in self.landmarks:
            if not lm.get("kf"):
                continue
            try:
                P = np.array(lm["cov_flat"], dtype=np.float64).reshape(3, 3)
                P = 0.5 * (P + P.T)
                eigvals, eigvecs = np.linalg.eigh(P)
            except Exception:
                continue
            eigvals = np.clip(eigvals, 1e-9, None)
            sigmas = np.sqrt(eigvals) * 2.0 * self.covariance_sigma_scale  # diameter

            # Build a quaternion from eigvecs (orthonormal, right-handed).
            if np.linalg.det(eigvecs) < 0.0:
                eigvecs[:, 0] = -eigvecs[:, 0]
            qx, qy, qz, qw = self.rot_to_quat(eigvecs)

            mean = lm.get("mean", [0.0, 0.0, 0.0])
            m = Marker()
            m.header.frame_id = self.map_frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "semantic_landmark_cov"
            m.id = marker_id
            marker_id += 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(mean[0])
            m.pose.position.y = float(mean[1])
            m.pose.position.z = float(mean[2])
            m.pose.orientation.x = float(qx)
            m.pose.orientation.y = float(qy)
            m.pose.orientation.z = float(qz)
            m.pose.orientation.w = float(qw)
            m.scale.x = max(0.02, float(sigmas[0]))
            m.scale.y = max(0.02, float(sigmas[1]))
            m.scale.z = max(0.02, float(sigmas[2]))
            r, g, b = self.color_for_status(lm.get("status", "candidate"))
            m.color.r = float(r)
            m.color.g = float(g)
            m.color.b = float(b)
            m.color.a = 0.30  # translucent so they don't hide the sphere markers
            m.lifetime.sec = 1
            ma.markers.append(m)
        return ma

    @staticmethod
    def rot_to_quat(R):
        """3x3 rotation matrix -> (qx, qy, qz, qw)."""
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0:
            s = math.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        return qx, qy, qz, qw

    def load_semantic_map(self):
        """
        Loads semantic memory from JSON if it exists.
        """
        if not self.semantic_map_path.exists():
            return

        try:
            data = json.loads(self.semantic_map_path.read_text())
            self.landmarks = data.get("landmarks", [])

            if self.load_only_permanent:
                self.landmarks = [
                    landmark
                    for landmark in self.landmarks
                    if landmark.get("status") in self.permanent_statuses
                ]

            for landmark in self.landmarks:
                landmark.setdefault("visibility", "not_recently_seen")

            self.next_landmark_index = self.next_index_from_landmarks()
            self.get_logger().info(
                f"Loaded {len(self.landmarks)} semantic landmarks."
            )
        except Exception as exc:
            self.get_logger().warn(
                f"Could not load semantic map {self.semantic_map_path}: {exc}"
            )
            self.landmarks = []
            self.next_landmark_index = 1

    def save_semantic_map(self):
        """
        Saves semantic memory to JSON. Kalman landmarks are written with the
        spec-§22 schema: `position_map` (3-vec) + `covariance` (3x3).
        Legacy landmarks fall back to `pose_map` + `uncertainty_radius`.
        """
        if not self.dirty:
            return

        try:
            self.semantic_map_path.parent.mkdir(parents=True, exist_ok=True)
            permanent = self.permanent_landmarks()
            data = {
                "version": 2,
                "frame_id": self.map_frame,
                "saved_at": self.now_sec(),
                "permanent_statuses": sorted(self.permanent_statuses),
                "live_landmark_count": len(self.landmarks),
                "permanent_landmark_count": len(permanent),
                "landmarks": [self.landmark_save_view(lm) for lm in permanent],
            }
            self.semantic_map_path.write_text(json.dumps(data, indent=2) + "\n")
            self.dirty = False
        except Exception as exc:
            self.get_logger().warn(
                f"Could not save semantic map {self.semantic_map_path}: {exc}",
                throttle_duration_sec=2.0,
            )

    def landmark_save_view(self, lm):
        """Return the spec-§22 JSON shape, preserving extra debug fields."""
        if lm.get("kf"):
            cov_flat = lm.get("cov_flat", [0.0] * 9)
            cov_3x3 = [list(cov_flat[i * 3:(i + 1) * 3]) for i in range(3)]
            return {
                "id": lm.get("id"),
                "class": lm.get("class_name", "unknown"),
                "class_id": lm.get("class_id", -1),
                "state": lm.get("status", "candidate"),
                "position_map": list(lm.get("mean", [0.0, 0.0, 0.0])),
                "covariance": cov_3x3,
                "hit_count": int(lm.get("hit_count", 0)),
                "miss_count": int(lm.get("miss_count", 0)),
                "last_seen": float(lm.get("last_seen", 0.0)),
                "last_d2": float(lm.get("last_d2", 0.0)),
                "confidence_avg": float(lm.get("confidence_avg", 0.0)),
                "source_sensor": lm.get("source_sensor", "unknown"),
            }
        # Legacy landmark (pre-Phase-1 detection) — keep its original shape.
        return lm

    def make_marker_array(
        self,
        statuses=None,
        only_visible=None,
        namespace_prefix="semantic_landmarks",
    ):
        """
        Builds markers for one semantic visualization layer.

        /perception/semantic_landmark_markers uses stable landmarks only.
        /perception/semantic_hypothesis_markers uses candidate/confirmed memory.
        /perception/semantic_current_markers uses objects visible right now.
        """
        marker_array = MarkerArray()

        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        marker_id = 1

        for landmark in self.public_landmarks():
            if statuses is not None and landmark.get("status") not in statuses:
                continue

            if only_visible is True and landmark.get("visibility") != "visible":
                continue

            if only_visible is False and landmark.get("visibility") == "visible":
                continue

            marker_array.markers.append(
                self.make_sphere_marker(marker_id, landmark, namespace_prefix)
            )
            marker_id += 1

            marker_array.markers.append(
                self.make_text_marker(marker_id, landmark, namespace_prefix)
            )
            marker_id += 1

        return marker_array

    def make_sphere_marker(self, marker_id, landmark, namespace_prefix):
        pose = landmark["pose_map"]
        radius = max(
            0.08,
            min(
                float(self.max_marker_radius),
                float(landmark.get("uncertainty_radius", 0.20)),
            ),
        )

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.map_frame
        marker.ns = namespace_prefix
        marker.id = int(marker_id)
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = float(pose["x"])
        marker.pose.position.y = float(pose["y"])
        marker.pose.position.z = float(pose["z"])
        marker.pose.orientation.w = 1.0

        marker.scale.x = radius
        marker.scale.y = radius
        marker.scale.z = radius

        r, g, b = self.color_for_status(landmark.get("status", "candidate"))
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b

        # Fade old-but-trusted landmarks instead of demoting stable memory.
        if landmark.get("visibility") == "visible":
            marker.color.a = 0.85
        else:
            marker.color.a = 0.35

        return marker

    def make_text_marker(self, marker_id, landmark, namespace_prefix):
        pose = landmark["pose_map"]

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.map_frame
        marker.ns = f"{namespace_prefix}_labels"
        marker.id = int(marker_id)
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = float(pose["x"])
        marker.pose.position.y = float(pose["y"])
        marker.pose.position.z = float(pose["z"]) + 0.20
        marker.pose.orientation.w = 1.0

        marker.scale.z = 0.12

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        marker.text = (
            f"{landmark['class_name']} "
            f"{landmark['status']} "
            f"{landmark.get('visibility', 'unknown')} "
            f"n={landmark['seen_count']} "
            f"c={landmark['confidence']:.2f}"
        )

        return marker

    def make_landmark_id(self, obj):
        class_name = str(obj.get("class_name", "object")).lower()
        clean_name = "".join(
            ch if ch.isalnum() else "_"
            for ch in class_name
        ).strip("_")
        if not clean_name:
            clean_name = "object"

        index = self.next_landmark_index
        self.next_landmark_index += 1
        return f"{clean_name}_{index:03d}"

    def next_index_from_landmarks(self):
        max_index = 0
        for landmark in self.landmarks:
            suffix = str(landmark.get("id", "")).rsplit("_", 1)[-1]
            if suffix.isdigit():
                max_index = max(max_index, int(suffix))
        return max_index + 1

    def public_landmarks(self):
        self.refresh_landmark_visibility()
        return [
            landmark
            for landmark in self.landmarks
            if landmark.get("status") != "rejected"
        ]

    def permanent_landmarks(self):
        """
        Returns landmarks trusted enough to save across runs.

        Candidates are useful live hypotheses, but they are not permanent map
        truth. By default only stable landmarks are written to semantic_map.json.
        """
        self.refresh_landmark_visibility()
        return [
            landmark
            for landmark in self.landmarks
            if landmark.get("status") in self.permanent_statuses
        ]

    def refresh_landmark_visibility(self):
        """
        Updates live visibility without changing long-term memory confidence.

        `status` means memory confidence: candidate / confirmed / stable.
        `visibility` means current observation state: visible / not_recently_seen.
        A stable sign should not become a candidate again just because it leaves
        the camera view.
        """
        now_sec = self.now_sec()
        for landmark in self.landmarks:
            last_seen = float(landmark.get("last_seen", 0.0))
            if now_sec - last_seen <= self.visible_timeout_sec:
                landmark["visibility"] = "visible"
            else:
                landmark["visibility"] = "not_recently_seen"
    
    def normalize_class_name(self, class_name):
        """
        Normalizes labels so stop_sign and stop sign compare as the same class.
        """
        return " ".join(
            str(class_name).lower().replace("_", " ").strip().split()
        )

    def is_dynamic_class(self, class_name):
        """
        Dynamic classes can pass through a landmark bubble without replacing it.
        """
        return self.normalize_class_name(class_name) in self.dynamic_class_names

    def status_for_seen_count(self, seen_count):
        if seen_count >= self.stable_seen_count:
            return "stable"
        if seen_count >= self.confirmed_seen_count:
            return "confirmed"
        return "candidate"

    def observation_weight(self, obj):
        confidence = float(obj.get("confidence", 0.0))
        quality = float(obj.get("quality_score", 0.0))
        valid_ratio = float(obj.get("valid_depth_ratio", 0.0))
        return max(0.05, min(1.0, confidence * (0.40 + 0.40 * quality + 0.20 * valid_ratio)))

    def history_entry(self, obj, point_map, now_sec):
        return {
            "stamp": now_sec,
            "detection_id": obj.get("detection_id", ""),
            "confidence": float(obj.get("confidence", 0.0)),
            "quality_score": float(obj.get("quality_score", 0.0)),
            "valid_depth_ratio": float(obj.get("valid_depth_ratio", 0.0)),
            "uncertainty_radius": float(obj.get("uncertainty_radius", 0.30)),
            "pose_map": self.point_dict(point_map),
        }

    def distance_to_landmark(self, landmark, point_map):
        pose = landmark.get("pose_map", {})
        dx = float(pose.get("x", 0.0)) - float(point_map.point.x)
        dy = float(pose.get("y", 0.0)) - float(point_map.point.y)
        dz = float(pose.get("z", 0.0)) - float(point_map.point.z)

        if self.association_use_xy_only:
            return math.sqrt(dx * dx + dy * dy)

        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def point_dict(self, point_map):
        return {
            "x": float(point_map.point.x),
            "y": float(point_map.point.y),
            "z": float(point_map.point.z),
        }

    def stamp_from_header(self, header):
        stamp = header.get("stamp", {})
        msg = self.get_clock().now().to_msg()
        msg.sec = int(stamp.get("sec", msg.sec))
        msg.nanosec = int(stamp.get("nanosec", msg.nanosec))
        return msg

    def stamp_dict(self):
        stamp = self.get_clock().now().to_msg()
        return {
            "sec": int(stamp.sec),
            "nanosec": int(stamp.nanosec),
        }

    def now_sec(self):
        now = self.get_clock().now().to_msg()
        return float(now.sec) + 1e-9 * float(now.nanosec)

    @staticmethod
    def blend(old_value, new_value, alpha):
        return (1.0 - alpha) * float(old_value) + alpha * float(new_value)

    @staticmethod
    def color_for_status(status):
        if status == "stable":
            return 0.10, 0.90, 0.25
        if status == "confirmed":
            return 0.15, 0.45, 1.0
        if status == "stale":
            return 0.55, 0.55, 0.55
        return 1.0, 0.62, 0.08


def main(args=None):
    rclpy.init(args=args)
    node = SemanticLandmarkMapper()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
