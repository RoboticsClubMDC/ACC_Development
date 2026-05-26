#!/usr/bin/env python3

import json
import math

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseWithCovarianceStamped
import tf2_ros
from rclpy.duration import Duration


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """Quaternion (x,y,z,w) → 3x3 rotation matrix. No external deps."""
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx = qx * qx * s
    yy = qy * qy * s
    zz = qz * qz * s
    xy = qx * qy * s
    xz = qx * qz * s
    yz = qy * qz * s
    wx = qw * qx * s
    wy = qw * qy * s
    wz = qw * qz * s
    return np.array([
        [1.0 - (yy + zz), xy - wz,         xz + wy],
        [xy + wz,         1.0 - (xx + zz), yz - wx],
        [xz - wy,         yz + wx,         1.0 - (xx + yy)],
    ])


def yaw_from_quaternion(qx, qy, qz, qw):
    """Yaw (z-axis rotation) from quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class Object3DEstimator(Node):
    """
    Converts YOLO 2D detections into 3D object observations.

    Responsibilities:
    - Subscribe to YOLO 2D detections.
    - Subscribe to aligned D435 depth.
    - Subscribe to D435 CameraInfo.
    - Estimate object position in aligned_camera_optical_frame.
    - Publish structured 3D object JSON.
    - Publish temporary RViz markers.

    Non-responsibilities:
    - Does not run YOLO.
    - Does not own the D435 camera.
    - Does not store semantic memory.
    - Does not decide vehicle behavior.
    """

    def __init__(self):
        super().__init__("object_3d_estimator")

        self.bridge = CvBridge()

        self.declare_parameter("detections_topic", "/perception/yolo/detections_2d")
        self.declare_parameter("depth_topic", "/perception/d435/depth/image_rect")
        self.declare_parameter("camera_info_topic", "/perception/d435/camera_info")
        self.declare_parameter("objects_topic", "/perception/objects_3d")
        self.declare_parameter("markers_topic", "/perception/object_markers")

        self.declare_parameter("min_depth", 0.05)
        self.declare_parameter("max_depth", 2.0)
        self.declare_parameter("depth_crop_ratio", 0.5)
        self.declare_parameter("min_valid_pixels", 10)
        self.declare_parameter("min_valid_ratio", 0.10)
        self.declare_parameter("min_quality", 0.0)
        self.declare_parameter("marker_lifetime_sec", 2.0)

        # Added 2026-05-20 13:47:02 EDT:
        # Keep weak/far depth estimates useful, but prevent huge Foxglove bubbles.
        self.declare_parameter("max_uncertainty_radius", 0.75)
        self.declare_parameter("max_marker_radius", 0.45)

        # Added 2026-05-25: Phase-1 of the formal R_obs covariance pipeline.
        # See semantic-landmark math spec §3-§11. These build the 3x3
        # observation covariance in the map frame, replacing the heuristic
        # uncertainty_radius scalar for downstream Kalman landmark filtering.
        self.declare_parameter("emit_map_frame_covariance", True)
        self.declare_parameter("pose_topic", "/qcar2_pose_fused")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        # pixel-sigma model from spec §3
        self.declare_parameter("sigma_u_min_px", 2.0)
        self.declare_parameter("sigma_v_min_px", 2.0)
        self.declare_parameter("alpha_min", 0.05)
        self.declare_parameter("alpha_max", 0.20)
        # depth-sigma model from spec §4:  sigma_d = a + b * d^2
        self.declare_parameter("depth_sigma_a", 0.01)
        self.declare_parameter("depth_sigma_b", 0.02)
        # mounting / RGB-depth alignment covariance, spec §10
        self.declare_parameter("sigma_mount_x", 0.01)
        self.declare_parameter("sigma_mount_y", 0.01)
        self.declare_parameter("sigma_mount_z", 0.02)
        self.declare_parameter("sigma_align_x", 0.02)
        self.declare_parameter("sigma_align_y", 0.02)
        self.declare_parameter("sigma_align_z", 0.03)

        self.min_depth = float(self.get_parameter("min_depth").value)
        self.max_depth = float(self.get_parameter("max_depth").value)
        self.depth_crop_ratio = float(self.get_parameter("depth_crop_ratio").value)
        self.min_valid_pixels = int(self.get_parameter("min_valid_pixels").value)
        self.min_valid_ratio = float(self.get_parameter("min_valid_ratio").value)
        self.min_quality = float(self.get_parameter("min_quality").value)
        self.marker_lifetime_sec = float(
            self.get_parameter("marker_lifetime_sec").value)
        self.max_uncertainty_radius = float(
            self.get_parameter("max_uncertainty_radius").value)
        self.max_marker_radius = float(
            self.get_parameter("max_marker_radius").value)

        # Phase-1 covariance params
        self.emit_map_frame_covariance = bool(self.get_parameter("emit_map_frame_covariance").value)
        self.map_frame = str(self.get_parameter("map_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.sigma_u_min_px = float(self.get_parameter("sigma_u_min_px").value)
        self.sigma_v_min_px = float(self.get_parameter("sigma_v_min_px").value)
        self.alpha_min = float(self.get_parameter("alpha_min").value)
        self.alpha_max = float(self.get_parameter("alpha_max").value)
        self.depth_sigma_a = float(self.get_parameter("depth_sigma_a").value)
        self.depth_sigma_b = float(self.get_parameter("depth_sigma_b").value)
        self.R_extrinsic = np.diag([
            float(self.get_parameter("sigma_mount_x").value) ** 2,
            float(self.get_parameter("sigma_mount_y").value) ** 2,
            float(self.get_parameter("sigma_mount_z").value) ** 2,
        ])
        self.R_align = np.diag([
            float(self.get_parameter("sigma_align_x").value) ** 2,
            float(self.get_parameter("sigma_align_y").value) ** 2,
            float(self.get_parameter("sigma_align_z").value) ** 2,
        ])
        pose_topic = str(self.get_parameter("pose_topic").value)

        # Latest robot pose + 3x3 planar covariance (xx, xy, xt; yx, yy, yt; tx, ty, tt).
        self.latest_pose_xy_theta = None       # np.array([x, y, theta]) in map frame
        self.latest_pose_cov_3x3 = None        # 3x3, planar
        self.latest_pose_stamp = None

        # TF buffer for map<-base and base<-aligned_camera_optical_frame composition
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            pose_topic,
            self.pose_cb,
            10,
        )

        detections_topic = str(self.get_parameter("detections_topic").value)
        depth_topic = str(self.get_parameter("depth_topic").value)
        camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        objects_topic = str(self.get_parameter("objects_topic").value)
        markers_topic = str(self.get_parameter("markers_topic").value)

        self.depth_img = None
        self.depth_stamp = None
        self.camera_info = None

        self.depth_sub = self.create_subscription(
            Image,
            depth_topic,
            self.depth_cb,
            qos_profile_sensor_data,
        )

        self.info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.info_cb,
            10,
        )

        self.detections_sub = self.create_subscription(
            String,
            detections_topic,
            self.detections_cb,
            10,
        )

        self.objects_pub = self.create_publisher(String, objects_topic, 10)
        self.markers_pub = self.create_publisher(MarkerArray, markers_topic, 10)

        self.frame_count = 0

        self.get_logger().info(
            f"object_3d_estimator ready. detections={detections_topic}, "
            f"depth={depth_topic}, camera_info={camera_info_topic}"
        )

    def pose_cb(self, msg):
        """Cache latest /qcar2_pose_fused as (x, y, theta) + 3x3 planar covariance."""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        theta = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        # PoseWithCovariance.covariance is row-major 6x6 over (x, y, z, roll, pitch, yaw).
        c = np.array(msg.pose.covariance, dtype=np.float64).reshape(6, 6)
        P = np.array([
            [c[0, 0], c[0, 1], c[0, 5]],
            [c[1, 0], c[1, 1], c[1, 5]],
            [c[5, 0], c[5, 1], c[5, 5]],
        ])
        if not np.all(np.isfinite(P)) or not np.isfinite(theta):
            return
        self.latest_pose_xy_theta = np.array([p.x, p.y, theta])
        self.latest_pose_cov_3x3 = P
        self.latest_pose_stamp = msg.header.stamp

    # ---------- R_obs math helpers (spec §3-§11) ----------

    def pixel_sigmas(self, u1, v1, u2, v2, conf):
        """Spec §3: confidence-aware pixel std dev from bbox + confidence."""
        bw = max(1.0, float(u2 - u1))
        bh = max(1.0, float(v2 - v1))
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * (1.0 - max(0.0, min(1.0, conf)))
        su = max(self.sigma_u_min_px, alpha * bw)
        sv = max(self.sigma_v_min_px, alpha * bh)
        return su, sv

    def depth_sigma_model(self, d):
        """Spec §4: sigma_d = a + b*d^2."""
        return max(1e-4, self.depth_sigma_a + self.depth_sigma_b * (float(d) ** 2))

    def jacobian_cam(self, u, v, d, fx, fy, cx, cy):
        """Spec §6: ∂p_c / ∂[u, v, d]. p_c = ((u-cx)d/fx, (v-cy)d/fy, d)."""
        return np.array([
            [d / fx, 0.0,    (u - cx) / fx],
            [0.0,    d / fy, (v - cy) / fy],
            [0.0,    0.0,    1.0],
        ])

    def jacobian_robot(self, x_b, y_b, theta):
        """Spec §9: ∂p_m / ∂[x, y, theta]. p_m,xy depends on robot pose."""
        st = math.sin(theta)
        ct = math.cos(theta)
        return np.array([
            [1.0, 0.0, -st * x_b - ct * y_b],
            [0.0, 1.0,  ct * x_b - st * y_b],
            [0.0, 0.0,  0.0],
        ])

    def lookup_base_from_cam(self, camera_frame):
        """Get static T_base_from_camera_optical (latest)."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, camera_frame, rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
        except Exception as exc:
            self.get_logger().warn(
                f"TF lookup {self.base_frame}<-{camera_frame} failed: {exc}",
                throttle_duration_sec=2.0,
            )
            return None, None
        t = tf.transform.translation
        q = tf.transform.rotation
        R = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
        return np.array([t.x, t.y, t.z]), R

    def build_R_obs(self, u, v, d, conf, bbox, fx, fy, cx, cy, camera_frame):
        """
        Build the full observation covariance in the map frame and return
        (position_map_xyz, R_obs_3x3) or (None, None) if anything is missing.

        Spec §11:
          R_obs = R_map<-cam * J_cam * R_m * J_cam^T * R_map<-cam^T
                + J_r * P_r * J_r^T
                + R_extrinsic + R_align
        """
        if self.latest_pose_xy_theta is None or self.latest_pose_cov_3x3 is None:
            return None, None

        # 1) pixel + depth sigmas → R_m
        su, sv = self.pixel_sigmas(bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"], conf)
        sd = self.depth_sigma_model(d)
        R_m = np.diag([su * su, sv * sv, sd * sd])

        # 2) point in camera frame + J_cam
        Xc = (u - cx) * d / fx
        Yc = (v - cy) * d / fy
        Zc = d
        p_c = np.array([Xc, Yc, Zc])
        J_cam = self.jacobian_cam(u, v, d, fx, fy, cx, cy)
        R_pc = J_cam @ R_m @ J_cam.T

        # 3) T_base<-cam (static)
        t_bc, R_bc = self.lookup_base_from_cam(camera_frame)
        if R_bc is None:
            return None, None
        p_b = R_bc @ p_c + t_bc          # point in base frame

        # 4) compose to map frame using latest robot pose (2D yaw + z trans = 0)
        x_r, y_r, theta = self.latest_pose_xy_theta
        st, ct = math.sin(theta), math.cos(theta)
        R_mb = np.array([
            [ct, -st, 0.0],
            [st,  ct, 0.0],
            [0.0, 0.0, 1.0],
        ])
        p_m = R_mb @ p_b + np.array([x_r, y_r, 0.0])

        # 5) rotate camera-projection covariance into map
        R_map_from_cam = R_mb @ R_bc
        R_cam_map = R_map_from_cam @ R_pc @ R_map_from_cam.T

        # 6) robot-pose Jacobian contribution
        J_r = self.jacobian_robot(p_b[0], p_b[1], theta)
        R_robot = J_r @ self.latest_pose_cov_3x3 @ J_r.T

        # 7) total
        R_obs = R_cam_map + R_robot + self.R_extrinsic + self.R_align
        # Symmetrize to fight numerical drift
        R_obs = 0.5 * (R_obs + R_obs.T)

        if not np.all(np.isfinite(p_m)) or not np.all(np.isfinite(R_obs)):
            return None, None
        return p_m, R_obs

    def depth_cb(self, msg):
        """
        Stores the latest aligned depth image.

        The depth source publishes 32FC1 depth in meters after distance_scale.
        """
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except Exception as exc:
            self.get_logger().warn(f"Could not decode depth image: {exc}")
            return

        if depth.ndim == 3:
            depth = depth[:, :, 0]

        self.depth_img = depth.astype(np.float32, copy=False)
        self.depth_stamp = msg.header.stamp

    def info_cb(self, msg):
        """
        Stores the latest camera intrinsics.

        We use K:
        fx = K[0], fy = K[4], cx = K[2], cy = K[5].
        """
        self.camera_info = msg

    def detections_cb(self, msg):
        """
        Main estimator callback.

        Receives YOLO 2D detections as JSON, combines them with latest
        aligned depth and camera intrinsics, then publishes 3D object estimates.
        """
        self.frame_count += 1

        if self.depth_img is None or self.camera_info is None:
            self.get_logger().warn(
                "Waiting for depth image and camera info.",
                throttle_duration_sec=2.0,
            )
            return

        try:
            detections_msg = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"Could not parse detections JSON: {exc}")
            return

        objects = []
        marker_array = MarkerArray()

        frame_id = detections_msg.get("header", {}).get(
            "frame_id",
            self.camera_info.header.frame_id,
        )

        detections = detections_msg.get("detections", [])

        marker_id = 0
        for det in detections:
            bbox = det.get("bbox", {})
            estimate = self.estimate_object_depth(bbox)

            if estimate is None:
                continue

            point = estimate["point_camera"]
            camera_distance = math.sqrt(
                float(point[0]) * float(point[0])
                + float(point[1]) * float(point[1])
                + float(point[2]) * float(point[2])
            )

            obj = {
                "detection_id": det.get("detection_id", ""),
                "class_id": det.get("class_id", -1),
                "class_name": det.get("class_name", "unknown"),
                "confidence": float(det.get("confidence", 0.0)),
                "source_sensor": det.get("source_sensor", "d435_aligned_rgb"),
                "frame_id": frame_id,
                "bbox": bbox,
                "pose_camera": {
                    "x": point[0],
                    "y": point[1],
                    "z": point[2],
                },
                "depth_median": estimate["median_depth"],
                "distance_camera_m": float(camera_distance),
                "valid_depth_ratio": estimate["valid_ratio"],
                "depth_sigma": estimate["depth_sigma"],
                "depth_span": estimate["depth_span"],
                "quality_components": estimate["quality_components"],
                "quality_score": estimate["quality_score"],
                "uncertainty_radius": estimate["uncertainty_radius"],
            }

            # Phase-1: build full R_obs in map frame (spec §11) and attach it
            # alongside the legacy fields. The Phase-2 mapper will consume
            # position_map + R_obs and ignore pose_camera/uncertainty_radius.
            if self.emit_map_frame_covariance:
                fx = float(self.camera_info.k[0])
                fy = float(self.camera_info.k[4])
                cx = float(self.camera_info.k[2])
                cy = float(self.camera_info.k[5])
                u_c = 0.5 * (float(bbox["x1"]) + float(bbox["x2"]))
                v_c = 0.5 * (float(bbox["y1"]) + float(bbox["y2"]))
                d_med = float(estimate["median_depth"])
                p_m, R_obs = self.build_R_obs(
                    u_c, v_c, d_med,
                    float(det.get("confidence", 0.0)),
                    bbox, fx, fy, cx, cy, frame_id,
                )
                if p_m is not None:
                    obj["position_map"] = [float(p_m[0]), float(p_m[1]), float(p_m[2])]
                    obj["R_obs"] = [[float(R_obs[i, j]) for j in range(3)] for i in range(3)]
                    obj["map_frame"] = self.map_frame
                    obj["pixel_uv"] = [float(u_c), float(v_c)]
                    obj["sigma_pixel"] = list(self.pixel_sigmas(
                        bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"],
                        float(det.get("confidence", 0.0)),
                    ))
                    obj["sigma_depth"] = self.depth_sigma_model(d_med)

            objects.append(obj)

            marker_array.markers.append(
                self.make_sphere_marker(marker_id, frame_id, obj)
            )
            marker_id += 1

            marker_array.markers.append(
                self.make_text_marker(marker_id, frame_id, obj)
            )
            marker_id += 1

        out = String()
        out.data = json.dumps({
            "header": detections_msg.get("header", {}),
            "frame_id": frame_id,
            "objects": objects,
        })

        self.objects_pub.publish(out)
        self.markers_pub.publish(marker_array)

        if self.frame_count % 30 == 1:
            self.get_logger().info(
                f"Published {len(objects)} 3D object estimates."
            )

    def estimate_object_depth(self, bbox):
        """
        Estimates one object's 3D point from its bbox and aligned depth.

        This adapts the visual-odometry depth filtering idea for YOLO boxes:
        - center crop
        - valid depth ratio
        - median depth
        - MAD / robust sigma
        - p10-p90 span
        - quality score
        - uncertainty radius

        It is intentionally less aggressive than VO. YOLO boxes often contain
        edges, background, poles, signs, and floor, so weak estimates are kept
        with larger uncertainty instead of being dropped too early.
        """
        depth = self.depth_img
        h, w = depth.shape[:2]

        try:
            x1 = float(bbox["x1"])
            y1 = float(bbox["y1"])
            x2 = float(bbox["x2"])
            y2 = float(bbox["y2"])
        except Exception:
            return None

        x1 = max(0.0, min(float(w - 1), x1))
        x2 = max(0.0, min(float(w), x2))
        y1 = max(0.0, min(float(h - 1), y1))
        y2 = max(0.0, min(float(h), y2))

        if x2 <= x1 or y2 <= y1:
            return None

        ratio = max(0.1, min(1.0, self.depth_crop_ratio))

        cx_box = 0.5 * (x1 + x2)
        cy_box = 0.5 * (y1 + y2)
        bw = x2 - x1
        bh = y2 - y1

        crop_w = bw * ratio
        crop_h = bh * ratio

        rx1 = int(round(cx_box - crop_w / 2.0))
        rx2 = int(round(cx_box + crop_w / 2.0))
        ry1 = int(round(cy_box - crop_h / 2.0))
        ry2 = int(round(cy_box + crop_h / 2.0))

        rx1 = max(0, min(w - 1, rx1))
        rx2 = max(0, min(w, rx2))
        ry1 = max(0, min(h - 1, ry1))
        ry2 = max(0, min(h, ry2))

        if rx2 <= rx1 or ry2 <= ry1:
            return None

        patch = depth[ry1:ry2, rx1:rx2]
        patch_area = max(int(patch.size), 1)

        valid_mask = (
            np.isfinite(patch)
            & (patch > self.min_depth)
            & (patch < self.max_depth)
        )

        valid_count = int(np.count_nonzero(valid_mask))
        valid_ratio = float(valid_count) / float(patch_area)

        if valid_count < self.min_valid_pixels:
            return None

        if valid_ratio < self.min_valid_ratio:
            return None

        z_vals = patch[valid_mask].astype(np.float32)
        z_med = float(np.nanmedian(z_vals))

        if not math.isfinite(z_med) or z_med <= 0.0:
            return None

        mad = float(np.nanmedian(np.abs(z_vals - z_med)))
        depth_sigma = 1.4826 * mad

        p10 = float(np.nanpercentile(z_vals, 10))
        p90 = float(np.nanpercentile(z_vals, 90))
        depth_span = p90 - p10

        sigma_limit = max(0.015, 0.035 * z_med)
        span_limit = max(0.040, 0.180 * z_med)

        q_valid = min(1.0, valid_ratio / 0.50)
        q_noise = math.exp(-((depth_sigma / max(sigma_limit, 1e-6)) ** 2))
        q_span = math.exp(-((depth_span / max(span_limit, 1e-6)) ** 2))
        q_range = 1.0 / (1.0 + (z_med / 2.0) ** 2)

        quality = float(q_valid * q_noise * q_span * q_range)

        # `quality` should normally inform uncertainty and semantic-landmark
        # weight, not erase otherwise stable object observations. Keep
        # min_quality at 0.0 by default and only use it as an explicit debug
        # gate when the operator asks for stricter filtering.
        if self.min_quality > 0.0 and quality < self.min_quality:
            return None

        fx = float(self.camera_info.k[0])
        fy = float(self.camera_info.k[4])
        cx = float(self.camera_info.k[2])
        cy = float(self.camera_info.k[5])

        u = cx_box
        v = cy_box

        x = (u - cx) * z_med / fx
        y = (v - cy) * z_med / fy
        z = z_med

        uncertainty_radius = self.estimate_uncertainty_radius(
            z_med,
            depth_sigma,
            depth_span,
            valid_ratio,
            quality,
        )

        return {
            "point_camera": [float(x), float(y), float(z)],
            "median_depth": float(z_med),
            "valid_ratio": float(valid_ratio),
            "depth_sigma": float(depth_sigma),
            "depth_span": float(depth_span),
            "quality_components": {
                "q_valid": float(q_valid),
                "q_noise": float(q_noise),
                "q_span": float(q_span),
                "q_range": float(q_range),
            },
            "quality_score": float(quality),
            "uncertainty_radius": float(uncertainty_radius),
        }

    def estimate_uncertainty_radius(
        self,
        z_med,
        depth_sigma,
        depth_span,
        valid_ratio,
        quality,
    ):
        """
        Converts depth quality metrics into a simple uncertainty radius.

        This is not a final calibrated covariance. It is a practical confidence
        bubble for RViz and landmark gating.
        """
        base = 0.04
        range_term = 0.03 * z_med
        noise_term = min(0.40, 2.0 * depth_sigma)
        span_term = min(0.40, 0.5 * depth_span)
        valid_penalty = 0.10 * (1.0 - min(1.0, valid_ratio))
        quality_penalty = 0.20 * (1.0 - min(1.0, quality))

        radius = (
            base
            + range_term
            + noise_term
            + span_term
            + valid_penalty
            + quality_penalty
        )

        # This caps the reported uncertainty itself. Far objects can still be
        # uncertain, but they should not grow without bound.
        return min(float(self.max_uncertainty_radius), radius)

    def make_sphere_marker(self, marker_id, frame_id, obj):
        """
        Makes a sphere marker at the object camera-frame point.
        """
        p = obj["pose_camera"]

        # This caps only the visible marker size. The JSON still carries the
        # uncertainty_radius used by mapping and debugging.
        radius = min(float(self.max_marker_radius), obj["uncertainty_radius"])

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = frame_id
        marker.ns = "perception_objects"
        marker.id = int(marker_id)
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = float(p["x"])
        marker.pose.position.y = float(p["y"])
        marker.pose.position.z = float(p["z"])
        marker.pose.orientation.w = 1.0

        marker.scale.x = max(0.05, float(radius))
        marker.scale.y = max(0.05, float(radius))
        marker.scale.z = max(0.05, float(radius))

        marker.color.r = 1.0
        marker.color.g = 0.35
        marker.color.b = 0.05
        marker.color.a = 0.85

        self.set_marker_lifetime(marker)
        return marker


    def make_text_marker(self, marker_id, frame_id, obj):
        """
        Makes a text label marker for the object estimate.
        """
        p = obj["pose_camera"]

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = frame_id
        marker.ns = "perception_object_labels"
        marker.id = int(marker_id)
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = float(p["x"])
        marker.pose.position.y = float(p["y"])
        marker.pose.position.z = float(p["z"]) + 0.12
        marker.pose.orientation.w = 1.0

        marker.scale.z = 0.10

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        marker.text = (
            f"{obj['class_name']} "
            f"q={obj['quality_score']:.2f} "
            f"z={obj['depth_median']:.2f}"
        )

        self.set_marker_lifetime(marker)
        return marker

    def set_marker_lifetime(self, marker):
        lifetime = max(0.1, float(self.marker_lifetime_sec))
        marker.lifetime.sec = int(lifetime)
        marker.lifetime.nanosec = int((lifetime - int(lifetime)) * 1_000_000_000)
    
def main(args=None):
    rclpy.init(args=args)
    node = Object3DEstimator()

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
