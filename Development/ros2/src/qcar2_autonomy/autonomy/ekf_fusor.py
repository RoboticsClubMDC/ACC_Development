#!/usr/bin/env python3
"""EKF Fusor — Quanser-spec robust localization for QCar 2.

Fuses IMU + wheel encoder + steering (prediction) with a global pose estimate
(correction) using an Extended Kalman Filter. The pose estimate comes from
either Cartographer's TF (map -> base_link) during mapping, or from AMCL's
/amcl_pose during runtime — selectable via the `correction_source` parameter.

Output: /qcar2_pose_fused (PoseWithCovarianceStamped) for autonomy + perception.
Diagnostics: /qcar2_ekf/{p_diag, k_diag, innovation, innovation_mahalanobis,
health, mode} for Foxglove visualization of filter convergence.

Architecturally a downstream consumer of Cartographer/AMCL — does NOT feed
back into them. Avoids the circular-reasoning trap of the earlier
pose_estimator design.

See: Quanser "ROS 2 For QCar 2" doc, "ekf_fusor" application node spec.
See: QCar 2 User Manual - System Hardware v1.0, Table 11 (wheelbase, steering).
    LC Commentary: Yes, Quanser did this, but I did a rebuilt
    with how I think is correct the full math of this EKF, and im 
    too lazy to check how to change my docker to the things that QUanser uploaded
    for this, but not for writting this and doing the math.
    Enjoy the art that is cooking here.
"""

import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import (
    PoseStamped,
    PoseWithCovarianceStamped,
    TransformStamped,
    Twist,
)
from nav_msgs.msg import Odometry
from qcar2_interfaces.msg import MotorCommands
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float32, Float32MultiArray, String
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

from autonomy.estimation import GyroKF, QcarEKF


# Encoder constants — same as pose_estimator.py, sourced from QCar 2 manual p.12.
ENCODER_TICKS_PER_REV = 720.0 * 4.0
GEAR_RATIO = (13.0 * 19.0) / (70.0 * 37.0)
WHEEL_RADIUS = 0.033


# Health states for /qcar2_ekf/health.
HEALTH_HEALTHY = "healthy"
HEALTH_DEGRADED = "degraded"
HEALTH_PREDICTION_ONLY = "prediction_only"
HEALTH_OUTLIER_STREAK = "outlier_streak"
HEALTH_BOOTSTRAPPING = "bootstrapping"


def yaw_to_quaternion_msg(yaw):
    from geometry_msgs.msg import Quaternion
    q = Quaternion()
    q.z = math.sin(0.5 * yaw)
    q.w = math.cos(0.5 * yaw)
    return q


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class EkfFusor(Node):

    def __init__(self):
        super().__init__("ekf_fusor")

        # --- Parameters ---
        self.declare_parameter("wheelbase", 0.256)
        self.declare_parameter("predict_rate", 80.0)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("correction_source", "tf")  # tf | amcl_pose | landmark | none
        self.declare_parameter("amcl_pose_topic", "/amcl_pose")
        # Phase-4 (added 2026-05-25): consume stable-landmark-derived robot pose
        # corrections published by semantic_landmark_mapper. The mapper
        # publishes only when `enable_landmark_correction:=true` over there.
        self.declare_parameter("landmark_pose_topic", "/perception/landmark_pose_correction")
        self.declare_parameter("r_landmark_default_diag", [0.10, 0.10, 1.0e6])  # x, y, yaw
        self.declare_parameter("use_gyro_kf", True)
        # 2026-05-28: blend gyro into the QcarEKF predict (matches
        # pose_estimator). omega = gyro_weight·gyro + (1−w)·steering_omega.
        # Was steering-only in the predict, which drifted under cornering at
        # speed because steering→yaw-rate (tan δ/L) lags the real yaw rate.
        # 0.65 trusts the gyro more than the steering model. Set 0.0 to
        # revert to steering-only.
        self.declare_parameter("gyro_weight", 0.65)
        self.declare_parameter("q_diag", [0.0005, 0.0005, 0.002])
        self.declare_parameter("r_carto_diag", [0.02, 0.02, 0.01])
        self.declare_parameter("r_amcl_default_diag", [0.05, 0.05, 0.03])
        self.declare_parameter("max_dt", 0.25)
        self.declare_parameter("tf_correction_rate", 10.0)
        self.declare_parameter("stale_correction_warning_sec", 2.0)
        self.declare_parameter("stale_correction_predict_only_sec", 5.0)
        self.declare_parameter("mahalanobis_gate_chi2", 11.345)  # χ²_3 @ 99%
        self.declare_parameter("publish_tf", False)
        self.declare_parameter("fused_child_frame", "base_link_fused")
        self.declare_parameter("initial_pose", [0.0, 0.0, 0.0])
        self.declare_parameter("steering_limit", 0.52)

        self.wheelbase = float(self.get_parameter("wheelbase").value)
        self.predict_rate = float(self.get_parameter("predict_rate").value)
        self.map_frame = self.get_parameter("map_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.correction_source = self.get_parameter("correction_source").value
        self.amcl_topic = self.get_parameter("amcl_pose_topic").value
        self.landmark_topic = self.get_parameter("landmark_pose_topic").value
        self.r_landmark_default_diag = list(self.get_parameter("r_landmark_default_diag").value)
        self.use_gyro_kf = bool(self.get_parameter("use_gyro_kf").value)
        self.gyro_weight = float(self.get_parameter("gyro_weight").value)
        self.q_diag = list(self.get_parameter("q_diag").value)
        self.r_carto_diag = list(self.get_parameter("r_carto_diag").value)
        self.r_amcl_default_diag = list(self.get_parameter("r_amcl_default_diag").value)
        self.max_dt = float(self.get_parameter("max_dt").value)
        self.tf_correction_rate = float(self.get_parameter("tf_correction_rate").value)
        self.stale_warn = float(self.get_parameter("stale_correction_warning_sec").value)
        self.stale_predict_only = float(self.get_parameter("stale_correction_predict_only_sec").value)
        self.maha_gate = float(self.get_parameter("mahalanobis_gate_chi2").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.fused_child_frame = self.get_parameter("fused_child_frame").value
        self.initial_pose = list(self.get_parameter("initial_pose").value)
        self.steering_limit = float(self.get_parameter("steering_limit").value)

        # --- Filters ---
        x0 = np.array([[self.initial_pose[0]], [self.initial_pose[1]], [self.initial_pose[2]]])
        P0 = np.eye(3) * 1.0  # large initial uncertainty; first correction will collapse it
        Q = np.diag(self.q_diag)
        R_init = np.diag(self.r_carto_diag)  # gets swapped per-correction by source
        self.qcar2_ekf = QcarEKF(x0=x0, P0=P0, Q=Q, R=R_init, L=self.wheelbase)
        self.gyro_kf = GyroKF(
            x0=np.zeros((2, 1)),
            P0=np.eye(2) * 0.1,
            Q=np.diag([0.001, 0.0001]),
            R=np.array([[0.05]]),
        )

        # --- Inputs (sensor measurements held as latest-value caches) ---
        self.speed = 0.0
        self.yaw_rate = 0.0
        self.steering = 0.0

        # --- State tracking ---
        self.last_predict_time = None
        self.last_correction_time = None
        self.bootstrapped = False  # set True after first correction OR initial_pose != [0,0,0]
        if any(abs(v) > 1e-9 for v in self.initial_pose):
            self.bootstrapped = True
        self.outlier_streak = 0
        self.last_K = np.zeros((3, 3))
        self.last_innovation = np.zeros(3)
        self.last_maha = 0.0
        self.last_pose_source = "init"

        # --- QoS for AMCL pose (often transient-local; subscribe defensively) ---
        amcl_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # --- Subscriptions ---
        self.create_subscription(JointState, "/qcar2_joint", self.joint_state_cb, 10)
        self.create_subscription(Imu, "/qcar2_imu", self.imu_cb, 20)
        self.create_subscription(Twist, "/cmd_vel_nav", self.cmd_vel_cb, 10)
        self.create_subscription(MotorCommands, "/qcar2_motor_speed_cmd", self.motor_cmd_cb, 10)
        if self.correction_source == "amcl_pose":
            self.create_subscription(
                PoseWithCovarianceStamped, self.amcl_topic, self.amcl_pose_cb, amcl_qos
            )
        elif self.correction_source == "landmark":
            # Phase-4: same QoS pattern as AMCL. The mapper publishes
            # PoseWithCovarianceStamped on /perception/landmark_pose_correction
            # only when it matches a stable landmark and the user opted in
            # via `enable_landmark_correction:=true` on the mapper side.
            self.create_subscription(
                PoseWithCovarianceStamped, self.landmark_topic, self.landmark_pose_cb, amcl_qos
            )

        # 2026-05-27: /initialpose hard-reset channel. trip_planner publishes
        # here on arrival at a known node — semantically a RESET signal, not a
        # regular measurement. We bypass the Mahalanobis gate (which would
        # otherwise reject the snap as an outlier when AMCL jumps to the new
        # pose). After the reset, /amcl_pose converges to the snapped pose
        # within ~1s and the normal correction path resumes seamlessly.
        # Subscribed regardless of correction_source — /initialpose is always
        # an authoritative reset.
        self.create_subscription(
            PoseWithCovarianceStamped, "/initialpose", self.initialpose_cb, amcl_qos
        )

        # --- Publishers ---
        self.pub_pose = self.create_publisher(PoseWithCovarianceStamped, "/qcar2_pose_fused", 10)
        self.pub_odom = self.create_publisher(Odometry, "/qcar2_ekf/odometry_fused", 10)
        self.pub_p_diag = self.create_publisher(Float32MultiArray, "/qcar2_ekf/p_diag", 10)
        self.pub_k_diag = self.create_publisher(Float32MultiArray, "/qcar2_ekf/k_diag", 10)
        self.pub_innovation = self.create_publisher(Float32MultiArray, "/qcar2_ekf/innovation", 10)
        self.pub_maha = self.create_publisher(Float32, "/qcar2_ekf/innovation_mahalanobis", 10)
        self.pub_health = self.create_publisher(String, "/qcar2_ekf/health", 10)
        self.pub_mode = self.create_publisher(String, "/qcar2_ekf/mode", 10)

        # --- TF setup ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        # --- Timers ---
        self.create_timer(1.0 / self.predict_rate, self.predict_step)
        if self.correction_source == "tf":
            self.create_timer(1.0 / self.tf_correction_rate, self.try_tf_correction)
        self.create_timer(1.0, self.health_timer_cb)
        self.create_timer(1.0, self.mode_timer_cb)

        self.get_logger().info(
            f"ekf_fusor: source={self.correction_source}, predict@{self.predict_rate}Hz, "
            f"wheelbase={self.wheelbase}, publish_tf={self.publish_tf}, bootstrapped={self.bootstrapped}"
        )

    # ===== Sensor callbacks =====

    def joint_state_cb(self, msg):
        if not msg.velocity:
            return
        ticks_per_sec = float(msg.velocity[0])
        if not np.isfinite(ticks_per_sec):
            self.get_logger().warn("Dropping non-finite encoder sample", throttle_duration_sec=2.0)
            return
        self.speed = (
            ticks_per_sec
            / ENCODER_TICKS_PER_REV
            * GEAR_RATIO
            * (2.0 * math.pi)
            * WHEEL_RADIUS
        )

    def imu_cb(self, msg):
        # NaN-guard: Quanser PIT IMU can emit bad samples. If we propagate a
        # NaN here, pose_estimator's /odom turns NaN and Cartographer's
        # imu_tracker.cc:67 CHECK SIGABRTs the whole cartographer_node.
        z = float(msg.angular_velocity.z)
        if not np.isfinite(z):
            self.get_logger().warn("Dropping non-finite IMU sample", throttle_duration_sec=2.0)
            return
        self.yaw_rate = z

    def cmd_vel_cb(self, msg):
        value = float(msg.angular.z)
        if not np.isfinite(value):
            return
        self.steering = float(np.clip(value, -self.steering_limit, self.steering_limit))

    def motor_cmd_cb(self, msg):
        for name, value in zip(msg.motor_names, msg.values):
            if name == "steering_angle":
                value = float(value)
                if not np.isfinite(value):
                    continue
                self.steering = float(np.clip(value, -self.steering_limit, self.steering_limit))

    def amcl_pose_cb(self, msg):
        z = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            yaw_from_quaternion(msg.pose.pose.orientation),
        ])
        # Use AMCL's reported covariance if non-zero, else fall back to default.
        c = msg.pose.covariance
        var_x = c[0]  if c[0]  > 1e-9 else self.r_amcl_default_diag[0]
        var_y = c[7]  if c[7]  > 1e-9 else self.r_amcl_default_diag[1]
        var_t = c[35] if c[35] > 1e-9 else self.r_amcl_default_diag[2]
        R_amcl = np.diag([var_x, var_y, var_t])
        self.apply_correction(z, R_amcl, source="amcl_pose")

    def initialpose_cb(self, msg):
        """
        2026-05-27: trusted hard-reset from /initialpose (RViz, trip_planner
        node-arrival snap, carto_to_amcl seed, amcl_load.sh seed).

        Bypasses the Mahalanobis gate — by design, /initialpose IS a reset
        signal, not a measurement to validate against the current belief.
        Sets xHat to the message pose with the message's reported covariance
        (or a tight default if the message covariance is empty), resets the
        outlier streak, and marks bootstrap done.

        After this fires, AMCL is ALSO resetting around /initialpose. Its
        next /amcl_pose will be close to the snapped pose so the regular
        amcl_pose_cb path resumes normally with a small innovation — no
        rejection cascade, no discontinuity in /qcar2_pose_fused.
        """
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        theta = yaw_from_quaternion(msg.pose.pose.orientation)
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(theta)):
            self.get_logger().warn("/initialpose contains non-finite values; ignored")
            return

        # Use sender-reported covariance for xHat's P; fall back to a tight
        # default if the sender didn't fill it in (some RViz GUIs zero it).
        c = msg.pose.covariance
        var_x = c[0]  if c[0]  > 1e-9 else 0.01
        var_y = c[7]  if c[7]  > 1e-9 else 0.01
        var_t = c[35] if c[35] > 1e-9 else 0.05

        # Hard reset of the planar EKF state.
        self.qcar2_ekf.xHat = np.array([[x], [y], [theta]])
        self.qcar2_ekf.P    = np.diag([var_x, var_y, var_t])

        if self.use_gyro_kf:
            self.gyro_kf.xHat[0, 0] = theta

        self.bootstrapped         = True
        self.outlier_streak       = 0
        self.last_correction_time = self.get_clock().now()
        self.last_pose_source     = "initialpose"
        self.last_innovation      = np.zeros(3)
        self.last_maha            = 0.0

        # Publish a zero-innovation diagnostic point so plots show the
        # reset cleanly (not as a spike that looks like an outlier).
        self._publish_correction_diagnostics(np.zeros(3), 0.0)

        self.get_logger().info(
            f"/initialpose reset: x={x:.3f}, y={y:.3f}, theta={math.degrees(theta):.1f}° "
            f"(var: x={var_x:.4f}, y={var_y:.4f}, theta={var_t:.4f})"
        )

    def landmark_pose_cb(self, msg):
        """
        Phase-4: landmark-derived implied robot pose correction.

        The mapper publishes implied x,y with a tiny covariance block AND a
        huge yaw variance (we cannot extract yaw from a single point
        landmark). We pass that yaw variance through verbatim — the EKF then
        weights yaw nearly to zero in this update, leaving the predicted
        yaw effectively unchanged.

        Same outlier gate / bootstrap / streak handling as AMCL via
        `apply_correction`. Yaw value comes from the mapper but should match
        the EKF's own current yaw anyway since the mapper reads
        /qcar2_pose_fused.
        """
        z = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            yaw_from_quaternion(msg.pose.pose.orientation),
        ])
        if not np.all(np.isfinite(z)):
            self.get_logger().warn("Dropping non-finite landmark pose correction", throttle_duration_sec=2.0)
            return
        c = msg.pose.covariance
        var_x = c[0]  if c[0]  > 1e-9 else self.r_landmark_default_diag[0]
        var_y = c[7]  if c[7]  > 1e-9 else self.r_landmark_default_diag[1]
        var_t = c[35] if c[35] > 1e-9 else self.r_landmark_default_diag[2]
        R_lm = np.diag([var_x, var_y, var_t])
        self.apply_correction(z, R_lm, source="landmark")

    # ===== Predict =====

    def predict_step(self):
        now = self.get_clock().now()
        if self.last_predict_time is None:
            self.last_predict_time = now
            return

        dt = (now - self.last_predict_time).nanoseconds * 1e-9
        self.last_predict_time = now

        # dt clamp — skip if too large (filter would linearize poorly)
        if dt <= 0.0 or dt > self.max_dt:
            return

        # Bootstrap: don't integrate before first absolute reference
        if not self.bootstrapped:
            return

        # Blended yaw rate: gyro_weight·gyro + (1−w)·steering_omega.
        # If the gyro sample is non-finite (already NaN-guarded in imu_cb,
        # but belt-and-suspenders), fall back to steering-only.
        steering_omega = self.speed * math.tan(self.steering) / self.wheelbase
        if np.isfinite(self.yaw_rate):
            omega = self.gyro_weight * self.yaw_rate + (1.0 - self.gyro_weight) * steering_omega
        else:
            omega = steering_omega

        # QcarEKF predict — x,y from speed+heading, yaw from blended omega.
        u = [self.speed, self.steering]
        self.qcar2_ekf.prediction(dt, u, omega=omega)

        # GyroKF predict (independent yaw+bias tracker, used in diagnostics).
        if self.use_gyro_kf:
            self.gyro_kf.prediction(dt, self.yaw_rate)

        self.publish_fused(now)
        self.publish_diagnostics()

    # ===== Correction =====

    def try_tf_correction(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time()
            )
        except TransformException:
            return

        z = np.array([
            t.transform.translation.x,
            t.transform.translation.y,
            yaw_from_quaternion(t.transform.rotation),
        ])
        R_carto = np.diag(self.r_carto_diag)
        self.apply_correction(z, R_carto, source="tf")

    def apply_correction(self, z, R_meas, source):
        # First correction triggers bootstrap if not already done
        if not self.bootstrapped:
            self.qcar2_ekf.xHat = np.array([[z[0]], [z[1]], [z[2]]])
            self.qcar2_ekf.P = np.eye(3) * 0.25
            if self.use_gyro_kf:
                self.gyro_kf.xHat[0, 0] = z[2]
            self.bootstrapped = True
            self.last_correction_time = self.get_clock().now()
            self.last_pose_source = source
            self.get_logger().info(
                f"Bootstrap from {source}: x={z[0]:.3f}, y={z[1]:.3f}, theta={z[2]:.3f}"
            )
            # Publish a zero-innovation point so the plot starts cleanly at bootstrap time.
            self._publish_correction_diagnostics(np.zeros(3), 0.0)
            return

        # Innovation (H = I)
        x_pred = self.qcar2_ekf.xHat.flatten()
        innovation = np.array([z[0] - x_pred[0], z[1] - x_pred[1], wrap(z[2] - x_pred[2])])

        # Innovation covariance S = P + R
        S = self.qcar2_ekf.P + R_meas

        # Mahalanobis distance: y^T S^-1 y, χ²_3 distributed under null hypothesis
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.get_logger().warn("Innovation covariance singular; skipping correction")
            return
        maha = float(innovation @ S_inv @ innovation)
        self.last_maha = maha
        self.last_innovation = innovation

        # Publish diagnostics BEFORE the accept/reject decision so the plot
        # shows outlier spikes too — that's the whole point of the gate.
        self._publish_correction_diagnostics(innovation, maha)

        if maha > self.maha_gate:
            self.outlier_streak += 1
            self.get_logger().warn(
                f"Outlier rejected ({source}): maha={maha:.2f} > gate={self.maha_gate:.2f} "
                f"(streak={self.outlier_streak})"
            )
            return

        self.outlier_streak = 0

        # Cache K for diagnostics
        self.last_K = self.qcar2_ekf.P @ S_inv

        # Run the correction through the filter (uses its own internal innovation calc)
        self.qcar2_ekf.R = R_meas
        y_col = np.array([[z[0]], [z[1]], [z[2]]])
        self.qcar2_ekf.correction(y_col)

        if self.use_gyro_kf:
            self.gyro_kf.correction(np.array([[z[2]]]))

        self.last_correction_time = self.get_clock().now()
        self.last_pose_source = source

    def _publish_correction_diagnostics(self, innovation, maha):
        """Publish innovation + mahalanobis on every correction attempt
        (accepted, rejected, or bootstrap) so the Foxglove plot reflects all
        gate activity, not just successful corrections."""
        inn_msg = Float32MultiArray()
        inn_msg.data = [float(innovation[0]), float(innovation[1]), float(innovation[2])]
        self.pub_innovation.publish(inn_msg)
        self.pub_maha.publish(Float32(data=float(maha)))

    # ===== Output =====

    def publish_fused(self, stamp_now):
        x = float(self.qcar2_ekf.xHat[0, 0])
        y = float(self.qcar2_ekf.xHat[1, 0])
        yaw = float(self.qcar2_ekf.xHat[2, 0])
        q = yaw_to_quaternion_msg(yaw)
        stamp = stamp_now.to_msg()

        # PoseWithCovarianceStamped — primary output
        pose = PoseWithCovarianceStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.map_frame
        pose.pose.pose.position.x = x
        pose.pose.pose.position.y = y
        pose.pose.pose.position.z = 0.0
        pose.pose.pose.orientation = q
        # Embed 3x3 P into 6x6 ROS covariance (x,y,z,roll,pitch,yaw)
        cov = np.zeros((6, 6))
        P = self.qcar2_ekf.P
        cov[0, 0] = P[0, 0]; cov[0, 1] = P[0, 1]; cov[0, 5] = P[0, 2]
        cov[1, 0] = P[1, 0]; cov[1, 1] = P[1, 1]; cov[1, 5] = P[1, 2]
        cov[5, 0] = P[2, 0]; cov[5, 1] = P[2, 1]; cov[5, 5] = P[2, 2]
        pose.pose.covariance = cov.flatten().tolist()
        self.pub_pose.publish(pose)

        # nav_msgs/Odometry — also publishes velocity in twist for consumers that want it
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.map_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose = pose.pose.pose
        odom.pose.covariance = pose.pose.covariance
        odom.twist.twist.linear.x = self.speed
        steering_omega = self.speed * math.tan(self.steering) / self.wheelbase
        omega = self.yaw_rate if self.use_gyro_kf else steering_omega
        odom.twist.twist.angular.z = float(omega)
        self.pub_odom.publish(odom)

        # Optional TF broadcast
        if self.tf_broadcaster is not None:
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self.map_frame
            tf.child_frame_id = self.fused_child_frame
            tf.transform.translation.x = x
            tf.transform.translation.y = y
            tf.transform.translation.z = 0.0
            tf.transform.rotation = q
            self.tf_broadcaster.sendTransform(tf)

    def publish_diagnostics(self):
        P = self.qcar2_ekf.P
        p_msg = Float32MultiArray()
        p_msg.data = [float(P[0, 0]), float(P[1, 1]), float(P[2, 2])]
        self.pub_p_diag.publish(p_msg)

        k_msg = Float32MultiArray()
        k_msg.data = [
            float(self.last_K[0, 0]),
            float(self.last_K[1, 1]),
            float(self.last_K[2, 2]),
        ]
        self.pub_k_diag.publish(k_msg)

    # ===== Health =====

    def health_timer_cb(self):
        msg = String()
        if not self.bootstrapped:
            msg.data = HEALTH_BOOTSTRAPPING
        elif self.outlier_streak >= 5:
            msg.data = HEALTH_OUTLIER_STREAK
        elif self.last_correction_time is None:
            msg.data = HEALTH_BOOTSTRAPPING
        else:
            now = self.get_clock().now()
            age = (now - self.last_correction_time).nanoseconds * 1e-9
            if age > self.stale_predict_only:
                msg.data = HEALTH_PREDICTION_ONLY
                # Inflate covariance to signal high uncertainty downstream
                self.qcar2_ekf.P = self.qcar2_ekf.P + np.diag([0.001, 0.001, 0.0001])
            elif age > self.stale_warn:
                msg.data = HEALTH_DEGRADED
            else:
                msg.data = HEALTH_HEALTHY
        self.pub_health.publish(msg)

    def mode_timer_cb(self):
        msg = String()
        msg.data = f"correction_source={self.correction_source};last={self.last_pose_source}"
        self.pub_mode.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = EkfFusor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
