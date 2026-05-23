#!/usr/bin/env python3

# EKF odometry source for QCar2.
#
# Owns the odom -> base_link TF and publishes nav_msgs/Odometry on /odom.
# Inputs: wheel speed from /qcar2_joint, IMU yaw rate from /qcar2_imu,
# steering command from /cmd_vel_nav and /qcar2_motor_speed_cmd.
#
# This replaces qcar2_odometry (encoder-only). Cartographer and AMCL should
# read this node's /odom and odom->base_link TF.

import math

import numpy as np
import rclpy
from geometry_msgs.msg import Quaternion, TransformStamped, Twist
from nav_msgs.msg import Odometry
from qcar2_interfaces.msg import MotorCommands
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from tf2_ros import TransformBroadcaster


ENCODER_TICKS_PER_REV = 720.0 * 4.0
GEAR_RATIO = (13.0 * 19.0) / (70.0 * 30.0)
WHEEL_RADIUS = 0.033
WHEELBASE = 0.257


def wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_to_quaternion_msg(yaw):
    half = 0.5 * yaw
    q = Quaternion()
    q.z = math.sin(half)
    q.w = math.cos(half)
    return q


class PoseEstimator(Node):
    """EKF that owns odom -> base_link and publishes /odom."""

    def __init__(self):
        super().__init__("pose_estimator")

        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("frequency", 80.0)
        self.declare_parameter("max_dt", 0.25)
        self.declare_parameter("steering_limit", 0.6)
        self.declare_parameter("gyro_weight", 0.65)
        self.declare_parameter("publish_tf", True)

        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.frequency = float(self.get_parameter("frequency").value)
        self.max_dt = float(self.get_parameter("max_dt").value)
        self.steering_limit = float(self.get_parameter("steering_limit").value)
        self.gyro_weight = float(self.get_parameter("gyro_weight").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)

        # Odom frame starts at the robot's pose at boot, by ROS convention.
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.P = np.zeros((3, 3))
        self.Q = np.diag([0.0005, 0.0005, 0.002])

        self.speed = 0.0
        self.steering = 0.0
        self.yaw_rate = 0.0
        self.omega = 0.0
        self.last_update_time = None

        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None
        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 20)

        self.create_subscription(JointState, "/qcar2_joint", self.joint_state_callback, 10)
        self.create_subscription(Imu, "/qcar2_imu", self.imu_callback, 20)
        self.create_subscription(Twist, "/cmd_vel_nav", self.cmd_vel_callback, 10)
        self.create_subscription(
            MotorCommands,
            "/qcar2_motor_speed_cmd",
            self.motor_cmd_callback,
            10,
        )

        self.timer = self.create_timer(1.0 / self.frequency, self.filter_timer)

    def joint_state_callback(self, msg):
        if not msg.velocity:
            return

        motor_ticks_per_sec = float(msg.velocity[0])
        self.speed = (
            motor_ticks_per_sec
            / ENCODER_TICKS_PER_REV
            * GEAR_RATIO
            * (2.0 * math.pi)
            * WHEEL_RADIUS
        )

    def imu_callback(self, msg):
        self.yaw_rate = float(msg.angular_velocity.z)

    def cmd_vel_callback(self, msg):
        self.steering = self.clip_steering(msg.angular.z)

    def motor_cmd_callback(self, msg):
        for name, value in zip(msg.motor_names, msg.values):
            if name == "steering_angle":
                self.steering = self.clip_steering(value)

    def clip_steering(self, value):
        return float(np.clip(value, -self.steering_limit, self.steering_limit))

    def filter_timer(self):
        now = self.get_clock().now()
        if self.last_update_time is None:
            self.last_update_time = now
            self.publish(now)
            return

        dt = (now - self.last_update_time).nanoseconds * 1e-9
        self.last_update_time = now
        if dt <= 0.0 or dt > self.max_dt:
            self.publish(now)
            return

        self.predict(dt)
        self.publish(now)

    def predict(self, dt):
        # Blend gyro yaw rate with bicycle-model yaw rate from steering+speed.
        steering_omega = self.speed * math.tan(self.steering) / WHEELBASE
        self.omega = self.gyro_weight * self.yaw_rate + (1.0 - self.gyro_weight) * steering_omega

        yaw0 = self.yaw
        self.x += self.speed * math.cos(yaw0) * dt
        self.y += self.speed * math.sin(yaw0) * dt
        self.yaw = wrap_to_pi(self.yaw + self.omega * dt)

        F = np.array([
            [1.0, 0.0, -self.speed * math.sin(yaw0) * dt],
            [0.0, 1.0, self.speed * math.cos(yaw0) * dt],
            [0.0, 0.0, 1.0],
        ])
        self.P = F @ self.P @ F.T + self.Q

    def publish(self, now):
        stamp = now.to_msg()
        q = yaw_to_quaternion_msg(self.yaw)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = q
        odom.twist.twist.linear.x = self.speed
        odom.twist.twist.angular.z = self.omega

        pose_cov = np.zeros((6, 6))
        pose_cov[0, 0] = self.P[0, 0]
        pose_cov[0, 1] = self.P[0, 1]
        pose_cov[1, 0] = self.P[1, 0]
        pose_cov[1, 1] = self.P[1, 1]
        pose_cov[5, 5] = self.P[2, 2]
        pose_cov[0, 5] = self.P[0, 2]
        pose_cov[5, 0] = self.P[2, 0]
        pose_cov[1, 5] = self.P[1, 2]
        pose_cov[5, 1] = self.P[2, 1]
        odom.pose.covariance = pose_cov.flatten().tolist()

        twist_cov = np.zeros((6, 6))
        twist_cov[0, 0] = 0.05
        twist_cov[5, 5] = 0.05
        odom.twist.covariance = twist_cov.flatten().tolist()

        self.odom_pub.publish(odom)

        if self.tf_broadcaster is not None:
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self.odom_frame
            tf.child_frame_id = self.base_frame
            tf.transform.translation.x = self.x
            tf.transform.translation.y = self.y
            tf.transform.translation.z = 0.0
            tf.transform.rotation = q
            self.tf_broadcaster.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
