#!/usr/bin/env python3
"""pd_tuner.py — Tkinter GUI with two sliders that publish PD gains live to
path_follower's tuning topics.

Why this exists: Foxglove's Publish panel only fires on button press, not
on slider drag. This script gives you continuous slider control over
Kp/Kd while watching the car drive in Foxglove side-by-side.

Topics published:
    /nav/kp_steering_set  (std_msgs/Float32)
    /nav/kd_steering_set  (std_msgs/Float32)

Usage (in dev container with X forwarding or in a desktop environment):
    python3 /workspaces/isaac_ros-dev/ros2/scripts/pd_tuner.py

Defaults match what nav_to_pose.py declares (Kp=1.0, Kd=0.3) so opening
the GUI doesn't immediately change the controller until you drag.
"""

import threading
import tkinter as tk

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


class PDTunerNode(Node):

    def __init__(self):
        super().__init__("pd_tuner_gui")
        self.pub_kp = self.create_publisher(Float32, "/nav/kp_steering_set", 5)
        self.pub_kd = self.create_publisher(Float32, "/nav/kd_steering_set", 5)

    def publish_kp(self, value):
        self.pub_kp.publish(Float32(data=float(value)))

    def publish_kd(self, value):
        self.pub_kd.publish(Float32(data=float(value)))


def main():
    rclpy.init()
    node = PDTunerNode()

    # Spin in a background thread so the Tk mainloop owns the foreground.
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    root = tk.Tk()
    root.title("QCar2 PD Tuner")
    root.geometry("520x260")

    tk.Label(root, text="Pure-pursuit PD live tuning",
             font=("Helvetica", 12, "bold")).pack(pady=8)

    tk.Label(root, text="Kp (proportional)").pack()
    kp_slider = tk.Scale(
        root, from_=0.0, to=2.0, resolution=0.01, orient=tk.HORIZONTAL,
        length=460, tickinterval=0.5,
        command=node.publish_kp,
    )
    kp_slider.set(1.0)
    kp_slider.pack(pady=4)

    tk.Label(root, text="Kd (derivative — gyro damping)").pack()
    kd_slider = tk.Scale(
        root, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
        length=460, tickinterval=0.25,
        command=node.publish_kd,
    )
    kd_slider.set(0.30)
    kd_slider.pack(pady=4)

    tk.Label(
        root, fg="gray",
        text="Drag a slider → publishes immediately to /nav/kp_steering_set or "
             "/nav/kd_steering_set\nWatch path_follower log for confirmation.",
    ).pack(pady=8)

    try:
        root.mainloop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
