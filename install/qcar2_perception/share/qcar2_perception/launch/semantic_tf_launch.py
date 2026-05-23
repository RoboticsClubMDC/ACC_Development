from launch import LaunchDescription
from launch_ros.actions import Node


def static_tf(name, x, y, z, qx, qy, qz, qw, child):
    return Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=name,
        arguments=[
            "--x", str(x),
            "--y", str(y),
            "--z", str(z),
            "--qx", str(qx),
            "--qy", str(qy),
            "--qz", str(qz),
            "--qw", str(qw),
            "--frame-id", "base_link",
            "--child-frame-id", child,
        ],
        output="screen",
    )

# CSI camera ID convention from QCar2 manual:
# right = 0
# rear  = 1
# front = 2
# left  = 3

def generate_launch_description():
    d435_tf = static_tf(
        "d435_aligned_static_tf",
        0.095, 0.032, 0.172,
        -0.5, 0.5, -0.5, 0.5,
        "aligned_camera_optical_frame",
    )

    csi_front_tf = static_tf(
        "csi_front_static_tf",
        0.183, 0.0, 0.110,
        -0.5, 0.5, -0.5, 0.5,
        "csi_front_optical_frame",
    )

    csi_left_tf = static_tf(
        "csi_left_static_tf",
        0.012, 0.033, 0.110,
        -0.707107, 0.0, 0.0, 0.707107,
        "csi_left_optical_frame",
    )

    csi_rear_tf = static_tf(
        "csi_rear_static_tf",
        -0.152, 0.0, 0.110,
        -0.5, -0.5, 0.5, 0.5,
        "csi_rear_optical_frame",
    )

    csi_right_tf = static_tf(
        "csi_right_static_tf",
        0.012, -0.053, 0.110,
        0.0, -0.707107, 0.707107, 0.0,
        "csi_right_optical_frame",
    )

    return LaunchDescription([
        d435_tf,
        csi_front_tf,
        csi_left_tf,
        csi_rear_tf,
        csi_right_tf,
    ])