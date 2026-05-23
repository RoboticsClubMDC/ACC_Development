from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    realsense_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="realsense_static_tf",
        arguments=[
            "--x", "0.095",
            "--y", "0.032",
            "--z", "0.172",
            "--qx", "-0.5",
            "--qy", "0.5",
            "--qz", "-0.5",
            "--qw", "0.5",
            "--frame-id", "base_link",
            "--child-frame-id", "aligned_camera_optical_frame",
        ],
        output="screen",
    )

    return LaunchDescription([
        realsense_tf,
    ])
