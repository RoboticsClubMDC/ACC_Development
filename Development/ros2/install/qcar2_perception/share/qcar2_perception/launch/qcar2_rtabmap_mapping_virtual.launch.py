from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Added 2026-05-22 22:35:00 EDT:
    # One-shot virtual RTAB mapping launch. This intentionally does not start
    # Cartographer or qcar2_nodes/rgbd. LiDAR supplies 2D geometry, the aligned
    # D435 source supplies RGB-D evidence, and RTAB owns rtab_map -> rtab_odom.
    lidar_node = Node(
        package="qcar2_nodes",
        executable="lidar",
        name="Lidar",
        parameters=[{"device_type": "virtual"}],
        output="screen",
    )

    qcar2_hardware = Node(
        package="qcar2_nodes",
        executable="qcar2_hardware",
        name="qcar2_hardware",
        parameters=[{"device_type": "virtual"}],
        output="screen",
    )

    lidar_tf = Node(
        package="qcar2_nodes",
        executable="fixed_lidar_frame_virtual",
        name="fixed_lidar_frame",
        output="screen",
    )

    d435_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="d435_aligned_static_tf",
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

    d435_aligned_source = Node(
        package="qcar2_perception",
        executable="d435_aligned_source",
        name="d435_aligned_source",
        parameters=[{
            "is_physical": False,
            "distance_scale": 0.1,
        }],
        output="screen",
    )

    rgbd_sync = Node(
        package="rtabmap_sync",
        executable="rgbd_sync",
        namespace="rtabmap",
        name="rgbd_sync",
        parameters=[{
            "approx_sync": True,
            "topic_queue_size": 50,
            "sync_queue_size": 50,
            "qos": 0,
            "qos_camera_info": 0,
            "approx_sync_max_interval": 0.1,
        }],
        remappings=[
            ("rgb/image", "/perception/d435/rgb/image_raw"),
            ("depth/image", "/perception/d435/depth/image_rect"),
            ("rgb/camera_info", "/perception/d435/camera_info"),
        ],
        output="screen",
    )

    rgbd_odometry = Node(
        package="rtabmap_odom",
        executable="rgbd_odometry",
        namespace="rtabmap",
        name="rgbd_odometry",
        parameters=[{
            "frame_id": "base_link",
            "odom_frame_id": "rtab_odom",
            "publish_tf": True,
            "approx_sync": True,
            "subscribe_rgbd": True,
        }],
        output="screen",
    )

    rtabmap = Node(
        package="rtabmap_slam",
        executable="rtabmap",
        namespace="rtabmap",
        name="rtabmap",
        parameters=[{
            "frame_id": "base_link",
            "map_frame_id": "rtab_map",
            "publish_tf": True,
            "subscribe_rgbd": True,
            "subscribe_scan": True,
            "approx_sync": True,
            "topic_queue_size": 50,
            "sync_queue_size": 50,
            "approx_sync_max_interval": 0.2,
            "RGBD/CreateOccupancyGrid": "true",
            "Rtabmap/DetectionRate": "1.0",
        }],
        remappings=[
            ("scan", "/scan"),
        ],
        output="screen",
    )

    return LaunchDescription([
        lidar_node,
        qcar2_hardware,
        lidar_tf,
        d435_tf,
        d435_aligned_source,
        rgbd_sync,
        rgbd_odometry,
        rtabmap,
    ])
