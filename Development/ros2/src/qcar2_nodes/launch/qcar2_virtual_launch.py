# This is the launch file that starts up the basic QCar2 nodes (VIRTUAL).

import subprocess

from launch import LaunchDescription
from launch.actions import (ExecuteProcess, LogInfo, RegisterEventHandler,
                            OpaqueFunction, TimerAction, DeclareLaunchArgument)
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration, PythonExpression
from launch.event_handlers import (OnProcessExit, OnProcessStart)
from launch.conditions import IfCondition

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # --- 2026-05-26: camera source selector (single-owner architecture) ---
    # Mirrors qcar2_launch.py (physical) so virtual and physical now expose
    # the same `camera_source` knob. Both options publish the same topic
    # names (/camera/color_image, /camera/depth_image) so every downstream
    # subscriber (yolo_detector, vo_node, cartographer's RGB-D consumers if
    # ever added, etc.) is agnostic.
    #
    # 'depth_aligned' (DEFAULT) -> qcar2_autonomy/camera_bridge owns the
    #                              RealSense via pit.YOLO.utils.QCar2DepthAligned
    #                              (virtual branch: QLabs Camera3D port 18965
    #                              + warpPerspective depth alignment) and
    #                              publishes aligned 32FC1 depth in METERS on
    #                              /camera/depth_image and bgr8 RGB on
    #                              /camera/color_image. This is what YOLO and
    #                              VO require — yolo_detector.py's distance
    #                              estimate depends on depth being aligned to
    #                              the RGB pixel grid, otherwise its mask-
    #                              median distance is reading the wrong pixels.
    # 'rgbd'                    -> legacy qcar2_nodes/rgbd (device_type:=virtual)
    #                              publishes MONO16 raw depth that is NOT
    #                              aligned to the color frame. Kept for parity
    #                              with the physical fallback path.
    # ----------------------------------------------------------------------
    camera_source_arg = DeclareLaunchArgument(
        'camera_source',
        default_value='depth_aligned',
        description=("RealSense owner in virtual mode: 'depth_aligned' "
                     "(qcar2_autonomy/camera_bridge, aligned 32FC1 meters via "
                     "QLabs Camera3D + warpPerspective) or 'rgbd' "
                     "(legacy qcar2_nodes/rgbd, MONO16 raw)."))
    camera_source = LaunchConfiguration('camera_source')

    lidar_node = Node(
            package='qcar2_nodes',
            executable='lidar',
            name='Lidar',
            parameters=[{"device_type":"virtual"}]
        )

    # Legacy RealSense owner (MONO16). Active only when camera_source == 'rgbd'.
    realsense_camera_node = Node(
            package='qcar2_nodes',
            executable='rgbd',
            name='RealsenseCamera',
            parameters=[{"device_type":"virtual"},
                        {"frame_width_rgb":640},
                        {"frame_height_rgb":480},
                        {"frame_width_depth":640},
                        {"frame_height_depth":480}],
            condition=IfCondition(
                PythonExpression(["'", camera_source, "' == 'rgbd'"]))
        )

    # Single-owner aligned-depth bridge — Python implementation in
    # qcar2_autonomy. Same node used by physical mode (qcar2_launch.py);
    # device_type:=virtual triggers QCar2DepthAligned's QLabs Camera3D
    # branch (utils.py lines 148-165: Camera3D on tcpip://localhost:18965,
    # depth_scale=5.5, warpPerspective M matrix for alignment). Republishes
    # frames as bgr8 RGB + 32FC1 depth in meters, plus CameraInfo and the
    # base_link -> camera_color_optical_frame static TF that standard
    # RGB-D tools (RTAB-Map, RViz, our yolo_detector / vo_node) expect.
    camera_bridge_node = Node(
            package='qcar2_autonomy',
            executable='camera_bridge',
            name='RealsenseCamera',
            parameters=[{"device_type":"virtual"}],
            condition=IfCondition(
                PythonExpression(["'", camera_source, "' == 'depth_aligned'"]))
        )

    csi_camera_node = Node(
            package='qcar2_nodes',
            executable='csi',
            name='csi_camera',
            parameters=[{"device_type":"virtual"},
                        {"frame_width":820},
                        {"frame_height":410},
                        {"frame_rate":15.0},
                        {"camera_num":3}]
        )

    qcar2_hardware = Node(
            package='qcar2_nodes',
            executable='qcar2_hardware',
            name='qcar2_hardware',
            parameters=[
                {"device_type":"virtual"},
                {"allow_boolean_led_cmd": False},
            ]

        )

    return LaunchDescription([
        camera_source_arg,
        lidar_node,
        realsense_camera_node,
        camera_bridge_node,
        csi_camera_node,
        qcar2_hardware,
    ])
