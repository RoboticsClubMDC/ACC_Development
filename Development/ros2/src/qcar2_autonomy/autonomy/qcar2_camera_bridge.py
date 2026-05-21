#!/usr/bin/env python3
"""qcar2_camera_bridge.py — single RealSense owner for physical QCar 2.

This node is the canonical replacement for `rgbd.cpp` in physical
launches under the single-owner camera architecture. It instantiates
`pit.YOLO.utils.QCar2DepthAligned` (Quanser's official aligned-depth
helper — the same class the legacy `yolo_detector.py` used to use)
and re-publishes the aligned frames as standard ROS topics so every
other consumer (VO, YOLO, traffic, lane, ...) simply subscribes.

Why Python and not the C++ bridge in qcar2_nodes:
- The C++ implementation in `qcar2_nodes/src/qcar2_camera_bridge.cpp`
  attempted to use Quanser's `stream_*` C API directly. After multiple
  rounds of fixes (path resolver, PIT-matching invocation, expected
  -QERR_WOULD_BLOCK handling, switch from `stream_receive` to
  `stream_receive_byte_array`, switch from `qcomm_*` to `stream_*`
  with explicit recv_buffer_size=4.9MB, and even bumping the kernel
  `net.core.rmem_max` to 16 MB), throughput stayed pinned at ~0.3 fps
  vs the expected ~30 fps PIT achieves on the same hardware. The
  Quanser library does something inside its Python wrapper that
  pure-C callers cannot reach from outside (likely an internal
  ring-buffer pull from the kernel that runs at the Python-binding
  level). Detailed forensic recorded in
  `Development/ros2/src/qcar2_autonomy/VO_CHANGELOG.md` under the
  2026-05-14 hotfix #1..#5 entries and the pivot entry that follows.
- The Python bridge uses `QCar2DepthAligned` exactly as legacy
  `yolo_detector.py` did, which is known to run at ~30 Hz on this
  Jetson. Same class, same protocol, same buffer behavior — just
  one new ROS2 publisher node around it.

Topics published (drop-in replacement for rgbd.cpp's topic names so
existing subscribers do not need updating):
    /camera/color_image  sensor_msgs/Image  bgr8         ~30 Hz
    /camera/depth_image  sensor_msgs/Image  32FC1 meters ~30 Hz
    (depth is aligned to the color pixel grid at the driver level)

Parameters:
    device_type          'physical' | 'virtual' (default 'physical')
    pit_python_path      Optional path to MDC_libraries/python; the
                         resolver also tries MDC_PYTHON_PATH env and a
                         few well-known repo-relative locations.
    pit_ip               Default 'localhost'.
    pit_port             Default '18777' (matches PIT's
                         QCar2DepthAligned default).
    pit_non_blocking     Default True.
    pit_manual_start     Default False; PIT spawns the
                         QCar2DepthAlign runtime via quarc_run.
    publish_rate         Hz of the polling timer (default 60.0). The
                         actual publish rate is bounded by PIT's
                         delivery (~30 Hz). Polling above that just
                         means the timer returns quickly when no new
                         frame is available.
"""

import os
import sys
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster

# Canonical intrinsics + camera->body extrinsic live in ONE place
# (DepthProjector). Importing them here keeps camera_info / TF
# consistent with the VO engine and avoids re-typing or
# cross-polluting the virtual/physical tables.
from autonomy.visual_odometry import DepthProjector


def _rotation_to_quaternion(R):
    """3x3 rotation matrix -> (x, y, z, w) unit quaternion."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w, x = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        y, z = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] >= R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w, x = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        y, z = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w, x = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        y, z = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)


def _resolve_pit_import(custom_path: str):
    """Locate and import `pit.YOLO.utils.QCar2DepthAligned`.

    The same resolver pattern used in `vo_node._load_qcar2_depth_aligned`
    so behavior stays consistent across the project.
    """
    try:
        from pit.YOLO.utils import QCar2DepthAligned  # type: ignore
        return QCar2DepthAligned, "sys.path(existing)"
    except Exception:
        pass

    candidates = []
    if custom_path:
        candidates.append(custom_path)
    env_path = os.getenv("MDC_PYTHON_PATH", "").strip()
    if env_path:
        candidates.extend(p for p in env_path.split(":") if p.strip())
    home = Path.home()
    candidates.extend([
        str(home / "Documents/ACC_Development/Development/MDC_libraries/python"),
        "/home/nvidia/Documents/ACC_Development/Development/MDC_libraries/python",
        "/workspaces/isaac_ros-dev/MDC_libraries/python",
    ])

    for path in candidates:
        if not path or not Path(path).exists():
            continue
        if path not in sys.path:
            sys.path.insert(0, path)
        try:
            from pit.YOLO.utils import QCar2DepthAligned  # type: ignore
            return QCar2DepthAligned, path
        except Exception:
            continue

    raise ImportError(
        "Could not import pit.YOLO.utils.QCar2DepthAligned. Set the "
        "'pit_python_path' parameter or the MDC_PYTHON_PATH env var to "
        "the directory containing the 'pit' package."
    )


class CameraBridge(Node):

    def __init__(self):
        super().__init__("qcar2_camera_bridge")

        # ── Parameters ──────────────────────────────────────────────
        self.declare_parameter("device_type", "physical")
        self.declare_parameter("pit_python_path", "")
        self.declare_parameter("pit_ip", "localhost")
        self.declare_parameter("pit_port", "18777")
        self.declare_parameter("pit_non_blocking", True)
        self.declare_parameter("pit_manual_start", False)
        self.declare_parameter("publish_rate", 60.0)
        # 2026-05-18: publish CameraInfo + a static base_link->camera
        # optical TF so standard ROS RGB-D tools (RTAB-Map, RViz) work.
        # Additive: new topic + static TF only, no behavior change to
        # the image streams. Default ON (correct ROS hygiene anyway).
        self.declare_parameter("publish_camera_info", True)
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_optical_frame",
                                "camera_color_optical_frame")

        device_type = str(self.get_parameter("device_type").value).strip().lower()
        if device_type not in ("physical", "virtual"):
            self.get_logger().warn(
                f"Invalid device_type '{device_type}', defaulting to 'physical'")
            device_type = "physical"
        is_physical = (device_type == "physical")

        # ── Import + start the Quanser helper ───────────────────────
        pit_path = str(self.get_parameter("pit_python_path").value).strip()
        QCar2DepthAligned, resolved_path = _resolve_pit_import(pit_path)
        self.get_logger().info(
            f"PIT import resolved via: {resolved_path}")

        try:
            self.cam = QCar2DepthAligned(
                ip=str(self.get_parameter("pit_ip").value),
                nonBlocking=bool(self.get_parameter("pit_non_blocking").value),
                manualStart=bool(self.get_parameter("pit_manual_start").value),
                port=str(self.get_parameter("pit_port").value),
                isPhyscial=is_physical,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize QCar2DepthAligned. Check that the "
                "Quanser depth-align runtime can spawn (quarc_run on PATH, "
                "runtime binary present in the resources tree) and that "
                "no other camera owner is running."
            ) from exc

        self.get_logger().info(
            "QCar2DepthAligned initialized; expecting frames at ~30 Hz "
            f"(device_type={device_type}).")

        # ── Publishers (drop-in replacement for rgbd.cpp topic names) ──
        # 2026-05-15: switched to qos_profile_sensor_data (BEST_EFFORT) so
        # `ros2 topic hz /camera/*` reports the same rate the bridge
        # heartbeat reports. The previous default (RELIABLE, depth=5) caused
        # half-drop at the DDS layer for high-rate large Image messages,
        # making topic_hz read ~17 fps even though the bridge was publishing
        # at 30 fps. BEST_EFFORT drops in flight instead of buffering, which
        # is the correct semantics for live sensor data.
        self.bridge = CvBridge()
        self.pub_rgb = self.create_publisher(
            Image, "/camera/color_image", qos_profile_sensor_data)
        self.pub_depth = self.create_publisher(
            Image, "/camera/depth_image", qos_profile_sensor_data)

        # ── CameraInfo + static base_link->camera optical TF ────────
        # RTAB-Map / RViz / any standard RGB-D tool need the intrinsics
        # (CameraInfo) and the camera's place on the robot (TF). The
        # depth is aligned to the COLOR grid, so the COLOR (RGB)
        # intrinsics describe both streams. Numbers come from the
        # canonical DepthProjector table (single source of truth),
        # selected by device_type — never cross-pollinated.
        self.publish_camera_info = bool(
            self.get_parameter("publish_camera_info").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.optical_frame = str(
            self.get_parameter("camera_optical_frame").value)
        self.pub_caminfo = self.create_publisher(
            CameraInfo, "/camera/camera_info", qos_profile_sensor_data)

        if is_physical:
            fx, fy = DepthProjector.PHYSICAL_FX_RGB, DepthProjector.PHYSICAL_FY_RGB
            cx, cy = DepthProjector.PHYSICAL_CX_RGB, DepthProjector.PHYSICAL_CY_RGB
            T_cb = DepthProjector.PHYSICAL_T_CAM2BODY
        else:
            fx, fy = DepthProjector.VIRTUAL_FX_RGB, DepthProjector.VIRTUAL_FY_RGB
            cx, cy = DepthProjector.VIRTUAL_CX_RGB, DepthProjector.VIRTUAL_CY_RGB
            T_cb = DepthProjector.VIRTUAL_T_CAM2BODY  # NOTE: QLabs units

        self._caminfo = CameraInfo()
        self._caminfo.width = 640
        self._caminfo.height = 480
        self._caminfo.distortion_model = "plumb_bob"
        self._caminfo.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # post-calib: none
        self._caminfo.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        self._caminfo.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self._caminfo.p = [fx, 0.0, cx, 0.0,
                           0.0, fy, cy, 0.0,
                           0.0, 0.0, 1.0, 0.0]

        # Static TF base_link -> camera optical frame, from the same
        # T_cam2body the VO engine uses (P_body = R@P_cam + t).
        self._static_tf = StaticTransformBroadcaster(self)
        T_cb = np.asarray(T_cb, dtype=np.float64)
        q = _rotation_to_quaternion(T_cb[:3, :3])
        st = TransformStamped()
        st.header.stamp = self.get_clock().now().to_msg()
        st.header.frame_id = self.base_frame
        st.child_frame_id = self.optical_frame
        st.transform.translation.x = float(T_cb[0, 3])
        st.transform.translation.y = float(T_cb[1, 3])
        st.transform.translation.z = float(T_cb[2, 3])
        st.transform.rotation.x = float(q[0])
        st.transform.rotation.y = float(q[1])
        st.transform.rotation.z = float(q[2])
        st.transform.rotation.w = float(q[3])
        if self.publish_camera_info:
            self._static_tf.sendTransform(st)
            self.get_logger().info(
                f"CameraInfo on /camera/camera_info "
                f"(fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}) and "
                f"static TF {self.base_frame}->{self.optical_frame} "
                "published (RTAB-Map / RViz ready).")

        # ── Polling timer ───────────────────────────────────────────
        rate = float(self.get_parameter("publish_rate").value)
        if rate <= 0.0:
            rate = 60.0
        self.timer = self.create_timer(1.0 / rate, self._tick)

        # ── Diagnostic state ────────────────────────────────────────
        self._frames_published = 0
        self._first_frame_logged = False
        self._last_hb = self.get_clock().now()

        self.get_logger().info(
            f"qcar2_camera_bridge up (poll rate={rate:.1f} Hz). "
            "Topics: /camera/color_image (bgr8), "
            "/camera/depth_image (32FC1 meters, aligned).")

    def _tick(self):
        # Heartbeat every ~5 s — helps confirm publish rate live.
        now = self.get_clock().now()
        dt = (now - self._last_hb).nanoseconds * 1e-9
        if dt >= 5.0:
            fps = (self._frames_published / dt) if dt > 0 else 0.0
            self.get_logger().info(
                f"[bridge hb] published={self._frames_published} "
                f"frames in {dt:.1f} s (~{fps:.1f} fps)")
            self._frames_published = 0
            self._last_hb = now

        try:
            new = self.cam.read()
        except Exception as exc:
            self.get_logger().warn(
                f"QCar2DepthAligned.read() raised: {exc}",
                throttle_duration_sec=5.0)
            return
        if not new:
            return  # no fresh packet this tick

        if not self._first_frame_logged:
            self.get_logger().info(
                "First aligned RGBD frame received from QCar2DepthAlign "
                "runtime; publishing on /camera/color_image and "
                "/camera/depth_image.")
            self._first_frame_logged = True

        # PIT's `read()` writes into `self.cam.rgb` (uint8 HxWx3, BGR
        # order because PIT pre-selects channels [3,2,1] before assigning)
        # and `self.cam.depth` (float32, HxWx1, meters, aligned to the
        # color pixel grid).
        rgb = np.ascontiguousarray(self.cam.rgb)

        depth = np.asarray(self.cam.depth)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        depth = np.ascontiguousarray(depth.astype(np.float32, copy=False))

        stamp = self.get_clock().now().to_msg()

        # frame_id aligned to the TF tree's optical frame so RTAB-Map /
        # RViz can place the camera correctly. (VO/overlay key off the
        # topic, not this string, so this is metadata-only.)
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb, encoding="bgr8")
        rgb_msg.header.stamp = stamp
        rgb_msg.header.frame_id = self.optical_frame
        self.pub_rgb.publish(rgb_msg)

        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = self.optical_frame
        self.pub_depth.publish(depth_msg)

        if self.publish_camera_info:
            self._caminfo.header.stamp = stamp
            self._caminfo.header.frame_id = self.optical_frame
            self.pub_caminfo.publish(self._caminfo)

        self._frames_published += 1

    def destroy_node(self):
        try:
            self.cam.terminate()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.try_shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
