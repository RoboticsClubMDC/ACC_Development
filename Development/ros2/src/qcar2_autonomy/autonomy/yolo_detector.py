#! /usr/bin/env python3
import sys
import threading

sys.path.insert(0, "/workspaces/isaac_ros-dev/MDC_libraries/python")

# Quanser PIT YOLOv8 wrapper. Deferred (in-method) import so a custom
# ultralytics model can be loaded without requiring PIT to be importable
# — useful when running off the QCar where the PIT package isn't on the
# python path. See _load_model() below; the deferred import happens only
# if model_type=='quanser_seg' is selected.
#
# from pit.YOLO.nets import YOLOv8
# --- 2026-05-14: camera ownership moved to qcar2_camera_bridge ---------
# This node previously instantiated QCar2DepthAligned itself, which made
# it a second owner of the RealSense alongside rgbd / the new bridge.
# Under the single-owner architecture the bridge is the sole PIT client;
# this node now subscribes to /camera/color_image and /camera/depth_image
# (aligned 32FC1 meters from the bridge, or MONO16 raw from legacy rgbd
# if camera_source:=rgbd is selected in the launch). YOLO inference,
# /motion_enable, /trip_planner/qcar_state, and the stop-override logic
# below are unchanged.
#
# To restore direct PIT-ownership behavior (not recommended outside
# diagnostic runs), uncomment the import + instantiation below and
# disable qcar2_camera_bridge in the launch file.
# from pit.YOLO.utils import QCar2DepthAligned
# -----------------------------------------------------------------------

# Generic python packages
import time  # Time library
import numpy as np
import cv2

# ROS specific packages
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool, UInt8   # <-- ADDED UInt8
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

'''
Description:

Node for detecting traffic light state and signs on the road. Provides flags
which define if a traffic signal has been detected and what action to take.

2026-05-26: ported Erick's rich semantic logic (traffic-light color via PIT
lightColor, yield-sign handling, per-class distance gating, annotated overlay
on /qcar_camera/rgb_yolo) onto Gabriel's subscriber-only camera path. Camera
ownership stays in qcar2_camera_bridge; this node only consumes
/camera/color_image + /camera/depth_image. New behavior versus Erick's
Erick-branch yolo_detector.py:
  * Subscriber pattern (no QCar2DepthAligned).
  * Traffic-light stop requires a FULLY VISIBLE bbox (margin from image
    edges) so we don't slam on the brakes when only the bottom of the TL
    has entered the frame. Justification (user, 2026-05-26): "we don't
    want to stop in the middle of the street" when only half the TL is
    in view.
  * Per-class thresholds promoted to declare_parameter() so they can be
    tuned at launch without rebuilding.
  * Diagnostic log: every detection prints PIT's mask-median distance
    AND a center-patch depth sample. Use this to see WHY the stop sign
    appears closer-than-reality and the TL appears farther-than-reality:
    on a TL the seg mask bleeds onto the sky / pole behind it, biasing
    the median high; on a small/distant stop sign the mask is jitter-noisy
    and biases short. See PIT post_processing in pit/YOLO/nets.py:373.
'''


# --- Per-class thresholds (defaults; overridable via ROS params) -------
# These come from Erick's tuned values, except where the user observed
# brake-too-early / brake-too-late issues at physical runs (2026-05-26):
#   stop_sign_dist_m: kept conservative (0.55) — pulled in from Erick's
#     1.0 m because user saw the car stop ~1.5 m before the sign on
#     Erick's branch. PIT mask-median over a small octagon at distance
#     biases shorter than truth; 0.55 m means the car only commits to a
#     stop when the depth median says ~half a meter, which empirically
#     lines up with "right in front of the sign".
#   tl_stop_dist_m: pushed out to 3.5 m (from Erick's 2.5) — user saw
#     TL stops trigger too late. The TL seg mask leaks onto sky / pole-
#     behind, biasing the median DEEPER than the real light face, so 2.5 m
#     was effectively never crossed until the car was on top of the light.
#   tl_edge_margin_px: only act on TLs whose bbox is at least this many
#     px inside the image border on all four sides (= "we can see the
#     full object").
# -----------------------------------------------------------------------
DEFAULTS = {
    "stop_sign_conf":      0.90,
    "stop_sign_dist_m":    0.55,
    "stop_sign_hold_s":    3.0,

    "yield_sign_conf":     0.90,
    "yield_sign_dist_m":   0.55,
    "yield_sign_hold_s":   1.5,

    # 2026-05-26 PM-2: predictive "stop beside the sign" approach.
    # Instead of "brake when depth < X" (which fires too late if the car
    # is moving and propagation latency makes us blow past the sign),
    # track depth-vs-time for each detected sign, linear-fit the slope to
    # estimate approach speed, and predict the absolute time when we'll
    # be `stop_target_offset_m` away from the sign. Then brake at that
    # absolute time even if the sign drops out of view in the final
    # approach (which happens — narrow camera FOV).
    "stop_target_offset_m":     0.30,  # stop this far before the sign face
    "stop_predict_min_samples": 3,     # need 3+ frames before fitting
    "stop_predict_max_depth_m": 8.0,   # ignore detections beyond this. Bumped
                                       # from 5.0 after 2026-05-26 run showed
                                       # first detections at 5.6 m were dropped
                                       # and tracking only started at 4.9 m.
    "stop_predict_commit_at_m": 4.5,   # commit depth-rate prediction once depth
                                       # crosses this. Bumped from 3.0 — when a
                                       # sign is on the side of the road, depth
                                       # decreases SLOWER than driving distance
                                       # (we're moving past it laterally) and
                                       # depth often never reaches 3 m before
                                       # the sign exits the FOV.
    "stop_predict_min_speed":   0.05,  # m/s; only commit if approaching faster

    # 2026-05-26 PM-3: lateral-edge commit trigger. PRIMARY brake signal
    # for signs on the side of the road. When the bbox center crosses
    # into the outer `lateral_edge_frac` of the frame, we're about to
    # pass the sign laterally — brake immediately. Subsumes the depth-
    # rate prediction in the common case; depth-rate stays as a fallback
    # for head-on / occluded signs.
    "lateral_edge_frac":        0.15,  # outer 15% of frame on either side
                                       # (= 96 px on a 640-wide image)

    "tl_conf":             0.50,
    "tl_min_dist_m":       0.30,   # ignore near-zero depth garbage
    "tl_stop_dist_m":     10.0,    # camera-depth gate. Bumped from 3.5 after
                                   # 2026-05-26 PM-3 run showed overhead TLs
                                   # never drop below ~6-9 m of camera depth
                                   # (we look up at them, depth is along camera
                                   # ray not road distance) -- 3.5 m was
                                   # literally never satisfied.
    "tl_min_height_px":   50,      # 2026-05-26 PM-4: bbox-height proximity
                                   # fallback. Independent of depth. A TL bbox
                                   # this tall (in a 480-tall frame) means we
                                   # are visually close enough to act. Either
                                   # this OR the depth gate above can engage
                                   # the brake (logical OR).
    "tl_hold_s":           0.60,   # refresh-each-frame brake hold; must exceed
                                   # one inference period (~0.033s @ 30Hz) by a
                                   # comfortable margin to avoid the stop-go-stop
                                   # flicker the user saw at 0.25s. Bumped 2026-05-26.
    "tl_edge_margin_px":   8,
    "tl_allow_top_clip":   True,   # 2026-05-26 PM-4: overhead TLs naturally
                                   # clip the top edge of the frame as we
                                   # approach (the camera tilts up, the TL
                                   # bracket exits the top). User log showed
                                   # repeated "TL RED ... bbox not fully visible"
                                   # because top y=0. Allowing top-clipping is
                                   # the right policy for overhead lights; we
                                   # still reject left/right/bottom clipping
                                   # (those indicate misdetect or wrong-lane TL).

    # 2026-05-26 PM-6: temporal color stabilization. User reported the
    # car decides red -> stops -> "nvm" -> green -> moves -> "actually red"
    # -> stops again. That's frame-to-frame color flicker in the HSV
    # check (a single weak detection flipping the decision). Majority
    # vote over the last N color readings stabilizes the decision.
    "tl_color_history_size":    5,

    # 2026-05-26 PM-6: pass-the-line rule. If the TL bbox is THIS tall
    # at the moment we first see red/yellow, we are already at the
    # intersection -- starting a NEW brake here would stop in the
    # middle of the road. Don't engage a fresh brake; existing brakes
    # from earlier in the approach continue normally (refresh-each-frame
    # keeps them active while the TL stays red/yellow).
    "tl_pass_line_height_px":  100,

    # 2026-05-26 PM-6: stricter HSV thresholds in _check_traffic_light_color.
    # The previous "relative threshold 0.25*(max-min) on V" trips on
    # weak gradients and reflections. Add absolute floors:
    #   * V (brightness) must be above this to count as "lit"
    #   * S (saturation) must be above this -- a real colored light
    #     has high saturation; a near-white reflection has low.
    "tl_color_min_v":           90,    # 0-255
    "tl_color_min_s":           70,    # 0-255  (the key reflection-rejection)

    # 2026-05-26 PM-6: bottom crop. The RealSense FOV on the QCar2 sees
    # the front CSI camera housing in the bottom rows -- user reports
    # it getting misclassified as "car" and "traffic light". Cropping
    # the bottom N rows before YOLO inference is the cleanest fix.
    # Default ~5% of 480 = 24 px. Adjust based on what shows up in
    # /qcar_camera/rgb_yolo (the published overlay reflects the crop).
    "crop_bottom_px":          24,

    # 2026-05-26 PM-8: TL approach state machine. Implements the user's
    # explicit semantic — commit to STOP-or-GO at first sighting based
    # on color, then HOLD that decision through subsequent color changes
    # until the TL exits the FOV. Prevents the "saw green, started
    # going, light turned yellow halfway through intersection, slammed
    # brakes" failure mode.
    "tl_fsm_lost_frames_to_reset":    15,  # ~0.5 s @ 30 fps
    "tl_fsm_green_frames_to_release": 3,   # ~0.1 s of stable green
                                           # to release a COMMIT_STOP

    "detection_cooldown_s": 10.0,  # for stop/yield only (TL is always-evaluated)

    # 2026-05-26: which depth measurement gates the stop decision.
    #   'center_patch' (DEFAULT) — median of valid depths in central 20%
    #     of the bbox. Robust to seg-mask leakage (stop sign mask
    #     dilating onto sky/pole-behind, TL mask leaking onto sky).
    #     This is "Luigi's center-only" idea generalized.
    #   'pit_median'             — PIT's torch.median(mask × depth) over
    #     the full segmentation mask. Kept as opt-in for A/B comparison.
    # The other measurement is always logged as the second opinion.
    "distance_source":     "center_patch",
}


class ObjectDetector(Node):

    def __init__(self):
        super().__init__('yolo_detector')

        # Additional parameters
        imageWidth  = 640
        imageHeight = 480
        self._img_w = imageWidth
        self._img_h = imageHeight

        # --- declare ROS params (tunable at launch) --------------------
        for k, v in DEFAULTS.items():
            self.declare_parameter(k, v)

        # 2026-05-26: backend selector for the YOLO model.
        #   model_path  — path to a .pt file. Empty string = use the
        #                 Quanser PIT seg model at the hardcoded default.
        #   model_type  — 'auto' | 'quanser_seg' | 'ultralytics'.
        #                 'auto' picks 'ultralytics' when model_path is
        #                 set, else 'quanser_seg'.
        # Custom ultralytics models use ultralytics.YOLO() directly
        # (no PIT wrapper, no seg mask required) and rely on the existing
        # _center_patch_depth_m gating + an in-process HSV traffic-light
        # color check ported from PIT (since custom models won't carry
        # PIT's TrafficLight.lightColor attribute).
        self.declare_parameter("model_path", "")
        self.declare_parameter("model_type", "auto")

        # --- 2026-05-14: camera ownership replaced by ROS subscribers --
        # Original line (kept commented so the direct-PIT path can be
        # restored quickly if needed):
        #     self.QCarImg = QCar2DepthAligned()
        # The replacement is two camera subscribers + a small frame
        # buffer with thread-safe access. See _rgb_cb / _depth_cb below.
        self._frame_lock = threading.Lock()
        self._rgb_latest = None      # numpy.ndarray HxWx3 uint8 (bgr)
        self._depth_latest = None    # numpy.ndarray HxW float32 meters
        # ---------------------------------------------------------------
        self._load_model(imageWidth, imageHeight)

        # Call on_timer function every second to receive pose info
        self.dt = 1/30
        self.timer = self.create_timer(self.dt, self.on_timer)

        self.motion_publisher = self.create_publisher(Bool,'/motion_enable',1)
        self.motion_enable = True
        # 2026-05-26: rewritten brake-time bookkeeping. Old design used a
        # `sign_detected` latch that skipped YOLO until disable_until
        # expired — that's what caused the TL stop-go-stop flicker (every
        # hold expiry was a one-tick brake-release before the next YOLO
        # tick re-engaged). New design evaluates every tick and tracks
        # brake state purely via absolute timestamps:
        #   brake_until_abs       — absolute time.time() when brake releases.
        #                           TL refreshes this each frame it's red/yellow;
        #                           stop/yield set it once per latch.
        #   sign_cooldown_until_abs — earliest time a stop/yield sign can
        #                             re-trigger (TL is exempt, always evaluates).
        self.brake_until_abs = 0.0
        self.sign_cooldown_until_abs = 0.0
        self.flag_value = True
        self.publish_motion_flag(True)

        # 2026-05-26 PM-2: per-sign-type approach trackers. Each tracks
        # depth history over consecutive frames and predicts the absolute
        # time at which the brake should engage so the car ends up
        # `stop_target_offset_m` away from the sign face. See
        # _SignApproachTracker below.
        self._stop_tracker  = ObjectDetector._SignApproachTracker(self.get_logger(), "stop sign")
        self._yield_tracker = ObjectDetector._SignApproachTracker(self.get_logger(), "yield sign")

        # 2026-05-26 PM-6: TL state for the temporal color stabilization.
        self._tl_color_history = []          # list of (t_abs, color_str)
        self._tl_last_bucket_str = ""        # for FSM-driver log
        # 2026-05-26 PM-8 (kept; PM-9 reverted): TL approach FSM owns
        # the stop/go logic. States: IDLE / COMMIT_STOP / COMMIT_GO.
        # First-sight commit-and-hold: once we commit to GO, no later
        # color change can engage a brake. Deliberate trade per user
        # (Turn 150): running a red is worse than stopping in the
        # middle of an intersection.
        self._tl_fsm = ObjectDetector._TLStateMachine(self.get_logger())

        # publish image aligned information
        self.bridge = CvBridge()
        self.publish_rgb = self.create_publisher(Image,'/qcar_camera/rgb',10)
        self.publish_depth = self.create_publisher(Image,'/qcar_camera/depth',10)
        # 2026-05-26: annotated overlay (boxes + masks + class+distance) so
        # rqt_image_view can show what YOLO is actually doing. This is the
        # topic the user inspects when debugging the brake-too-soon / TL-
        # brake-too-late problem.
        self.publish_rgb_yolo = self.create_publisher(Image, '/qcar_camera/rgb_yolo', 10)

        # --- 2026-05-14: camera ownership replaced by ROS subscribers --
        # Subscribe to whichever publisher is supplying camera data:
        #   - qcar2_camera_bridge (default, physical) -> 32FC1 meters
        #   - rgbd.cpp            (fallback)          -> MONO16 raw counts
        # Encoding is detected per frame in _depth_cb.
        # 2026-05-15: sensor_data QoS (BEST_EFFORT) to match the bridge
        # publishers. Reliable QoS on Image topics half-dropped at the DDS
        # layer; BEST_EFFORT is the correct semantics for live sensor data.
        self.create_subscription(
            Image, '/camera/color_image', self._rgb_cb, qos_profile_sensor_data)
        self.create_subscription(
            Image, '/camera/depth_image', self._depth_cb, qos_profile_sensor_data)
        # ---------------------------------------------------------------

        # --- ADDED: publish qcar_state override to existing topic ---
        self.qcar_state_pub = self.create_publisher(UInt8, '/trip_planner/qcar_state', 10)
        self.stop_override_active = False
        self.stop_override_until = 0.0
        # -----------------------------------------------------------

        self.timer2 = self.create_timer(1/500, self.flag_publisher)

        self.get_logger().info(
            "yolo_detector ready (subscriber mode): "
            "/camera/color_image + /camera/depth_image -> "
            "/motion_enable, /trip_planner/qcar_state, /qcar_camera/rgb_yolo"
        )

    def flag_publisher(self):
        # keep existing behavior
        self.publish_motion_flag(self.flag_value)

        # --- ADDED: while stop override active, spam state=1 so it wins over trip_planner 10Hz ---
        now = time.time()
        if self.stop_override_active and now < self.stop_override_until:
            msg = UInt8()
            msg.data = 1
            self.qcar_state_pub.publish(msg)
        elif self.stop_override_active and now >= self.stop_override_until:
            self.stop_override_active = False
        # --------------------------------------------------------------------------------------

    # --- 2026-05-14: ROS subscriber callbacks (replace direct PIT read) ---
    def _rgb_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:
            self.get_logger().warn(
                f"yolo_detector RGB decode error: {exc}",
                throttle_duration_sec=5.0)
            return
        with self._frame_lock:
            self._rgb_latest = frame

    def _depth_cb(self, msg):
        """Accept either 32FC1 (aligned meters from bridge) or MONO16
        (raw counts from rgbd.cpp). Store as float32 meters internally."""
        enc = getattr(msg, "encoding", "") or ""
        try:
            if enc == "32FC1":
                d = self.bridge.imgmsg_to_cv2(msg, "32FC1")
                if d.ndim == 3:
                    d = d[:, :, 0]
                depth = d.astype(np.float32, copy=False)
            else:
                # Legacy MONO16: divide by RealSense depth-units default
                # (0.001 m/unit -> divisor 1000) to get meters.
                d = self.bridge.imgmsg_to_cv2(msg, "passthrough")
                if d.ndim == 3:
                    d = d[:, :, 0]
                depth = d.astype(np.float32) / 1000.0
        except Exception as exc:
            self.get_logger().warn(
                f"yolo_detector depth decode error: {exc}",
                throttle_duration_sec=5.0)
            return
        with self._frame_lock:
            self._depth_latest = depth
    # ----------------------------------------------------------------------

    def on_timer(self):
        # --- 2026-05-14: read frames from subscribers instead of PIT ---
        # Original (commented; lines that drove direct PIT ownership):
        #     self.QCarImg.read()
        #     rgb = self.QCarImg.rgb
        #     depth = self.QCarImg.depth
        with self._frame_lock:
            rgb = None if self._rgb_latest is None else self._rgb_latest.copy()
            depth = None if self._depth_latest is None else self._depth_latest.copy()
        if rgb is None or depth is None:
            # Bridge / rgbd has not delivered a frame yet; skip this tick.
            return

        # 2026-05-26 PM-6: bottom crop -- hide the CSI bumper that the
        # RealSense sees in the bottom rows. Crop BEFORE inference so the
        # YOLO model never sees those pixels. The published overlay
        # (/qcar_camera/rgb_yolo) is also cropped so the user can verify
        # in rqt what the model actually sees. Bbox coordinates are
        # therefore in the cropped frame's coordinate system, which is
        # consistent with the cropped depth.
        crop_bot = int(self.get_parameter("crop_bottom_px").value)
        if crop_bot > 0:
            h, _ = rgb.shape[:2]
            if crop_bot < h:
                rgb   = rgb[:h - crop_bot, :, :]
                depth = depth[:h - crop_bot, :]
        # Refresh cached image dimensions so downstream visibility
        # checks (_bbox_fully_in_frame) use the cropped height.
        self._img_h = rgb.shape[0]
        self._img_w = rgb.shape[1]

        # Cache the latest frames for yolo_detect() which still references
        # them by name (and previously accessed self.QCarImg.rgb / .depth).
        self._current_rgb = rgb
        self._current_depth = depth

        # depth is already float32 in meters (32FC1 path) or converted from
        # MONO16 to meters in _depth_cb; no extra processing required.
        msg_rgb = self.bridge.cv2_to_imgmsg(rgb, "bgr8")
        msg_depth = self.bridge.cv2_to_imgmsg(depth, "32FC1")

        self.publish_rgb.publish(msg_rgb)
        self.publish_depth.publish(msg_depth)

        # 2026-05-26: always evaluate every tick. yolo_detect() updates
        # self.brake_until_abs directly (TL refreshes it each frame, stop
        # and yield use the cooldown latch). flag_value is then derived
        # purely from absolute time — no per-tick latch, no flicker gap.
        self.yolo_detect()

        now = time.time()
        new_flag = (now >= self.brake_until_abs)
        # 2026-05-26 PM-3: log every motion_enable state transition so we
        # can see in this node's own console when the brake intent flips
        # — independent of whether downstream consumers (path_follower)
        # are actually acting on it. If you see BRAKE ENGAGED here and
        # the car keeps moving, the bug is downstream, not in yolo.
        if new_flag != self.flag_value:
            if new_flag:
                self.get_logger().info(
                    f">>> BRAKE RELEASED (motion_enable -> True) "
                    f"at t={now:.2f}")
            else:
                hold = self.brake_until_abs - now
                self.get_logger().info(
                    f">>> BRAKE ENGAGED  (motion_enable -> False) "
                    f"at t={now:.2f}, hold for {hold:.2f}s")
        self.flag_value = new_flag

    # ------------------------------------------------------------------ #
    # 2026-05-26: helpers for the rich detection logic                   #
    # ------------------------------------------------------------------ #
    def _bbox_fully_in_frame(self, x, y, w, h, margin, allow_top_clip=False):
        """True if the bbox (top-left x,y, width w, height h) is fully
        inside the image with at least `margin` pixels of slack on every
        side. Used to gate traffic-light stops on the full-object
        visibility requirement (user, 2026-05-26).

        2026-05-26 PM-4: added `allow_top_clip`. For overhead traffic
        lights the bracket naturally extends above the frame as we
        approach (camera tilts up, TL sweeps to top of image). The
        full-visibility rule was designed against half-visible signs
        ENTERING from the side -- top clipping doesn't carry the same
        risk. Skip the top-edge check when allow_top_clip=True."""
        if x < margin: return False
        if (x + w) > (self._img_w - margin): return False
        if (y + h) > (self._img_h - margin): return False
        if (not allow_top_clip) and y < margin: return False
        return True

    def _center_patch_depth_m(self, depth, x, y, w, h, frac=0.2):
        """Robust depth at the *center* of a detection bbox, computed as
        the median of valid depths in a small central patch (frac of bbox
        side length). Returned as a float (meters) or NaN if no valid
        pixels.

        This is a SECOND OPINION on PIT's mask-median distance. We log
        both per detection so the user can see the bias direction on
        the physical car (see the 2026-05-26 changelog entry)."""
        if depth is None:
            return float("nan")
        H, W = depth.shape[:2]
        pw = max(2, int(w * frac))
        ph = max(2, int(h * frac))
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        x0 = max(0, cx - pw // 2); x1 = min(W, x0 + pw)
        y0 = max(0, cy - ph // 2); y1 = min(H, y0 + ph)
        patch = depth[y0:y1, x0:x1]
        valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 10.0)]
        if valid.size == 0:
            return float("nan")
        return float(np.median(valid))

    # ------------------------------------------------------------------ #
    # 2026-05-26 (PM): backend dispatch + custom-model helpers           #
    # ------------------------------------------------------------------ #
    def _load_model(self, image_width, image_height):
        """Choose YOLO backend based on (model_type, model_path) params
        and load it. Sets:
            self._is_ultralytics  — True for custom ultralytics model,
                                    False for Quanser PIT seg model.
            self.myYolo           — model instance (wrapper varies).
            self._ult_names       — dict[int, str] of class names, only
                                    populated for ultralytics path.
        """
        model_type = str(self.get_parameter("model_type").value).strip().lower()
        model_path = str(self.get_parameter("model_path").value).strip()

        if model_type == "auto":
            model_type = "ultralytics" if model_path else "quanser_seg"

        if model_type == "quanser_seg":
            # Deferred PIT import — only fail-load if this path is actually
            # selected (lets the custom-model path run on machines where
            # the PIT python package isn't present, e.g. fresh containers).
            from pit.YOLO.nets import YOLOv8
            self.myYolo = YOLOv8(
                modelPath=(model_path or
                           "./ros2/src/qcar2_autonomy/models/quanser_yolov8s-seg.pt"),
                imageHeight=image_height,
                imageWidth=image_width,
                convert_tensorrt=False,
            )
            self._is_ultralytics = False
            self._ult_names = None
            self.get_logger().info(
                f"yolo_detector: backend=quanser_seg (PIT YOLOv8 wrapper, "
                f"seg masks + lightColor available)")
        elif model_type == "ultralytics":
            if not model_path:
                raise RuntimeError(
                    "model_type=ultralytics requires model_path to be set "
                    "to a .pt file path.")
            from ultralytics import YOLO
            self.myYolo = YOLO(model_path)
            self._is_ultralytics = True
            self._ult_names = dict(self.myYolo.names)
            self.get_logger().info(
                f"yolo_detector: backend=ultralytics (path={model_path})")
            self.get_logger().info(
                f"  classes: {self._ult_names}")
        else:
            raise RuntimeError(
                f"Unknown model_type='{model_type}'. "
                f"Expected one of: auto, quanser_seg, ultralytics.")

    def _check_traffic_light_color(self, bgr, x1, y1, x2, y2,
                                   min_v_abs=90, min_s_abs=70):
        """Pure-numpy port of pit.YOLO.nets.YOLOv8.check_traffic_light,
        with stricter absolute thresholds (2026-05-26 PM-6).

        Used by the ultralytics backend, which detects traffic lights as
        a single class with no built-in color awareness. Three vertical
        circular patches in the bbox (top=red, mid=yellow, bottom=green)
        — a patch is considered "lit" when:
          1. Its mean V (brightness) is above `min_v_abs` AND its mean
             S (saturation) is above `min_s_abs` — absolute floors that
             reject reflections (high V, low S) and shadows (low V).
          2. Its V is above the cross-patch mean by at least 25% of the
             V dynamic range across the three patches — PIT's original
             relative comparison.

        Returns a lowercase string like 'red', 'yellow', 'green',
        'red yellow', or 'idle' if no patch is meaningfully lit.
        Backwards-compat with PIT's name format."""
        H, W = bgr.shape[:2]
        x1 = max(0, int(x1)); x2 = min(W, int(x2))
        y1 = max(0, int(y1)); y2 = min(H, int(y2))
        if x2 - x1 < 3 or y2 - y1 < 3:
            return "idle"

        d = max(2, int(0.3 * (x2 - x1)))     # patch diameter (PIT formula)
        cx = (x1 + x2) // 2
        # Three vertical patch centers
        rR = (cx, (3 * y1 + y2) // 4)         # top quarter -> red
        rY = (cx, (y1 + y2) // 2)             # middle      -> yellow
        rG = (cx, (y1 + 3 * y2) // 4)         # bottom qtr  -> green

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]                      # saturation -- real color vs gray
        v = hsv[:, :, 2]                      # value      -- brightness

        def _patch_means(center):
            """Returns (mean_v, mean_s) over the circular patch, or
            (0.0, 0.0) if the patch is empty."""
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.circle(mask, center, max(1, d // 2), 1, -1)
            sel_v = v[mask > 0]
            sel_s = s[mask > 0]
            if sel_v.size == 0:
                return 0.0, 0.0
            return float(sel_v.mean()), float(sel_s.mean())

        vR, sR = _patch_means(rR)
        vY, sY = _patch_means(rY)
        vG, sG = _patch_means(rG)

        mean_v = (vR + vY + vG) / 3.0
        max_v  = max(vR, vY, vG)
        min_v  = min(vR, vY, vG)
        if (max_v - min_v) < 30:               # PIT's "all roughly equal" floor
            return "idle"
        rel_thresh = (max_v - min_v) * 0.25

        def _is_lit(v_patch, s_patch):
            # 1) absolute brightness floor   2) absolute saturation floor
            # 3) brighter than the mean by the relative threshold
            return (v_patch >= min_v_abs
                    and s_patch >= min_s_abs
                    and v_patch > mean_v
                    and (v_patch - mean_v) > rel_thresh)

        colors = []
        if _is_lit(vR, sR): colors.append("red")
        if _is_lit(vY, sY): colors.append("yellow")
        if _is_lit(vG, sG): colors.append("green")
        return " ".join(colors) if colors else "idle"

    class _SignApproachTracker:
        """Per-sign-type approach tracker. Builds depth-vs-time history
        across consecutive detections, fits a linear slope to estimate
        approach speed (m/s), and predicts the absolute wall-clock time
        when the car will be `target_offset_m` away from the sign face
        (i.e. "right beside it"). Commits to the prediction once the
        sign is close (depth < commit_at_m) so the brake still fires
        even if the sign drops out of the camera FOV in the last few
        meters (which always happens — D435 has ~70 deg HFOV, sign
        leaves the frame ~1-2 m before we reach it).

        Lifecycle per sign instance:
          update(t, d)             — feed each fresh detection
          should_brake(t)          — True once now >= target_arrival
          on_brake_fired()         — clear state for the NEXT sign

        Drop logic: if more than 1.0 s passes without an update, we
        assume the sign is gone. If we hadn't committed yet, reset
        the history (it was probably a misdetection). If we HAD
        committed, keep the prediction alive — that's the point.
        """
        STALE_S = 1.0

        def __init__(self, logger, label):
            self._log = logger
            self._label = label
            self._history = []                    # list[(t_abs, depth_m)]
            self._target_arrival_abs = None
            self._committed_at = None             # debug: when did we commit
            self._last_update_t = None

        def update(self, t_abs, depth_m, bbox_center_x, img_w, *,
                   target_offset_m, min_samples, commit_at_m, min_speed,
                   max_depth_m, lateral_edge_frac):
            """Record a new detection and possibly commit a brake-time
            prediction. Returns the current target_arrival_abs (or None
            if not yet committed).

            Two independent commit triggers, in priority order:

            1) LATERAL EDGE (primary, ALWAYS WINS). When bbox center x
               has crossed into the outer `lateral_edge_frac` of the
               frame, we are physically passing the sign — brake
               immediately (target_arrival = now). **Overrides** an
               earlier depth-rate commit if the depth-rate target is
               in the future (it almost always is for side-of-road
               signs because depth_rate is small).

            2) DEPTH-RATE (fallback). Fit slope of depth-vs-time, derive
               approach_speed, predict t_arrival when depth reaches
               target_offset_m. Commit once depth < commit_at_m. This
               trigger covers head-on approaches and occluded signs that
               disappear before reaching the lateral edge."""
            now = t_abs

            # Stale check — if we never committed and history is old,
            # wipe it (probably a flicker on something else).
            if self._last_update_t is not None \
                    and (now - self._last_update_t) > self.STALE_S \
                    and self._target_arrival_abs is None:
                self._history.clear()

            self._last_update_t = now

            # ---- Trigger 1: LATERAL EDGE — ALWAYS CHECKED ----------------
            # Run BEFORE the "already committed" early-return so a stale
            # depth-rate commit (which can land in the future) gets
            # overridden the moment we're physically passing the sign.
            edge_px = float(img_w) * float(lateral_edge_frac)
            if bbox_center_x < edge_px or bbox_center_x > (img_w - edge_px):
                # Override iff lateral-edge target (now) is sooner than
                # any prior commit. For lateral_edge the target is `now`
                # so this overrides any future-target commit.
                if (self._target_arrival_abs is None
                        or now < self._target_arrival_abs):
                    prior = self._target_arrival_abs
                    self._target_arrival_abs = float(now)
                    self._committed_at = now
                    if prior is not None:
                        self._log.info(
                            f"[predict] {self._label} COMMIT (lateral edge) "
                            f"OVERRIDES prior commit (was brake_in="
                            f"{prior - now:.2f}s): bbox_x={bbox_center_x:.0f}px "
                            f"depth={depth_m:.2f}m -> BRAKE NOW"
                        )
                    else:
                        self._log.info(
                            f"[predict] {self._label} COMMIT (lateral edge): "
                            f"bbox_x={bbox_center_x:.0f}px depth={depth_m:.2f}m "
                            f"-> BRAKE NOW"
                        )
                return self._target_arrival_abs

            # Already depth-rate committed — keep that commit, don't refit
            if self._target_arrival_abs is not None:
                return self._target_arrival_abs

            # ---- Trigger 2: DEPTH-RATE fallback --------------------------
            # Ignore obviously-out-of-range readings for the fit
            if depth_m <= 0.05 or depth_m > max_depth_m:
                return None

            self._history.append((now, depth_m, bbox_center_x))
            if len(self._history) > 8:
                self._history.pop(0)

            if len(self._history) < min_samples:
                return None

            # Linear fit: d(t) = slope*t + intercept ; approaching => slope<0
            ts = np.array([t for t, _, _ in self._history], dtype=np.float64)
            ds = np.array([d for _, d, _ in self._history], dtype=np.float64)
            slope, intercept = np.polyfit(ts, ds, 1)
            approach_speed = -slope                       # m/s (positive = closing)

            if approach_speed < min_speed:
                # Not really approaching (could be drifting, stationary,
                # or noisy). Don't commit yet.
                return None

            # Predict time when depth = target_offset_m :
            #   target_offset = slope*t + intercept
            #   t = (target_offset - intercept) / slope = (intercept - target_offset) / approach_speed
            t_target = (intercept - target_offset_m) / approach_speed

            # Only commit when we're close enough that the fit is trustworthy.
            if depth_m <= commit_at_m:
                self._target_arrival_abs = float(t_target)
                self._committed_at = now
                dt_to_brake = self._target_arrival_abs - now
                self._log.info(
                    f"[predict] {self._label} COMMIT (depth-rate): depth={depth_m:.2f}m "
                    f"approach={approach_speed:.2f}m/s "
                    f"brake_in={dt_to_brake:.2f}s "
                    f"(target offset={target_offset_m:.2f}m, {len(self._history)} samples). "
                    f"NOTE: depth-rate commits often land too far in the future for "
                    f"side-of-road signs; lateral-edge will override when bbox reaches "
                    f"frame edge."
                )
            return self._target_arrival_abs

        def should_brake(self, t_abs):
            return (self._target_arrival_abs is not None
                    and t_abs >= self._target_arrival_abs)

        def on_brake_fired(self):
            """Call after the brake actually engages. Clears state so the
            NEXT sign of this type can be tracked from scratch."""
            self._history.clear()
            self._target_arrival_abs = None
            self._committed_at = None
            self._last_update_t = None

    class _TLStateMachine:
        """Traffic-light approach state machine (2026-05-26 PM-8, kept
        after PM-9 was reverted at user's explicit request: "going on a
        red is better than staying in the middle of the intersection").

        Encodes commit-and-hold semantics:

          IDLE
            * first sighting + bbox_h > pass_line          -> COMMIT_GO
            * first sighting + color in {red, yellow}      -> COMMIT_STOP (brake)
            * first sighting + color in {green, idle}      -> COMMIT_GO

          COMMIT_STOP   (brake refreshed every tick)
            * effective color = green for K frames         -> COMMIT_GO (release)
            * otherwise                                    -> stay

          COMMIT_GO     (no brake, color-change-IMMUNE)
            * anything                                     -> stay

          any non-IDLE + TL not seen for M frames          -> IDLE (reset)

        Critical invariant: COMMIT_GO ignores ALL subsequent color
        changes. Once we've committed to drive (because we first saw
        green, or because we passed the line, or because we released
        from a sustained-green wait), no late yellow or red can
        re-engage the brake. This trades correctness ("don't run a
        red light") for the more important safety property ("don't
        stop in the middle of the intersection") -- a deliberate user
        decision (Turn 150).

        Returns one of:
          'brake'   refresh brake this tick
          'release' just transitioned, hard-release brake
          None      no action"""

        IDLE        = "IDLE"
        COMMIT_STOP = "COMMIT_STOP"
        COMMIT_GO   = "COMMIT_GO"

        def __init__(self, logger):
            self._log = logger
            self._state = self.IDLE
            self._lost_count = 0
            self._green_count = 0
            self._enter_time = 0.0

        @property
        def state(self):
            return self._state

        def update(self, t_abs, present, eff_color, bbox_h, *,
                   pass_line_height_px, lost_frames_to_reset,
                   green_frames_to_release):
            if not present:
                self._lost_count += 1
                if self._state != self.IDLE \
                        and self._lost_count >= lost_frames_to_reset:
                    self._enter(self.IDLE, t_abs,
                        f"lost for {self._lost_count} frames -> reset")
                return None

            # TL is present this tick
            self._lost_count = 0
            is_stop_color = eff_color in ("red", "yellow")
            past_line     = bbox_h > pass_line_height_px

            if self._state == self.IDLE:
                if past_line or not is_stop_color:
                    self._enter(self.COMMIT_GO, t_abs,
                        f"first sight {'past line ' if past_line else ''}"
                        f"color={eff_color} bh={bbox_h} -> GO")
                    return None
                else:
                    self._enter(self.COMMIT_STOP, t_abs,
                        f"first sight red/yellow at bh={bbox_h} -> STOP")
                    return "brake"

            if self._state == self.COMMIT_STOP:
                if eff_color == "green":
                    self._green_count += 1
                    if self._green_count >= green_frames_to_release:
                        self._enter(self.COMMIT_GO, t_abs,
                            f"sustained green ({self._green_count} frames) "
                            f"-> GO (release)")
                        return "release"
                    return "brake"
                else:
                    self._green_count = 0
                    return "brake"   # refresh

            # state == COMMIT_GO  -- ignore all color changes
            return None

        def _enter(self, new_state, t, reason):
            if new_state != self._state:
                self._log.info(
                    f"[TL-FSM] {self._state} -> {new_state}: {reason}")
            self._state = new_state
            self._enter_time = t
            if new_state == self.IDLE:
                self._green_count = 0
                self._lost_count = 0

    class _Detection:
        """Backend-agnostic detection record. Mirrors the .__dict__ shape
        of PIT's Obstacle/TrafficLight so the existing per-detection loop
        in yolo_detect() can consume either backend's output unchanged.
        For the ultralytics path, `distance` stays -1.0 (we use the
        center-patch depth instead, which is the gating default anyway)
        and `lightColor` is filled in by _check_traffic_light_color()
        for TL detections."""
        def __init__(self, name, conf, x, y, distance=-1.0, lightColor=""):
            self.name = name
            self.conf = float(conf)
            self.x = float(x)
            self.y = float(y)
            self.distance = float(distance)
            self.lightColor = lightColor

    # Map from the v3 ultralytics model's class names to the per-class
    # naming convention the existing detection loop expects. Custom-model
    # classes that don't map to a Quanser equivalent (crosswalk,
    # speedlimit) pass through unchanged and are handled by the new
    # elif branches in the loop.
    _ULT_NAME_MAP = {
        "stop":         "stop sign",
        "trafficlight": "traffic light",
        # crosswalk + speedlimit pass through as-is
    }
    # ------------------------------------------------------------------ #

    def yolo_detect(self):
        """Run YOLO on the latest buffered frame. Updates
        self.brake_until_abs based on what's currently in view:
          * Traffic light red/yellow + valid range + fully visible:
            refresh brake_until_abs = max(brake_until_abs, now+tl_hold).
            Refreshes every tick so the brake stays continuously engaged
            while the TL is red/yellow.
          * Stop sign / yield sign: latched one-shot. Sets brake_until_abs
            once, then enforces sign_cooldown_until_abs before re-firing.
          * Green/idle TL or other detections: no-op."""

        # --- 2026-05-14: feed YOLO from the buffered frames instead of
        #     directly from QCar2DepthAligned. Equivalent data (aligned
        #     RGB + 32FC1 depth in meters) sourced from the bridge.
        # Original (commented):
        #     rgbProcessed = self.myYolo.pre_process(self.QCarImg.rgb)
        #     ...post_processing(alignedDepth=self.QCarImg.depth, ...)
        rgb_for_yolo = getattr(self, "_current_rgb", None)
        depth_for_yolo = getattr(self, "_current_depth", None)
        if rgb_for_yolo is None or depth_for_yolo is None:
            return

        # 2026-05-26 (PM): dispatch on backend. Both branches must end
        # up with `processedResults` (list of objects exposing .name,
        # .conf, .x, .y, .distance, .lightColor via __dict__) and
        # `xyxy` (Nx4 numpy array of [x1,y1,x2,y2]) so the unified
        # per-detection loop below is backend-agnostic.
        if self._is_ultralytics:
            processedResults, xyxy = self._run_ultralytics(rgb_for_yolo)
        else:
            processedResults, xyxy = self._run_quanser_seg(rgb_for_yolo,
                                                            depth_for_yolo)

        # Cache param reads once per tick (cheap but tidy)
        p = self.get_parameter
        stop_conf      = float(p("stop_sign_conf").value)
        stop_hold      = float(p("stop_sign_hold_s").value)
        yield_conf     = float(p("yield_sign_conf").value)
        yield_hold     = float(p("yield_sign_hold_s").value)
        tl_conf        = float(p("tl_conf").value)
        tl_min_dist    = float(p("tl_min_dist_m").value)
        tl_stop_dist   = float(p("tl_stop_dist_m").value)
        tl_min_height  = int(p("tl_min_height_px").value)
        tl_hold        = float(p("tl_hold_s").value)
        tl_margin      = int(p("tl_edge_margin_px").value)
        tl_allow_top_clip = bool(p("tl_allow_top_clip").value)
        # 2026-05-26 PM-6: temporal stabilization + pass-line params
        tl_history_size       = int(p("tl_color_history_size").value)
        tl_pass_line_height   = int(p("tl_pass_line_height_px").value)
        # 2026-05-26 PM-8: FSM params
        tl_fsm_lost_reset     = int(p("tl_fsm_lost_frames_to_reset").value)
        tl_fsm_green_release  = int(p("tl_fsm_green_frames_to_release").value)
        cooldown_def   = float(p("detection_cooldown_s").value)
        dsource        = str(p("distance_source").value).strip().lower()
        # 2026-05-26 PM-2: predictive-approach params (used by the
        # _SignApproachTracker for stop / yield signs)
        stop_target_offset_m     = float(p("stop_target_offset_m").value)
        stop_predict_min_samples = int(p("stop_predict_min_samples").value)
        stop_predict_max_depth   = float(p("stop_predict_max_depth_m").value)
        stop_predict_commit_at_m = float(p("stop_predict_commit_at_m").value)
        stop_predict_min_speed   = float(p("stop_predict_min_speed").value)
        # 2026-05-26 PM-3: lateral-edge commit (primary stop-sign trigger)
        lateral_edge_frac        = float(p("lateral_edge_frac").value)

        now = time.time()

        # `xyxy` came from the backend dispatch above (uniform Nx4 numpy
        # array of [x1,y1,x2,y2] for both Quanser-seg and ultralytics
        # paths).

        # 2026-05-26 PM-8: TL FSM input collection. Across all
        # detections this tick we pick the MOST PROMINENT (largest
        # bbox_h) TL that passes visibility + confidence and feed it
        # to the FSM after the loop. If no TL passes, we feed the FSM
        # an empty signal (present=False) so it can count toward the
        # "lost frames to reset" threshold.
        tl_best_h        = 0.0
        tl_best_eff_clr  = "idle"
        tl_best_inst_clr = ""
        tl_best_used_d   = float('nan')
        tl_best_bx       = 0.0
        tl_best_by       = 0.0
        tl_best_bw       = 0.0
        tl_best_present  = False

        for i, object in enumerate(processedResults):
            labelName = object.__dict__.get("name", "")
            labelConf = float(object.__dict__.get("conf", 0.0))
            pit_d = float(object.__dict__.get("distance", -1.0))

            # bbox geometry for this detection
            if xyxy is not None and i < len(xyxy):
                x1, y1, x2, y2 = xyxy[i].tolist()
                bx, by = float(x1), float(y1)
                bw, bh = float(x2 - x1), float(y2 - y1)
            else:
                bx = float(object.__dict__.get("x", 0))
                by = float(object.__dict__.get("y", 0))
                bw = bh = 0.0

            # Two distance estimates per detection. We GATE on `used_d`
            # (chosen by the distance_source param) and LOG the other
            # one as a second opinion. center_patch is the default and
            # is what the user asked for ("focus on 50% center of the
            # stop sign" — Luigi's approach generalized).
            center_d = self._center_patch_depth_m(depth_for_yolo, bx, by, bw, bh)
            if dsource == "pit_median":
                used_d, other_d, other_label = pit_d, center_d, "centerD"
            else:
                used_d, other_d, other_label = center_d, pit_d, "PITdist"

            # NaN-safe range checks below assume used_d is finite. If the
            # center patch had no valid depth (occlusion, sky), fall back
            # to PIT's median so we never silently fail to detect.
            if not np.isfinite(used_d) or used_d <= 0.0:
                used_d = pit_d

            self.get_logger().info(
                f"[YOLO] {labelName} conf={labelConf:.2f} "
                f"used={used_d:.3f}m({dsource}) {other_label}={other_d:.3f}m "
                f"bbox=({bx:.0f},{by:.0f},{bw:.0f}x{bh:.0f})"
            )

            # -----------------------------------------------------------
            # TRAFFIC LIGHT — collect only. The actual brake decision
            # is made by the _TLStateMachine after this per-detection
            # loop completes (2026-05-26 PM-8). We extract instantaneous
            # color, update the temporal-vote history, gate on
            # confidence + visibility + close-enough, and remember the
            # MOST PROMINENT (largest bbox_h) TL of the tick. The FSM
            # gets one (eff_color, bbox_h) per tick.
            # -----------------------------------------------------------
            if str(labelName).startswith("traffic light"):
                inst_color = str(object.__dict__.get("lightColor", "")).strip().lower()

                # Temporal stabilization (majority vote, N frames)
                self._tl_color_history.append((now, inst_color))
                while len(self._tl_color_history) > tl_history_size:
                    self._tl_color_history.pop(0)
                bucket_counts = {"red": 0, "yellow": 0, "green": 0, "idle": 0}
                for _t, _c in self._tl_color_history:
                    if "red" in _c:    bucket_counts["red"]    += 1
                    elif "yellow" in _c: bucket_counts["yellow"] += 1
                    elif "green" in _c:  bucket_counts["green"]  += 1
                    else:                bucket_counts["idle"]   += 1
                priority = ["red", "yellow", "green", "idle"]
                eff_color = max(priority,
                                key=lambda c: (bucket_counts[c],
                                               -priority.index(c)))

                depth_ok = (np.isfinite(used_d)
                            and used_d > tl_min_dist
                            and used_d < tl_stop_dist)
                height_ok = (bh >= tl_min_height)
                close_enough = depth_ok or height_ok
                full_visible = self._bbox_fully_in_frame(
                    bx, by, bw, bh, tl_margin,
                    allow_top_clip=tl_allow_top_clip,
                )

                detection_usable = ((labelConf >= tl_conf)
                                    and close_enough
                                    and full_visible)

                if detection_usable and bh > tl_best_h:
                    # New "most prominent" TL for this tick.
                    tl_best_h        = bh
                    tl_best_eff_clr  = eff_color
                    tl_best_inst_clr = inst_color
                    tl_best_used_d   = used_d
                    tl_best_bx       = bx
                    tl_best_by       = by
                    tl_best_bw       = bw
                    tl_best_present  = True
                    # Stash the bucket string for the FSM-driver log
                    self._tl_last_bucket_str = (
                        f"r{bucket_counts['red']}/y{bucket_counts['yellow']}/"
                        f"g{bucket_counts['green']}/i{bucket_counts['idle']}"
                    )
                elif not detection_usable and (labelConf >= tl_conf):
                    # TL detected but failed visibility / proximity --
                    # log WHY so the user can tune. Only logged when
                    # nothing else picked this TL up as usable.
                    reasons = []
                    if not full_visible:
                        if bx < tl_margin: reasons.append(f"left x={bx:.0f}")
                        if (bx + bw) > (self._img_w - tl_margin):
                            reasons.append(f"right x+w={bx+bw:.0f}")
                        if (by + bh) > (self._img_h - tl_margin):
                            reasons.append(f"bottom y+h={by+bh:.0f}")
                        if (not tl_allow_top_clip) and by < tl_margin:
                            reasons.append(f"top y={by:.0f}")
                    if not close_enough:
                        reasons.append(
                            f"too far (depth={used_d:.2f}m, h={bh:.0f}px)")
                    self.get_logger().info(
                        f"TL {eff_color.upper()} (inst={inst_color}) "
                        f"h={bh:.0f}px: {', '.join(reasons) or 'rejected'} "
                        f"-- not feeding FSM")

            # -----------------------------------------------------------
            # STOP SIGN — PREDICTIVE "stop right beside the sign".
            # User request 2026-05-26 PM: don't gate on "depth < X" because
            # the car blows past before brake propagation completes. Instead,
            # track depth-vs-time on the approach, fit a slope, predict the
            # absolute time when we'll be `stop_target_offset_m` away from
            # the sign face, then brake at that time even if the sign leaves
            # the FOV in the last 1-2 m. See _SignApproachTracker.
            # -----------------------------------------------------------
            elif labelName == "stop sign":
                if labelConf < stop_conf:
                    self.get_logger().info(
                        f"  stop sign conf={labelConf:.2f} < {stop_conf:.2f} -- not gating")
                elif not (0.0 < used_d <= stop_predict_max_depth):
                    self.get_logger().info(
                        f"  stop sign depth={used_d:.2f}m out of [0, {stop_predict_max_depth:.1f}] "
                        f"-- not gating")
                elif now < self.sign_cooldown_until_abs:
                    self.get_logger().info(
                        f"  stop sign within cooldown ({self.sign_cooldown_until_abs - now:.1f}s "
                        f"remaining) -- not gating")
                else:
                    # Feed the tracker. Returns committed brake-arrival time
                    # (None if still building history / not approaching).
                    bbox_cx = bx + bw / 2.0
                    target = self._stop_tracker.update(
                        now, used_d, bbox_cx, self._img_w,
                        target_offset_m   = stop_target_offset_m,
                        min_samples       = stop_predict_min_samples,
                        commit_at_m       = stop_predict_commit_at_m,
                        min_speed         = stop_predict_min_speed,
                        max_depth_m       = stop_predict_max_depth,
                        lateral_edge_frac = lateral_edge_frac,
                    )
                    if target is None:
                        self.get_logger().info(
                            f"  stop sign @ {used_d:.2f}m -- tracking (no commit yet, "
                            f"{len(self._stop_tracker._history)} samples)")
                    elif self._stop_tracker.should_brake(now):
                        # Fire the brake.
                        self.brake_until_abs = max(self.brake_until_abs, now + stop_hold)
                        self.sign_cooldown_until_abs = now + cooldown_def
                        self.stop_override_active = True
                        self.stop_override_until = max(self.stop_override_until,
                                                       now + stop_hold)
                        msg = UInt8(); msg.data = 1
                        self.qcar_state_pub.publish(msg)
                        self.get_logger().info(
                            f"Stop Sign -> BRAKE NOW (predicted arrival, "
                            f"depth={used_d:.2f}m, hold {stop_hold:.1f}s, "
                            f"cooldown {cooldown_def:.0f}s)"
                        )
                        self._stop_tracker.on_brake_fired()
                    else:
                        dt = target - now
                        self.get_logger().info(
                            f"  stop sign @ {used_d:.2f}m committed, brake in {dt:.2f}s")

            # -----------------------------------------------------------
            # YIELD SIGN — same predictive pattern (when the v4 model
            # adds yield back). Currently dead-code on the v3 ultralytics
            # backend (no 'yield sign' class) and live on Quanser-seg.
            # -----------------------------------------------------------
            elif labelName == "yield sign":
                if labelConf < yield_conf or not (0.0 < used_d <= stop_predict_max_depth):
                    self.get_logger().info(
                        f"  yield sign conf={labelConf:.2f} dist={used_d:.2f}m -- not gating")
                elif now < self.sign_cooldown_until_abs:
                    pass
                else:
                    bbox_cx = bx + bw / 2.0
                    target = self._yield_tracker.update(
                        now, used_d, bbox_cx, self._img_w,
                        target_offset_m   = stop_target_offset_m,
                        min_samples       = stop_predict_min_samples,
                        commit_at_m       = stop_predict_commit_at_m,
                        min_speed         = stop_predict_min_speed,
                        max_depth_m       = stop_predict_max_depth,
                        lateral_edge_frac = lateral_edge_frac,
                    )
                    if target is not None and self._yield_tracker.should_brake(now):
                        # Yield is a SLOW, not a full stop — keep the shorter hold.
                        self.brake_until_abs = max(self.brake_until_abs, now + yield_hold)
                        self.sign_cooldown_until_abs = now + cooldown_def
                        self.get_logger().info(
                            f"Yield Sign -> SLOW NOW (predicted arrival, "
                            f"depth={used_d:.2f}m, hold {yield_hold:.1f}s)")
                        self._yield_tracker.on_brake_fired()

            # -----------------------------------------------------------
            # CAR (informational only — gating is done by other nodes).
            # NOTE the user wanted 'car' dropped from the v3 ultralytics
            # model because the RealSense was misclassifying the front
            # CSI bumper as a car, so this elif is dead-code on the
            # ultralytics backend (model has no 'car' class). Kept for
            # the Quanser-seg backend, which still emits class 2.
            # -----------------------------------------------------------
            elif labelName == 'car' and labelConf > 0.9 and 0.0 < used_d < 0.45:
                self.get_logger().info(f"Car found @ {used_d:.2f}m")

            # -----------------------------------------------------------
            # CROSSWALK — v3 ultralytics class. Informational for now;
            # the "stop before crosswalk if TL red" interaction the user
            # described (Turn 138) is a future enhancement that will
            # condition on simultaneous TL+crosswalk detection.
            # -----------------------------------------------------------
            elif labelName == "crosswalk" and labelConf >= 0.5:
                self.get_logger().info(
                    f"Crosswalk @ {used_d:.2f}m conf={labelConf:.2f} "
                    f"(no auto-stop)")

            # -----------------------------------------------------------
            # SPEED LIMIT — v3 ultralytics class. User explicitly asked
            # for classify-only, no speed change. Just log.
            # -----------------------------------------------------------
            elif labelName == "speedlimit" and labelConf >= 0.5:
                self.get_logger().info(
                    f"Speed Limit sign @ {used_d:.2f}m conf={labelConf:.2f} "
                    f"(classify-only)")

        # 2026-05-26 PM-8: drive the TL state machine ONCE per tick with
        # the most prominent TL we collected (or "not present" if no
        # usable TL this tick). The FSM decides brake/release/no-op
        # based on first-sighting color + held commitment.
        fsm_action = self._tl_fsm.update(
            now,
            present=tl_best_present,
            eff_color=tl_best_eff_clr,
            bbox_h=tl_best_h,
            pass_line_height_px     = tl_pass_line_height,
            lost_frames_to_reset    = tl_fsm_lost_reset,
            green_frames_to_release = tl_fsm_green_release,
        )
        if fsm_action == "brake":
            self.brake_until_abs = max(self.brake_until_abs, now + tl_hold)
            self.stop_override_active = True
            self.stop_override_until = max(
                self.stop_override_until, now + tl_hold)
            msg = UInt8(); msg.data = 1
            self.qcar_state_pub.publish(msg)
            self.get_logger().info(
                f"Traffic Light {tl_best_eff_clr.upper()} "
                f"(inst={tl_best_inst_clr}, buckets {self._tl_last_bucket_str}) "
                f"@ depth={tl_best_used_d:.2f}m h={tl_best_h:.0f}px "
                f"[{self._tl_fsm.state}] -> STOP (brake +{tl_hold:.2f}s)"
            )
        elif fsm_action == "release":
            # Hard release: pull brake_until_abs back to now so the
            # outer on_timer sees new_flag=True immediately.
            self.brake_until_abs = min(self.brake_until_abs, now)
            self.get_logger().info(
                f"Traffic Light -> RELEASE (FSM committed to GO; "
                f"brake_until_abs pulled to now)"
            )
        # action None -> no-op this tick

        # 2026-05-26 PM-5: poll ARMED trackers after the per-detection
        # loop. Critical for the case where a sign committed a future
        # brake-arrival time (depth-rate fallback) and then EXITED the
        # camera FOV before that time arrived — without this poll the
        # brake never fires because should_brake() is only checked
        # inside the per-detection elif which never runs again. Now the
        # poll runs every tick regardless of detections.
        poll_now = time.time()
        for tracker, hold, label in (
            (self._stop_tracker,  stop_hold,  "Stop Sign"),
            (self._yield_tracker, yield_hold, "Yield Sign"),
        ):
            if tracker.should_brake(poll_now) \
                    and poll_now >= self.sign_cooldown_until_abs:
                self.brake_until_abs = max(
                    self.brake_until_abs, poll_now + hold)
                self.sign_cooldown_until_abs = poll_now + cooldown_def
                if label == "Stop Sign":
                    self.stop_override_active = True
                    self.stop_override_until = max(
                        self.stop_override_until, poll_now + hold)
                    msg = UInt8(); msg.data = 1
                    self.qcar_state_pub.publish(msg)
                self.get_logger().info(
                    f"{label} -> BRAKE NOW (armed-tracker poll; sign may be "
                    f"out of FOV) hold {hold:.2f}s cooldown {cooldown_def:.0f}s"
                )
                tracker.on_brake_fired()

    # ------------------------------------------------------------------ #
    # Backend implementations — invoked by yolo_detect()                 #
    # ------------------------------------------------------------------ #
    def _run_quanser_seg(self, rgb, depth):
        """Quanser PIT YOLOv8 seg backend. Returns (processedResults, xyxy).
        Preserves the pre-2026-05-26-PM behavior exactly.
        processedResults: list[Obstacle|TrafficLight] (PIT objects).
        xyxy: numpy Nx4 of bbox corners, or None if extraction fails."""
        rgbProcessed = self.myYolo.pre_process(rgb)
        prediction = self.myYolo.predict(
            inputImg   = rgbProcessed,
            classes    = [2, 9, 11, 33],   # car, TL, stop, yield
            confidence = 0.3,
            half       = True,
            verbose    = False,
        )
        # Annotated overlay
        try:
            ann = None
            if hasattr(prediction, "plot"):
                ann = prediction.plot()
            elif isinstance(prediction, (list, tuple)) and len(prediction) > 0 \
                    and hasattr(prediction[0], "plot"):
                ann = prediction[0].plot()
            if ann is not None and isinstance(ann, np.ndarray) and ann.size:
                self.publish_rgb_yolo.publish(self.bridge.cv2_to_imgmsg(ann, "bgr8"))
        except Exception as exc:
            self.get_logger().warn(
                f"YOLO overlay publish failed: {exc}", throttle_duration_sec=5.0)

        processedResults = self.myYolo.post_processing(
            alignedDepth     = depth,
            clippingDistance = 5,
        )
        try:
            xyxy = prediction.boxes.xyxy.cpu().numpy()
        except Exception:
            xyxy = None
        return processedResults, xyxy

    def _run_ultralytics(self, rgb):
        """Custom ultralytics detection backend. Returns (processedResults,
        xyxy) in the same shape as the Quanser path.
        processedResults: list[_Detection] (our backend-agnostic record).
        xyxy: numpy Nx4 of bbox corners.

        Notes:
          * Detection (not seg) model: no segmentation masks => no PIT-
            style mask-median distance; gating uses center_patch (already
            the default).
          * Traffic-light color: computed in-process via
            _check_traffic_light_color (HSV brightness, ported from PIT).
          * Class-name normalization via _ULT_NAME_MAP so the existing
            per-detection loop (which checks 'stop sign', 'traffic light')
            keeps working unchanged."""
        results = self.myYolo.predict(
            source  = rgb,
            conf    = 0.30,
            verbose = False,
            classes = None,    # let the model emit all classes; loop filters
        )
        if not results:
            return [], np.zeros((0, 4))
        r0 = results[0]

        # Annotated overlay (boxes + labels + scores)
        try:
            ann = r0.plot()
            if isinstance(ann, np.ndarray) and ann.size:
                self.publish_rgb_yolo.publish(self.bridge.cv2_to_imgmsg(ann, "bgr8"))
        except Exception as exc:
            self.get_logger().warn(
                f"YOLO overlay publish failed: {exc}", throttle_duration_sec=5.0)

        if r0.boxes is None or len(r0.boxes) == 0:
            return [], np.zeros((0, 4))

        cls_arr  = r0.boxes.cls.cpu().numpy().astype(int)
        conf_arr = r0.boxes.conf.cpu().numpy()
        xyxy_arr = r0.boxes.xyxy.cpu().numpy()

        # Cache HSV color thresholds once
        tl_min_v = int(self.get_parameter("tl_color_min_v").value)
        tl_min_s = int(self.get_parameter("tl_color_min_s").value)

        detections = []
        for i in range(len(cls_arr)):
            cls_id  = int(cls_arr[i])
            conf    = float(conf_arr[i])
            x1, y1, x2, y2 = xyxy_arr[i].tolist()
            raw     = self._ult_names.get(cls_id, str(cls_id))
            norm    = self._ULT_NAME_MAP.get(raw, raw)

            if norm == "traffic light":
                color = self._check_traffic_light_color(
                    rgb, x1, y1, x2, y2,
                    min_v_abs=tl_min_v, min_s_abs=tl_min_s,
                )
                # Match PIT's .name format so the existing
                # `str(labelName).startswith("traffic light")` branch hits.
                det_name = f"traffic light ({color})"
                det = ObjectDetector._Detection(
                    name=det_name, conf=conf, x=x1, y=y1,
                    distance=-1.0, lightColor=color,
                )
            else:
                det = ObjectDetector._Detection(
                    name=norm, conf=conf, x=x1, y=y1, distance=-1.0,
                )
            detections.append(det)

        return detections, xyxy_arr

    def publish_motion_flag(self, enable:bool):
       msg = Bool()
       msg.data = enable
       self.motion_publisher.publish(msg)

    def terminate(self):
       # --- 2026-05-14: nothing to terminate; this node no longer owns
       #     QCar2DepthAligned. The bridge (qcar2_camera_bridge) handles
       #     runtime shutdown via its destructor.
       # Original (commented):
       #     self.QCarImg.terminate()
       pass


def main():
  rclpy.init()
  node = ObjectDetector()
  try:
      rclpy.spin(node)
  except KeyboardInterrupt:
      node.terminate()
      pass
  rclpy.shutdown()

if __name__ == '__main__':
  main()
