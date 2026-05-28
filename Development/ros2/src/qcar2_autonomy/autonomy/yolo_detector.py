#! /usr/bin/env python3

import sys
sys.path.insert(0, "/workspaces/isaac_ros-dev/MDC_libraries/python")

from pit.YOLO.nets import YOLOv8
from pit.YOLO.utils import QCar2DepthAligned

import time
import numpy as np
import cv2
from pathlib import Path
import urllib.request
import tempfile
import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def ensure_model_exists(model_path: Path, url: str, logger=None) -> None:
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if model_path.exists() and model_path.stat().st_size > 1024 * 1024:
        if logger:
            logger.info(f"YOLO model found: {model_path} ({model_path.stat().st_size/1e6:.1f} MB)")
        return

    if logger:
        logger.warn(f"YOLO model not found, downloading to: {model_path}")

    with tempfile.NamedTemporaryFile(dir=str(model_path.parent), delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with urllib.request.urlopen(url) as r, open(tmp_path, "wb") as f:
            chunk = 1024 * 1024
            while True:
                data = r.read(chunk)
                if not data:
                    break
                f.write(data)

        if tmp_path.stat().st_size < 1024 * 1024:
            raise RuntimeError(f"Downloaded file too small, refusing to use it.")

        tmp_path.replace(model_path)
        if logger:
            logger.info(f"YOLO model downloaded OK: {model_path}")
    finally:
        if tmp_path.exists() and (not model_path.exists() or tmp_path != model_path):
            try:
                tmp_path.unlink()
            except Exception:
                pass


class SignApproachTracker:
    """Predictive approach tracker for one sign type. Builds depth-vs-time
    history, fits approach speed, and predicts the wall-clock time when the car
    will be `target_offset_m` from the sign face ("right beside it"). Two
    triggers: (1) lateral-edge — bbox center reaches the outer
    `lateral_edge_frac` of the frame => brake now (primary for side-of-road
    signs, which leave the FOV ~1-2 m before arrival); (2) depth-rate fit =>
    predicted arrival (fallback / head-on). Ported from the Gabriel branch."""
    STALE_S = 1.0

    def __init__(self, logger, label):
        self._log = logger
        self._label = label
        self._history = []
        self._target_arrival_abs = None
        self._last_update_t = None

    def update(self, t_abs, depth_m, bbox_center_x, img_w, *,
               target_offset_m, min_samples, commit_at_m, min_speed,
               max_depth_m, lateral_edge_frac):
        now = t_abs
        if (self._last_update_t is not None
                and (now - self._last_update_t) > self.STALE_S
                and self._target_arrival_abs is None):
            self._history.clear()
        self._last_update_t = now

        # Trigger 1: lateral edge (always checked; overrides a future commit).
        edge_px = float(img_w) * float(lateral_edge_frac)
        if bbox_center_x < edge_px or bbox_center_x > (img_w - edge_px):
            if (self._target_arrival_abs is None or now < self._target_arrival_abs):
                self._target_arrival_abs = float(now)
                self._log.info(
                    f"[predict] {self._label} COMMIT (lateral edge): "
                    f"bbox_x={bbox_center_x:.0f}px depth={depth_m:.2f}m -> BRAKE NOW")
            return self._target_arrival_abs

        if self._target_arrival_abs is not None:
            return self._target_arrival_abs

        # Trigger 2: depth-rate fallback. Skipped when depth is NaN/out-of-range
        # (the lateral-edge trigger above is depth-free and still works).
        if (not np.isfinite(depth_m)) or depth_m <= 0.05 or depth_m > max_depth_m:
            return None
        self._history.append((now, depth_m, bbox_center_x))
        if len(self._history) > 8:
            self._history.pop(0)
        if len(self._history) < min_samples:
            return None
        ts = np.array([t for t, _, _ in self._history], dtype=np.float64)
        ds = np.array([d for _, d, _ in self._history], dtype=np.float64)
        slope, intercept = np.polyfit(ts, ds, 1)
        approach_speed = -slope
        if approach_speed < min_speed:
            return None
        t_target = (intercept - target_offset_m) / approach_speed
        if depth_m <= commit_at_m:
            self._target_arrival_abs = float(t_target)
            self._log.info(
                f"[predict] {self._label} COMMIT (depth-rate): depth={depth_m:.2f}m "
                f"approach={approach_speed:.2f}m/s brake_in={t_target - now:.2f}s")
        return self._target_arrival_abs

    def should_brake(self, t_abs):
        return (self._target_arrival_abs is not None
                and t_abs >= self._target_arrival_abs)

    def on_brake_fired(self):
        self._history.clear()
        self._target_arrival_abs = None
        self._last_update_t = None


class TLStateMachine:
    """Traffic-light approach FSM (ported from Gabriel branch). Commit-and-hold:
      IDLE: first sight green/idle OR bbox past the line -> COMMIT_GO (no brake);
            first sight red/yellow -> COMMIT_STOP (brake).
      COMMIT_STOP: sustained green for K frames -> COMMIT_GO (release); else brake.
      COMMIT_GO: ignores ALL later color changes (a late yellow/red after we
            committed to go does NOT re-brake — deliberate: better to clear the
            intersection than freeze in it).
      Not seen for M frames -> IDLE (reset).
    Returns 'brake', 'release', or None."""
    IDLE        = "IDLE"
    COMMIT_STOP = "COMMIT_STOP"
    COMMIT_GO   = "COMMIT_GO"

    def __init__(self, logger):
        self._log = logger
        self._state = self.IDLE
        self._lost_count = 0
        self._green_count = 0
        self._enter_time = 0.0

    def update(self, t_abs, present, eff_color, bbox_h, *,
               pass_line_height_px, lost_frames_to_reset, green_frames_to_release):
        if not present:
            self._lost_count += 1
            if self._state != self.IDLE and self._lost_count >= lost_frames_to_reset:
                self._enter(self.IDLE, t_abs, f"lost for {self._lost_count} frames -> reset")
            return None
        self._lost_count = 0
        is_stop_color = eff_color in ("red", "yellow")
        past_line = bbox_h > pass_line_height_px
        if self._state == self.IDLE:
            if past_line or not is_stop_color:
                self._enter(self.COMMIT_GO, t_abs,
                    f"first sight {'past-line ' if past_line else ''}color={eff_color} "
                    f"bh={bbox_h:.0f} -> GO")
                return None
            self._enter(self.COMMIT_STOP, t_abs, f"first sight {eff_color} bh={bbox_h:.0f} -> STOP")
            return "brake"
        if self._state == self.COMMIT_STOP:
            if eff_color == "green":
                self._green_count += 1
                if self._green_count >= green_frames_to_release:
                    self._enter(self.COMMIT_GO, t_abs, "sustained green -> GO (release)")
                    return "release"
                return "brake"
            self._green_count = 0
            return "brake"
        return None  # COMMIT_GO — color-change immune

    def _enter(self, new_state, t, reason):
        if new_state != self._state:
            self._log.info(f"[TL-FSM] {self._state} -> {new_state}: {reason}")
        self._state = new_state
        self._enter_time = t
        if new_state == self.IDLE:
            self._green_count = 0
            self._lost_count = 0


class ObjectDetector(Node):

    def __init__(self):
        super().__init__('yolo_detector')

        imageWidth  = 640
        imageHeight = 480
        self.QCarImg = QCar2DepthAligned()

        model_dir  = Path("/workspaces/isaac_ros-dev/ros2/src/qcar2_autonomy/models")
        model_path = model_dir / "quanser_yolov8s-seg.pt"
        model_url  = "https://quanserinc.box.com/shared/static/ce0gxomeg4b12wlcch9cmlh0376nditf.pt"

        ensure_model_exists(model_path, model_url, logger=self.get_logger())

        self.myYolo = YOLOv8(
            modelPath=str(model_path),
            imageHeight=imageHeight,
            imageWidth=imageWidth,
            convert_tensorrt=False,
        )

        self.dt    = 1 / 30
        self.timer = self.create_timer(self.dt, self.on_timer)

        self.motion_publisher = self.create_publisher(Bool, '/motion_enable', 1)
        self.flag_value       = False
        self.publish_motion_flag(True)

        # Time-based braking: brake while now < brake_until_abs (go otherwise).
        self.brake_until_abs         = 0.0
        self.sign_cooldown_until_abs = 0.0

        # --- Predictive stop/yield "stop beside the sign" (ported from Gabriel) ---
        self._img_w = imageWidth
        self._img_h = imageHeight
        # Erick's model tops out ~0.745 conf on stop signs, so 0.90 rejected
        # every frame and the tracker was never fed. Lowered to match the model.
        self.declare_parameter('stop_sign_conf',           0.40)
        self.declare_parameter('yield_sign_conf',          0.40)
        self.declare_parameter('stop_sign_hold_s',         3.0)
        self.declare_parameter('yield_sign_hold_s',        1.5)
        self.declare_parameter('stop_target_offset_m',     0.30)
        self.declare_parameter('stop_predict_min_samples', 3)
        self.declare_parameter('stop_predict_max_depth_m', 8.0)
        self.declare_parameter('stop_predict_commit_at_m', 4.5)
        self.declare_parameter('stop_predict_min_speed',   0.05)
        self.declare_parameter('lateral_edge_frac',        0.30)
        self.declare_parameter('stop_brake_height_px',     120.0)
        self.declare_parameter('detection_cooldown_s',     10.0)
        self.stop_conf                = float(self.get_parameter('stop_sign_conf').value)
        self.yield_conf               = float(self.get_parameter('yield_sign_conf').value)
        self.stop_hold                = float(self.get_parameter('stop_sign_hold_s').value)
        self.yield_hold               = float(self.get_parameter('yield_sign_hold_s').value)
        self.stop_target_offset_m     = float(self.get_parameter('stop_target_offset_m').value)
        self.stop_predict_min_samples = int(self.get_parameter('stop_predict_min_samples').value)
        self.stop_predict_max_depth   = float(self.get_parameter('stop_predict_max_depth_m').value)
        self.stop_predict_commit_at_m = float(self.get_parameter('stop_predict_commit_at_m').value)
        self.stop_predict_min_speed   = float(self.get_parameter('stop_predict_min_speed').value)
        self.lateral_edge_frac        = float(self.get_parameter('lateral_edge_frac').value)
        self.stop_brake_height_px     = float(self.get_parameter('stop_brake_height_px').value)
        self.cooldown_def             = float(self.get_parameter('detection_cooldown_s').value)
        self._stop_tracker  = SignApproachTracker(self.get_logger(), 'stop sign')
        self._yield_tracker = SignApproachTracker(self.get_logger(), 'yield sign')

        # --- Traffic-light FSM + color majority-vote (no-flicker, ported from Gabriel) ---
        self.declare_parameter('tl_conf',                        0.50)
        self.declare_parameter('tl_min_dist_m',                  0.30)
        self.declare_parameter('tl_stop_dist_m',                 5.0)
        self.declare_parameter('tl_min_height_px',               50)
        self.declare_parameter('tl_hold_s',                      0.60)
        self.declare_parameter('tl_edge_margin_px',              8)
        self.declare_parameter('tl_allow_top_clip',             True)
        self.declare_parameter('tl_color_history_size',          8)
        self.declare_parameter('tl_pass_line_height_px',         100)
        self.declare_parameter('tl_fsm_lost_frames_to_reset',    15)
        self.declare_parameter('tl_fsm_green_frames_to_release', 3)
        self.tl_conf              = float(self.get_parameter('tl_conf').value)
        self.tl_min_dist          = float(self.get_parameter('tl_min_dist_m').value)
        self.tl_stop_dist         = float(self.get_parameter('tl_stop_dist_m').value)
        self.tl_min_height        = int(self.get_parameter('tl_min_height_px').value)
        self.tl_hold              = float(self.get_parameter('tl_hold_s').value)
        self.tl_margin            = int(self.get_parameter('tl_edge_margin_px').value)
        self.tl_allow_top_clip    = bool(self.get_parameter('tl_allow_top_clip').value)
        self.tl_history_size      = int(self.get_parameter('tl_color_history_size').value)
        self.tl_pass_line_height  = int(self.get_parameter('tl_pass_line_height_px').value)
        self.tl_fsm_lost_reset    = int(self.get_parameter('tl_fsm_lost_frames_to_reset').value)
        self.tl_fsm_green_release = int(self.get_parameter('tl_fsm_green_frames_to_release').value)
        self._tl_color_history = []
        self._tl_fsm = TLStateMachine(self.get_logger())

        self.bridge          = CvBridge()
        self.publish_rgb     = self.create_publisher(Image, '/qcar_camera/rgb',       10)
        self.publish_depth   = self.create_publisher(Image, '/qcar_camera/depth',     10)
        self.publish_rgb_yolo = self.create_publisher(Image, '/qcar_camera/rgb_yolo', 10)
        self.timer2          = self.create_timer(1 / 500, self.flag_publisher)

    def flag_publisher(self):
        self.publish_motion_flag(self.flag_value)

    def on_timer(self):
        self.QCarImg.read()

        rgb   = self.QCarImg.rgb
        depth = self.QCarImg.depth

        if depth is not None:
            depth = np.asarray(depth)
            if depth.ndim == 3 and depth.shape[2] == 1:
                depth = depth[:, :, 0]
            depth = depth.astype(np.float32, copy=False)

        self.publish_rgb.publish(self.bridge.cv2_to_imgmsg(rgb,   "bgr8"))
        self.publish_depth.publish(self.bridge.cv2_to_imgmsg(depth, "32FC1"))

        # Run perception every tick (feeds the trackers + TL FSM), then derive
        # the motion flag purely from the time-based brake. True = go.
        self.yolo_detect()
        self.flag_value = (time.time() >= self.brake_until_abs)

    def _bbox_fully_in_frame(self, x, y, w, h, margin, allow_top_clip=True):
        if x < margin:
            return False
        if (x + w) > (self._img_w - margin):
            return False
        if (y + h) > (self._img_h - margin):
            return False
        if (not allow_top_clip) and y < margin:
            return False
        return True

    def _center_patch_depth_m(self, depth, x, y, w, h, frac=0.2):
        """Median of valid depths in the central `frac` of a bbox — a robust
        distance estimate that avoids mask bleed onto sky/pole behind the sign.
        Returns meters, or NaN if no valid pixels."""
        if depth is None or w <= 0 or h <= 0:
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

    def _sign_brake_decision(self, tracker, conf_thresh, label_conf,
                             used_d, bbox_cx, bw, bh, now, label):
        """Feed the predictive tracker; return True the instant it commits a
        brake for this sign. Falls back to a simple distance gate if no usable
        bbox is available."""
        if label_conf < conf_thresh:
            return False
        if bw <= 1.0:
            # No usable bbox this frame — simple distance fallback (old behavior).
            return 0.0 < used_d < 1.0
        # Depth-free "we're basically at it" trigger: this model's depth is
        # unreliable (frozen/NaN), so a tall bbox is the most reliable "close"
        # signal. Tune via stop_brake_height_px (0 disables).
        if self.stop_brake_height_px > 0.0 and bh >= self.stop_brake_height_px:
            tracker.on_brake_fired()
            self.get_logger().info(
                f"{label} Sign -> BRAKE NOW (bbox height {bh:.0f}px >= "
                f"{self.stop_brake_height_px:.0f}px, depth={used_d:.2f}m)")
            return True
        # Always feed the tracker. The lateral-edge trigger is geometric
        # (depth-free) and is the robust "stop beside the sign" signal even when
        # the depth estimate is tiny/NaN; the depth-rate trigger is skipped
        # internally when depth is invalid. Do NOT gate this on depth.
        target = tracker.update(
            now, used_d, bbox_cx, self._img_w,
            target_offset_m=self.stop_target_offset_m,
            min_samples=self.stop_predict_min_samples,
            commit_at_m=self.stop_predict_commit_at_m,
            min_speed=self.stop_predict_min_speed,
            max_depth_m=self.stop_predict_max_depth,
            lateral_edge_frac=self.lateral_edge_frac,
        )
        if target is not None and tracker.should_brake(now):
            tracker.on_brake_fired()
            self.get_logger().info(
                f"{label} Sign -> BRAKE NOW (depth={used_d:.2f}m, bbox_x={bbox_cx:.0f}px, "
                f"edge<{self.lateral_edge_frac*self._img_w:.0f}|>{self._img_w-self.lateral_edge_frac*self._img_w:.0f})")
            return True
        return False

    def yolo_detect(self):
        rgbProcessed = self.myYolo.pre_process(self.QCarImg.rgb)
        predicion    = self.myYolo.predict(
            inputImg=rgbProcessed,
            classes=[9, 11, 33],
            confidence=0.3,
            half=True,
            verbose=False
        )

        try:
            ann  = None
            pred = predicion
            if isinstance(pred, (list, tuple)) and len(pred) > 0 and hasattr(pred[0], "plot"):
                ann = pred[0].plot()
            elif hasattr(pred, "plot"):
                ann = pred.plot()
            if ann is not None and isinstance(ann, np.ndarray) and ann.size:
                self.publish_rgb_yolo.publish(self.bridge.cv2_to_imgmsg(ann, "bgr8"))
        except Exception as e:
            self.get_logger().warn(f"YOLO overlay publish failed: {e}")

        processedResults = self.myYolo.post_processing(
            alignedDepth=self.QCarImg.depth,
            clippingDistance=5
        )

        # bbox array (Nx4 xyxy), same order as processedResults (PIT sets it).
        xyxy = getattr(self.myYolo, "bounding", None)
        depth_arr = self.QCarImg.depth
        if depth_arr is not None:
            depth_arr = np.asarray(depth_arr)
            if depth_arr.ndim == 3 and depth_arr.shape[2] == 1:
                depth_arr = depth_arr[:, :, 0]
            depth_arr = depth_arr.astype(np.float32, copy=False)

        now = time.time()

        # Per-tick "most prominent" traffic light for the FSM (after the loop).
        tl_best_present = False
        tl_best_eff     = "idle"
        tl_best_h       = 0.0

        for i, obj in enumerate(processedResults):
            labelName  = obj.__dict__.get("name",     "")
            labelConf  = float(obj.__dict__.get("conf",     0.0))
            objectDist = float(obj.__dict__.get("distance", -1.0))

            # bbox + center-patch depth (used by the stop/yield predictive tracker)
            if xyxy is not None and i < len(xyxy):
                x1, y1, x2, y2 = [float(v) for v in xyxy[i][:4]]
                bx, by, bw, bh = x1, y1, x2 - x1, y2 - y1
            else:
                bx = by = bw = bh = 0.0
            bbox_cx  = bx + bw / 2.0
            center_d = self._center_patch_depth_m(depth_arr, bx, by, bw, bh)
            used_d   = center_d if (np.isfinite(center_d) and center_d > 0.0) else objectDist

            self.get_logger().info(
                f"{labelName} @ {labelConf:.3f} conf. pit={objectDist:.3f}m "
                f"used={used_d:.3f}m bbox={bw:.0f}x{bh:.0f} cx={bbox_cx:.0f}")

            if str(labelName).startswith("traffic light"):
                inst_color = str(obj.__dict__.get("lightColor", "")).strip().lower()
                # Temporal majority vote over the last N frames — kills color flicker.
                self._tl_color_history.append((now, inst_color))
                while len(self._tl_color_history) > self.tl_history_size:
                    self._tl_color_history.pop(0)
                counts = {"red": 0, "yellow": 0, "green": 0, "idle": 0}
                for _t, _c in self._tl_color_history:
                    if "red" in _c:      counts["red"]    += 1
                    elif "yellow" in _c: counts["yellow"] += 1
                    elif "green" in _c:  counts["green"]  += 1
                    else:                counts["idle"]   += 1
                priority = ["red", "yellow", "green", "idle"]
                eff_color = max(priority, key=lambda c: (counts[c], -priority.index(c)))

                depth_ok  = (np.isfinite(used_d) and used_d > self.tl_min_dist
                             and used_d < self.tl_stop_dist)
                height_ok = (bh >= self.tl_min_height)
                close_enough = depth_ok or height_ok
                full_visible = self._bbox_fully_in_frame(
                    bx, by, bw, bh, self.tl_margin, allow_top_clip=self.tl_allow_top_clip)
                if (labelConf >= self.tl_conf) and close_enough and full_visible and bh > tl_best_h:
                    tl_best_h       = bh
                    tl_best_eff     = eff_color
                    tl_best_present = True

            elif labelName == "stop sign":
                if (now >= self.sign_cooldown_until_abs
                        and self._sign_brake_decision(self._stop_tracker, self.stop_conf,
                                                      labelConf, used_d, bbox_cx, bw, bh, now, "Stop")):
                    self.brake_until_abs = max(self.brake_until_abs, now + self.stop_hold)
                    self.sign_cooldown_until_abs = now + self.cooldown_def

            elif labelName == "yield sign":
                if (now >= self.sign_cooldown_until_abs
                        and self._sign_brake_decision(self._yield_tracker, self.yield_conf,
                                                      labelConf, used_d, bbox_cx, bw, bh, now, "Yield")):
                    self.brake_until_abs = max(self.brake_until_abs, now + self.yield_hold)
                    self.sign_cooldown_until_abs = now + self.cooldown_def

        # Drive the TL state machine once per tick with the most prominent TL
        # (or "not present"). Commit-on-green: brake only if red/yellow on first
        # sight; once committed to GO it ignores later color changes.
        action = self._tl_fsm.update(
            now,
            present=tl_best_present,
            eff_color=tl_best_eff,
            bbox_h=tl_best_h,
            pass_line_height_px=self.tl_pass_line_height,
            lost_frames_to_reset=self.tl_fsm_lost_reset,
            green_frames_to_release=self.tl_fsm_green_release,
        )
        if action == "brake":
            self.brake_until_abs = max(self.brake_until_abs, now + self.tl_hold)
        elif action == "release":
            # Hard-release only a TL-scale brake; never cancel a stop-sign brake.
            if self.brake_until_abs <= now + self.tl_hold + 0.05:
                self.brake_until_abs = min(self.brake_until_abs, now)

        # If the depth-rate tracker committed a future brake time, the sign can
        # leave the camera before that time arrives. Poll armed trackers every
        # tick so the planned stop/yield still fires out of view.
        poll_now = time.time()
        for tracker, hold, label in (
            (self._stop_tracker, self.stop_hold, "Stop"),
            (self._yield_tracker, self.yield_hold, "Yield"),
        ):
            if tracker.should_brake(poll_now) and poll_now >= self.sign_cooldown_until_abs:
                self.brake_until_abs = max(self.brake_until_abs, poll_now + hold)
                self.sign_cooldown_until_abs = poll_now + self.cooldown_def
                self.get_logger().info(
                    f"{label} Sign -> BRAKE NOW (armed tracker poll; sign may be out of FOV, "
                    f"hold {hold:.1f}s)")
                tracker.on_brake_fired()

    def publish_motion_flag(self, enable: bool):
        msg      = Bool()
        msg.data = enable
        self.motion_publisher.publish(msg)

    def terminate(self):
        self.QCarImg.terminate()


def main():
    rclpy.init()
    node = ObjectDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.terminate()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
