#! /usr/bin/env python3

"""Standalone D435 viewer for learned traffic-light color detection.

This script uses the `yoloObjDetBP01.pt` detection model directly on a USB
Intel RealSense D435. By default it filters to the model's explicit traffic
light color classes: Green, Red, and Yellow.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from Traffic_light_green_yellow_red import DEFAULT_IP as DEFAULT_TRAFFIC_LIGHT_IP
from Traffic_light_green_yellow_red import run_traffic_light_loop


SCRIPT_DIR = Path(__file__).resolve().parent
DEVELOPMENT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_MODEL_PATH = (
    DEVELOPMENT_ROOT
    / "ros2"
    / "src"
    / "qcar2_autonomy"
    / "models"
    / "yoloObjDetBP01.pt"
)
DEFAULT_COLOR_CLASSES = [1, 3, 6]
DEFAULT_CAPTURE_DIR = SCRIPT_DIR / "captures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run learned traffic-light color detection live on a USB D435."
    )
    parser.set_defaults(run_light_loop=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the YOLO traffic-light color model (.pt).",
    )
    parser.add_argument("--width", type=int, default=640, help="Color/depth stream width.")
    parser.add_argument("--height", type=int, default=480, help="Color/depth stream height.")
    parser.add_argument("--fps", type=int, default=30, help="Camera frame rate.")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.30,
        help="Minimum detection confidence.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold.",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=25,
        help="Maximum detections per frame.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda:0, 0, etc.",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=DEFAULT_COLOR_CLASSES,
        help="Class ids to keep. Defaults to the learned color classes [1, 3, 6].",
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Disable the default color-only filter and show every class in the model.",
    )
    parser.add_argument(
        "--no-light-loop",
        dest="run_light_loop",
        action="store_false",
        help="Do not start the physical traffic light cycle in the background.",
    )
    parser.add_argument(
        "--light-ip",
        default=DEFAULT_TRAFFIC_LIGHT_IP,
        help="Traffic light IP used by the built-in physical light loop.",
    )
    parser.add_argument(
        "--light-red",
        type=float,
        default=30.0,
        help="Seconds to hold red in the built-in physical light loop.",
    )
    parser.add_argument(
        "--light-green",
        type=float,
        default=30.0,
        help="Seconds to hold green in the built-in physical light loop.",
    )
    parser.add_argument(
        "--light-yellow",
        type=float,
        default=3.0,
        help="Seconds to hold yellow in the built-in physical light loop.",
    )
    parser.add_argument(
        "--light-cycles",
        type=int,
        default=0,
        help="Number of physical light cycles to run. Use 0 to loop until the viewer exits.",
    )
    parser.add_argument(
        "--leave-light-on",
        action="store_true",
        help="Leave the physical traffic light on its last color when the viewer exits.",
    )
    parser.add_argument(
        "--show-depth",
        action="store_true",
        help="Show a second window with the aligned depth image.",
    )
    parser.add_argument(
        "--depth-max",
        type=float,
        default=5.0,
        help="Maximum distance in meters used for depth preview scaling.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_CAPTURE_DIR,
        help="Where snapshots are saved when you press 's'.",
    )
    return parser.parse_args()


def load_runtime_dependencies():
    missing = []

    try:
        import pyrealsense2 as rs
    except ImportError:
        rs = None
        missing.append("pyrealsense2")

    try:
        from ultralytics import YOLO
    except ImportError:
        YOLO = None
        missing.append("ultralytics")

    try:
        import torch
    except ImportError:
        torch = None

    if missing:
        raise RuntimeError(
            "Missing Python packages: "
            f"{', '.join(missing)}. Install the RealSense and YOLO runtime first."
        )

    return rs, YOLO, torch


def choose_device(requested: str, torch_module) -> str:
    if requested != "auto":
        return requested

    if torch_module is not None and torch_module.cuda.is_available():
        return "cuda:0"

    return "cpu"


def class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))

    if 0 <= class_id < len(names):
        return str(names[class_id])

    return str(class_id)


def format_model_classes(names) -> str:
    if isinstance(names, dict):
        entries = sorted(names.items())
    else:
        entries = list(enumerate(names))

    return ", ".join(f"{idx}:{name}" for idx, name in entries)


def build_depth_preview(depth_m: np.ndarray, max_distance: float) -> np.ndarray:
    clipped = np.clip(depth_m, 0.0, max_distance)
    scaled = np.zeros_like(clipped, dtype=np.uint8)

    valid = clipped > 0.0
    if np.any(valid):
        scaled[valid] = np.clip(
            255.0 * (1.0 - (clipped[valid] / max_distance)),
            0,
            255,
        ).astype(np.uint8)

    return cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)


def center_patch_distance(
    xyxy: np.ndarray,
    depth_m: np.ndarray,
    patch_radius: int = 4,
) -> float | None:
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cx = max(0, min(depth_m.shape[1] - 1, (x1 + x2) // 2))
    cy = max(0, min(depth_m.shape[0] - 1, (y1 + y2) // 2))

    x0 = max(0, cx - patch_radius)
    x3 = min(depth_m.shape[1], cx + patch_radius + 1)
    y0 = max(0, cy - patch_radius)
    y3 = min(depth_m.shape[0], cy + patch_radius + 1)

    patch = depth_m[y0:y3, x0:x3]
    valid_depth = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid_depth.size == 0:
        return None

    return float(np.median(valid_depth))


def draw_detection_details(annotated: np.ndarray, result, depth_m: np.ndarray) -> list[str]:
    detection_summaries: list[str] = []

    if result.boxes is None or len(result.boxes) == 0:
        return detection_summaries

    names = result.names
    for index in range(len(result.boxes)):
        xyxy = result.boxes.xyxy[index].detach().cpu().numpy()
        conf = float(result.boxes.conf[index].item())
        cls_id = int(result.boxes.cls[index].item())
        label = class_name(names, cls_id)
        distance_m = center_patch_distance(xyxy, depth_m)

        x1, _, _, y2 = [int(v) for v in xyxy]
        if distance_m is None:
            distance_text = "depth n/a"
            detection_summaries.append(f"{label} {conf:.2f}")
        else:
            distance_text = f"{distance_m:.2f} m"
            detection_summaries.append(f"{label} {conf:.2f} {distance_m:.2f}m")

        text_position = (max(5, x1), max(20, min(annotated.shape[0] - 10, y2 + 18)))
        cv2.putText(
            annotated,
            distance_text,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            distance_text,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return detection_summaries


def draw_status_banner(
    frame: np.ndarray,
    fps: float,
    detection_count: int,
    active_classes: list[int] | None,
    light_loop_active: bool,
) -> None:
    classes_text = "all" if active_classes is None else ",".join(str(v) for v in active_classes)
    light_text = "on" if light_loop_active else "off"
    text = (
        f"FPS: {fps:5.1f} | Detections: {detection_count:2d} | "
        f"Classes: {classes_text} | Light loop: {light_text} | q/esc quit | s snapshot"
    )
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (20, 20, 20), -1)
    cv2.putText(
        frame,
        text,
        (10, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def save_snapshot(save_dir: Path, annotated: np.ndarray, depth_preview: np.ndarray | None) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    image_path = save_dir / f"traffic_light_color_{timestamp}.png"
    cv2.imwrite(str(image_path), annotated)

    if depth_preview is not None:
        depth_path = save_dir / f"traffic_light_color_depth_{timestamp}.png"
        cv2.imwrite(str(depth_path), depth_preview)

    print(f"Saved snapshot: {image_path}")


def main() -> int:
    args = parse_args()
    active_classes = None if args.all_classes else (args.classes or DEFAULT_COLOR_CLASSES)

    if not args.model.exists():
        print(f"Model file not found: {args.model}", file=sys.stderr)
        return 1

    try:
        rs, YOLO, torch = load_runtime_dependencies()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        print(
            "Suggested install: python -m pip install ultralytics pyrealsense2",
            file=sys.stderr,
        )
        return 1

    device = choose_device(args.device, torch)
    print(f"Using model: {args.model}")
    print(f"Using device: {device}")

    try:
        model = YOLO(str(args.model), task="detect")
    except Exception as exc:
        print(f"Failed to load model: {exc}", file=sys.stderr)
        return 1

    print("Model classes:")
    print(format_model_classes(model.names))
    if active_classes is None:
        print("Filtering to classes: all model classes")
    else:
        print(f"Filtering to classes: {active_classes}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    try:
        pipeline_profile = pipeline.start(config)
    except Exception as exc:
        print(f"Failed to start D435 stream: {exc}", file=sys.stderr)
        print("Make sure the RealSense D435 is connected and not in use elsewhere.", file=sys.stderr)
        return 1

    align = rs.align(rs.stream.color)
    depth_sensor = pipeline_profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())

    light_thread = None
    light_stop_event = None
    if args.run_light_loop:
        light_stop_event = threading.Event()
        light_thread = threading.Thread(
            target=run_traffic_light_loop,
            kwargs={
                "light_ip": args.light_ip,
                "red": args.light_red,
                "green": args.light_green,
                "yellow": args.light_yellow,
                "cycles": args.light_cycles,
                "leave_on": args.leave_light_on,
                "stop_event": light_stop_event,
            },
            daemon=True,
        )
        light_thread.start()
        print(f"Started physical traffic light loop at {args.light_ip}")

    print(
        f"Streaming RealSense at {args.width}x{args.height}@{args.fps} "
        f"(depth scale {depth_scale:.6f} m/unit)"
    )

    previous_time = time.perf_counter()
    last_log_time = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_bgr = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_m = depth_raw.astype(np.float32) * depth_scale

            results = model.predict(
                source=color_bgr,
                conf=args.confidence,
                iou=args.iou,
                classes=active_classes,
                device=device,
                max_det=args.max_det,
                verbose=False,
            )

            result = results[0]
            annotated = result.plot()
            detections = draw_detection_details(annotated, result, depth_m)

            now = time.perf_counter()
            fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now
            draw_status_banner(
                annotated,
                fps,
                len(detections),
                active_classes,
                args.run_light_loop,
            )

            if detections and (time.perf_counter() - last_log_time) >= 0.25:
                print(" | ".join(detections))
                last_log_time = time.perf_counter()

            cv2.imshow("Traffic Light Color D435 Viewer", annotated)

            depth_preview = None
            if args.show_depth:
                depth_preview = build_depth_preview(depth_m, args.depth_max)
                cv2.imshow("Traffic Light Color D435 Depth", depth_preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                save_snapshot(args.save_dir, annotated, depth_preview)

    except KeyboardInterrupt:
        pass
    finally:
        if light_stop_event is not None:
            light_stop_event.set()
        if light_thread is not None:
            light_thread.join(timeout=5.0)
        pipeline.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
