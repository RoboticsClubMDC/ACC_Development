#! /usr/bin/env python3

"""Standalone Quanser YOLOv8 D435 viewer.

This script opens a USB Intel RealSense D435 directly with pyrealsense2,
loads the Quanser YOLOv8 segmentation model, and shows live detections in
an OpenCV window. It does not depend on ROS nodes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEVELOPMENT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_MODEL_PATH = (
    DEVELOPMENT_ROOT
    / "ros2"
    / "src"
    / "qcar2_autonomy"
    / "models"
    / "quanser_yolov8s-seg.pt"
)
DEFAULT_CAPTURE_DIR = SCRIPT_DIR / "captures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Quanser YOLOv8 model live on a USB Intel RealSense D435."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the Quanser YOLOv8 segmentation model (.pt).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Color/depth stream width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Color/depth stream height.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Camera frame rate.",
    )
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
        default=50,
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
        default=None,
        help="Optional class ids to keep. Leave empty to use all model classes.",
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
        package_list = ", ".join(missing)
        raise RuntimeError(
            "Missing Python packages: "
            f"{package_list}. Install the RealSense and YOLO runtime first."
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


def draw_status_banner(frame: np.ndarray, fps: float, detection_count: int) -> None:
    text = (
        f"FPS: {fps:5.1f} | Detections: {detection_count:2d} | "
        "q/esc quit | s snapshot"
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


def build_depth_preview(depth_m: np.ndarray, max_distance: float) -> np.ndarray:
    clipped = np.clip(depth_m, 0.0, max_distance)
    scaled = np.zeros_like(clipped, dtype=np.uint8)

    valid = clipped > 0.0
    if np.any(valid):
        scaled[valid] = np.clip(
            (255.0 * (1.0 - (clipped[valid] / max_distance))),
            0,
            255,
        ).astype(np.uint8)

    return cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)


def median_distance_from_mask(result, detection_index: int, depth_m: np.ndarray) -> float | None:
    if result.masks is None or result.masks.data is None:
        return None

    if detection_index >= len(result.masks.data):
        return None

    mask = result.masks.data[detection_index].detach().cpu().numpy()
    if mask.shape != depth_m.shape:
        mask = cv2.resize(
            mask,
            (depth_m.shape[1], depth_m.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    valid_depth = depth_m[(mask > 0.5) & np.isfinite(depth_m) & (depth_m > 0.0)]
    if valid_depth.size == 0:
        return None

    return float(np.median(valid_depth))


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


def annotate_distances(
    annotated: np.ndarray,
    result,
    depth_m: np.ndarray,
) -> list[str]:
    detection_summaries: list[str] = []

    if result.boxes is None or len(result.boxes) == 0:
        return detection_summaries

    names = result.names

    for index in range(len(result.boxes)):
        xyxy = result.boxes.xyxy[index].detach().cpu().numpy()
        conf = float(result.boxes.conf[index].item())
        cls_id = int(result.boxes.cls[index].item())
        label = class_name(names, cls_id)

        distance_m = median_distance_from_mask(result, index, depth_m)
        if distance_m is None:
            distance_m = center_patch_distance(xyxy, depth_m)

        x1, y1, x2, y2 = [int(v) for v in xyxy]

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


def save_snapshot(
    save_dir: Path,
    annotated: np.ndarray,
    depth_preview: np.ndarray | None,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    image_path = save_dir / f"quanser_yolo_{timestamp}.png"
    cv2.imwrite(str(image_path), annotated)

    if depth_preview is not None:
        depth_path = save_dir / f"quanser_yolo_depth_{timestamp}.png"
        cv2.imwrite(str(depth_path), depth_preview)

    print(f"Saved snapshot: {image_path}")


def main() -> int:
    args = parse_args()
    args.classes = args.classes or None

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
        model = YOLO(str(args.model), task="segment")
    except Exception as exc:
        print(f"Failed to load model: {exc}", file=sys.stderr)
        return 1

    print("Model classes:")
    print(format_model_classes(model.names))
    if args.classes:
        print(f"Filtering to classes: {args.classes}")

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
                classes=args.classes,
                device=device,
                max_det=args.max_det,
                retina_masks=True,
                verbose=False,
            )

            result = results[0]
            annotated = result.plot()
            detections = annotate_distances(annotated, result, depth_m)

            now = time.perf_counter()
            fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now
            draw_status_banner(annotated, fps, len(detections))

            if detections and (time.perf_counter() - last_log_time) >= 0.25:
                print(" | ".join(detections))
                last_log_time = time.perf_counter()

            cv2.imshow("Quanser YOLOv8 D435 Viewer", annotated)

            depth_preview = None
            if args.show_depth:
                depth_preview = build_depth_preview(depth_m, args.depth_max)
                cv2.imshow("Quanser YOLOv8 D435 Depth", depth_preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                save_snapshot(args.save_dir, annotated, depth_preview)

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
