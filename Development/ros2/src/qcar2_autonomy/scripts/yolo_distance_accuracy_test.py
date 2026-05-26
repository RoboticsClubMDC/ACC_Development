#!/usr/bin/env python3
"""
Run the Quanser YOLOv8 model and estimate object distance for quick accuracy tests.

Primary distance source:
  - aligned D435 depth median inside the YOLO mask or bbox center crop.

Optional fallback / calibration check:
  - monocular pinhole estimate: distance = known_width_m * focal_length_px / bbox_width_px.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_MODEL = (
    REPO_ROOT
    / "Development"
    / "ros2"
    / "src"
    / "qcar2_autonomy"
    / "models"
    / "quanser_yolov8s-seg.pt"
)
DEFAULT_MDC_PATH = REPO_ROOT / "Development" / "MDC_libraries" / "python"


def add_mdc_paths(extra_path):
    candidates = [
        extra_path,
        os.getenv("MDC_PYTHON_PATH", ""),
        str(DEFAULT_MDC_PATH),
        "/workspaces/isaac_ros-dev/MDC_libraries/python",
        "/home/nvidia/Documents/ACC_Development/Development/MDC_libraries/python",
    ]
    for item in candidates:
        if not item:
            continue
        for path in str(item).split(":"):
            if path and Path(path).exists() and path not in sys.path:
                sys.path.insert(0, path)


def parse_class_filter(text):
    text = str(text).strip()
    if not text or text.lower() in ("all", "none"):
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_known_widths(text):
    if not text:
        return {}
    path = Path(text).expanduser()
    if path.exists():
        return json.loads(path.read_text())
    return json.loads(text)


def class_known_width(known_widths, class_id, class_name):
    for key in (str(class_id), str(class_name), str(class_name).lower()):
        if key in known_widths:
            return float(known_widths[key])
    return None


def load_camera(is_physical, mdc_path):
    add_mdc_paths(mdc_path)
    try:
        from pit.YOLO.utils import QCar2DepthAligned
    except Exception as exc:
        raise RuntimeError(
            "Could not import QCar2DepthAligned. Run inside the Quanser/PAL "
            "environment or pass --image/--video instead."
        ) from exc
    return QCar2DepthAligned(isPhyscial=is_physical)


def open_source(args):
    if args.image:
        image = cv2.imread(str(Path(args.image).expanduser()))
        if image is None:
            raise RuntimeError(f"Could not read image: {args.image}")
        return "image", image, None

    if args.video:
        cap = cv2.VideoCapture(str(Path(args.video).expanduser()))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {args.video}")
        return "video", cap, None

    if args.camera_index is not None:
        cap = cv2.VideoCapture(int(args.camera_index))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index: {args.camera_index}")
        return "video", cap, None

    qcar_camera = load_camera(args.physical, args.mdc_path)
    return "qcar_depth_aligned", qcar_camera, qcar_camera


def read_frame(source_kind, source):
    if source_kind == "image":
        return False, source.copy(), None

    if source_kind == "video":
        ok, frame = source.read()
        return ok, frame, None

    ok = source.read()
    rgb = np.asarray(source.rgb)
    depth = np.asarray(source.depth)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return ok, rgb.copy(), depth.astype(np.float32, copy=False)


def resize_depth(depth, width, height):
    if depth is None:
        return None
    if depth.shape[:2] == (height, width):
        return depth
    return cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)


def robust_depth_for_detection(depth, xyxy, mask, crop_ratio, min_depth, max_depth):
    if depth is None:
        return None

    h, w = depth.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(0, min(w, int(round(x2))))
    y2 = max(0, min(h, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None

    if mask is not None:
        mask_u8 = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        roi_mask = mask_u8[y1:y2, x1:x2] > 0
        roi_depth = depth[y1:y2, x1:x2]
        values = roi_depth[roi_mask]
    else:
        ratio = max(0.05, min(1.0, float(crop_ratio)))
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bw = (x2 - x1) * ratio
        bh = (y2 - y1) * ratio
        rx1 = max(0, min(w - 1, int(round(cx - bw / 2.0))))
        rx2 = max(0, min(w, int(round(cx + bw / 2.0))))
        ry1 = max(0, min(h - 1, int(round(cy - bh / 2.0))))
        ry2 = max(0, min(h, int(round(cy + bh / 2.0))))
        values = depth[ry1:ry2, rx1:rx2].reshape(-1)

    values = values[np.isfinite(values)]
    values = values[(values >= min_depth) & (values <= max_depth)]
    if values.size == 0:
        return None

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return {
        "median": median,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "mad_sigma": 1.4826 * mad,
        "count": int(values.size),
    }


def get_masks(result):
    if getattr(result, "masks", None) is None or result.masks is None:
        return []
    try:
        return result.masks.data.detach().cpu().numpy()
    except Exception:
        return []


def annotate_detection(
    image,
    xyxy,
    class_name,
    conf,
    depth_stats,
    mono_distance,
    known_distance,
):
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    cv2.rectangle(image, (x1, y1), (x2, y2), (35, 210, 80), 2)

    labels = [f"{class_name} {conf:.2f}"]
    if depth_stats is not None:
        labels.append(f"depth {depth_stats['median']:.2f}m")
    if mono_distance is not None:
        labels.append(f"mono {mono_distance:.2f}m")
    if known_distance is not None and depth_stats is not None:
        err = depth_stats["median"] - known_distance
        labels.append(f"err {err:+.2f}m")

    y = max(18, y1 - 8)
    for line in labels:
        cv2.putText(
            image,
            line,
            (x1, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            (x1, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        y += 18


def print_detection(frame_idx, row):
    depth = row.get("depth_m")
    mono = row.get("mono_m")
    known = row.get("known_m")
    parts = [
        f"frame={frame_idx:05d}",
        f"class={row['class_name']}",
        f"conf={row['confidence']:.2f}",
        f"bbox_w={row['bbox_width_px']:.1f}px",
    ]
    if depth is not None:
        parts.append(f"depth={depth:.3f}m")
        parts.append(f"sigma={row['depth_mad_sigma']:.3f}m")
    if mono is not None:
        parts.append(f"mono={mono:.3f}m")
    if row.get("fx_calibrated_px") is not None:
        parts.append(f"fx_cal={row['fx_calibrated_px']:.1f}px")
    if known is not None and depth is not None:
        parts.append(f"depth_err={depth - known:+.3f}m")
    if known is not None and mono is not None:
        parts.append(f"mono_err={mono - known:+.3f}m")
    print("  ".join(parts), flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run Quanser YOLOv8 and print depth/monocular distance estimates."
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="YOLOv8 .pt/.onnx model path.")
    parser.add_argument("--image", help="Single image path. Depth is unavailable unless using QCar source.")
    parser.add_argument("--video", help="Video path.")
    parser.add_argument("--camera-index", type=int, help="OpenCV camera index.")
    parser.add_argument("--physical", action="store_true", help="Use physical QCar2DepthAligned mode.")
    parser.add_argument("--virtual", dest="physical", action="store_false", help="Use virtual QCar2DepthAligned mode.")
    parser.set_defaults(physical=False)
    parser.add_argument("--mdc-path", default="", help="Optional MDC_libraries/python path.")
    parser.add_argument("--classes", default="2,9,11", help="Comma-separated class ids, or all.")
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--frames", type=int, default=0, help="Number of frames to process. 0 means until stopped.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. cpu, 0, cuda:0.")
    parser.add_argument("--no-window", action="store_true", help="Do not show OpenCV preview.")
    parser.add_argument("--save-dir", default="yolo_distance_runs", help="Directory for annotated frames and CSV.")
    parser.add_argument("--save-every", type=int, default=15, help="Save every Nth annotated frame. 0 disables images.")
    parser.add_argument("--crop-ratio", type=float, default=0.50, help="BBox center crop ratio when no segmentation mask exists.")
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--fx", type=float, default=455.20, help="Focal length in pixels for monocular estimate.")
    parser.add_argument(
        "--known-width",
        type=float,
        help="Single real object width in meters to use for every detection.",
    )
    parser.add_argument(
        "--known-widths",
        default="",
        help='JSON string/file mapping class name or id to width meters, e.g. \'{"stop sign":0.75,"9":0.30}\'.',
    )
    parser.add_argument("--known-distance", type=float, help="Measured test distance in meters for error reporting.")
    parser.add_argument(
        "--calibrate-fx",
        action="store_true",
        help="Use --known-width and --known-distance to print measured webcam focal length estimates.",
    )
    args = parser.parse_args()

    if args.calibrate_fx and (args.known_width is None or args.known_distance is None):
        raise SystemExit("--calibrate-fx requires both --known-width and --known-distance.")

    add_mdc_paths(args.mdc_path)
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: ultralytics. Install/activate the QCar YOLO environment, "
            "then rerun this script."
        ) from exc

    model_path = Path(args.model).expanduser()
    if not model_path.exists():
        raise SystemExit(f"YOLO model not found: {model_path}")

    known_widths = parse_known_widths(args.known_widths)
    class_filter = parse_class_filter(args.classes)
    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / f"distance_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    csv = csv_path.open("w", buffering=1)
    csv.write(
        "frame,class_id,class_name,confidence,bbox_width_px,depth_m,depth_mad_sigma,"
        "depth_pixels,known_width_m,fx_px,fx_calibrated_px,mono_m,known_m,depth_error_m,mono_error_m\n"
    )

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    source_kind, source, qcar_camera = open_source(args)
    print(f"Source: {source_kind}")
    print(f"Logging CSV: {csv_path}")

    frame_idx = 0
    try:
        while args.frames <= 0 or frame_idx < args.frames:
            ok, frame, depth = read_frame(source_kind, source)
            if not ok and source_kind != "image":
                break
            if frame is None:
                break

            frame_idx += 1
            h, w = frame.shape[:2]
            depth = resize_depth(depth, w, h)

            results = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.confidence,
                classes=class_filter,
                device=args.device,
                verbose=False,
            )
            result = results[0]
            masks = get_masks(result)
            boxes = getattr(result, "boxes", None)
            annotated = frame.copy()

            if boxes is not None and len(boxes) > 0:
                for i, box in enumerate(boxes):
                    xyxy = box.xyxy[0].detach().cpu().numpy().astype(float)
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    class_name = str(result.names.get(cls_id, cls_id))
                    bbox_width = max(1.0, float(xyxy[2] - xyxy[0]))
                    mask = masks[i] if i < len(masks) else None
                    depth_stats = robust_depth_for_detection(
                        depth,
                        xyxy,
                        mask,
                        args.crop_ratio,
                        args.min_depth,
                        args.max_depth,
                    )

                    known_width = args.known_width
                    if known_width is None:
                        known_width = class_known_width(known_widths, cls_id, class_name)
                    mono_distance = None
                    if known_width is not None:
                        mono_distance = known_width * float(args.fx) / bbox_width
                    fx_calibrated = None
                    if args.calibrate_fx and known_width is not None:
                        fx_calibrated = bbox_width * float(args.known_distance) / known_width

                    row = {
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": conf,
                        "bbox_width_px": bbox_width,
                        "depth_m": None if depth_stats is None else depth_stats["median"],
                        "depth_mad_sigma": None if depth_stats is None else depth_stats["mad_sigma"],
                        "depth_pixels": 0 if depth_stats is None else depth_stats["count"],
                        "mono_m": mono_distance,
                        "fx_calibrated_px": fx_calibrated,
                        "known_m": args.known_distance,
                    }
                    print_detection(frame_idx, row)

                    depth_err = (
                        ""
                        if row["known_m"] is None or row["depth_m"] is None
                        else row["depth_m"] - row["known_m"]
                    )
                    mono_err = (
                        ""
                        if row["known_m"] is None or row["mono_m"] is None
                        else row["mono_m"] - row["known_m"]
                    )
                    csv.write(
                        f"{frame_idx},{cls_id},{class_name},{conf:.6f},{bbox_width:.3f},"
                        f"{'' if row['depth_m'] is None else f'{row['depth_m']:.6f}'},"
                        f"{'' if row['depth_mad_sigma'] is None else f'{row['depth_mad_sigma']:.6f}'},"
                        f"{row['depth_pixels']},"
                        f"{'' if known_width is None else f'{known_width:.6f}'},"
                        f"{float(args.fx):.6f},"
                        f"{'' if fx_calibrated is None else f'{fx_calibrated:.6f}'},"
                        f"{'' if mono_distance is None else f'{mono_distance:.6f}'},"
                        f"{'' if args.known_distance is None else f'{args.known_distance:.6f}'},"
                        f"{depth_err if depth_err == '' else f'{depth_err:.6f}'},"
                        f"{mono_err if mono_err == '' else f'{mono_err:.6f}'}\n"
                    )

                    annotate_detection(
                        annotated,
                        xyxy,
                        class_name,
                        conf,
                        depth_stats,
                        mono_distance,
                        args.known_distance,
                    )

            if args.save_every > 0 and frame_idx % args.save_every == 0:
                cv2.imwrite(str(save_dir / f"frame_{frame_idx:05d}.jpg"), annotated)

            if not args.no_window:
                cv2.imshow("Quanser YOLO distance accuracy test", annotated)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

            if source_kind == "image":
                if not args.no_window:
                    cv2.waitKey(0)
                break

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        csv.close()
        if qcar_camera is not None:
            try:
                qcar_camera.terminate()
            except Exception:
                pass
        if source_kind == "video":
            source.release()
        cv2.destroyAllWindows()
        print(f"Saved log: {csv_path}")


if __name__ == "__main__":
    main()
