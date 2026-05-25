#! /usr/bin/env python3

"""Prepare a merged Quanser-plus-cone YOLOv8 segmentation dataset.

The provided Roboflow dataset contains a single class, "Traffic Cone", with
class id 0. The Quanser model uses the 80-class COCO label map plus a custom
"yield sign" class at id 33. This script preserves the model's existing class
layout and appends "Traffic Cone" as class id 80.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[6]
PACKAGE_ROOT = REPO_ROOT / "Development" / "ros2" / "src" / "qcar2_autonomy"
MODEL_PATH = PACKAGE_ROOT / "models" / "quanser_yolov8s-seg.pt"
RAW_DATASET_DIR = PACKAGE_ROOT / "datasets" / "traffic_cone_roboflow_raw"
PREPARED_DATASET_DIR = PACKAGE_ROOT / "datasets" / "traffic_cone_quanser81"
CONE_CLASS_NAME = "Traffic Cone"


def extract_frame_number(path: Path) -> int:
    match = re.search(r"-([0-9]+)_jpg", path.name)
    if match:
        return int(match.group(1))
    return 0


def sorted_image_paths(images_dir: Path) -> list[Path]:
    return sorted(images_dir.glob("*"), key=lambda path: (extract_frame_number(path), path.name))


def split_single_train_set(train_images_dir: Path) -> dict[str, list[Path]]:
    images = sorted_image_paths(train_images_dir)
    splits = {"train": [], "valid": [], "test": []}

    for index, image_path in enumerate(images):
        bucket = index % 10
        if bucket == 0:
            splits["test"].append(image_path)
        elif bucket == 5:
            splits["valid"].append(image_path)
        else:
            splits["train"].append(image_path)

    return splits


def paired_label_path(image_path: Path, labels_dir: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"


def remap_label_contents(label_text: str, target_class_id: int) -> str:
    remapped_lines = []
    for raw_line in label_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        parts[0] = str(target_class_id)
        remapped_lines.append(" ".join(parts))
    return "\n".join(remapped_lines) + ("\n" if remapped_lines else "")


def prepare_output_dirs(base_dir: Path, clean: bool = True) -> None:
    if clean and base_dir.exists():
        shutil.rmtree(base_dir)

    for split in ("train", "valid", "test"):
        (base_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (base_dir / split / "labels").mkdir(parents=True, exist_ok=True)


def write_split(
    image_paths: list[Path],
    raw_labels_dir: Path,
    output_split_dir: Path,
    target_class_id: int,
) -> int:
    written = 0

    for image_path in image_paths:
        label_path = paired_label_path(image_path, raw_labels_dir)
        if not label_path.exists():
            continue

        output_image_path = output_split_dir / "images" / image_path.name
        output_label_path = output_split_dir / "labels" / label_path.name

        shutil.copy2(image_path, output_image_path)

        remapped = remap_label_contents(label_path.read_text(encoding="utf-8"), target_class_id)
        output_label_path.write_text(remapped, encoding="utf-8")
        written += 1

    return written


def load_quanser_names(model_path: Path) -> list[str]:
    model = YOLO(str(model_path), task="segment")
    names = model.names
    ordered = [names[index] for index in sorted(names)]

    if CONE_CLASS_NAME in ordered:
        raise RuntimeError(f"{CONE_CLASS_NAME} already exists in the source model.")

    if len(ordered) != 80:
        raise RuntimeError(f"Expected 80 base classes in {model_path.name}, found {len(ordered)}.")

    return ordered


def write_data_yaml(dataset_dir: Path, class_names: list[str]) -> Path:
    data = {
        "path": dataset_dir.as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }

    output_path = dataset_dir / "data.yaml"
    output_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return output_path


def prepare_dataset(clean: bool = True) -> dict[str, object]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Quanser model not found: {MODEL_PATH}")

    if not RAW_DATASET_DIR.exists():
        raise FileNotFoundError(f"Raw Roboflow dataset not found: {RAW_DATASET_DIR}")

    base_names = load_quanser_names(MODEL_PATH)
    merged_names = base_names + [CONE_CLASS_NAME]
    cone_class_id = len(merged_names) - 1

    raw_train_images = RAW_DATASET_DIR / "train" / "images"
    raw_train_labels = RAW_DATASET_DIR / "train" / "labels"
    raw_valid_images = RAW_DATASET_DIR / "valid" / "images"
    raw_valid_labels = RAW_DATASET_DIR / "valid" / "labels"
    raw_test_images = RAW_DATASET_DIR / "test" / "images"
    raw_test_labels = RAW_DATASET_DIR / "test" / "labels"

    prepare_output_dirs(PREPARED_DATASET_DIR, clean=clean)

    if raw_valid_images.exists() and raw_test_images.exists():
        split_images = {
            "train": sorted_image_paths(raw_train_images),
            "valid": sorted_image_paths(raw_valid_images),
            "test": sorted_image_paths(raw_test_images),
        }
        split_label_dirs = {
            "train": raw_train_labels,
            "valid": raw_valid_labels,
            "test": raw_test_labels,
        }
    else:
        split_images = split_single_train_set(raw_train_images)
        split_label_dirs = {
            "train": raw_train_labels,
            "valid": raw_train_labels,
            "test": raw_train_labels,
        }

    split_counts = {}
    for split_name, image_paths in split_images.items():
        count = write_split(
            image_paths=image_paths,
            raw_labels_dir=split_label_dirs[split_name],
            output_split_dir=PREPARED_DATASET_DIR / split_name,
            target_class_id=cone_class_id,
        )
        split_counts[split_name] = count

    data_yaml_path = write_data_yaml(PREPARED_DATASET_DIR, merged_names)

    summary = {
        "dataset_dir": PREPARED_DATASET_DIR,
        "data_yaml": data_yaml_path,
        "cone_class_id": cone_class_id,
        "class_names": merged_names,
        "split_counts": split_counts,
    }

    return summary


def main() -> int:
    summary = prepare_dataset()
    print(f"Prepared dataset: {summary['dataset_dir']}")
    print(f"Data YAML: {summary['data_yaml']}")
    print(f"Cone class id: {summary['cone_class_id']}")
    for split_name, count in summary["split_counts"].items():
        print(f"{split_name}: {count} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
