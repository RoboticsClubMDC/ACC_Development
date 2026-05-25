#! /usr/bin/env python3

"""Fine-tune Quanser's YOLOv8 segmentation model to add traffic cone support."""

from __future__ import annotations

import argparse
import shutil
from itertools import repeat
from pathlib import Path

import torch
from ultralytics import YOLO

from prepare_quanser_cone_dataset import MODEL_PATH, PACKAGE_ROOT, prepare_dataset


RUNS_DIR = PACKAGE_ROOT / "runs" / "segment"
OUTPUT_MODEL_PATH = PACKAGE_ROOT / "models" / "quanser_yolov8s-seg-cone.pt"
OUTPUT_LAST_MODEL_PATH = PACKAGE_ROOT / "models" / "quanser_yolov8s-seg-cone-last.pt"


def install_ultralytics_cache_workaround() -> None:
    """Avoid a Windows multiprocessing pipe failure during label caching."""

    import ultralytics.data.dataset as dataset_module
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import HELP_URL, get_hash, save_dataset_cache_file, verify_image_label
    from ultralytics.utils import LOGGER, TQDM

    def cache_labels_sequential(self, path: Path = Path("./labels.cache")) -> dict:
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))

        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )

        iterable = zip(
            self.im_files,
            self.label_files,
            repeat(self.prefix),
            repeat(self.use_keypoints),
            repeat(len(self.data["names"])),
            repeat(nkpt),
            repeat(ndim),
            repeat(self.single_cls),
        )
        pbar = TQDM(iterable, desc=desc, total=total)

        for args in pbar:
            im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg = verify_image_label(args)
            nm += nm_f
            nf += nf_f
            ne += ne_f
            nc += nc_f

            if im_file:
                x["labels"].append(
                    {
                        "im_file": im_file,
                        "shape": shape,
                        "cls": lb[:, 0:1],
                        "bboxes": lb[:, 1:],
                        "segments": segments,
                        "keypoints": keypoint,
                        "normalized": True,
                        "bbox_format": "xywh",
                    }
                )
            if msg:
                msgs.append(msg)

            pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"

        pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}No labels found in {path}. {HELP_URL}")

        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs
        if x["labels"]:
            save_dataset_cache_file(self.prefix, path, x, dataset_module.DATASET_CACHE_VERSION)
        return x

    YOLODataset.cache_labels = cache_labels_sequential


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the Quanser YOLOv8 segmentation model on the cone dataset."
    )
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=480, help="Training image size.")
    parser.add_argument("--batch", type=int, default=2, help="Batch size.")
    parser.add_argument(
        "--device",
        default="cpu" if not torch.cuda.is_available() else "0",
        help="Training device, for example 'cpu' or '0'.",
    )
    parser.add_argument("--workers", type=int, default=0, help="Data loader workers.")
    parser.add_argument("--freeze", type=int, default=10, help="Number of layers to freeze.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--name", default="quanser_cone_finetune", help="Run name.")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow overwriting an existing run directory.",
    )
    return parser.parse_args()


def copy_trained_weights(run_dir: Path) -> None:
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"

    if best_path.exists():
        shutil.copy2(best_path, OUTPUT_MODEL_PATH)
        print(f"Copied best weights to: {OUTPUT_MODEL_PATH}")

    if last_path.exists():
        shutil.copy2(last_path, OUTPUT_LAST_MODEL_PATH)
        print(f"Copied last weights to: {OUTPUT_LAST_MODEL_PATH}")


def main() -> int:
    args = parse_args()
    summary = prepare_dataset(clean=False)
    install_ultralytics_cache_workaround()

    print(f"Using source model: {MODEL_PATH}")
    print(f"Using dataset config: {summary['data_yaml']}")
    print(f"Traffic cone class id: {summary['cone_class_id']}")
    print(f"Run output: {RUNS_DIR / args.name}")

    model = YOLO(str(MODEL_PATH), task="segment")
    results = model.train(
        data=str(summary["data_yaml"]),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        freeze=args.freeze,
        patience=args.patience,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=args.exist_ok,
        task="segment",
        amp=False,
    )

    run_dir = Path(results.save_dir)
    print(f"Training artifacts saved to: {run_dir}")
    copy_trained_weights(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
