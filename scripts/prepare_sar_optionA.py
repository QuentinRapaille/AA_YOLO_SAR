#!/usr/bin/env python3
"""Prepare a SAR dataset for AA-YOLO Option A.

- Reads SAR images from an input images directory
- Converts each image to 8-bit PNG, 3 channels (grayscale duplicated)
- Copies YOLO labels from an input labels directory (same relative path and stem)
- Creates train/val/test txt split files in data/datasets/

This keeps AA-YOLO code unchanged (ch=3) and avoids JPEG artifacts.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".dng", ".webp", ".mpo"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare SAR dataset for AA-YOLO Option A")
    p.add_argument("--images-dir", type=Path, required=True, help="Input SAR images root")
    p.add_argument("--labels-dir", type=Path, required=True, help="Input YOLO labels root")
    p.add_argument("--dataset-name", type=str, default="SAR_OPTION_A", help="Output dataset folder name")
    p.add_argument("--output-root", type=Path, default=Path("data/datasets"), help="Output datasets root")
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--p-low", type=float, default=1.0, help="Lower percentile for clipping")
    p.add_argument("--p-high", type=float, default=99.0, help="Upper percentile for clipping")
    p.add_argument("--flat", action="store_true", help="Flatten output filenames (avoid subdirs)")
    return p.parse_args()


def to_uint8(img: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    if img.ndim == 3:
        if img.shape[2] == 1:
            img = img[:, :, 0]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    x = img.astype(np.float32)
    lo = float(np.percentile(x, p_low))
    hi = float(np.percentile(x, p_high))

    if hi <= lo:
        lo = float(x.min())
        hi = float(x.max())

    if hi <= lo:
        return np.zeros_like(x, dtype=np.uint8)

    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo)
    x = (x * 255.0).round().astype(np.uint8)
    return x


def main() -> None:
    args = parse_args()

    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    images_dir = args.images_dir.resolve()
    labels_dir = args.labels_dir.resolve()
    if not images_dir.exists():
        raise FileNotFoundError(f"images-dir not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels-dir not found: {labels_dir}")

    dataset_dir = (args.output_root / args.dataset_name).resolve()
    out_images = dataset_dir / "images"
    out_labels = dataset_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])
    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    converted_rel_paths = []
    missing_labels = 0

    for src in image_paths:
        rel = src.relative_to(images_dir)
        if args.flat:
            rel_out = Path(f"{src.stem}.png")
            rel_label = Path(f"{src.stem}.txt")
        else:
            rel_out = rel.with_suffix(".png")
            rel_label = rel.with_suffix(".txt")

        dst_img = out_images / rel_out
        dst_lbl = out_labels / rel_label
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] Skipping unreadable image: {src}")
            continue

        gray_u8 = to_uint8(img, p_low=args.p_low, p_high=args.p_high)
        rgb = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
        ok = cv2.imwrite(str(dst_img), rgb)
        if not ok:
            print(f"[WARN] Failed to write: {dst_img}")
            continue

        src_lbl = labels_dir / rel.with_suffix(".txt")
        if src_lbl.exists():
            dst_lbl.write_text(src_lbl.read_text())
            converted_rel_paths.append(rel_out)
        else:
            missing_labels += 1

    if not converted_rel_paths:
        raise RuntimeError("No image+label pairs were produced. Check label naming/path.")

    rng = random.Random(args.seed)
    rng.shuffle(converted_rel_paths)

    n = len(converted_rel_paths)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    train_rel = converted_rel_paths[:n_train]
    val_rel = converted_rel_paths[n_train:n_train + n_val]
    test_rel = converted_rel_paths[n_train + n_val:]

    split_prefix = f"{args.dataset_name}"
    txt_root = args.output_root.resolve()

    def write_split(name: str, rel_paths: list[Path]) -> Path:
        txt_path = txt_root / f"{split_prefix}_{name}.txt"
        lines = [f"./{args.dataset_name}/images/{p.as_posix()}\n" for p in rel_paths]
        txt_path.write_text("".join(lines))
        return txt_path

    train_txt = write_split("train", train_rel)
    val_txt = write_split("val", val_rel)
    test_txt = write_split("test", test_rel)

    print(f"[OK] Output dataset: {dataset_dir}")
    print(f"[OK] Paired samples: {n}")
    print(f"[OK] Missing labels skipped: {missing_labels}")
    print(f"[OK] Splits: train={len(train_rel)} val={len(val_rel)} test={len(test_rel)}")
    print(f"[OK] Split files:")
    print(f"      {train_txt}")
    print(f"      {val_txt}")
    print(f"      {test_txt}")


if __name__ == "__main__":
    main()
