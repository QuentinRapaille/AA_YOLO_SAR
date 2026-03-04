#!/usr/bin/env python3
"""Prepare Official SSDD (BBox VOC) in YOLO format for AA-YOLO.

Input expected:
- Official-SSDD-OPEN/BBox_SSDD/voc_style/JPEGImages/*.jpg
- Official-SSDD-OPEN/BBox_SSDD/voc_style/Annotations/*.xml
- Official-SSDD-OPEN/BBox_SSDD/voc_style/ImageSets/Main/train.txt
- Official-SSDD-OPEN/BBox_SSDD/voc_style/ImageSets/Main/test.txt

Output:
- data/datasets/SSDD_YOLO/images/*.jpg
- data/datasets/SSDD_YOLO/labels/*.txt
- data/datasets/SSDD_YOLO/SSDD_train.txt
- data/datasets/SSDD_YOLO/SSDD_val.txt
- data/datasets/SSDD_YOLO/SSDD_test.txt
"""

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare SSDD in YOLO format")
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Official-SSDD-OPEN/BBox_SSDD/voc_style"),
        help="Path to Official SSDD BBox voc_style root",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/datasets/SSDD_YOLO"),
        help="Output directory for YOLO dataset",
    )
    p.add_argument("--train-count", type=int, default=742, help="Target train count")
    p.add_argument("--val-count", type=int, default=186, help="Target val count")
    p.add_argument("--test-count", type=int, default=232, help="Target test count")
    p.add_argument("--seed", type=int, default=42, help="Seed for train/val split from train set")
    return p.parse_args()


def load_ids(txt_path: Path) -> list[str]:
    return [line.strip() for line in txt_path.read_text().splitlines() if line.strip()]


def clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def xml_to_yolo(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in {xml_path}")

    w = float(size.findtext("width", "0"))
    h = float(size.findtext("height", "0"))
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image size in {xml_path}: width={w}, height={h}")

    lines: list[str] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name and name != "ship":
            continue
        b = obj.find("bndbox")
        if b is None:
            continue

        xmin = clamp(float(b.findtext("xmin", "0")), 0.0, w)
        ymin = clamp(float(b.findtext("ymin", "0")), 0.0, h)
        xmax = clamp(float(b.findtext("xmax", "0")), 0.0, w)
        ymax = clamp(float(b.findtext("ymax", "0")), 0.0, h)
        if xmax <= xmin or ymax <= ymin:
            continue

        xc = ((xmin + xmax) * 0.5) / w
        yc = ((ymin + ymax) * 0.5) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def write_split(path: Path, ids: list[str]) -> None:
    lines = [f"./images/{img_id}.jpg" for img_id in ids]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def main() -> None:
    args = parse_args()

    ds = args.dataset_root.resolve()
    imgs = ds / "JPEGImages"
    anns = ds / "Annotations"
    main = ds / "ImageSets" / "Main"
    train_txt = main / "train.txt"
    test_txt = main / "test.txt"

    for p in [imgs, anns, train_txt, test_txt]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required path: {p}")

    train_ids = load_ids(train_txt)
    test_ids = load_ids(test_txt)

    if len(train_ids) != args.train_count + args.val_count:
        raise ValueError(
            f"train.txt size mismatch: got {len(train_ids)}, expected {args.train_count + args.val_count}"
        )
    if len(test_ids) != args.test_count:
        raise ValueError(f"test.txt size mismatch: got {len(test_ids)}, expected {args.test_count}")

    rng = random.Random(args.seed)
    shuffled = train_ids[:]
    rng.shuffle(shuffled)
    out_train_ids = sorted(shuffled[: args.train_count])
    out_val_ids = sorted(shuffled[args.train_count : args.train_count + args.val_count])
    out_test_ids = test_ids[:]  # keep official test ordering

    out = args.output_root.resolve()
    out_images = out / "images"
    out_labels = out / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    all_ids = sorted(set(out_train_ids + out_val_ids + out_test_ids))
    missing_images = 0
    missing_xml = 0
    total_boxes = 0
    converted = 0

    for img_id in all_ids:
        src_img = imgs / f"{img_id}.jpg"
        src_xml = anns / f"{img_id}.xml"
        dst_img = out_images / f"{img_id}.jpg"
        dst_lbl = out_labels / f"{img_id}.txt"

        if not src_img.exists():
            missing_images += 1
            continue
        if not src_xml.exists():
            missing_xml += 1
            continue

        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)
        yolo_lines = xml_to_yolo(src_xml)
        dst_lbl.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

        total_boxes += len(yolo_lines)
        converted += 1

    write_split(out / "SSDD_train.txt", out_train_ids)
    write_split(out / "SSDD_val.txt", out_val_ids)
    write_split(out / "SSDD_test.txt", out_test_ids)

    print(f"[OK] Output dataset: {out}")
    print(f"[OK] Converted samples: {converted}")
    print(f"[OK] Total boxes: {total_boxes}")
    print(f"[OK] Missing images skipped: {missing_images}")
    print(f"[OK] Missing xml skipped: {missing_xml}")
    print(f"[OK] train={len(out_train_ids)} val={len(out_val_ids)} test={len(out_test_ids)}")
    print(f"[OK] seed={args.seed} (val drawn from official train split)")


if __name__ == "__main__":
    main()
