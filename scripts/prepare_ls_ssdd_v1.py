#!/usr/bin/env python3
"""Convert LS-SSDD-v1.0 (VOC XML) to YOLO format for AA-YOLO.

Expected LS-SSDD-v1.0 structure:
- Annotations_sub/Annotations_sub/*.xml
- JPEGImages_sub_train/JPEGImages_sub_train/*.jpg
- JPEGImages_sub_test/JPEGImages_sub_test/*.jpg
- train.txt, val.txt, test.txt (stems without extension)

Output structure:
- data/datasets/LS_SSDD_v1_YOLO/images/*.jpg
- data/datasets/LS_SSDD_v1_YOLO/labels/*.txt
- data/datasets/LS_SSDD_v1_YOLO/LS_SSDD_v1_{train,val,test}.txt
"""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare LS-SSDD-v1.0 in YOLO format")
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("LS-SSDD-v1.0"),
        help="Path to LS-SSDD-v1.0 root directory",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/datasets/LS_SSDD_v1_YOLO"),
        help="Output directory for YOLO dataset",
    )
    p.add_argument(
        "--train-split-file",
        type=str,
        default="train_w_val.txt",
        help="Training split file name inside dataset-root (use train_w_val.txt for disjoint train/val)",
    )
    return p.parse_args()


def load_split_ids(split_file: Path) -> list[str]:
    return [line.strip() for line in split_file.read_text().splitlines() if line.strip()]


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def xml_to_yolo_labels(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    size_node = root.find("size")
    if size_node is None:
        raise ValueError(f"Missing <size> in {xml_path}")

    width = float(size_node.findtext("width", "0"))
    height = float(size_node.findtext("height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}: width={width}, height={height}")

    lines: list[str] = []
    for obj in root.findall("object"):
        cls_name = obj.findtext("name", "").strip().lower()
        if cls_name and cls_name != "ship":
            continue

        bnd = obj.find("bndbox")
        if bnd is None:
            continue

        xmin = float(bnd.findtext("xmin", "0"))
        ymin = float(bnd.findtext("ymin", "0"))
        xmax = float(bnd.findtext("xmax", "0"))
        ymax = float(bnd.findtext("ymax", "0"))

        xmin = max(0.0, min(xmin, width))
        xmax = max(0.0, min(xmax, width))
        ymin = max(0.0, min(ymin, height))
        ymax = max(0.0, min(ymax, height))

        if xmax <= xmin or ymax <= ymin:
            continue

        x_center = ((xmin + xmax) / 2.0) / width
        y_center = ((ymin + ymax) / 2.0) / height
        box_w = (xmax - xmin) / width
        box_h = (ymax - ymin) / height

        x_center = clamp01(x_center)
        y_center = clamp01(y_center)
        box_w = clamp01(box_w)
        box_h = clamp01(box_h)

        # Single class dataset: ship -> class id 0
        lines.append(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

    return lines


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset-root not found: {dataset_root}")

    ann_dir = dataset_root / "Annotations_sub" / "Annotations_sub"
    train_img_dir = dataset_root / "JPEGImages_sub_train" / "JPEGImages_sub_train"
    test_img_dir = dataset_root / "JPEGImages_sub_test" / "JPEGImages_sub_test"

    if not ann_dir.exists():
        raise FileNotFoundError(f"Missing annotations directory: {ann_dir}")
    if not train_img_dir.exists():
        raise FileNotFoundError(f"Missing train images directory: {train_img_dir}")
    if not test_img_dir.exists():
        raise FileNotFoundError(f"Missing test images directory: {test_img_dir}")

    out_root = args.output_root.resolve()
    out_images = out_root / "images"
    out_labels = out_root / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    train_split_path = dataset_root / args.train_split_file
    if not train_split_path.exists():
        raise FileNotFoundError(f"Training split file not found: {train_split_path}")

    split_ids = {
        "train": load_split_ids(train_split_path),
        "val": load_split_ids(dataset_root / "val.txt"),
        "test": load_split_ids(dataset_root / "test.txt"),
    }

    train_img_map = {p.stem: p for p in train_img_dir.glob("*.jpg")}
    test_img_map = {p.stem: p for p in test_img_dir.glob("*.jpg")}

    txt_paths: dict[str, Path] = {}
    converted = 0
    missing_xml = 0
    missing_img = 0
    total_boxes = 0
    total_empty = 0

    for split_name, ids in split_ids.items():
        split_txt = out_root / f"LS_SSDD_v1_{split_name}.txt"
        txt_paths[split_name] = split_txt
        rel_lines: list[str] = []

        for stem in ids:
            src_img = train_img_map.get(stem) or test_img_map.get(stem)
            if src_img is None:
                missing_img += 1
                continue

            xml_path = ann_dir / f"{stem}.xml"
            if not xml_path.exists():
                missing_xml += 1
                continue

            dst_img = out_images / f"{stem}.jpg"
            dst_lbl = out_labels / f"{stem}.txt"

            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)

            label_lines = xml_to_yolo_labels(xml_path)
            dst_lbl.write_text("\n".join(label_lines) + ("\n" if label_lines else ""))

            if not label_lines:
                total_empty += 1
            total_boxes += len(label_lines)
            converted += 1
            # Keep paths relative to the split txt directory (YOLO loader behavior).
            rel_lines.append(f"./images/{stem}.jpg")

        split_txt.write_text("\n".join(rel_lines) + ("\n" if rel_lines else ""))

    print(f"[OK] Output dataset: {out_root}")
    print(f"[OK] Converted samples: {converted}")
    print(f"[OK] Total boxes: {total_boxes}")
    print(f"[OK] Empty-label samples: {total_empty}")
    print(f"[OK] Missing images skipped: {missing_img}")
    print(f"[OK] Missing xml skipped: {missing_xml}")
    for split_name in ("train", "val", "test"):
        n = len(load_split_ids(txt_paths[split_name]))
        print(f"[OK] {split_name}: {n} -> {txt_paths[split_name]}")


if __name__ == "__main__":
    main()
