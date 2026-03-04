# scripts/convert_hrsid_coco_to_yolo.py
from pathlib import Path
import json
import random
from collections import defaultdict

ROOT = Path("HRSID_png")
ANN = ROOT / "annotations"
OUT = Path("data/datasets/HRSID_YOLO")
OUT_LABELS = OUT / "labels"
OUT_LABELS.mkdir(parents=True, exist_ok=True)
VAL_FROM_TRAIN = 500
SEED = 42

def resolve_image(file_name: str) -> Path:
    cands = [
        ROOT / file_name,
        ROOT / "JPEGImages" / file_name,
        ROOT / "images" / file_name,
        ROOT / "train2017" / file_name,
        ROOT / "test2017" / file_name,
    ]
    for p in cands:
        if p.exists():
            return p
    return cands[0]

def convert(json_file: Path, split_name: str):
    data = json.loads(json_file.read_text())
    images = data["images"]
    anns = data["annotations"]

    anns_by_img = defaultdict(list)
    for a in anns:
        anns_by_img[a["image_id"]].append(a)

    split_txt = OUT / f"{split_name}.txt"
    lines = []

    for im in images:
        im_id = im["id"]
        w, h = float(im["width"]), float(im["height"])
        img_path = resolve_image(im["file_name"])
        if not img_path.exists():
            continue

        yolo_lines = []
        for a in anns_by_img.get(im_id, []):
            if a.get("iscrowd", 0) == 1:
                continue
            x, y, bw, bh = map(float, a["bbox"])
            if bw <= 0 or bh <= 0:
                continue
            xc = (x + bw / 2.0) / w
            yc = (y + bh / 2.0) / h
            nw = bw / w
            nh = bh / h
            yolo_lines.append(f"0 {xc:.8f} {yc:.8f} {nw:.8f} {nh:.8f}")

        (OUT_LABELS / f"{Path(im['file_name']).stem}.txt").write_text(
            "\n".join(yolo_lines) + ("\n" if yolo_lines else "")
        )
        lines.append(img_path.as_posix())

    split_txt.write_text("\n".join(lines) + ("\n" if lines else ""))

convert(ANN / "train2017.json", "HRSID_train")
convert(ANN / "test2017.json", "HRSID_test")

# Keep test split intact.
# Build val by sampling from train split.
train_txt = OUT / "HRSID_train.txt"
train_lines = [l.strip() for l in train_txt.read_text().splitlines() if l.strip()]
if not train_lines:
    raise RuntimeError("HRSID_train.txt is empty; cannot build val split from train.")

rng = random.Random(SEED)
val_count = min(VAL_FROM_TRAIN, len(train_lines))
val_lines = rng.sample(train_lines, val_count)
val_set = set(val_lines)
new_train_lines = [l for l in train_lines if l not in val_set]

train_txt.write_text("\n".join(new_train_lines) + ("\n" if new_train_lines else ""))
(OUT / "HRSID_val.txt").write_text("\n".join(val_lines) + ("\n" if val_lines else ""))

# optionnel: train_test global
if (ANN / "train_test2017.json").exists():
    convert(ANN / "train_test2017.json", "HRSID_train_test")

(OUT / "hrsid.yaml").write_text(
    f"train: {(OUT / 'HRSID_train.txt').as_posix()}\n"
    f"val: {(OUT / 'HRSID_val.txt').as_posix()}\n"
    f"test: {(OUT / 'HRSID_test.txt').as_posix()}\n\n"
    "nc: 1\n"
    "names: ['ship']\n"
)

print("Done.")
