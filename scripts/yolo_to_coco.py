"""Route B / Task 2: convert the existing BDD100K YOLO labels to COCO json for mmdet.

Reads the layout produced by scripts/preprocess_data.py::preprocess_bdd100k():
    datasets/bdd100k_yolo/{split}/images/*.jpg
    datasets/bdd100k_yolo/{split}/labels/*.txt   (class_id cx cy w h, normalized 0-1)

Writes:
    datasets/bdd100k_yolo/annotations/{split}.json   (COCO format, category_id 1-based)

IMPORTANT: class order below matches the class_to_id mapping actually used when the
.txt labels were generated (scripts/preprocess_data.py BDD_CLASSES / download_datasets.py
class_to_id). This is NOT the same order as docs/MMDETECTION_ROUTE_B_RUNBOOK.md's §1 list —
that list is wrong relative to the labels already on disk. Using the runbook's order here
would silently remap every category (e.g. label id 0 = 'pedestrian' on disk would be read
back as 'car'). Do not "fix" this order to match the runbook without re-checking the
labels; fix the runbook instead.
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
SRC_DIR = DATASETS_DIR / "bdd100k_yolo"
OUT_DIR = SRC_DIR / "annotations"

# Order matches class_to_id in scripts/preprocess_data.py / scripts/download_datasets.py.
CLASSES = (
    'pedestrian', 'rider', 'car', 'truck', 'bus',
    'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign',
)

# BDD100K images are natively 1280x720 (scripts/preprocess_data.py hardcodes the same
# constants when normalizing the YOLO labels in the first place). Reusing them here
# instead of opening every image with PIL guarantees an exact round-trip and avoids a
# per-image file open over Drive's FUSE mount, which is what made the original run look
# hung and get Ctrl-C'd (tens of thousands of small Drive reads, no progress feedback).
IMG_W, IMG_H = 1280, 720


def convert_split(split):
    img_dir = SRC_DIR / split / "images"
    lbl_dir = SRC_DIR / split / "labels"
    if not img_dir.exists() or not lbl_dir.exists():
        print(f"  [SKIP] {split}: missing {img_dir} or {lbl_dir}")
        return None

    images, annotations = [], []
    img_id = 0
    ann_id = 0
    img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.jpeg")) + sorted(img_dir.glob("*.png"))
    total = len(img_paths)
    print(f"  {split}: converting {total} images...", flush=True)

    for i, img_path in enumerate(img_paths, 1):
        W, H = IMG_W, IMG_H

        img_id += 1
        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": W,
            "height": H,
        })

        if i % 500 == 0 or i == total:
            print(f"    {split}: {i}/{total}", flush=True)

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        text = lbl_path.read_text().strip()
        if not text:
            continue

        for line in text.splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id, cx, cy, w, h = parts
            class_id = int(class_id)
            cx, cy, w, h = float(cx), float(cy), float(w), float(h)

            box_w = w * W
            box_h = h * H
            x_min = (cx - w / 2) * W
            y_min = (cy - h / 2) * H

            ann_id += 1
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": class_id + 1,  # COCO category_id is 1-based
                "bbox": [x_min, y_min, box_w, box_h],
                "area": box_w * box_h,
                "iscrowd": 0,
                "segmentation": [],
            })

    categories = [{"id": i + 1, "name": name} for i, name in enumerate(CLASSES)]
    coco = {"images": images, "annotations": annotations, "categories": categories}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{split}.json"
    with open(out_path, "w") as f:
        json.dump(coco, f)

    print(f"  {split}: {len(images)} images, {len(annotations)} annotations -> {out_path}")
    return out_path


def main():
    if not SRC_DIR.exists():
        print(f"[ERROR] {SRC_DIR} not found. Run scripts/preprocess_data.py first.")
        sys.exit(1)

    print(f"YOLO -> COCO conversion ({SRC_DIR})")
    for split in ["train", "val"]:
        convert_split(split)


if __name__ == "__main__":
    main()
