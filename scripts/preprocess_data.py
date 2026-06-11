"""
Phase 1: Data Preprocessing Script
Resizes, normalizes images and splits datasets into train/val/test sets.
For BDD100K: converts JSON detection labels to YOLO .txt format.

Optimized for low-resource machines (16GB RAM, HDD):
  - Processes images in small chunks to limit memory usage
  - Frees memory between chunks with explicit gc
  - Skips already-processed images for resume support
  - Reduces JPEG quality slightly to save disk space on HDD
"""

import os
import gc
import cv2
import sys
import json
import shutil
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"

TARGET_SIZE = (640, 640)
SPLIT_RATIO = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42
CHUNK_SIZE = 200
JPEG_QUALITY = 90

def collect_images(dataset_path):
    """Collect all image file paths from a dataset directory recursively."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = []
    for ext in extensions:
        images.extend(dataset_path.rglob(f"*{ext}"))
        images.extend(dataset_path.rglob(f"*{ext.upper()}"))
    return sorted(set(images))

def resize_image(img, target_size=TARGET_SIZE):
    """Resize image while maintaining aspect ratio with letterbox padding."""
    h, w = img.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    return canvas

def split_dataset(images, split_ratio=SPLIT_RATIO, seed=RANDOM_SEED):
    """Split image paths into train/val/test sets."""
    random.seed(seed)
    shuffled = images.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * split_ratio["train"])
    n_val = int(n * split_ratio["val"])

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }

def get_output_path(img_path, split_dir):
    """Generate output path, handling duplicate filenames."""
    out_path = split_dir / img_path.name
    if out_path.exists():
        stem = img_path.stem
        suffix = img_path.suffix
        out_path = split_dir / f"{stem}_{hash(str(img_path)) % 10000}{suffix}"
    return out_path

def process_chunk(chunk, split_dir, jpeg_quality=JPEG_QUALITY):
    """Process a chunk of images: read, resize, save. Returns (processed, skipped, failed)."""
    processed = 0
    skipped = 0
    failed = 0

    for img_path in chunk:
        out_path = get_output_path(img_path, split_dir)

        if out_path.exists():
            skipped += 1
            continue

        try:
            img = cv2.imread(str(img_path))
            if img is None:
                failed += 1
                continue

            img_resized = resize_image(img, TARGET_SIZE)

            ext = out_path.suffix.lower()
            if ext in (".jpg", ".jpeg"):
                cv2.imwrite(str(out_path), img_resized,
                            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            elif ext == ".png":
                cv2.imwrite(str(out_path), img_resized,
                            [cv2.IMWRITE_PNG_COMPRESSION, 6])
            else:
                cv2.imwrite(str(out_path), img_resized)

            processed += 1

            del img, img_resized

        except Exception as e:
            failed += 1

    return processed, skipped, failed

def preprocess_dataset(dataset_name, chunk_size=CHUNK_SIZE):
    """Preprocess a single dataset in memory-safe chunks."""
    src_dir = DATASETS_DIR / dataset_name
    out_dir = DATASETS_DIR / f"{dataset_name}_processed"

    if not src_dir.exists():
        print(f"  [SKIP] {dataset_name}: source directory not found at {src_dir}")
        return

    print(f"\n{'='*60}")
    print(f"PREPROCESSING: {dataset_name}")
    print(f"{'='*60}")

    images = collect_images(src_dir)
    if not images:
        print(f"  [SKIP] No images found in {src_dir}")
        return

    print(f"  Found {len(images)} images")
    print(f"  Chunk size: {chunk_size} images | JPEG quality: {JPEG_QUALITY}")

    splits = split_dataset(images)

    total_processed = 0
    total_skipped = 0
    total_failed = 0

    for split_name, split_images in splits.items():
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        n_images = len(split_images)
        n_chunks = (n_images + chunk_size - 1) // chunk_size

        print(f"\n  {split_name}: {n_images} images in {n_chunks} chunks")

        split_processed = 0
        split_skipped = 0
        split_failed = 0

        pbar = tqdm(total=n_images, desc=f"    {split_name}", ncols=80, unit="img")

        for i in range(0, n_images, chunk_size):
            chunk = split_images[i:i + chunk_size]

            p, s, f = process_chunk(chunk, split_dir, JPEG_QUALITY)
            split_processed += p
            split_skipped += s
            split_failed += f

            pbar.update(len(chunk))

            gc.collect()

        pbar.close()

        print(f"    -> processed: {split_processed} | skipped: {split_skipped} | failed: {split_failed}")

        total_processed += split_processed
        total_skipped += split_skipped
        total_failed += split_failed

    print(f"\n  {'─'*40}")
    print(f"  {dataset_name} DONE")
    print(f"    Total processed: {total_processed}")
    print(f"    Already existed: {total_skipped}")
    print(f"    Failed/corrupt:  {total_failed}")

    for split_name in ["train", "val", "test"]:
        split_dir = out_dir / split_name
        if split_dir.exists():
            count = len(list(split_dir.glob("*")))
            print(f"    {split_name}: {count} images")

    print(f"  Output: {out_dir}")

def preprocess_bdd100k():
    """Convert BDD100K dataset to YOLO format with proper detection labels.

    BDD100K comes with JSON labels that must be converted to YOLO .txt format.
    Images are NOT resized (Ultralytics handles that during training).
    Uses BDD100K's built-in train/val split instead of random splitting.

    Output structure:
        datasets/bdd100k_yolo/
        ├── train/
        │   ├── images/   (symlinks or copies of original images)
        │   └── labels/   (YOLO format .txt files)
        └── val/
            ├── images/
            └── labels/
    """
    src_dir = DATASETS_DIR / "bdd100k"
    out_dir = DATASETS_DIR / "bdd100k_yolo"

    if not src_dir.exists():
        print(f"  [SKIP] BDD100K not found at {src_dir}")
        return

    print(f"\n{'='*60}")
    print("PREPROCESSING: BDD100K → YOLO format")
    print(f"{'='*60}")

    train_labels = out_dir / "train" / "labels"
    if train_labels.exists() and len(list(train_labels.glob("*.txt"))) > 1000:
        count = len(list(train_labels.glob("*.txt")))
        print(f"  [SKIP] Already converted ({count} train labels exist)")
        return

    label_files = {}
    per_image_dirs = {}
    for split in ['train', 'val']:
        consolidated = (
            list(src_dir.rglob(f'det_{split}.json')) +
            list(src_dir.rglob(f'bdd100k_labels_images_{split}.json'))
        )
        if consolidated:
            label_files[split] = consolidated[0]
            print(f"  Found {split} labels (consolidated): {consolidated[0]}")
            continue

        candidates = []
        for d in src_dir.rglob(split):
            if not d.is_dir():
                continue
            n = sum(1 for _ in d.glob('*.json'))
            if n > 0:
                candidates.append((n, d))
        if candidates:
            candidates.sort(reverse=True)
            n, best = candidates[0]
            per_image_dirs[split] = best
            print(f"  Found {split} labels (per-image): {best} ({n} JSON files)")

    if not label_files and not per_image_dirs:
        # Aggressive fallback: find any JSON file whose name or ancestor path
        # contains 'train' or 'val' – handles non-standard Kaggle dataset layouts.
        all_jsons = sorted(src_dir.rglob('*.json'),
                           key=lambda p: p.stat().st_size, reverse=True)
        for j in all_jsons:
            name_lower = j.name.lower()
            path_lower = str(j).lower()
            for split in ['train', 'val']:
                if split not in label_files:
                    if split in name_lower or split in path_lower:
                        label_files[split] = j
                        print(f"  Found {split} labels (fallback scan): {j.relative_to(src_dir)} ({j.stat().st_size/1e6:.0f} MB)")
        # last resort: assign by file size (largest = train)
        if not label_files and len(all_jsons) >= 2:
            label_files['train'] = all_jsons[0]
            label_files['val'] = all_jsons[1]
            print(f"  Found train labels (size-rank): {all_jsons[0].name} ({all_jsons[0].stat().st_size/1e6:.0f} MB)")
            print(f"  Found val   labels (size-rank): {all_jsons[1].name} ({all_jsons[1].stat().st_size/1e6:.0f} MB)")

    if not label_files and not per_image_dirs:
        print("  [ERROR] No BDD100K label JSON files found!")
        print("  Searched for:")
        print("    - det_{train,val}.json")
        print("    - bdd100k_labels_images_{train,val}.json")
        print("    - labels/100k/{train,val}/*.json (per-image)")
        print("  Make sure bdd100k_labels.zip was downloaded and extracted.")
        return

    img_dirs = {}
    for split in ['train', 'val']:

        candidates = list(src_dir.rglob(f'100k/{split}'))
        if not candidates:

            for d in src_dir.rglob(split):
                if d.is_dir() and any(d.glob('*.jpg')):
                    candidates.append(d)
                    break
        if candidates:
            img_dirs[split] = candidates[0]
            img_count = len(list(candidates[0].glob('*.jpg')))
            print(f"  Found {split} images: {candidates[0]} ({img_count} files)")

    if not img_dirs:
        print("  [ERROR] No BDD100K image directories found!")
        print("  Expected: bdd100k/images/100k/train/ or similar")
        return

    BDD_CLASSES = [
        'pedestrian', 'rider', 'car', 'truck', 'bus',
        'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign'
    ]
    class_to_id = {c: i for i, c in enumerate(BDD_CLASSES)}

    IMG_W, IMG_H = 1280, 720

    total_images = 0
    total_labels = 0

    for split in ['train', 'val']:
        has_consolidated = split in label_files
        has_per_image = split in per_image_dirs
        if not (has_consolidated or has_per_image) or split not in img_dirs:
            print(f"  [SKIP] {split}: missing labels or images")
            continue

        img_out = out_dir / split / "images"
        lbl_out = out_dir / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        print(f"\n  Loading {split} labels...")
        if has_consolidated:
            with open(label_files[split], 'r') as f:
                annotations = json.load(f)
            print(f"  {len(annotations)} annotations loaded (consolidated)")
        else:
            json_files = sorted(per_image_dirs[split].glob('*.json'))
            print(f"  {len(json_files)} per-image annotation files found")
            def _iter_per_image():
                for jp in json_files:
                    try:
                        with open(jp, 'r') as f:
                            data = json.load(f)
                    except (json.JSONDecodeError, OSError):
                        continue

                    name = data.get('name', jp.stem)
                    if not name.endswith('.jpg'):
                        name = name + '.jpg'
                    frames = data.get('frames') or []
                    objs = frames[0].get('objects', []) if frames else data.get('labels', [])
                    yield {'name': name, 'labels': objs}
            annotations = list(_iter_per_image())
            print(f"  {len(annotations)} annotations loaded (per-image)")

        converted = 0
        skipped = 0
        no_labels = 0

        for ann in tqdm(annotations, desc=f"    {split}", ncols=80):
            img_name = ann.get('name', '')
            if not img_name:
                skipped += 1
                continue

            img_path = img_dirs[split] / img_name
            if not img_path.exists():
                skipped += 1
                continue

            labels = ann.get('labels', [])
            yolo_lines = []

            for lbl in labels:
                cat = lbl.get('category', '')
                if cat not in class_to_id:
                    continue

                box = lbl.get('box2d', {})
                if not box:
                    continue

                x1 = float(box.get('x1', 0))
                y1 = float(box.get('y1', 0))
                x2 = float(box.get('x2', 0))
                y2 = float(box.get('y2', 0))

                x_center = ((x1 + x2) / 2) / IMG_W
                y_center = ((y1 + y2) / 2) / IMG_H
                w = (x2 - x1) / IMG_W
                h = (y2 - y1) / IMG_H

                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                w = max(0.001, min(1.0, w))
                h = max(0.001, min(1.0, h))

                class_id = class_to_id[cat]
                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

            img_dest = img_out / img_name
            if not img_dest.exists():
                try:
                    os.symlink(img_path.resolve(), img_dest)
                except (OSError, NotImplementedError):
                    shutil.copy2(str(img_path), str(img_dest))

            lbl_name = Path(img_name).stem + '.txt'
            lbl_dest = lbl_out / lbl_name
            with open(lbl_dest, 'w') as f:
                f.write('\n'.join(yolo_lines))

            if yolo_lines:
                total_labels += len(yolo_lines)
            else:
                no_labels += 1

            converted += 1

        total_images += converted
        print(f"    {split}: {converted} images converted, {skipped} skipped, {no_labels} backgrounds")

    print(f"\n  {'─'*40}")
    print(f"  BDD100K → YOLO conversion DONE")
    print(f"    Total images: {total_images}")
    print(f"    Total object labels: {total_labels}")
    print(f"    Output: {out_dir}")

    for split in ['train', 'val']:
        lbl_dir = out_dir / split / "labels"
        img_dir = out_dir / split / "images"
        if lbl_dir.exists():
            lbl_count = len(list(lbl_dir.glob("*.txt")))
            img_count = len(list(img_dir.glob("*.jpg")))
            non_empty = sum(1 for f in lbl_dir.glob("*.txt") if f.stat().st_size > 0)
            print(f"    {split}: {img_count} images, {lbl_count} labels ({non_empty} with objects)")

def main():
    print("Phase 1: Data Preprocessing")
    print(f"Target size: {TARGET_SIZE}")
    print(f"Split ratio: {SPLIT_RATIO}")
    print(f"Chunk size: {CHUNK_SIZE} images")
    print(f"JPEG quality: {JPEG_QUALITY}")

    all_datasets = ["lfw", "widerface", "rtts", "bdd100k", "exdark"]
    targets = sys.argv[1:] if len(sys.argv) > 1 else all_datasets

    for name in targets:
        if name not in all_datasets:
            print(f"  [WARNING] Unknown dataset: {name}")
            continue
        try:

            if name == "bdd100k":
                preprocess_bdd100k()
            else:
                preprocess_dataset(name)
            gc.collect()
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
