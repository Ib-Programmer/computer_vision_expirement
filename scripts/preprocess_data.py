"""
Phase 1: Data Preprocessing Script
Resizes, normalizes images and splits datasets into train/val/test sets.

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
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"

# Configuration
TARGET_SIZE = (640, 640)
SPLIT_RATIO = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42
CHUNK_SIZE = 200        # images per chunk before gc
JPEG_QUALITY = 90       # save disk space on HDD (100=lossless, 90=good balance)


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

    # Pad to target size (black letterbox)
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

        # Skip if already processed (resume support)
        if out_path.exists():
            skipped += 1
            continue

        try:
            img = cv2.imread(str(img_path))
            if img is None:
                failed += 1
                continue

            img_resized = resize_image(img, TARGET_SIZE)

            # Save with controlled quality
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

            # Release image memory immediately
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

    # Collect paths only (no image data in memory)
    images = collect_images(src_dir)
    if not images:
        print(f"  [SKIP] No images found in {src_dir}")
        return

    print(f"  Found {len(images)} images")
    print(f"  Chunk size: {chunk_size} images | JPEG quality: {JPEG_QUALITY}")

    # Split into train/val/test
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

        # Process in chunks with progress bar
        pbar = tqdm(total=n_images, desc=f"    {split_name}", ncols=80, unit="img")

        for i in range(0, n_images, chunk_size):
            chunk = split_images[i:i + chunk_size]

            p, s, f = process_chunk(chunk, split_dir, JPEG_QUALITY)
            split_processed += p
            split_skipped += s
            split_failed += f

            pbar.update(len(chunk))

            # Free memory after each chunk
            gc.collect()

        pbar.close()

        print(f"    -> processed: {split_processed} | skipped: {split_skipped} | failed: {split_failed}")

        total_processed += split_processed
        total_skipped += split_skipped
        total_failed += split_failed

    # Summary
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


def main():
    print("Phase 1: Data Preprocessing")
    print(f"Target size: {TARGET_SIZE}")
    print(f"Split ratio: {SPLIT_RATIO}")
    print(f"Chunk size: {CHUNK_SIZE} images")
    print(f"JPEG quality: {JPEG_QUALITY}")

    # Allow processing specific datasets via CLI args
    all_datasets = ["lfw", "widerface", "rtts", "bdd100k"]
    targets = sys.argv[1:] if len(sys.argv) > 1 else all_datasets

    for name in targets:
        if name not in all_datasets:
            print(f"  [WARNING] Unknown dataset: {name}")
            continue
        try:
            preprocess_dataset(name)
            gc.collect()  # clean up between datasets
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print("Next: run augment_data.py")


if __name__ == "__main__":
    main()
