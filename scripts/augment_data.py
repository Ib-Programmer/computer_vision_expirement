"""
Phase 1: Data Augmentation Script
Applies outdoor-condition augmentations: fog, low-light, motion blur, rain.
Saves augmented copies alongside originals for controlled experiments.

Optimized for low-resource machines (16GB RAM, HDD):
  - Processes images in small chunks with gc between chunks
  - Skips already-augmented images for resume support
  - Controlled JPEG quality to save disk space
  - CLI args to target specific datasets
"""

import gc
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import albumentations as A

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
OUTPUT_DIR = BASE_DIR / "outputs" / "augmented"

CHUNK_SIZE = 150        # images per chunk before gc
JPEG_QUALITY = 90       # save disk space on HDD


# ── Augmentation Transforms ─────────────────────────────────────────────────

def get_fog_transform():
    return A.Compose([
        A.RandomFog(
            fog_coef_lower=0.3,
            fog_coef_upper=0.7,
            alpha_coef=0.08,
            p=1.0
        )
    ])


def get_low_light_transform():
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=(-0.5, -0.2),
            contrast_limit=(-0.3, 0.0),
            p=1.0
        ),
        A.RandomGamma(gamma_limit=(150, 300), p=0.7),
    ])


def get_motion_blur_transform():
    return A.Compose([
        A.MotionBlur(blur_limit=(7, 15), p=1.0)
    ])


def get_rain_transform():
    return A.Compose([
        A.RandomRain(
            slant_lower=-10,
            slant_upper=10,
            drop_length=20,
            drop_width=1,
            drop_color=(200, 200, 200),
            blur_value=3,
            brightness_coefficient=0.7,
            p=1.0
        )
    ])


def get_combined_transform():
    return A.Compose([
        A.OneOf([
            A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.5, alpha_coef=0.08, p=1.0),
            A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1,
                         drop_color=(200, 200, 200), blur_value=3, brightness_coefficient=0.7, p=1.0),
        ], p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=(-0.4, -0.1), contrast_limit=(-0.2, 0.0), p=1.0),
            A.MotionBlur(blur_limit=(7, 12), p=1.0),
        ], p=0.5),
        A.GaussNoise(var_limit=(10.0, 40.0), p=0.3),
    ])


AUGMENTATIONS = {
    "fog": get_fog_transform(),
    "low_light": get_low_light_transform(),
    "motion_blur": get_motion_blur_transform(),
    "rain": get_rain_transform(),
    "combined": get_combined_transform(),
}


# ── Chunk-based Augmentation ────────────────────────────────────────────────

def augment_chunk(chunk, transform, aug_name, out_dir, jpeg_quality=JPEG_QUALITY):
    """Augment a chunk of images. Returns (processed, skipped, failed)."""
    processed = 0
    skipped = 0
    failed = 0

    for img_path in chunk:
        out_path = out_dir / f"{img_path.stem}_{aug_name}{img_path.suffix}"

        # Skip if already augmented (resume support)
        if out_path.exists():
            skipped += 1
            continue

        try:
            img = cv2.imread(str(img_path))
            if img is None:
                failed += 1
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            augmented = transform(image=img_rgb)["image"]
            aug_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

            ext = out_path.suffix.lower()
            if ext in (".jpg", ".jpeg"):
                cv2.imwrite(str(out_path), aug_bgr,
                            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            else:
                cv2.imwrite(str(out_path), aug_bgr)

            processed += 1

            del img, img_rgb, augmented, aug_bgr

        except Exception:
            failed += 1

    return processed, skipped, failed


def augment_dataset(dataset_name, aug_types=None, chunk_size=CHUNK_SIZE):
    """Apply augmentations to a preprocessed dataset in chunks."""
    src_dir = DATASETS_DIR / f"{dataset_name}_processed"

    if not src_dir.exists():
        print(f"  [SKIP] {dataset_name}: preprocessed data not found at {src_dir}")
        print(f"         Run preprocess_data.py first.")
        return

    if aug_types is None:
        aug_types = list(AUGMENTATIONS.keys())

    print(f"\n{'='*60}")
    print(f"AUGMENTING: {dataset_name}")
    print(f"{'='*60}")

    for split in ["train"]:  # Only augment training set
        split_dir = src_dir / split
        if not split_dir.exists():
            continue

        images = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
        print(f"  Split: {split} ({len(images)} images)")
        print(f"  Chunk size: {chunk_size} | Augmentations: {aug_types}")

        for aug_name in aug_types:
            transform = AUGMENTATIONS[aug_name]
            out_dir = OUTPUT_DIR / dataset_name / split / aug_name
            out_dir.mkdir(parents=True, exist_ok=True)

            existing = len(list(out_dir.glob("*")))
            if existing >= len(images):
                print(f"    [SKIP] {aug_name}: already done ({existing} files)")
                continue

            n_images = len(images)
            n_chunks = (n_images + chunk_size - 1) // chunk_size

            print(f"    {aug_name}: {n_images} images in {n_chunks} chunks")

            total_p, total_s, total_f = 0, 0, 0
            pbar = tqdm(total=n_images, desc=f"      {aug_name}", ncols=80, unit="img")

            for i in range(0, n_images, chunk_size):
                chunk = images[i:i + chunk_size]
                p, s, f = augment_chunk(chunk, transform, aug_name, out_dir)
                total_p += p
                total_s += s
                total_f += f
                pbar.update(len(chunk))
                gc.collect()

            pbar.close()
            print(f"      -> new: {total_p} | skipped: {total_s} | failed: {total_f}")

    print(f"  Augmentation complete for {dataset_name}")


def create_comparison_grid(dataset_name, num_samples=4):
    """Create a visual comparison grid: original vs each augmentation."""
    src_dir = DATASETS_DIR / f"{dataset_name}_processed" / "train"
    if not src_dir.exists():
        print(f"  Cannot create grid: {src_dir} not found")
        return

    images = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
    if not images:
        return

    samples = images[:num_samples]
    aug_names = list(AUGMENTATIONS.keys())

    grid_dir = OUTPUT_DIR / dataset_name / "comparison"
    grid_dir.mkdir(parents=True, exist_ok=True)

    for idx, img_path in enumerate(samples):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        row_images = [img]

        for aug_name in aug_names:
            transform = AUGMENTATIONS[aug_name]
            augmented = transform(image=img_rgb)["image"]
            aug_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
            row_images.append(aug_bgr)

        h, w = 320, 320
        resized = [cv2.resize(im, (w, h)) for im in row_images]
        grid_row = np.hstack(resized)

        out_path = grid_dir / f"comparison_{idx}.jpg"
        cv2.imwrite(str(out_path), grid_row)

        del img, img_rgb, row_images, resized, grid_row
        gc.collect()

    print(f"  Comparison grids saved to {grid_dir}")


def main():
    print("Phase 1: Data Augmentation")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Chunk size: {CHUNK_SIZE} images")
    print(f"Augmentation types: {list(AUGMENTATIONS.keys())}")

    all_datasets = ["lfw", "widerface", "rtts", "bdd100k"]
    targets = sys.argv[1:] if len(sys.argv) > 1 else all_datasets

    for name in targets:
        if name not in all_datasets:
            print(f"  [WARNING] Unknown dataset: {name}")
            continue
        try:
            augment_dataset(name)
            create_comparison_grid(name)
            gc.collect()
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    print("\n" + "="*60)
    print("AUGMENTATION COMPLETE")
    print("="*60)
    print("Next: run dataset_stats.py to verify everything")


if __name__ == "__main__":
    main()
