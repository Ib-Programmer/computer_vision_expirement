"""
Phase 1: Dataset Statistics & Verification Script
Reports dataset stats: image counts, dimensions, class distribution, condition breakdown.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
OUTPUT_DIR = BASE_DIR / "outputs" / "augmented"


def count_images(directory):
    """Count image files in a directory."""
    if not directory.exists():
        return 0
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    count = 0
    for ext in extensions:
        count += len(list(directory.rglob(f"*{ext}")))
        count += len(list(directory.rglob(f"*{ext.upper()}")))
    return count


def get_image_stats(directory, sample_size=100):
    """Get image dimension statistics from a sample."""
    if not directory.exists():
        return None

    images = list(directory.rglob("*.jpg")) + list(directory.rglob("*.png"))
    if not images:
        return None

    sample = images[:min(sample_size, len(images))]
    heights, widths = [], []

    for img_path in sample:
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            heights.append(h)
            widths.append(w)

    if not heights:
        return None

    return {
        "min_size": f"{min(widths)}x{min(heights)}",
        "max_size": f"{max(widths)}x{max(heights)}",
        "avg_size": f"{int(np.mean(widths))}x{int(np.mean(heights))}",
    }


def report_raw_datasets():
    """Report statistics for raw downloaded datasets."""
    print("\n" + "="*60)
    print("RAW DATASET STATISTICS")
    print("="*60)

    datasets = ["lfw", "widerface", "rtts", "bdd100k"]

    for name in datasets:
        d = DATASETS_DIR / name
        count = count_images(d)
        stats = get_image_stats(d)

        print(f"\n  {name.upper()}")
        print(f"    Path:   {d}")
        print(f"    Exists: {d.exists()}")
        print(f"    Images: {count}")
        if stats:
            print(f"    Min size: {stats['min_size']}")
            print(f"    Max size: {stats['max_size']}")
            print(f"    Avg size: {stats['avg_size']}")

        # Count subdirectories
        if d.exists():
            subdirs = [x for x in d.iterdir() if x.is_dir()]
            if subdirs:
                print(f"    Subdirs: {len(subdirs)}")
                for sd in subdirs[:5]:
                    sd_count = count_images(sd)
                    print(f"      - {sd.name}: {sd_count} images")
                if len(subdirs) > 5:
                    print(f"      ... and {len(subdirs) - 5} more")


def report_processed_datasets():
    """Report statistics for preprocessed datasets."""
    print("\n" + "="*60)
    print("PREPROCESSED DATASET STATISTICS")
    print("="*60)

    datasets = ["lfw", "widerface", "rtts", "bdd100k"]

    for name in datasets:
        d = DATASETS_DIR / f"{name}_processed"
        if not d.exists():
            print(f"\n  {name.upper()}_PROCESSED: not found (run preprocess_data.py)")
            continue

        print(f"\n  {name.upper()}_PROCESSED")
        print(f"    Path: {d}")

        total = 0
        for split in ["train", "val", "test"]:
            split_dir = d / split
            count = count_images(split_dir)
            total += count
            print(f"    {split:5s}: {count:>6d} images")

        print(f"    {'total':5s}: {total:>6d} images")


def report_augmented_datasets():
    """Report statistics for augmented datasets."""
    print("\n" + "="*60)
    print("AUGMENTED DATASET STATISTICS")
    print("="*60)

    if not OUTPUT_DIR.exists():
        print("  No augmented data found (run augment_data.py)")
        return

    datasets = ["lfw", "widerface", "rtts", "bdd100k"]
    aug_types = ["fog", "low_light", "motion_blur", "rain", "combined"]

    for name in datasets:
        d = OUTPUT_DIR / name
        if not d.exists():
            continue

        print(f"\n  {name.upper()} AUGMENTED")

        for split in ["train"]:
            split_dir = d / split
            if not split_dir.exists():
                continue

            for aug in aug_types:
                aug_dir = split_dir / aug
                count = count_images(aug_dir)
                if count > 0:
                    print(f"    {split}/{aug:12s}: {count:>6d} images")


def report_summary():
    """Print overall summary."""
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)

    total_raw = 0
    total_processed = 0
    total_augmented = 0

    for name in ["lfw", "widerface", "rtts", "bdd100k"]:
        total_raw += count_images(DATASETS_DIR / name)
        total_processed += count_images(DATASETS_DIR / f"{name}_processed")
        total_augmented += count_images(OUTPUT_DIR / name) if (OUTPUT_DIR / name).exists() else 0

    print(f"  Raw images:         {total_raw:>8d}")
    print(f"  Preprocessed images:{total_processed:>8d}")
    print(f"  Augmented images:   {total_augmented:>8d}")
    print(f"  Total images:       {total_raw + total_processed + total_augmented:>8d}")

    # Check readiness for Phase 2
    print("\n  Phase 1 Readiness Check:")
    checks = {
        "Raw datasets downloaded": total_raw > 0,
        "Data preprocessed": total_processed > 0,
        "Augmentations applied": total_augmented > 0,
    }
    for check, passed in checks.items():
        status = "PASS" if passed else "PENDING"
        print(f"    [{status:7s}] {check}")


def main():
    print("Phase 1: Dataset Statistics & Verification")
    print(f"Datasets directory: {DATASETS_DIR}")

    report_raw_datasets()
    report_processed_datasets()
    report_augmented_datasets()
    report_summary()


if __name__ == "__main__":
    main()
