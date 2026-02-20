"""
Phase 1: Dataset Download Script
Downloads public datasets for outdoor object detection and face recognition.

Datasets:
  - RTTS (Real-world Task-driven Testing Set) - foggy/hazy outdoor images
  - WiderFace - face detection benchmark
  - LFW (Labeled Faces in the Wild) - face recognition benchmark
  - BDD100K - requires manual download (account needed)
  - ExDark (Exclusively Dark Image Dataset) - low-light object detection
"""

import os
import sys
import zipfile
import tarfile
import urllib.request
import shutil
from pathlib import Path
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, dest_path, desc="Downloading"):
    """Download a file with progress bar."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        print(f"  [SKIP] {dest_path.name} already exists")
        return dest_path

    print(f"  Downloading: {url}")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, dest_path, reporthook=t.update_to)
    print(f"  Saved to: {dest_path}")
    return dest_path


def extract_zip(zip_path, extract_to):
    """Extract a zip archive."""
    print(f"  Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)
    print(f"  Extracted to: {extract_to}")


def extract_tar(tar_path, extract_to):
    """Extract a tar/tar.gz archive."""
    print(f"  Extracting {tar_path.name}...")
    with tarfile.open(tar_path, 'r:*') as t:
        t.extractall(extract_to)
    print(f"  Extracted to: {extract_to}")


# ── 1. LFW (Labeled Faces in the Wild) ──────────────────────────────────────

def download_lfw():
    """Download LFW dataset (~173MB) via sklearn or gdown fallback."""
    print("\n" + "="*60)
    print("DOWNLOADING: LFW (Labeled Faces in the Wild)")
    print("="*60)

    dest_dir = DATASETS_DIR / "lfw"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    if (dest_dir / "lfw").exists() and len(list((dest_dir / "lfw").glob("*"))) > 100:
        print("  [SKIP] LFW already downloaded and extracted")
        return

    # Try sklearn fetch (handles mirrors automatically)
    try:
        print("  Trying sklearn.datasets.fetch_lfw_people...")
        from sklearn.datasets import fetch_lfw_people
        dataset = fetch_lfw_people(data_home=str(dest_dir), download_if_missing=True)
        print(f"  Downloaded via sklearn: {dataset.images.shape[0]} images")
        print("  LFW download complete!")
        print(f"  Location: {dest_dir}")
        return
    except Exception as e:
        print(f"  sklearn method failed: {e}")

    # Fallback: gdown from Google Drive mirror
    try:
        import gdown
        print("  Trying Google Drive mirror...")
        file_id = "1CPSeum3HpopfomUEK1gts2elo4F1Uuey"
        archive_path = dest_dir / "lfw.tgz"
        if not archive_path.exists():
            gdown.download(id=file_id, output=str(archive_path), quiet=False)
        if archive_path.exists():
            extract_tar(archive_path, dest_dir)
            print("  LFW download complete!")
            print(f"  Location: {dest_dir / 'lfw'}")
            return
    except Exception as e:
        print(f"  gdown method failed: {e}")

    # Fallback: direct URL (original server)
    print("  Trying original UMass server...")
    url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    archive_path = dest_dir / "lfw.tgz"
    download_file(url, archive_path, desc="LFW")
    extract_tar(archive_path, dest_dir)

    pairs_url = "http://vis-www.cs.umass.edu/lfw/pairs.txt"
    download_file(pairs_url, dest_dir / "pairs.txt", desc="LFW pairs")

    print("  LFW download complete!")
    print(f"  Location: {dest_dir / 'lfw'}")


# ── 2. WiderFace ────────────────────────────────────────────────────────────

def download_widerface():
    """Download WiderFace dataset via gdown (Google Drive)."""
    print("\n" + "="*60)
    print("DOWNLOADING: WiderFace")
    print("="*60)

    dest_dir = DATASETS_DIR / "widerface"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if (dest_dir / "WIDER_train").exists() and (dest_dir / "WIDER_val").exists():
        print("  [SKIP] WiderFace already downloaded and extracted")
        return

    try:
        import gdown
    except ImportError:
        print("  [ERROR] gdown not installed. Run: pip install gdown")
        return

    # Google Drive file IDs for WiderFace
    files = {
        "WIDER_train.zip": "15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M",
        "WIDER_val.zip": "1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q",
        "wider_face_split.zip": "1H68E4FCjjLdIny4Gp-6BFYNNSO9eClJq",
    }

    # Hugging Face mirror for labels (Google Drive often rate-limits)
    hf_labels_url = "https://huggingface.co/datasets/wider_face/resolve/main/data/wider_face_split.zip"

    for filename, file_id in files.items():
        output_path = dest_dir / filename
        if output_path.exists():
            print(f"  [SKIP] {filename} already exists")
        else:
            downloaded = False

            # For labels zip, try Hugging Face first (more reliable)
            if filename == "wider_face_split.zip":
                try:
                    print(f"  Downloading {filename} from Hugging Face mirror...")
                    download_file(hf_labels_url, output_path, desc=filename)
                    if output_path.exists() and output_path.stat().st_size > 10000:
                        downloaded = True
                    else:
                        if output_path.exists():
                            output_path.unlink()
                except Exception as e:
                    print(f"  Hugging Face mirror failed: {e}")

            # Fallback to Google Drive
            if not downloaded:
                try:
                    print(f"  Downloading {filename} from Google Drive...")
                    gdown.download(id=file_id, output=str(output_path), quiet=False)
                except Exception as e:
                    print(f"  Google Drive download failed for {filename}: {e}")

        # Extract
        if output_path.exists() and output_path.stat().st_size > 10000:
            extracted_name = filename.replace(".zip", "")
            if not (dest_dir / extracted_name).exists():
                extract_zip(output_path, dest_dir)
        elif not output_path.exists():
            print(f"  [WARNING] {filename} not downloaded - skipping extraction")

    print("  WiderFace download complete!")
    print(f"  Location: {dest_dir}")


# ── 3. RTTS (Real-world Task-driven Testing Set) ────────────────────────────

def download_rtts():
    """Download RTTS dataset for hazy/foggy images."""
    print("\n" + "="*60)
    print("DOWNLOADING: RTTS (Real-world Task-driven Testing Set)")
    print("="*60)

    dest_dir = DATASETS_DIR / "rtts"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if (dest_dir / "RTTS").exists() or len(list(dest_dir.glob("*.png"))) > 100:
        print("  [SKIP] RTTS already downloaded")
        return

    archive_path = dest_dir / "RTTS.zip"

    # Method 1: Kaggle (if authenticated)
    if not archive_path.exists():
        try:
            print("  Trying Kaggle API (tuncnguyn/rtts-dataset)...")
            import kaggle
            kaggle.api.dataset_download_files(
                "tuncnguyn/rtts-dataset", path=str(dest_dir), unzip=True
            )
            print("  Downloaded via Kaggle!")
            print(f"  Location: {dest_dir}")
            return
        except (Exception, SystemExit) as e:
            print(f"  Kaggle method failed: {e}")

    # Method 2: UT Austin Box mirror (official RESIDE-beta link)
    if not archive_path.exists():
        try:
            print("  Trying UT Austin Box mirror (official RESIDE-beta)...")
            box_url = "https://utexas.app.box.com/index.php?rm=box_download_shared_file&shared_name=2yekra41udg9rgyzi3ysi513cps621qz&file_id=f_766454923366"
            download_file(box_url, archive_path, desc="RTTS")
            # Verify it's actually a zip, not an HTML error page
            if archive_path.exists() and archive_path.stat().st_size < 10000:
                archive_path.unlink()
                print("  Box download returned an error page, removing...")
        except Exception as e:
            print(f"  Box method failed: {e}")

    # Method 3: gdown from Google Drive (original, may be rate-limited)
    if not archive_path.exists():
        try:
            import gdown
            print("  Trying Google Drive (may be rate-limited)...")
            file_id = "1SiMgiAEJqOGoIINrupISLNUcFBJb_3tU"
            gdown.download(id=file_id, output=str(archive_path), quiet=False)
        except Exception as e:
            print(f"  Google Drive method failed: {e}")

    # Extract if we got the archive
    if archive_path.exists() and archive_path.stat().st_size > 1000:
        extract_zip(archive_path, dest_dir)
        print("  RTTS download complete!")
        print(f"  Location: {dest_dir}")
    else:
        # Clean up partial/empty file
        if archive_path.exists():
            archive_path.unlink()
        print("  [ERROR] All download methods failed for RTTS.")
        print("  Manual download options:")
        print("    1. Kaggle: https://www.kaggle.com/datasets/tuncnguyn/rtts-dataset")
        print("    2. Dropbox: https://bit.ly/3c4gl3z")
        print(f"    3. Extract to: {dest_dir}")


# ── 4. BDD100K ──────────────────────────────────────────────────────────────

def download_bdd100k():
    """Download BDD100K from archive.org mirror (~6.5GB images + 147MB labels)."""
    print("\n" + "="*60)
    print("DOWNLOADING: BDD100K (Berkeley DeepDrive)")
    print("="*60)

    dest_dir = DATASETS_DIR / "bdd100k"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if sum(1 for _ in dest_dir.rglob("*.jpg")) > 1000:
        print("  [SKIP] BDD100K images already downloaded")
        return

    # Archive.org mirror - no registration needed
    files = {
        "bdd100k_images.zip": {
            "url": "https://archive.org/download/bdd100k/bdd100k_images.zip",
            "desc": "BDD100K Images (~6.5GB)",
        },
        "bdd100k_labels.zip": {
            "url": "https://archive.org/download/bdd100k/bdd100k_labels.zip",
            "desc": "BDD100K Labels (~147MB)",
        },
    }

    for filename, info in files.items():
        archive_path = dest_dir / filename

        if archive_path.exists():
            print(f"  [SKIP] {filename} already downloaded")
        else:
            print(f"  Downloading {info['desc']}...")
            try:
                download_file(info["url"], archive_path, desc=filename)
            except Exception as e:
                print(f"  [ERROR] Failed to download {filename}: {e}")
                continue

        # Extract
        if archive_path.exists() and archive_path.stat().st_size > 10000:
            try:
                extract_zip(archive_path, dest_dir)
            except Exception as e:
                print(f"  [ERROR] Failed to extract {filename}: {e}")

    # Verify
    # Verify - archive.org extracts to bdd100k/bdd100k/images/ or bdd100k/images/
    img_count = sum(1 for _ in dest_dir.rglob("*.jpg"))
    if img_count > 100:
        print(f"  BDD100K download complete! ({img_count} images found)")
        print(f"  Location: {dest_dir}")
    else:
        print("  [WARNING] Download may have failed. Manual fallback:")
        print("    https://archive.org/download/bdd100k")
        print(f"    Extract to: {dest_dir}")


# ── 5. ExDark (Exclusively Dark Image Dataset) ───────────────────────────────

def download_exdark():
    """Download ExDark dataset for low-light object detection.

    ExDark contains 7,363 low-light images across 12 object classes:
    Bicycle, Boat, Bottle, Bus, Car, Cat, Chair, Cup, Dog, Motorbike, People, Table.

    Sources:
      - GitHub: https://github.com/cs-chan/Exclusively-Dark-Image-Dataset
      - Kaggle: soumikrakshit/exclusively-dark-image-dataset
    """
    print("\n" + "="*60)
    print("DOWNLOADING: ExDark (Exclusively Dark Image Dataset)")
    print("="*60)

    dest_dir = DATASETS_DIR / "exdark"
    dest_dir.mkdir(parents=True, exist_ok=True)

    EXDARK_CLASSES = [
        "Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat",
        "Chair", "Cup", "Dog", "Motorbike", "People", "Table",
    ]

    # Check if already downloaded (ExDark has ~7363 images across 12 class folders)
    existing_images = (
        sum(1 for _ in dest_dir.rglob("*.jpg"))
        + sum(1 for _ in dest_dir.rglob("*.png"))
        + sum(1 for _ in dest_dir.rglob("*.JPEG"))
        + sum(1 for _ in dest_dir.rglob("*.JPG"))
        + sum(1 for _ in dest_dir.rglob("*.PNG"))
    )
    if existing_images > 5000:
        print(f"  [SKIP] ExDark already downloaded ({existing_images} images found)")
        return

    downloaded = False

    # ── Method 1: GitHub releases ──
    github_files = {
        "ExDark.tar": "https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/releases/download/Img/ExDark.tar",
        "ExDark_Annno.tar": "https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/releases/download/Annno/ExDark_Annno.tar",
    }

    try:
        print("  Trying GitHub releases...")
        all_ok = True
        for filename, url in github_files.items():
            archive_path = dest_dir / filename
            if archive_path.exists() and archive_path.stat().st_size > 10000:
                print(f"  [SKIP] {filename} already downloaded")
            else:
                try:
                    download_file(url, archive_path, desc=filename)
                    if not archive_path.exists() or archive_path.stat().st_size < 10000:
                        print(f"  [WARNING] {filename} download appears too small or failed")
                        if archive_path.exists():
                            archive_path.unlink()
                        all_ok = False
                except Exception as e:
                    print(f"  GitHub download failed for {filename}: {e}")
                    if archive_path.exists():
                        archive_path.unlink()
                    all_ok = False

        # Extract archives
        for filename in github_files:
            archive_path = dest_dir / filename
            if archive_path.exists() and archive_path.stat().st_size > 10000:
                try:
                    extract_tar(archive_path, dest_dir)
                except Exception as e:
                    print(f"  [ERROR] Failed to extract {filename}: {e}")
                    all_ok = False

        # Check if images were extracted
        img_count = (
            sum(1 for _ in dest_dir.rglob("*.jpg"))
            + sum(1 for _ in dest_dir.rglob("*.png"))
            + sum(1 for _ in dest_dir.rglob("*.JPEG"))
            + sum(1 for _ in dest_dir.rglob("*.JPG"))
            + sum(1 for _ in dest_dir.rglob("*.PNG"))
        )
        if img_count > 1000:
            downloaded = True
            print(f"  GitHub download successful ({img_count} images extracted)")

    except Exception as e:
        print(f"  GitHub method failed: {e}")

    # ── Method 2: Kaggle fallback ──
    if not downloaded:
        try:
            print("  Trying Kaggle (soumikrakshit/exclusively-dark-image-dataset)...")
            import kaggle
            kaggle.api.dataset_download_files(
                "soumikrakshit/exclusively-dark-image-dataset",
                path=str(dest_dir),
                unzip=True,
            )
            img_count = (
                sum(1 for _ in dest_dir.rglob("*.jpg"))
                + sum(1 for _ in dest_dir.rglob("*.png"))
                + sum(1 for _ in dest_dir.rglob("*.JPEG"))
                + sum(1 for _ in dest_dir.rglob("*.JPG"))
                + sum(1 for _ in dest_dir.rglob("*.PNG"))
            )
            if img_count > 1000:
                downloaded = True
                print(f"  Kaggle download successful ({img_count} images extracted)")
            else:
                print("  [WARNING] Kaggle download produced too few images")
        except ImportError:
            print("  [WARNING] kaggle package not installed. Run: pip install kaggle")
        except (Exception, SystemExit) as e:
            print(f"  Kaggle method failed: {e}")

    # ── Verify ──
    if downloaded:
        final_count = (
            sum(1 for _ in dest_dir.rglob("*.jpg"))
            + sum(1 for _ in dest_dir.rglob("*.png"))
            + sum(1 for _ in dest_dir.rglob("*.JPEG"))
            + sum(1 for _ in dest_dir.rglob("*.JPG"))
            + sum(1 for _ in dest_dir.rglob("*.PNG"))
        )
        anno_count = sum(1 for _ in dest_dir.rglob("*.txt"))
        print(f"\n  ExDark download complete!")
        print(f"  Images: {final_count}")
        print(f"  Annotation files: {anno_count}")
        print(f"  Classes: {', '.join(EXDARK_CLASSES)}")
        print(f"  Location: {dest_dir}")
    else:
        print("\n  [ERROR] All download methods failed for ExDark.")
        print("  Manual download options:")
        print("    1. GitHub: https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/releases")
        print("    2. Kaggle: https://www.kaggle.com/datasets/soumikrakshit/exclusively-dark-image-dataset")
        print(f"    3. Extract to: {dest_dir}")


# ── 6. FoggyCityscapes ─────────────────────────────────────────────────────

def download_foggy_cityscapes():
    """Download FoggyCityscapes dataset for foggy scene detection.

    FoggyCityscapes applies synthetic fog to Cityscapes images at three
    density levels (beta = 0.005, 0.01, 0.02).

    Sources:
      - Official: https://www.cityscapes-dataset.com/foggydownload/ (requires registration)
      - Kaggle subset: yessicatuteja/foggy-cityscapes-image-dataset
    """
    print("\n" + "="*60)
    print("DOWNLOADING: FoggyCityscapes")
    print("="*60)

    dest_dir = DATASETS_DIR / "foggy_cityscapes"
    dest_dir.mkdir(parents=True, exist_ok=True)

    existing = sum(1 for _ in dest_dir.rglob("*.png")) + sum(1 for _ in dest_dir.rglob("*.jpg"))
    if existing > 500:
        print(f"  [SKIP] FoggyCityscapes already downloaded ({existing} images found)")
        return

    downloaded = False

    # Method 1: Kaggle dataset (subset, no registration needed)
    try:
        print("  Trying Kaggle (yessicatuteja/foggy-cityscapes-image-dataset)...")
        import kaggle
        kaggle.api.dataset_download_files(
            "yessicatuteja/foggy-cityscapes-image-dataset",
            path=str(dest_dir),
            unzip=True,
        )
        img_count = sum(1 for _ in dest_dir.rglob("*.png")) + sum(1 for _ in dest_dir.rglob("*.jpg"))
        if img_count > 100:
            downloaded = True
            print(f"  Kaggle download successful ({img_count} images)")
    except ImportError:
        print("  [WARNING] kaggle package not installed. Run: pip install kaggle")
    except (Exception, SystemExit) as e:
        print(f"  Kaggle method failed: {e}")

    # Method 2: Direct synthesis from Cityscapes (if available)
    if not downloaded:
        cityscapes_dir = DATASETS_DIR / "cityscapes"
        if cityscapes_dir.exists():
            print("  Found Cityscapes locally. You can generate foggy images using:")
            print("    https://github.com/sakaridis/fog_simulation_DBF")
        else:
            print("  No local Cityscapes found.")

    if downloaded:
        final_count = sum(1 for _ in dest_dir.rglob("*.png")) + sum(1 for _ in dest_dir.rglob("*.jpg"))
        print(f"\n  FoggyCityscapes download complete!")
        print(f"  Images: {final_count}")
        print(f"  Location: {dest_dir}")
    else:
        print("\n  [INFO] FoggyCityscapes requires Cityscapes account for full dataset.")
        print("  Options:")
        print("    1. Register at: https://www.cityscapes-dataset.com/login/")
        print("    2. Download foggy images: https://www.cityscapes-dataset.com/foggydownload/")
        print("    3. Kaggle subset: https://www.kaggle.com/datasets/yessicatuteja/foggy-cityscapes-image-dataset")
        print(f"    4. Extract to: {dest_dir}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Phase 1: Dataset Download")
    print(f"Base directory: {BASE_DIR}")
    print(f"Datasets directory: {DATASETS_DIR}")

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "lfw": download_lfw,
        "widerface": download_widerface,
        "rtts": download_rtts,
        "bdd100k": download_bdd100k,
        "exdark": download_exdark,
        "foggy_cityscapes": download_foggy_cityscapes,
    }

    # Allow downloading specific datasets via CLI args
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(datasets.keys())

    for name in targets:
        if name in datasets:
            try:
                datasets[name]()
            except Exception as e:
                print(f"\n  [ERROR] Failed to download {name}: {e}")
                print("  You can retry this dataset later.")
        else:
            print(f"\n  [WARNING] Unknown dataset: {name}")
            print(f"  Available: {', '.join(datasets.keys())}")

    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    for name in targets:
        d = DATASETS_DIR / name
        if d.exists():
            count = sum(1 for _ in d.rglob("*.jpg")) + sum(1 for _ in d.rglob("*.png"))
            print(f"  {name:12s} -> {count:>6d} images found")
        else:
            print(f"  {name:12s} -> NOT DOWNLOADED")

    print("\nDone! Next: run preprocess_data.py")


if __name__ == "__main__":
    main()
