"""
Phase 1: Dataset Download Script
Downloads public datasets for outdoor object detection and face recognition.

Datasets:
  - LFW   - face recognition benchmark
  - WiderFace - face detection benchmark
  - RTTS  - real-world foggy/hazy outdoor images
  - BDD100K - driving dataset (object detection)
  - exdark key -> LOL dataset: paired low/normal-light images (better for PSNR/SSIM than ExDark)
  - FoggyCityscapes - synthetic fog; requires Cityscapes account or Kaggle auth
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

def download_lfw():
    """Download LFW dataset (~173MB) via sklearn or gdown fallback."""
    print("\n" + "="*60)
    print("DOWNLOADING: LFW (Labeled Faces in the Wild)")
    print("="*60)

    dest_dir = DATASETS_DIR / "lfw"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if (dest_dir / "lfw").exists() and len(list((dest_dir / "lfw").glob("*"))) > 100:
        print("  [SKIP] LFW already downloaded and extracted")
        return

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

    print("  Trying original UMass server...")
    url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    archive_path = dest_dir / "lfw.tgz"
    download_file(url, archive_path, desc="LFW")
    extract_tar(archive_path, dest_dir)

    pairs_url = "http://vis-www.cs.umass.edu/lfw/pairs.txt"
    download_file(pairs_url, dest_dir / "pairs.txt", desc="LFW pairs")

    print("  LFW download complete!")
    print(f"  Location: {dest_dir / 'lfw'}")

def download_widerface():
    """Download WiderFace from the Hugging Face mirror (CUHK-CSE/wider_face).

    Google Drive (gdown) rate-limits aggressively and was the cause of empty
    WiderFace downloads, so Hugging Face is now the PRIMARY source (gdown is
    kept only as a fallback).

    Phase 4 uses a PRETRAINED SCRFD detector, so only WIDER_val + the split
    annotations are needed (~366 MB). WIDER_train.zip (~1.4 GB) is skipped by
    default to respect the compute/disk budget; set the env var WIDERFACE_TRAIN=1
    to also fetch it (only needed if you intend to train a detector yourself).
    """
    print("\n" + "="*60)
    print("DOWNLOADING: WiderFace (Hugging Face mirror)")
    print("="*60)

    dest_dir = DATASETS_DIR / "widerface"
    dest_dir.mkdir(parents=True, exist_ok=True)

    want_train = os.environ.get("WIDERFACE_TRAIN", "0") == "1"

    targets = ["WIDER_val.zip", "wider_face_split.zip"]
    if want_train:
        targets.append("WIDER_train.zip")

    needed = {"WIDER_val", "wider_face_split"} | ({"WIDER_train"} if want_train else set())
    if all((dest_dir / n).exists() for n in needed):
        print("  [SKIP] WiderFace already downloaded and extracted")
        return

    HF_REPO = "CUHK-CSE/wider_face"
    GDRIVE_IDS = {
        "WIDER_train.zip": "15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M",
        "WIDER_val.zip": "1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q",
        "wider_face_split.zip": "1H68E4FCjjLdIny4Gp-6BFYNNSO9eClJq",
    }

    for filename in targets:
        extracted_name = filename.replace(".zip", "")
        if (dest_dir / extracted_name).exists():
            print(f"  [SKIP] {extracted_name} already extracted")
            continue

        src_zip = None

        try:
            from huggingface_hub import hf_hub_download
            print(f"  Downloading {filename} from Hugging Face ({HF_REPO})...")
            src_zip = Path(hf_hub_download(repo_id=HF_REPO, repo_type="dataset",
                                           filename=f"data/{filename}"))
            print(f"    HF OK: {src_zip.stat().st_size / 1e6:.1f} MB")
        except Exception as e:
            print(f"    Hugging Face failed: {e}")

        if src_zip is None or not src_zip.exists():
            try:
                import gdown
                out = dest_dir / filename
                print(f"  Falling back to Google Drive for {filename}...")
                gdown.download(id=GDRIVE_IDS[filename], output=str(out), quiet=False)
                if out.exists() and out.stat().st_size > 100_000:
                    src_zip = out
            except Exception as e:
                print(f"    Google Drive failed: {e}")

        if src_zip is None or not src_zip.exists() or src_zip.stat().st_size < 100_000:
            print(f"  [WARNING] {filename} not downloaded - skipping extraction")
            continue

        try:
            with zipfile.ZipFile(src_zip) as z:
                z.namelist()
        except zipfile.BadZipFile:
            print(f"  [ERROR] {filename} is not a valid zip archive - skipping")
            continue
        extract_zip(src_zip, dest_dir)

        if src_zip.parent == dest_dir:
            try:
                src_zip.unlink()
            except OSError:
                pass

    val_dir = dest_dir / "WIDER_val"
    val_imgs = sum(1 for _ in val_dir.rglob("*.jpg")) if val_dir.exists() else 0
    print(f"\n  WiderFace ready: WIDER_val images = {val_imgs}")
    print(f"  Location: {dest_dir}")
    if val_imgs == 0:
        print("  [WARNING] No WIDER_val images found. Manual fallback:")
        print("    huggingface-cli download CUHK-CSE/wider_face data/WIDER_val.zip \\")
        print("      --repo-type dataset --local-dir datasets/widerface")

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

    if not archive_path.exists():
        try:
            print("  Trying UT Austin Box mirror (official RESIDE-beta)...")
            box_url = "https://utexas.app.box.com/index.php?rm=box_download_shared_file&shared_name=2yekra41udg9rgyzi3ysi513cps621qz&file_id=f_766454923366"
            download_file(box_url, archive_path, desc="RTTS")

            if archive_path.exists() and archive_path.stat().st_size < 10000:
                archive_path.unlink()
                print("  Box download returned an error page, removing...")
        except Exception as e:
            print(f"  Box method failed: {e}")

    if not archive_path.exists():
        try:
            import gdown
            print("  Trying Google Drive (may be rate-limited)...")
            file_id = "1SiMgiAEJqOGoIINrupISLNUcFBJb_3tU"
            gdown.download(id=file_id, output=str(archive_path), quiet=False)
        except Exception as e:
            print(f"  Google Drive method failed: {e}")

    if archive_path.exists() and archive_path.stat().st_size > 1000:
        extract_zip(archive_path, dest_dir)
        print("  RTTS download complete!")
        print(f"  Location: {dest_dir}")
    else:

        if archive_path.exists():
            archive_path.unlink()
        print("  [ERROR] All download methods failed for RTTS.")
        print("  Manual download options:")
        print("    1. Kaggle: https://www.kaggle.com/datasets/tuncnguyn/rtts-dataset")
        print("    2. Dropbox: https://bit.ly/3c4gl3z")
        print(f"    3. Extract to: {dest_dir}")

def download_bdd100k():
    """Download BDD100K with multi-source fallback and integrity validation.

    Phase 3 outdoor detection requires this dataset, so this function hard-fails
    if all sources fail (rather than silently warning and letting Phase 3 fall
    back to COCO128, which is not defensible as outdoor-scene detection).

    Sources tried in order:
      1. Kaggle: solesensei/solesensei_bdd100k (single archive, images + labels)
      2. archive.org: bdd100k_images.zip + bdd100k_labels.zip

    Each archive.org zip is size-checked and header-validated before extraction —
    the prior `size > 10000 bytes` gate accepted HTML error pages as valid 6.5 GB
    files, which is what was causing Phase 3 to silently lose its dataset.
    """
    print("\n" + "="*60)
    print("DOWNLOADING: BDD100K (Berkeley DeepDrive)")
    print("="*60)

    dest_dir = DATASETS_DIR / "bdd100k"
    dest_dir.mkdir(parents=True, exist_ok=True)

    def _state():
        imgs = sum(1 for _ in dest_dir.rglob("*.jpg"))
        jsons = sum(1 for _ in dest_dir.rglob("*.json"))
        return imgs, jsons

    img_count, json_count = _state()
    if img_count > 10000 and json_count > 0:
        print(f"  [SKIP] BDD100K already present ({img_count} images, {json_count} JSONs)")
        return

    try:
        print("  Trying Kaggle (solesensei/solesensei_bdd100k)...")
        import kaggle
        kaggle.api.dataset_download_files(
            "solesensei/solesensei_bdd100k",
            path=str(dest_dir),
            unzip=True,
        )
        img_count, json_count = _state()
        if img_count > 10000 and json_count > 0:
            print(f"  Kaggle download successful ({img_count} images, {json_count} JSONs)")
            print(f"  Location: {dest_dir}")
            return
        print(f"  Kaggle returned {img_count} images, {json_count} JSONs — falling back to archive.org")
    except ImportError:
        print("  [WARNING] kaggle package not installed. Run: pip install kaggle")
    except (Exception, SystemExit) as e:
        print(f"  Kaggle method failed: {e}")

    files = {
        "bdd100k_images.zip": {
            "url": "https://archive.org/download/bdd100k/bdd100k_images.zip",
            "min_bytes": 5 * 1024**3,
        },
        "bdd100k_labels.zip": {
            "url": "https://archive.org/download/bdd100k/bdd100k_labels.zip",
            "min_bytes": 50 * 1024**2,
        },
    }

    for filename, info in files.items():
        archive_path = dest_dir / filename

        if archive_path.exists() and archive_path.stat().st_size < info["min_bytes"]:
            print(f"  Removing undersized {filename} ({archive_path.stat().st_size} bytes)")
            archive_path.unlink()

        if not archive_path.exists():
            print(f"  Downloading {filename} from archive.org...")
            try:
                download_file(info["url"], archive_path, desc=filename)
            except Exception as e:
                print(f"  [ERROR] {filename} download failed: {e}")
                continue

        actual = archive_path.stat().st_size if archive_path.exists() else 0
        if actual < info["min_bytes"]:
            print(f"  [ERROR] {filename}: {actual} bytes < {info['min_bytes']} floor — likely error page")
            if archive_path.exists():
                archive_path.unlink()
            continue

        try:
            with zipfile.ZipFile(archive_path) as z:
                z.namelist()
        except zipfile.BadZipFile:
            print(f"  [ERROR] {filename} is not a valid zip archive")
            archive_path.unlink()
            continue

        try:
            extract_zip(archive_path, dest_dir)
        except Exception as e:
            print(f"  [ERROR] {filename} extraction failed: {e}")

    img_count, json_count = _state()
    if img_count > 10000 and json_count > 0:
        print(f"  BDD100K ready: {img_count} images, {json_count} JSON files")
        print(f"  Location: {dest_dir}")
        return

    raise RuntimeError(
        f"BDD100K download failed across all sources "
        f"(got {img_count} images, {json_count} JSON files). "
        f"Phase 3 outdoor detection requires this dataset. Manual options:\n"
        f"  1. Kaggle: kaggle datasets download -d solesensei/solesensei_bdd100k "
        f"-p datasets/bdd100k --unzip\n"
        f"  2. archive.org: https://archive.org/details/bdd100k\n"
        f"  3. Official (registration): https://bdd-data.berkeley.edu/"
    )

def download_exdark():
    """Download LOL (Low-light) dataset for Phase 2 low-light enhancement evaluation.

    LOL (Wei et al., BMVC 2018) has 500 paired low/normal-light images —
    better than ExDark for this experiment because paired images allow PSNR/SSIM evaluation.

    Sources (in order of reliability):
      1. HuggingFace: geekyrakshit/LoL-Dataset
      2. Kaggle: aryan022/low-light-image-enhancement-dataset
    """
    print("\n" + "="*60)
    print("DOWNLOADING: LOL Dataset (Low-light, paired for PSNR/SSIM)")
    print("="*60)

    dest_dir = DATASETS_DIR / "exdark"
    dest_dir.mkdir(parents=True, exist_ok=True)

    existing = (
        sum(1 for _ in dest_dir.rglob("*.jpg"))
        + sum(1 for _ in dest_dir.rglob("*.png"))
    )
    if existing > 200:
        print(f"  [SKIP] Low-light dataset already present ({existing} images)")
        return

    downloaded = False

    try:
        print("  Trying HuggingFace (geekyrakshit/LoL-Dataset)...")
        from huggingface_hub import snapshot_download
        repo_dir = snapshot_download(
            repo_id="geekyrakshit/LoL-Dataset",
            repo_type="dataset",
            local_dir=str(dest_dir / "lol_hf"),
            ignore_patterns=["*.csv", "*.json", "*.md"],
        )

        for zp in (dest_dir / "lol_hf").rglob("*.zip"):
            try:
                extract_zip(zp, dest_dir)
            except Exception as e:
                print(f"  [WARN] Failed to extract {zp.name}: {e}")
        img_count = sum(1 for _ in dest_dir.rglob("*.png")) + sum(1 for _ in dest_dir.rglob("*.jpg"))
        if img_count > 100:
            downloaded = True
            print(f"  HuggingFace download successful ({img_count} images)")
    except Exception as e:
        print(f"  HuggingFace method failed: {e}")

    if not downloaded:
        try:
            print("  Trying Kaggle (aryan022/low-light-image-enhancement-dataset)...")
            import kaggle
            kaggle.api.dataset_download_files(
                "aryan022/low-light-image-enhancement-dataset",
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

    if downloaded:
        final_count = sum(1 for _ in dest_dir.rglob("*.png")) + sum(1 for _ in dest_dir.rglob("*.jpg"))
        print(f"\n  LOL dataset download complete!")
        print(f"  Images: {final_count} (low-light + reference pairs)")
        print(f"  Location: {dest_dir}")
    else:
        print("\n  [INFO] Automatic download failed. Phase 2 will use synthetic low-light images.")
        print("  Manual options:")
        print("    1. HuggingFace: https://huggingface.co/datasets/geekyrakshit/LoL-Dataset")
        print("    2. Kaggle: https://www.kaggle.com/datasets/aryan022/low-light-image-enhancement-dataset")
        print(f"    3. Extract to: {dest_dir}")

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
