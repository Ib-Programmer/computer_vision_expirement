"""
Patch Phase2_Image_Enhancement.ipynb — fix 5 bugs found from Colab output:

1. Cell 06: synthetic_pairs=0 — search more dirs (raw RTTS, not only _processed/test)
2. Cell 06: albumentations deprecation — fog_coef_lower/upper -> fog_coef_range,
            slant_lower/upper -> slant_range
3. Cell 05: HF LOL snapshot returns 0 images — search all subdirs, remove 'low' filter
4. Cell 08: ZeroDCE++ training fallback finds 0 images — add RTTS raw dir to search
5. Cell 13: print bug — model_dir printed outside loop shows last value (FFA-Net)
"""

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "Phase2_Image_Enhancement.ipynb"
nb = json.load(open(NB_PATH, encoding="utf-8"))
cells = nb["cells"]

# ── Fix 1 & 2: Cell 06 — synthetic pairs ──────────────────────────────────
# Find by content: looks for 'synthetic_pairs' and 'albumentations'
c6_idx = None
for i, c in enumerate(cells):
    src = "".join(c["source"])
    if "synthetic_pairs" in src and "fog_coef_lower" in src:
        c6_idx = i
        break

if c6_idx is not None:
    cells[c6_idx]["source"] = ["""\
# Build synthetic paired test set for PSNR/SSIM evaluation.
# Clean source: search multiple dirs (raw + preprocessed) to find real outdoor images.
# Degrade synthetically -> we have ground-truth clean reference for PSNR/SSIM.
import albumentations as A
import glob
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

DATASETS_DIR = '/content/computer_vision_expirement/datasets'

# Search order: preprocessed splits first, then raw download dirs
search_dirs = [
    f'{DATASETS_DIR}/rtts_processed/test',
    f'{DATASETS_DIR}/rtts_processed/val',
    f'{DATASETS_DIR}/rtts_processed/train',
    f'{DATASETS_DIR}/rtts',           # raw RTTS download (always present after Cell 02)
    f'{DATASETS_DIR}/lfw_processed/test',
    f'{DATASETS_DIR}/lfw_processed/val',
    f'{DATASETS_DIR}/widerface_processed/test',
]

clean_sources = []
for d in search_dirs:
    found = (glob.glob(f'{d}/*.jpg') + glob.glob(f'{d}/*.png') +
             glob.glob(f'{d}/**/*.jpg', recursive=True) +
             glob.glob(f'{d}/**/*.png', recursive=True))
    found = sorted(set(found))
    if found:
        clean_sources.extend(found[:20 - len(clean_sources)])
        if len(clean_sources) >= 20:
            break

# Last resort: use paths from already-loaded foggy_images (real RTTS images)
if not clean_sources:
    clean_sources = [p for p, _ in globals().get('foggy_images', [])[:20]]

# ── Albumentations 1.4+ API (use *_range tuples instead of *_lower/*_upper) ──
def _make_fog():
    try:
        return A.RandomFog(fog_coef_range=(0.4, 0.7), alpha_coef=0.08, p=1.0)
    except TypeError:
        return A.RandomFog(fog_coef_lower=0.4, fog_coef_upper=0.7, alpha_coef=0.08, p=1.0)

def _make_rain():
    try:
        return A.RandomRain(slant_range=(-10, 10), drop_length=20, drop_width=1,
                            drop_color=(200, 200, 200), blur_value=3,
                            brightness_coefficient=0.7, p=1.0)
    except TypeError:
        return A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1,
                            drop_color=(200, 200, 200), blur_value=3,
                            brightness_coefficient=0.7, p=1.0)

deg_transforms = {
    'fog':       A.Compose([_make_fog()]),
    'low_light': A.Compose([A.RandomBrightnessContrast(
                     brightness_limit=(-0.5, -0.3), contrast_limit=(-0.3, 0.0), p=1.0)]),
    'rain':      A.Compose([_make_rain()]),
}

synthetic_pairs = []   # list of (clean_bgr, degraded_bgr, condition_name)
failed = 0
for img_path in clean_sources[:20]:
    img = cv2.imread(img_path)
    if img is None:
        failed += 1
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for cond, transform in deg_transforms.items():
        try:
            deg_rgb = transform(image=img_rgb)['image']
            deg_bgr = cv2.cvtColor(deg_rgb, cv2.COLOR_RGB2BGR)
            synthetic_pairs.append((img, deg_bgr, cond))
        except Exception as e:
            print(f'  [WARN] {cond} augmentation failed: {e}')

print(f'Synthetic paired images : {len(synthetic_pairs)}')
print(f'  ({len(clean_sources)} clean sources x {len(deg_transforms)} conditions, {failed} read failures)')
if clean_sources:
    print(f'  Source dir: {clean_sources[0][:70]}...')
print('PSNR/SSIM will compare enhanced output against the clean original.')
"""]
    print(f"Fixed Cell {c6_idx}: synthetic pairs + albumentations deprecation")
else:
    print("WARNING: Cell 06 (synthetic pairs) not found")

# ── Fix 3: Cell 05 — HuggingFace LOL snapshot returns 0 images ────────────
# Find the cell with the HuggingFace LOL snapshot_download block
c5_idx = None
for i, c in enumerate(cells):
    src = "".join(c["source"])
    if "geekyrakshit/LoL-Dataset" in src and "low_paths" in src:
        c5_idx = i
        break

if c5_idx is not None:
    old_src = "".join(cells[c5_idx]["source"])
    # Replace the HF LoL block only (between the HF fallback comment and end of that block)
    old_block = """\
if not exdark_images:
    print("[FALLBACK] Trying HuggingFace for low-light images (LoL-Dataset)...")
    try:
        from huggingface_hub import snapshot_download
        repo_dir = snapshot_download(repo_id='geekyrakshit/LoL-Dataset',
                                     repo_type='dataset', ignore_patterns=['*.csv', '*.json'])
        # Collect all images in the repo; prefer paths containing 'low'
        all_repo_imgs = []
        for ext in ['jpg', 'png', 'JPG', 'PNG', 'jpeg', 'JPEG']:
            all_repo_imgs.extend(glob.glob(f'{repo_dir}/**/*.{ext}', recursive=True))
        # Prefer low-light images (path contains 'low' but not 'highlight')
        low_paths = [p for p in all_repo_imgs
                     if 'low' in p.lower() and 'highlight' not in p.lower()]
        chosen = (low_paths if low_paths else all_repo_imgs)[:200]
        exdark_images = []
        for p in chosen:
            img = cv2.imread(p)
            if img is not None:
                exdark_images.append((p, img))
        print(f"  HuggingFace LOL: {len(exdark_images)} low-light images")
    except Exception as e:
        print(f"  HuggingFace LoL failed: {e}")"""
    new_block = """\
if not exdark_images:
    print("[FALLBACK] Trying HuggingFace for low-light images (LoL-Dataset)...")
    try:
        from huggingface_hub import snapshot_download
        repo_dir = snapshot_download(repo_id='geekyrakshit/LoL-Dataset',
                                     repo_type='dataset', ignore_patterns=['*.csv', '*.json'])
        # Collect ALL image files — repo structure varies, don't filter by path name
        all_repo_imgs = []
        for ext in ['jpg', 'png', 'JPG', 'PNG', 'jpeg', 'JPEG']:
            all_repo_imgs.extend(glob.glob(f'{repo_dir}/**/*.{ext}', recursive=True))
        # Prefer low-light images (path contains 'low' but not 'high')
        low_paths = [p for p in all_repo_imgs if 'low' in p.lower() and '/high' not in p.lower()]
        chosen = (low_paths if low_paths else all_repo_imgs)[:200]
        exdark_images = []
        for p in chosen:
            img = cv2.imread(p)
            if img is not None:
                exdark_images.append((p, img))
        if not exdark_images and all_repo_imgs:
            print(f"  NOTE: snapshot has {len(all_repo_imgs)} files but cv2 could not read them.")
            print(f"  Sample paths: {all_repo_imgs[:3]}")
        print(f"  HuggingFace LOL: {len(exdark_images)} low-light images")
    except Exception as e:
        print(f"  HuggingFace LoL failed: {e}")"""

    if old_block in old_src:
        cells[c5_idx]["source"] = [old_src.replace(old_block, new_block)]
        print(f"Fixed Cell {c5_idx}: HF LOL snapshot image search")
    else:
        print("WARNING: HF LOL block not found in Cell 05 (may already be fixed)")
else:
    print("WARNING: Cell 05 not found")

# ── Fix 4: Cell 08 — ZeroDCE++ training fallback finds 0 images ───────────
c8_idx = None
for i, c in enumerate(cells):
    src = "".join(c["source"])
    if "ZeroDCEpp" in src and "train_images" in src and "lol_dir" in src:
        c8_idx = i
        break

if c8_idx is not None:
    old_src = "".join(cells[c8_idx]["source"])
    # Fix the fallback training block — add RTTS raw dirs to search
    old_fallback = """\
    train_images = []
    for d in [lol_dir, '/content/LOL']:
        if os.path.exists(d):
            for p in (_glob.glob(f'{d}/**/*.png', recursive=True) +
                      _glob.glob(f'{d}/**/*.jpg', recursive=True)):
                img = cv2.imread(p)
                if img is not None:
                    train_images.append((p, img))
            if train_images:
                print(f"  Loaded {len(train_images)} LOL images from {d}")
                break

    # If LOL unavailable, use ExDark TRAIN split (explicitly not test split)
    if not train_images:
        print("  LOL unavailable; using ExDark train split...")
        train_images = load_dataset_images(DATASETS_DIR, 'exdark', split='train', max_samples=300)
        if not train_images:
            train_images = load_dataset_images(DATASETS_DIR, 'exdark', split='', max_samples=300)
        print(f"  Using {len(train_images)} ExDark images.")"""
    new_fallback = """\
    # Search in order: LOL official -> RTTS (always downloaded) -> ExDark
    train_images = []
    candidate_dirs = [
        lol_dir,
        '/content/LOL',
        '/content/LOL/our485/low',
        # RTTS is downloaded by Cell 02 — reliable source for zero-reference training
        '/content/computer_vision_expirement/datasets/rtts',
        '/content/computer_vision_expirement/datasets/rtts_processed/train',
        f'{DATASETS_DIR}/rtts',
        f'{DATASETS_DIR}/rtts_processed/train',
    ]
    for d in candidate_dirs:
        if not os.path.exists(d):
            continue
        found = (_glob.glob(f'{d}/**/*.png', recursive=True) +
                 _glob.glob(f'{d}/**/*.jpg', recursive=True))
        for p in found[:300]:
            img = cv2.imread(p)
            if img is not None:
                train_images.append((p, img))
        if train_images:
            print(f"  Loaded {len(train_images)} training images from {d}")
            break

    # If still empty, use ExDark train split
    if not train_images:
        print("  Trying ExDark train split...")
        train_images = load_dataset_images(DATASETS_DIR, 'exdark', split='train', max_samples=300)
        if not train_images:
            train_images = load_dataset_images(DATASETS_DIR, 'exdark', split='', max_samples=300)
        print(f"  Using {len(train_images)} ExDark images.")

    if not train_images:
        print("  [WARN] No training images found — Zero-DCE++ will use random init.")
        print("  Tip: ensure Cell 02 (download+preprocess) ran before this cell.")"""

    if old_fallback in old_src:
        cells[c8_idx]["source"] = [old_src.replace(old_fallback, new_fallback)]
        print(f"Fixed Cell {c8_idx}: ZeroDCE++ training fallback dirs")
    else:
        print("WARNING: ZeroDCE++ training fallback block not found in Cell 08")
else:
    print("WARNING: Cell 08 (ZeroDCE++) not found")

# ── Fix 5: Cell 13 — print bug (model_dir printed outside loop) ───────────
c13_idx = None
for i, c in enumerate(cells):
    src = "".join(c["source"])
    if "ENHANCED_DIR" in src and "model_dir" in src and "saved[name]" in src:
        c13_idx = i
        break

if c13_idx is not None:
    old_src = "".join(cells[c13_idx]["source"])
    old_print_block = """\
print('Enhanced outputs saved to Drive:')
for name, n in saved.items():
    print(f'  {name}: {n} images -> {model_dir}')"""
    new_print_block = """\
print('Enhanced outputs saved to Drive:')
for name, n in saved.items():
    _dir = f'{ENHANCED_DIR}/{name.replace("+","plus").replace(" ","_")}'
    print(f'  {name}: {n} images -> {_dir}')"""
    if old_print_block in old_src:
        cells[c13_idx]["source"] = [old_src.replace(old_print_block, new_print_block)]
        print(f"Fixed Cell {c13_idx}: print bug (model_dir outside loop)")
    else:
        print("WARNING: print block not found in Cell 13 (may already be fixed)")
else:
    print("WARNING: Cell 13 not found")

# ── Save ───────────────────────────────────────────────────────────────────
nb["cells"] = cells
json.dump(nb, open(NB_PATH, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"\nDone. Phase2 notebook saved ({len(cells)} cells).")
