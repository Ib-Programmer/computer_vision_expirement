"""
Run this in a Colab cell (after Phases 2-5) to upload all model weights to HF Hub.

Usage in Colab:
    !python scripts/upload_models_to_hf.py

Required secrets (Colab → Secrets):
    HF_TOKEN       your HuggingFace write token
    HF_USERNAME    your HuggingFace username (e.g. ibmuhd557)
"""
import os, shutil, glob, sys
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
try:
    from google.colab import userdata
    HF_TOKEN    = userdata.get("HF_TOKEN")    or os.environ.get("HF_TOKEN", "")
    HF_USERNAME = userdata.get("HF_USERNAME") or os.environ.get("HF_USERNAME", "")
except Exception:
    HF_TOKEN    = os.environ.get("HF_TOKEN", "")
    HF_USERNAME = os.environ.get("HF_USERNAME", "")

if not HF_TOKEN or not HF_USERNAME:
    sys.exit(
        "ERROR: Set HF_TOKEN and HF_USERNAME in Colab Secrets "
        "(Secrets tab in the left sidebar), then re-run."
    )

REPO_ID      = f"{HF_USERNAME}/cv-thesis-models"
PROJECT_DIR  = "/content/drive/MyDrive/computer_vision/results"
WEIGHTS_DIR  = "/content/computer_vision_expirement/weights"

# ── login & create repo ───────────────────────────────────────────────────────
from huggingface_hub import HfApi, login
login(token=HF_TOKEN, add_to_git_credential=False)
api = HfApi()
api.create_repo(REPO_ID, repo_type="model", exist_ok=True, private=False)
print(f"Repo ready: https://huggingface.co/{REPO_ID}\n")

def upload(local_path: str, repo_filename: str):
    if not os.path.exists(local_path):
        print(f"  [SKIP] not found: {local_path}")
        return
    size_mb = os.path.getsize(local_path) / 1e6
    print(f"  Uploading {repo_filename}  ({size_mb:.1f} MB) ...")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_filename,
        repo_id=REPO_ID,
        repo_type="model",
    )
    print(f"  ✓ {repo_filename}")

# ── Phase 5 models (primary — use ONNX for deployment) ───────────────────────
print("=== Phase 5: Optimized models ===")
upload(f"{PROJECT_DIR}/phase5/yolov8n_best.onnx",  "yolov8n_best.onnx")
upload(f"{PROJECT_DIR}/phase5/yolov8n_int8.onnx",  "yolov8n_int8.onnx")
upload(f"{PROJECT_DIR}/phase5/yolov8n_pruned.pt",  "yolov8n_pruned.pt")
# TRT FP16 engine is hardware-specific — skip uploading

# ── Phase 3 models (fallback .pt weights) ────────────────────────────────────
print("\n=== Phase 3: Trained detection models ===")
upload(f"{PROJECT_DIR}/phase3/yolov8n_outdoor_aug/weights/best.pt", "yolov8n_outdoor_aug_best.pt")
upload(f"{PROJECT_DIR}/phase3/yolov8n_baseline/weights/best.pt",    "yolov8n_baseline_best.pt")
upload(f"{PROJECT_DIR}/phase3/rtdetr_outdoor_aug/weights/best.pt",  "rtdetr_outdoor_aug_best.pt")

# ── Phase 2 models (enhancement — must re-download if session is fresh) ───────
print("\n=== Phase 2: Enhancement models ===")

# Zero-DCE++ — re-clone from GitHub if not present
zdce_pth = "weights/zero_dce_pp.pth"
if not os.path.exists(zdce_pth):
    print("  Re-downloading Zero-DCE++ from GitHub...")
    os.system("git clone --depth 1 https://github.com/Li-Chongyi/Zero-DCE_extension.git /tmp/zdce_repo")
    cand = (glob.glob("/tmp/zdce_repo/**/Epoch99.pth", recursive=True) or
            glob.glob("/tmp/zdce_repo/**/*.pth", recursive=True))
    if cand:
        os.makedirs("weights", exist_ok=True)
        shutil.copy(cand[0], zdce_pth)
        print(f"  Zero-DCE++ weights saved to {zdce_pth}")
    else:
        print("  [WARN] Zero-DCE++ checkpoint not found in repo")
upload(zdce_pth, "zero_dce_pp.pth")

# FFA-Net — re-download from Kaggle if not present
ffa_pth = "weights/ffa_net_outdoor.pk"
if not os.path.exists(ffa_pth):
    print("  Re-downloading FFA-Net from Kaggle...")
    os.makedirs("weights", exist_ok=True)
    os.system("kaggle datasets download -d balraj98/ffanet-pretrained-weights -p weights/ --unzip -q")
    cands = glob.glob("weights/**/ots_train_ffa_3_19.pk", recursive=True)
    if cands:
        shutil.copy(cands[0], ffa_pth)
        print(f"  FFA-Net weights saved to {ffa_pth}")
    else:
        print("  [WARN] FFA-Net .pk file not found after download")
upload(ffa_pth, "ffa_net_outdoor.pk")

# Restormer — already public on HF Hub at deepinv/Restormer
# The Space downloads it directly from there; no need to re-upload.
print("  [OK] Restormer: pulled from deepinv/Restormer at Space startup (already public)")

print(f"\n✓ All done — models at https://huggingface.co/{REPO_ID}")
print("\nNext: go to your HF Space → Settings → Variables and secrets, set:")
print(f"  HF_MODEL_REPO = {REPO_ID}")
print(f"  HF_TOKEN      = <your read token>")
print( "  INTERNAL_TOKEN = <match Spring Boot INFERENCE_TOKEN>")
