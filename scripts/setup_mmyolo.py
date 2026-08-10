"""One-shot Colab setup: install MMEngine/MMCV/MMDetection/MMYOLO, convert
the existing Phase 3 Ultralytics YOLOv8n checkpoint to MMYOLO format, then
run both models on the same held-out images and compare detections so you
know the conversion is faithful before building anything on top of it.

Run from a Colab cell (needs GPU runtime + the Phase 3 checkpoint already
in Google Drive, same layout as notebooks/Phase3_Object_Detection.ipynb):

    !python scripts/setup_mmyolo.py \
        --src-weights /content/drive/MyDrive/computer_vision/results/yolov8n_outdoor_aug_best.pt \
        --val-images  /content/computer_vision_expirement/datasets/bdd100k/val/images \
        --n-check     20

This does NOT touch deploy/app.py or the live HF Space. It only proves the
converted model works and matches the original — wiring it into the FastAPI
app is a separate, deliberate step once this passes.
"""
import argparse
import subprocess
import sys
from pathlib import Path

MMYOLO_REPO = "https://github.com/open-mmlab/mmyolo.git"
MMYOLO_DIR = Path("/content/mmyolo")
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "mmyolo_yolov8n_bdd100k.py"


def run(cmd, **kwargs):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def install_stack():
    run([sys.executable, "-m", "pip", "install", "-U", "openmim"])
    run(["mim", "install", "mmengine"])
    run(["mim", "install", "mmcv>=2.0.0"])
    run(["mim", "install", "mmdet>=3.0.0"])

    if not MMYOLO_DIR.exists():
        run(["git", "clone", "--depth", "1", MMYOLO_REPO, str(MMYOLO_DIR)])
    run([sys.executable, "-m", "pip", "install", "-e", str(MMYOLO_DIR)])


def convert_weights(src_weights: str, dst_weights: str):
    converter = MMYOLO_DIR / "tools" / "model_converters" / "yolov8_to_mmyolo.py"
    if not converter.exists():
        raise FileNotFoundError(
            f"{converter} not found — MMYOLO may have moved/renamed this script "
            "since this was written (repo last released Aug 2023). Check "
            "https://github.com/open-mmlab/mmyolo/tree/main/tools/model_converters "
            "for the current filename before proceeding."
        )
    run([sys.executable, str(converter), "--src", src_weights, "--dst", dst_weights])


def verify_parity(src_weights: str, dst_weights: str, val_images: str, n_check: int):
    """Run both the original Ultralytics model and the converted MMYOLO model
    on the same images; print per-image detection counts side by side.
    A large systematic mismatch (not just minor NMS-order differences) means
    the conversion or the config's num_classes/class order is wrong.
    """
    import glob

    from ultralytics import YOLO
    from mmdet.apis import init_detector, inference_detector

    images = sorted(glob.glob(f"{val_images}/*.jpg"))[:n_check]
    if not images:
        print(f"[warn] no .jpg images found under {val_images} — skipping parity check")
        return

    ul_model = YOLO(src_weights)
    mm_model = init_detector(str(CONFIG_PATH), dst_weights, device="cuda:0")

    print(f"\n{'image':40s} {'ultralytics dets':>18s} {'mmyolo dets':>14s}")
    for img in images:
        ul_res = ul_model(img, verbose=False, conf=0.45, iou=0.45)[0]
        ul_count = len(ul_res.boxes)

        mm_res = inference_detector(mm_model, img)
        mm_scores = mm_res.pred_instances.scores.cpu().numpy()
        mm_count = int((mm_scores >= 0.45).sum())

        flag = "  <-- check this one" if abs(ul_count - mm_count) > 2 else ""
        print(f"{Path(img).name:40s} {ul_count:18d} {mm_count:14d}{flag}")

    print(
        "\nIf counts track closely across images, conversion is faithful — "
        "safe to wire into deploy/app.py next. If they diverge, first suspect "
        "configs/mmyolo_yolov8n_bdd100k.py's num_classes/class-order fields "
        "before assuming the converter itself is broken."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src-weights", required=True, help="Path to yolov8n_outdoor_aug_best.pt")
    p.add_argument("--dst-weights", default="yolov8n_outdoor_aug_mmyolo.pth")
    p.add_argument("--val-images", default=None, help="Dir of .jpg images for the parity check")
    p.add_argument("--n-check", type=int, default=20)
    p.add_argument("--skip-install", action="store_true", help="Stack already installed this session")
    args = p.parse_args()

    if not args.skip_install:
        install_stack()

    convert_weights(args.src_weights, args.dst_weights)

    if args.val_images:
        verify_parity(args.src_weights, args.dst_weights, args.val_images, args.n_check)
    else:
        print("\n[skip] --val-images not given — skipping parity check. "
              "Strongly recommend running it before trusting the converted weights.")


if __name__ == "__main__":
    main()
