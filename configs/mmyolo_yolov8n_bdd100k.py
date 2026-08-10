"""MMYOLO delta config: YOLOv8n fine-tuned on 7% BDD100K (Phase 3 weights).

Overrides only the project-specific fields on top of MMYOLO's stock
YOLOv8n COCO config. Deliberately does NOT redefine backbone/neck/head
architecture, loss, or the anchor-free head internals — those are
inherited unchanged from the base config so the converted Phase 3
checkpoint (yolov8n_outdoor_aug_best.pt -> .pth via
tools/model_converters/yolov8_to_mmyolo.py) loads with matching shapes.

Usage (after `git clone https://github.com/open-mmlab/mmyolo.git` and
`pip install -e .` per scripts/setup_mmyolo.sh):

    from mmdet.apis import init_detector, inference_detector
    model = init_detector(
        "configs/mmyolo_yolov8n_bdd100k.py",
        "yolov8n_outdoor_aug_mmyolo.pth",
        device="cuda:0",   # or "cpu"
    )

If the `mmyolo::` package-config alias below does not resolve in your
environment (depends on mmengine version), replace it with the literal
path to the cloned repo, e.g.:
    _base_ = '/content/mmyolo/configs/yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco.py'
"""

_base_ = 'mmyolo::yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco.py'

# BDD100K detection classes (10), same order as scripts/download_datasets.py /
# EXPERIMENTS.md Phase 3 training run — order matters, it maps class index -> name.
class_name = (
    'car', 'truck', 'bus', 'person', 'rider',
    'bicycle', 'motorcycle', 'traffic light', 'traffic sign', 'train',
)
num_classes = len(class_name)
metainfo = dict(classes=class_name)

# NOTE: verify this dotted path against the cloned repo's actual config
# (`python -c "from mmengine import Config; c = Config.fromfile('mmyolo::yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco.py'); print(c.model.bbox_head.head_module.num_classes)"`)
# before trusting it — MMYOLO has changed this field's exact location
# across releases and this project can't run mmyolo locally to confirm.
model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
    ),
    train_cfg=dict(
        assigner=dict(num_classes=num_classes),
    ),
)

train_dataloader = dict(dataset=dict(metainfo=metainfo))
val_dataloader = dict(dataset=dict(metainfo=metainfo))
test_dataloader = val_dataloader

# Phase 3 fine-tuned weights, converted from Ultralytics .pt format.
load_from = 'yolov8n_outdoor_aug_mmyolo.pth'
