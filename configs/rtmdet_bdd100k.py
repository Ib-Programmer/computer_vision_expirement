"""Route B / Task 3: RTMDet-s fine-tune on the BDD100K subset, native mmdet.

Run inside the conda `mm` env (see docs/MMDETECTION_ROUTE_B_RUNBOOK.md §3):
    !MPLBACKEND=Agg conda run -n mm python -m mmdet.tools.train configs/rtmdet_bdd100k.py

Verify this base config path and the bbox_head field path against the installed
mmdet==3.3.0 package before trusting them (they can move between releases):
    conda run -n mm python -c "from mmengine import Config; \
        c = Config.fromfile('mmdet::rtmdet/rtmdet_s_8xb32-300e_coco.py'); \
        print(c.model.bbox_head.num_classes)"
"""

_base_ = 'mmdet::rtmdet/rtmdet_s_8xb32-300e_coco.py'

data_root = 'datasets/bdd100k_yolo/'

# Order matches scripts/yolo_to_coco.py CLASSES, which matches the class_to_id used
# when the on-disk YOLO labels were generated (scripts/preprocess_data.py). This is
# NOT alphabetical and NOT the order in earlier runbook drafts — do not "fix" it.
classes = (
    'pedestrian', 'rider', 'car', 'truck', 'bus',
    'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign',
)
metainfo = dict(classes=classes)

model = dict(bbox_head=dict(num_classes=len(classes)))

train_dataloader = dict(
    batch_size=4,            # T4-safe; raise if memory allows
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json',
        data_prefix=dict(img='train/images/'),
    ),
)
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/val.json',
        data_prefix=dict(img='val/images/'),
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/val.json')
test_evaluator = val_evaluator

# Start from COCO-pretrained weights (transfer learning).
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth'

max_epochs = 25
train_cfg = dict(max_epochs=max_epochs, val_interval=5)

# RTMDet's stage-2 pipeline switch (turns off mosaic/mixup near the end of training,
# via PipelineSwitchHook) is inherited from the base config and fires at a fixed
# epoch offset from the base's own max_epochs (300). For a 25-epoch fine-tune that
# switch epoch is almost certainly >= max_epochs here, i.e. it never fires and
# mosaic/mixup stay on for the whole short run. Confirm the switch epoch in the base
# config and override train_pipeline_stage2 / the hook's `switch_epoch` explicitly if
# you want the stage-2 (no-mosaic) pipeline to actually kick in before training ends.
