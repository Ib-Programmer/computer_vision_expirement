"""Route B / Task 6: RTMDet-s eval WITH the EnhanceImage transform (the thesis result).

Delta config on top of configs/rtmdet_bdd100k.py -- same model/data/checkpoint, only
test_pipeline changes (inserts scripts/mm_transforms.EnhanceImage right after
LoadImageFromFile). Used only for eval (Cell 35 / Task 6), never for training: see
scripts/mm_transforms.py's docstring on why the inline transform is eval/latency-only
and Task 4 training should use precomputed enhanced images instead.

    !conda run -n mm python /content/mmdetection/tools/test.py \
        configs/rtmdet_bdd100k_enhanced.py work_dirs/rtmdet_bdd100k/latest.pth \
        --out results_enhanced.pkl

Why test_pipeline is fully rewritten instead of spliced onto the inherited one:
configs/rtmdet_bdd100k.py uses the *old-style* string `_base_ = 'mmdet::rtmdet/...'`.
Old-style `_base_` configs are dict-merged by mmengine's Config.fromfile at load time --
the inherited fields (like test_pipeline, which rtmdet_bdd100k.py never redefines) are
NOT importable Python objects inside this file's own exec namespace, so there is no way
to "insert into the existing list" here; it must be redefined whole. This is the same
reason every RTMDet variant in the upstream mmdetection config zoo (s/m/l/x, and every
task-specific derivative) fully rewrites train_pipeline/test_pipeline rather than
patching it -- not a shortcut, the standard pattern for this exact situation.

VERIFY before trusting this file (fail-loud is fine, fail-silent is not): the pipeline
below is RTMDet's standard test_pipeline as of mmdet 3.3.0's rtmdet_s_8xb32-300e_coco.py.
Confirm it actually matches what Task 4's baseline run used, so "with enhancement" vs
"baseline" differ ONLY by the EnhanceImage step and stay a fair comparison:
    conda run -n mm python -c "from mmengine import Config; \
        c = Config.fromfile('mmdet::rtmdet/rtmdet_s_8xb32-300e_coco.py'); \
        print(c.test_pipeline)"
If it differs from the list below, fix this file to match -- do not "fix" the diff away
by editing the printed reference.
"""

_base_ = 'rtmdet_bdd100k.py'

custom_imports = dict(imports=['scripts.mm_transforms'], allow_failed_imports=False)

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='EnhanceImage', method='zero_dce'),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', pad_val=dict(img=(114, 114, 114)), size=(640, 640)),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
