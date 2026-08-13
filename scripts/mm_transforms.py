"""Route B / Task 5 — THE CONTRIBUTION: enhancement/defogging transform for mmdet.

Wires scripts/enhancers.py (Zero-DCE++ / FFA-Net / Restormer) into the mmdet data
pipeline as a swappable pre-detection stage.

Real-time claim (resolved 2026-08-13, target ~25-30 FPS end-to-end): only
method='zero_dce' (~33-39ms/img) qualifies. FFA-Net (~1.3s/img) and Restormer
(~3.9-4.4s/img) are a separate "quality mode" comparison in Task 6 — never the
real-time claim. Default is 'zero_dce' for that reason.

Register in a config with:
    custom_imports = dict(imports=['scripts.mm_transforms'], allow_failed_imports=False)
and insert into train_pipeline / test_pipeline right after LoadImageFromFile:
    dict(type='EnhanceImage', method='zero_dce')

Training cost note: with num_workers>1 each worker loads its own copy of the
enhancer model. For Task 4 training runs, prefer precomputing enhanced images
offline (run this transform once over the dataset, save outputs, point
train_dataloader at the enhanced copies) instead of paying the enhancer cost
every batch. Use this inline transform for eval (Task 6) and for the real-time
latency demonstration, where per-frame cost is the thing being measured.
"""

import time

from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS

import scripts.enhancers as E

_FN = {
    'zero_dce': E.enhance_zero_dce,
    'ffanet': E.enhance_ffanet,
    'restormer': E.enhance_restormer,
}


@TRANSFORMS.register_module()
class EnhanceImage(BaseTransform):
    """Apply a defogging/low-light enhancement model to results['img'].

    method: 'zero_dce' (real-time path, default), 'ffanet' or 'restormer'
    (quality-mode ablation, not real-time — see module docstring).
    log_latency: if True, stash per-call latency in results['enhance_latency_ms']
    so Task 6 eval can report measured, not assumed, real-time numbers.
    """

    def __init__(self, method='zero_dce', log_latency=True):
        if method not in _FN:
            raise ValueError(f"Unknown enhancement method '{method}', expected one of {list(_FN)}")
        self.method = method
        self.fn = _FN[method]
        self.log_latency = log_latency

    def transform(self, results):
        t0 = time.perf_counter()
        results['img'] = self.fn(results['img'])  # HWC BGR uint8 in/out
        if self.log_latency:
            results['enhance_latency_ms'] = (time.perf_counter() - t0) * 1000.0
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(method='{self.method}')"
