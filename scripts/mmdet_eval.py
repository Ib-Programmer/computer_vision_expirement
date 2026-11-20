"""Evaluate a COCO-pretrained MMDetection model on RTTS / ExDark, with or without
an image-enhancement step in front of the detector.

This is the thesis experiment: same detector, same images, only the
enhancement changes, so any mAP difference is caused by the enhancement.

Each run uses MMDetection's own Runner + CocoMetric and appends one JSON line
to --out with accuracy (mAP, mAP50, mAP75, per-class AP) and cost (enhancement
ms/img, detector ms/img and FPS, parameters, peak GPU memory).

Models are looked up by name in mmdet's bundled metafiles, so config and
checkpoint always come from the official OpenMMLab model zoo, e.g.
  rtmdet_s_8xb32-300e_coco, rtmdet_tiny_8xb32-300e_coco,
  faster-rcnn_r50_fpn_1x_coco, retinanet_r50_fpn_1x_coco, ssd300_coco

Usage (inside the Python 3.10 env with mmcv 2.1 / mmdet 3.3):
  python scripts/mmdet_eval.py --dataset rtts --enhance ffanet
  python scripts/mmdet_eval.py --dataset exdark --enhance zero_dce --limit 20
"""
import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

# Colab exports MPLBACKEND=module://matplotlib_inline.backend_inline, which does
# not exist in the separate mmdet env; mmdet imports matplotlib on load, so force
# a headless backend before anything pulls it in.
os.environ['MPLBACKEND'] = 'Agg'

REPO =Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ENHANCERS = ('none', 'clahe', 'zero_dce', 'ffanet', 'restormer')
_LATENCIES = []     # ms per image, filled by EvalEnhance


def _clahe(img_bgr):
    import cv2
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _enhancer_fn(method):
    if method == 'clahe':
        return _clahe
    import scripts.enhancers as E
    return {'zero_dce': E.enhance_zero_dce,
            'ffanet': E.enhance_ffanet,
            'restormer': E.enhance_restormer}[method]


def _register_transform():
    import torch
    from mmcv.transforms import BaseTransform
    from mmdet.registry import TRANSFORMS

    @TRANSFORMS.register_module(force=True)
    class EvalEnhance(BaseTransform):
        """Enhance results['img'] (HWC BGR uint8) and record per-image latency."""

        def __init__(self, method):
            self.method = method
            self.fn = _enhancer_fn(method)

        def transform(self, results):
            t0 = time.perf_counter()
            results['img'] = self.fn(results['img'])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _LATENCIES.append((time.perf_counter() - t0) * 1000.0)
            return results


def find_model(name):
    """Return (config_path, checkpoint_url, zoo_box_ap) from mmdet's metafiles."""
    import yaml
    import mmdet
    mim = Path(mmdet.__file__).parent / '.mim'
    for meta in glob.glob(str(mim / 'configs' / '*' / 'metafile.yml')):
        data = yaml.safe_load(Path(meta).read_text(encoding='utf-8')) or {}
        models = data.get('Models', []) if isinstance(data, dict) else data
        for m in models or []:
            if isinstance(m, dict) and m.get('Name') == name:
                ap = None
                for r in m.get('Results', []) or []:
                    if r.get('Task') == 'Object Detection':
                        ap = r.get('Metrics', {}).get('box AP')
                return str(mim / m['Config']), m['Weights'], ap
    raise SystemExit(f"model '{name}' not found in mmdet metafiles")


def _subset_gt(ann_file, limit, work_dir):
    gt = json.loads(Path(ann_file).read_text())
    if limit and limit < len(gt['images']):
        gt['images'] = gt['images'][:limit]
        keep = {im['id'] for im in gt['images']}
        gt['annotations'] = [a for a in gt['annotations'] if a['image_id'] in keep]
    out = Path(work_dir) / 'annotations.json'
    out.write_text(json.dumps(gt))
    return str(out), gt


def _clean(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _benchmark(cfg_path, weights, img_paths, n=50, warmup=5):
    """Detector-only speed on preloaded images (no enhancement, no data loading)."""
    import cv2
    import torch
    from mmdet.apis import inference_detector, init_detector
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = init_detector(cfg_path, weights, device=device)
    imgs = [cv2.imread(p) for p in img_paths[:n]]
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    for im in imgs[:warmup]:
        inference_detector(model, im)
    if device != 'cpu':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for im in imgs:
        inference_detector(model, im)
    if device != 'cpu':
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / len(imgs)
    mem = torch.cuda.max_memory_allocated() / 2**20 if device != 'cpu' else None
    return {'det_ms_per_img': round(ms, 2), 'fps': round(1000.0 / ms, 1),
            'params_M': round(params_m, 2), 'gpu_mem_MB': mem and round(mem, 1),
            'device': torch.cuda.get_device_name(0) if device != 'cpu' else 'cpu'}


def run(args):
    import torch
    import mmdet
    from mmengine.config import Config
    from mmengine.runner import Runner

    root = Path(args.root) / args.dataset
    ann_file, img_dir = root / 'annotations.json', root / 'images'
    if not ann_file.exists():
        raise SystemExit(f"{ann_file} missing - run scripts/prepare_eval_datasets.py first")

    tag = f"{args.model}__{args.dataset}__{args.enhance}" + (f"__n{args.limit}" if args.limit else '')
    work_dir = Path(args.work_dir) / tag
    work_dir.mkdir(parents=True, exist_ok=True)
    ann, gt = _subset_gt(ann_file, args.limit, work_dir)

    cfg_path, weights, zoo_ap = find_model(args.model)
    cfg = Config.fromfile(cfg_path)
    pipeline = cfg.test_dataloader.dataset.pipeline
    assert pipeline[0]['type'].endswith('LoadImageFromFile'), pipeline[0]
    if args.enhance != 'none':
        _register_transform()
        pipeline.insert(1, dict(type='EvalEnhance', method=args.enhance))

    ds = cfg.test_dataloader.dataset
    ds.data_root = str(root)
    ds.ann_file = ann
    ds.data_prefix = dict(img=str(img_dir) + os.sep)
    ds.test_mode = True
    ds.pipeline = pipeline
    # Enhancers run on the GPU; CUDA cannot be used inside forked dataloader workers.
    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.persistent_workers = False
    cfg.test_dataloader.batch_size = 1
    # Keep each model's own evaluator settings (e.g. RTMDet's proposal_nums);
    # only point it at our ground truth and turn on per-class AP.
    assert cfg.test_evaluator['type'].endswith('CocoMetric'), cfg.test_evaluator
    cfg.test_evaluator.update(ann_file=ann, metric='bbox', classwise=True,
                              format_only=False, outfile_prefix=None)
    cfg.load_from = weights
    cfg.work_dir = str(work_dir)
    cfg.launcher = 'none'

    runner = Runner.from_cfg(cfg)
    _LATENCIES.clear()
    t0 = time.perf_counter()
    metrics = runner.test()
    elapsed = time.perf_counter() - t0
    n = len(gt['images'])

    per_class = {k[len('coco/'):-len('_precision')]: _clean(v)
                 for k, v in metrics.items() if k.endswith('_precision')}
    per_class = {k: v for k, v in per_class.items() if v is not None}
    rec = {
        'model': args.model, 'framework': f"mmdetection {mmdet.__version__}",
        'dataset': args.dataset, 'enhance': args.enhance, 'n_images': n,
        'n_boxes': len(gt['annotations']),
        'mAP': _clean(metrics.get('coco/bbox_mAP')),
        'mAP50': _clean(metrics.get('coco/bbox_mAP_50')),
        'mAP75': _clean(metrics.get('coco/bbox_mAP_75')),
        'mAP_s': _clean(metrics.get('coco/bbox_mAP_s')),
        'mAP_m': _clean(metrics.get('coco/bbox_mAP_m')),
        'mAP_l': _clean(metrics.get('coco/bbox_mAP_l')),
        'per_class_AP': per_class,
        'enhance_ms_per_img': round(sum(_LATENCIES) / len(_LATENCIES), 2) if _LATENCIES else 0.0,
        'pipeline_ms_per_img': round(elapsed * 1000.0 / n, 2),
        'zoo_coco_box_AP': zoo_ap,
        'torch': torch.__version__,
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    if args.enhance == 'none' and not args.no_benchmark:
        imgs = [str(img_dir / im['file_name']) for im in gt['images']]
        rec.update(_benchmark(cfg_path, weights, imgs))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('a', encoding='utf-8') as f:
        f.write(json.dumps(rec) + '\n')
    print(f"[mmdet_eval] {tag}: mAP={rec['mAP']} mAP50={rec['mAP50']} "
          f"enhance={rec['enhance_ms_per_img']} ms/img -> {out}")
    return rec


def already_done(out, model, dataset, enhance, limit):
    p = Path(out)
    if not p.exists():
        return False
    for line in p.read_text(encoding='utf-8').splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if (r['model'], r['dataset'], r['enhance']) == (model, dataset, enhance) and \
                (limit is None or r['n_images'] == limit):
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model', default='rtmdet_s_8xb32-300e_coco')
    ap.add_argument('--dataset', choices=['rtts', 'exdark'], required=True)
    ap.add_argument('--enhance', choices=ENHANCERS, default='none')
    ap.add_argument('--root', default='datasets/eval')
    ap.add_argument('--limit', type=int, default=None, help='first N images only')
    ap.add_argument('--out', default='results/mmdet_eval.jsonl')
    ap.add_argument('--work-dir', default='work_dirs/mmdet_eval')
    ap.add_argument('--skip-existing', action='store_true')
    ap.add_argument('--no-benchmark', action='store_true')
    args = ap.parse_args()
    if args.skip_existing and already_done(args.out, args.model, args.dataset,
                                           args.enhance, args.limit):
        print(f"[mmdet_eval] skip (already in {args.out}): "
              f"{args.model} {args.dataset} {args.enhance}")
        return
    run(args)


if __name__ == '__main__':
    main()
