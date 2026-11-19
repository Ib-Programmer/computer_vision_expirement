"""Download RTTS (real fog) and ExDark (real low light) and write COCO-format
ground truth that a COCO-pretrained MMDetection model can be scored on directly.

Every class in both datasets has a COCO equivalent, so the ground-truth files
declare all 80 COCO categories (ids 1-80, names exactly as in mmdet's
CocoDataset) and only the mapped classes carry boxes. COCOeval skips categories
without ground truth, so mAP is averaged over the dataset's own classes.

Sources (public Hugging Face mirrors, no Kaggle account needed):
  RTTS   fuhao7i/RTTS          4,322 images, COCO json (test split = 433 images)
  ExDark dronefreak/ExDark     YOLO labels, test split ~730 images (640x640)

Usage:
  python scripts/prepare_eval_datasets.py --dataset all
  python scripts/prepare_eval_datasets.py --dataset rtts --split all   # all 4,322
  python scripts/prepare_eval_datasets.py --dataset exdark --limit 20  # quick test

Output (default root datasets/eval):
  rtts/images/*.png    rtts/annotations.json
  exdark/images/*.jpg  exdark/annotations.json
"""
import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

COCO_CLASSES = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush')
COCO_ID = {name: i + 1 for i, name in enumerate(COCO_CLASSES)}

RTTS_TO_COCO = {'person': 'person', 'bicycle': 'bicycle', 'car': 'car',
                'motorbike': 'motorcycle', 'bus': 'bus'}
EXDARK_TO_COCO = {'Bicycle': 'bicycle', 'Boat': 'boat', 'Bottle': 'bottle', 'Bus': 'bus',
                  'Car': 'car', 'Cat': 'cat', 'Chair': 'chair', 'Cup': 'cup', 'Dog': 'dog',
                  'Motorbike': 'motorcycle', 'People': 'person', 'Table': 'dining table'}


def _coco_skeleton(description):
    return {
        'info': {'description': description},
        'images': [], 'annotations': [],
        'categories': [{'id': COCO_ID[n], 'name': n, 'supercategory': 'none'}
                       for n in COCO_CLASSES],
    }


def _write(gt, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'annotations.json'
    path.write_text(json.dumps(gt))
    used = sorted({a['category_id'] for a in gt['annotations']})
    names = [COCO_CLASSES[i - 1] for i in used]
    print(f"  wrote {path}  ({len(gt['images'])} images, {len(gt['annotations'])} boxes, "
          f"classes: {', '.join(names)})")
    return path


def prepare_rtts(root, split='test', limit=None):
    from huggingface_hub import hf_hub_download, snapshot_download
    repo = 'fuhao7i/RTTS'
    out = root / 'rtts'
    print(f"[RTTS] split={split}")

    sources = ['coco/instances_test2017.json']
    if split == 'all':
        sources.append('coco/instances_trainval2017.json')
    src_gts = [json.loads(Path(hf_hub_download(repo, f, repo_type='dataset')).read_text())
               for f in sources]

    gt = _coco_skeleton(f'RTTS ({split}) remapped to COCO-80 category ids')
    next_img, next_ann = 1, 1
    for src in src_gts:
        cat_name = {c['id']: c['name'] for c in src['categories']}
        images = src['images'][:limit] if limit else src['images']
        keep = {}
        for im in images:
            keep[im['id']] = next_img
            gt['images'].append({'id': next_img, 'file_name': im['file_name'],
                                 'width': im['width'], 'height': im['height']})
            next_img += 1
        for a in src['annotations']:
            if a['image_id'] not in keep:
                continue
            name = RTTS_TO_COCO[cat_name[a['category_id']]]
            x, y, w, h = a['bbox']
            gt['annotations'].append({
                'id': next_ann, 'image_id': keep[a['image_id']],
                'category_id': COCO_ID[name], 'bbox': [x, y, w, h],
                'area': float(w * h), 'iscrowd': int(a.get('iscrowd', 0)),
            })
            next_ann += 1
        if limit:
            break

    files = [im['file_name'] for im in gt['images']]
    local = Path(snapshot_download(
        repo, repo_type='dataset',
        allow_patterns=[f'VOCdevkit/VOC2007/JPEGImages/{f}' for f in files]))
    img_dir = out / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    missing = 0
    for f in files:
        src = local / 'VOCdevkit' / 'VOC2007' / 'JPEGImages' / f
        if src.exists():
            if not (img_dir / f).exists():
                shutil.copy(src, img_dir / f)
        else:
            missing += 1
    if missing:
        drop = {im['id'] for im in gt['images'] if not (img_dir / im['file_name']).exists()}
        gt['images'] = [im for im in gt['images'] if im['id'] not in drop]
        gt['annotations'] = [a for a in gt['annotations'] if a['image_id'] not in drop]
        print(f"  [WARN] {missing} images missing from the mirror; dropped from ground truth")
    return _write(gt, out)


def prepare_exdark(root, limit=None):
    from huggingface_hub import snapshot_download
    repo = 'dronefreak/ExDark'
    out = root / 'exdark'
    print("[ExDark] split=test")
    local = Path(snapshot_download(repo, repo_type='dataset',
                                   allow_patterns=['data/data.yaml', 'data/test/*']))
    names = None
    for line in (local / 'data' / 'data.yaml').read_text().splitlines():
        if line.strip().startswith('names:'):
            names = json.loads(line.split(':', 1)[1].strip().replace("'", '"'))
    assert names and len(names) == 12, f"unexpected ExDark class list: {names}"

    img_src = local / 'data' / 'test' / 'images'
    lbl_src = local / 'data' / 'test' / 'labels'
    images = sorted(p for p in img_src.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    if limit:
        images = images[:limit]

    gt = _coco_skeleton('ExDark (test) remapped to COCO-80 category ids')
    img_dir = out / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    next_ann = 1
    for img_id, p in enumerate(images, start=1):
        with Image.open(p) as im:
            w, h = im.size
        gt['images'].append({'id': img_id, 'file_name': p.name, 'width': w, 'height': h})
        if not (img_dir / p.name).exists():
            shutil.copy(p, img_dir / p.name)
        lbl = lbl_src / (p.stem + '.txt')
        if not lbl.exists():
            continue
        for row in lbl.read_text().split('\n'):
            parts = row.split()
            if len(parts) != 5:
                continue
            cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
            bw, bh = bw * w, bh * h
            x, y = cx * w - bw / 2, cy * h - bh / 2
            gt['annotations'].append({
                'id': next_ann, 'image_id': img_id,
                'category_id': COCO_ID[EXDARK_TO_COCO[names[cls]]],
                'bbox': [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                'area': round(bw * bh, 2), 'iscrowd': 0,
            })
            next_ann += 1
    return _write(gt, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dataset', choices=['rtts', 'exdark', 'all'], default='all')
    ap.add_argument('--split', choices=['test', 'all'], default='test',
                    help="RTTS only: 'test' = 433 images, 'all' = 4,322 images")
    ap.add_argument('--root', default='datasets/eval')
    ap.add_argument('--limit', type=int, default=None, help='first N images (for quick tests)')
    args = ap.parse_args()
    root = Path(args.root)
    if args.dataset in ('rtts', 'all'):
        prepare_rtts(root, args.split, args.limit)
    if args.dataset in ('exdark', 'all'):
        prepare_exdark(root, args.limit)


if __name__ == '__main__':
    main()
