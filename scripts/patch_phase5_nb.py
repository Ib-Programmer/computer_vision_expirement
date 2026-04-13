"""Patch Phase5_Model_Optimization.ipynb to align with experiment guide §5."""

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "Phase5_Model_Optimization.ipynb"

nb = json.load(open(NB_PATH, encoding="utf-8"))
cells = nb["cells"]

# ── 1. Add Colab badge at index 0 ─────────────────────────────────────────
badge_src = (
    '<a href="https://colab.research.google.com/github/Ib-Programmer/'
    'computer_vision_expirement/blob/main/notebooks/Phase5_Model_Optimization.ipynb" target="_parent">'
    '<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'
)
badge_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [badge_src],
}
cells.insert(0, badge_cell)

# ── 2. Fix Cell 08 (INT8 export) — stray 'from ultralytics import YOLO' ──
# After badge insertion old cell 07 is now cell 08.
# The bad line: "from ultralytics import YOLO\n" sits alone with wrong indent.
c8 = cells[8]
src = "".join(c8["source"])
# Remove the stray line that appears after the onnxruntime fallback block
bad = "from ultralytics import YOLO\n        onnx_path"
good = "onnx_path"
if bad in src:
    src = src.replace(bad, good)
    c8["source"] = [src]
    print("Fixed Cell 08: removed stray 'from ultralytics import YOLO'")
else:
    print("Cell 08: stray import not found (may already be fixed)")

# ── 3. Rename section headings to match guide exactly ─────────────────────
# §5.3 Apply Quantization — currently "## 5.3 Export to TensorRT (INT8)"
c7_md = cells[7]  # markdown heading before INT8 cell (after badge, was cell 6)
if "## 5.3" in "".join(c7_md["source"]):
    c7_md["source"] = [
        "## 5.3 Apply Quantization\n",
        "\n",
        "INT8 quantization: ~4× smaller, ~2–4× faster with slight accuracy drop.\n",
        "Tries TensorRT INT8 first (needs calibration data); falls back to ONNX dynamic quantization.\n",
    ]
    print("Fixed Cell 07 heading -> ## 5.3 Apply Quantization")

# §5.4 Benchmark — find the heading "## 5.7 Full Optimization Comparison"
for i, c in enumerate(cells):
    if c["cell_type"] == "markdown" and "5.7" in "".join(c["source"]):
        c["source"] = [
            "## 5.4 Benchmark Optimized vs. Original\n",
            "\n",
            "Compare PyTorch FP32 baseline against ONNX Runtime, TensorRT FP16, INT8, pruned, and distilled variants.\n",
        ]
        print(f"Fixed Cell {i} heading -> ## 5.4 Benchmark Optimized vs. Original")
        break

# Also fix §5.6 ONNX Runtime heading to be a sub-section
for i, c in enumerate(cells):
    if c["cell_type"] == "markdown" and "5.6" in "".join(c["source"]):
        c["source"] = [
            "### 5.6 ONNX Runtime Inference Benchmark\n",
        ]
        print(f"Cell {i}: § 5.6 -> sub-section heading")
        break

# ── 4. Replace bare Cell 16 (just a print) with deliverables cell ─────────
deliverables_src = """\
import os, pandas as pd

PROJECT_DIR = '/content/drive/MyDrive/computer_vision'
RESULTS_DIR = f'{PROJECT_DIR}/results/phase5'

# ── Literature Benchmarks ──────────────────────────────────────────────────
lit = pd.DataFrame([
    {'Format': 'FP32 (baseline)', 'Size Reduction': '1×',   'Speed Gain': '1×',     'Accuracy Impact': 'None',         'Source': '—'},
    {'Format': 'FP16',            'Size Reduction': '~2×',  'Speed Gain': '~1.5-2×', 'Accuracy Impact': 'Minimal',      'Source': 'NVIDIA TRT docs'},
    {'Format': 'INT8',            'Size Reduction': '~4×',  'Speed Gain': '~2-4×',  'Accuracy Impact': 'Slight drop',  'Source': 'NVIDIA TRT docs'},
    {'Format': 'Pruning (30%)',   'Size Reduction': '~1.4×','Speed Gain': '~1.2×',  'Accuracy Impact': 'Needs fine-tune','Source': 'Han et al. 2015'},
    {'Format': 'Distillation',    'Size Reduction': '1×',   'Speed Gain': '1×',     'Accuracy Impact': 'Often improves','Source': 'Hinton et al. 2015'},
])

print('=' * 70)
print('TABLE 5.1: Optimization Trade-offs (Literature)')
print('=' * 70)
print(lit.to_string(index=False))

# ── Deliverables Checklist ─────────────────────────────────────────────────
print('\\n' + '=' * 70)
print('PHASE 5 DELIVERABLES CHECKLIST')
print('=' * 70)

onnx_ok  = os.path.exists(f'{RESULTS_DIR}/yolov8n_best.onnx')
bench_ok = os.path.exists(f'{RESULTS_DIR}/optimization_benchmark.csv')

items = [
    ('ONNX model file',                    onnx_ok,  'run Cell 5.1 (ONNX export)'),
    ('Optimization benchmark CSV',         bench_ok, 'run Cell 5.4 (benchmark)'),
    ('TensorRT FP16 engine',
        os.path.exists(f'{RESULTS_DIR}/yolov8n_fp16.engine'),  'run Cell 5.2'),
    ('INT8 quantized model',
        os.path.exists(f'{RESULTS_DIR}/yolov8n_int8.onnx') or
        os.path.exists(f'{RESULTS_DIR}/yolov8n_int8.engine'),  'run Cell 5.3'),
]

all_ok = True
for label, ok, hint in items:
    status = '[OK]' if ok else f'[MISSING — {hint}]'
    print(f'  {status:<45} {label}')
    if not ok:
        all_ok = False

print()
if bench_ok:
    df = pd.read_csv(f'{RESULTS_DIR}/optimization_benchmark.csv')
    print('Current benchmark results:')
    print(df[['Format', 'Latency_ms', 'FPS', 'Model_Size_MB']].to_string(index=False))

print('\\n' + '=' * 70)
print(f'Phase 5 Status: {"COMPLETE" if all_ok else "IN PROGRESS"}')
print('=' * 70)
if all_ok:
    print('Next: Open Phase6_Deployment.ipynb')
"""

# Find the last code cell (currently just a print)
last_code = None
for i in range(len(cells) - 1, -1, -1):
    if cells[i]["cell_type"] == "code":
        last_code = i
        break

# Check if it's the bare "Next: Open Phase6" print cell
if last_code is not None and "Phase 5 results saved" in "".join(cells[last_code]["source"]):
    cells[last_code]["source"] = [deliverables_src]
    print(f"Replaced Cell {last_code} with deliverables + checklist")
else:
    # Append new deliverables cell before the bare print
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [deliverables_src],
    })
    print("Appended deliverables cell")

# ── 5. Add ## 5.5 Deliverables heading before the deliverables code cell ──
# Find index of the deliverables cell we just modified
deliv_idx = None
for i, c in enumerate(cells):
    if c["cell_type"] == "code" and "PHASE 5 DELIVERABLES" in "".join(c["source"]):
        deliv_idx = i
        break

if deliv_idx is not None:
    # Check if heading already exists just before it
    prev = cells[deliv_idx - 1] if deliv_idx > 0 else None
    if prev is None or "5.5 Deliverables" not in "".join(prev["source"]):
        heading_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 5.5 Deliverables\n"],
        }
        cells.insert(deliv_idx, heading_cell)
        print(f"Inserted ## 5.5 Deliverables heading at index {deliv_idx}")

# ── Save ───────────────────────────────────────────────────────────────────
nb["cells"] = cells
json.dump(nb, open(NB_PATH, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

print(f"\nDone. Notebook now has {len(cells)} cells.")
for i, c in enumerate(cells):
    src = "".join(c["source"])[:100].replace("\n", " ")
    print(f"  [{i:02d}] {c['cell_type']:<8} | {src}")
