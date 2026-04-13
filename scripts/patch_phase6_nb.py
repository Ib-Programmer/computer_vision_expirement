"""Patch Phase6_Deployment.ipynb to align with experiment guide §6."""

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "Phase6_Deployment.ipynb"

nb = json.load(open(NB_PATH, encoding="utf-8"))
cells = nb["cells"]

# ── 1. Add Colab badge at index 0 ─────────────────────────────────────────
badge_src = (
    '<a href="https://colab.research.google.com/github/Ib-Programmer/'
    'computer_vision_expirement/blob/main/notebooks/Phase6_Deployment.ipynb" target="_parent">'
    '<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'
)
cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source": [badge_src]})
print("Added Colab badge")

# ── 2. Replace app.py cell with full 4-endpoint version ───────────────────
# After badge insertion, old cell 05 is now cell 06.
app_py_cell = None
for i, c in enumerate(cells):
    if c["cell_type"] == "code" and "%%writefile app.py" in "".join(c["source"]):
        app_py_cell = i
        break

if app_py_cell is not None:
    cells[app_py_cell]["source"] = ["""\
%%writefile app.py
\"\"\"
Outdoor Detection & Face Recognition REST API
Endpoints (guide §6.1):
  POST /enhance    — image enhancement (ZeroDCE++)
  POST /detect     — object detection (YOLOv8)
  POST /recognize  — face recognition (ArcFace)
  POST /pipeline   — full end-to-end pipeline
  GET  /health     — service health check
\"\"\"
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2, numpy as np, io, time, os

app = FastAPI(title="Outdoor Detection & Face Recognition API")

# Globals initialised on startup
detector  = None
face_app  = None
enh_model = None

@app.on_event("startup")
async def startup():
    global detector, face_app, enh_model
    PROJECT_DIR = os.environ.get("PROJECT_DIR", "/content/drive/MyDrive/computer_vision")
    MODELS_DIR  = f"{PROJECT_DIR}/results"

    # Object detector (YOLOv8)
    try:
        from ultralytics import YOLO
        for p in [
            f"{MODELS_DIR}/phase3/yolov8n_outdoor_aug/weights/best.pt",
            f"{MODELS_DIR}/phase3/yolov8n_baseline/weights/best.pt",
            "yolov8n.pt",
        ]:
            if os.path.exists(p):
                detector = YOLO(p)
                print(f"Detector loaded: {p}")
                break
    except Exception as e:
        print(f"Detector load failed: {e}")

    # Face analyzer (RetinaFace + ArcFace)
    try:
        from insightface.app import FaceAnalysis
        face_app = FaceAnalysis(name="buffalo_l",
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        print("Face analyzer loaded: RetinaFace + ArcFace")
    except Exception as e:
        print(f"Face analyzer load failed: {e}")

    # Enhancement model (ZeroDCE++)
    ckpt = f"{MODELS_DIR}/phase2/zerodce_best.pth"
    if os.path.exists(ckpt):
        try:
            import torch, sys
            sys.path.insert(0, "/content/computer_vision_expirement")
            from models.zerodce import ZeroDCEpp
            enh_model = ZeroDCEpp()
            enh_model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            enh_model.eval()
            print("Enhancement model loaded: ZeroDCE++")
        except Exception as e:
            print(f"Enhancement model load failed: {e}")
    else:
        print("ZeroDCE++ weights not found — /enhance will use CLAHE fallback")


def _decode_image(contents: bytes):
    nparr = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def _enhance_image(img_bgr):
    \"\"\"Apply ZeroDCE++ or CLAHE fallback.\"\"\"
    if enh_model is not None:
        import torch
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0)
        with torch.no_grad():
            out = enh_model(t)[0].squeeze(0).permute(1, 2, 0).numpy()
        out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    # CLAHE fallback
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/enhance")
async def enhance_image(file: UploadFile = File(...)):
    \"\"\"Enhance a degraded outdoor image (ZeroDCE++ or CLAHE fallback).\"\"\"
    img = _decode_image(await file.read())
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})
    t0 = time.time()
    enhanced = _enhance_image(img)
    latency_ms = (time.time() - t0) * 1000
    _, buf = cv2.imencode(".jpg", enhanced)
    return JSONResponse(content={
        "enhanced_bytes_b64": buf.tobytes().hex(),
        "latency_ms": round(latency_ms, 1),
        "method": "zerodce" if enh_model else "clahe",
    })


@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    \"\"\"Run object detection (YOLOv8) on an image.\"\"\"
    img = _decode_image(await file.read())
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})
    if detector is None:
        return JSONResponse(status_code=503, content={"error": "Detector not loaded"})
    t0 = time.time()
    det_results = detector(img, verbose=False)
    latency_ms = (time.time() - t0) * 1000
    detections = []
    for r in det_results:
        for box in r.boxes:
            detections.append({
                "class": r.names[int(box.cls)],
                "confidence": round(float(box.conf), 4),
                "bbox": [round(x, 1) for x in box.xyxy[0].tolist()],
            })
    return {"detections": detections, "latency_ms": round(latency_ms, 1)}


@app.post("/recognize")
async def recognize_faces(file: UploadFile = File(...)):
    \"\"\"Detect and extract face embeddings (RetinaFace + ArcFace).\"\"\"
    img = _decode_image(await file.read())
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})
    if face_app is None:
        return JSONResponse(status_code=503, content={"error": "Face analyzer not loaded"})
    t0 = time.time()
    faces = face_app.get(img)
    latency_ms = (time.time() - t0) * 1000
    face_list = [
        {"bbox": [round(x, 1) for x in f.bbox.tolist()],
         "det_score": round(float(f.det_score), 4),
         "embedding_dim": len(f.embedding)}
        for f in faces
    ]
    return {"faces": face_list, "latency_ms": round(latency_ms, 1)}


@app.post("/pipeline")
async def full_pipeline(file: UploadFile = File(...)):
    \"\"\"Full end-to-end pipeline: enhance -> detect -> recognize faces.\"\"\"
    img = _decode_image(await file.read())
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})
    t_start = time.time()
    # Enhance
    t0 = time.time()
    enhanced = _enhance_image(img)
    enh_ms = (time.time() - t0) * 1000
    # Detect
    detections = []
    det_ms = 0.0
    if detector:
        t0 = time.time()
        for r in detector(enhanced, verbose=False):
            for box in r.boxes:
                detections.append({
                    "class": r.names[int(box.cls)],
                    "confidence": round(float(box.conf), 4),
                    "bbox": [round(x, 1) for x in box.xyxy[0].tolist()],
                })
        det_ms = (time.time() - t0) * 1000
    # Recognize
    face_list = []
    face_ms = 0.0
    if face_app:
        t0 = time.time()
        faces = face_app.get(enhanced)
        face_ms = (time.time() - t0) * 1000
        face_list = [
            {"bbox": [round(x, 1) for x in f.bbox.tolist()],
             "det_score": round(float(f.det_score), 4)}
            for f in faces
        ]
    total_ms = (time.time() - t_start) * 1000
    return {
        "detections": detections,
        "faces": face_list,
        "timing": {
            "enhance_ms": round(enh_ms, 1),
            "detect_ms":  round(det_ms, 1),
            "face_ms":    round(face_ms, 1),
            "total_ms":   round(total_ms, 1),
        },
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "detector":  detector  is not None,
        "face_app":  face_app  is not None,
        "enh_model": enh_model is not None,
    }

print("app.py written — endpoints: /enhance /detect /recognize /pipeline /health")
"""]
    print(f"Replaced Cell {app_py_cell} app.py with all 4 endpoints")

# ── 3. Replace final cell with deliverables checklist ─────────────────────
# Find the last cell (project complete print)
last_idx = len(cells) - 1
last_src = "".join(cells[last_idx]["source"])
if "PROJECT COMPLETE" in last_src or "Phase Summary" in last_src:
    cells[last_idx]["source"] = ["""\
import os, pandas as pd

PROJECT_DIR = '/content/drive/MyDrive/computer_vision'
RESULTS_DIR = f'{PROJECT_DIR}/results/phase6'

# ── Literature Benchmarks ─────────────────────────────────────────────────
lit = pd.DataFrame([
    {'Metric': 'End-to-End Latency',    'Target': '< 2 s',  'Tool': 'custom timer'},
    {'Metric': 'Concurrent Users',      'Target': '10+',     'Tool': 'Locust / k6'},
    {'Metric': 'Task Logging Accuracy', 'Target': '100%',    'Tool': 'DB audit'},
    {'Metric': 'System Stability',      'Target': '> 99%',   'Tool': 'uptime monitor'},
])

print('=' * 60)
print('TABLE 6.1: System-Level Testing Targets (guide §6.4)')
print('=' * 60)
print(lit.to_string(index=False))

# ── Deliverables Checklist ─────────────────────────────────────────────────
print('\\n' + '=' * 60)
print('PHASE 6 DELIVERABLES CHECKLIST')
print('=' * 60)

api_ok    = os.path.exists('/content/computer_vision_expirement/app.py')
docker_ok = os.path.exists('/content/computer_vision_expirement/Dockerfile')
bench_ok  = os.path.exists(f'{RESULTS_DIR}/system_benchmark.csv')

lat_ok = False
if bench_ok:
    df = pd.read_csv(f'{RESULTS_DIR}/system_benchmark.csv')
    avg_row = df[df['Metric'] == 'Avg Latency']
    if not avg_row.empty:
        val_str = avg_row.iloc[0]['Value']  # e.g. "450.3 ms"
        try:
            lat_ms = float(val_str.split()[0])
            lat_ok = lat_ms < 2000
        except Exception:
            pass

items = [
    ('REST API (app.py) with all endpoints',      api_ok,    'run Cell 6.2 (%%writefile app.py)'),
    ('Dockerfile + docker-compose.yml',           docker_ok, 'run Cell 6.4 (%%writefile Dockerfile)'),
    ('System benchmark CSV',                      bench_ok,  'run Cell 6.6 (system test)'),
    ('End-to-end latency < 2 s',                  lat_ok,    'check Cell 6.6 latency results'),
]

all_ok = True
for label, ok, hint in items:
    status = '[OK]' if ok else f'[MISSING -- {hint}]'
    print(f'  {status:<52} {label}')
    if not ok:
        all_ok = False

print()
if bench_ok:
    print('System benchmark:')
    print(pd.read_csv(f'{RESULTS_DIR}/system_benchmark.csv').to_string(index=False))

print('\\n' + '=' * 60)
print(f'Phase 6 Status: {"COMPLETE" if all_ok else "IN PROGRESS"}')
print('=' * 60)
if all_ok:
    print('All 6 phases complete. Project ready for write-up.')
"""]
    print(f"Replaced Cell {last_idx} with deliverables + checklist")

# Insert ## 6.5 Deliverables heading before the deliverables code cell
deliv_idx = None
for i, c in enumerate(cells):
    if c["cell_type"] == "code" and "PHASE 6 DELIVERABLES" in "".join(c["source"]):
        deliv_idx = i
        break

if deliv_idx is not None:
    prev = cells[deliv_idx - 1] if deliv_idx > 0 else None
    if prev is None or "6.5 Deliverables" not in "".join(prev["source"]):
        cells.insert(deliv_idx, {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 6.5 Deliverables\n"],
        })
        print(f"Inserted ## 6.5 Deliverables heading at index {deliv_idx}")

# ── Save ───────────────────────────────────────────────────────────────────
nb["cells"] = cells
json.dump(nb, open(NB_PATH, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

print(f"\nDone. Notebook now has {len(cells)} cells.")
for i, c in enumerate(cells):
    src = "".join(c["source"])[:100].replace("\n", " ")
    print(f"  [{i:02d}] {c['cell_type']:<8} | {src}")
