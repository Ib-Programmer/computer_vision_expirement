"""
Outdoor Detection & Face Recognition REST API — HuggingFace Spaces Edition

Endpoints:
  POST /pipeline              download → enhance → detect → recognize
  POST /enrol                 register a named face identity (in-memory)
  DELETE /enrol/{id}          remove a registered identity
  GET  /health                service status

Spring Boot sends JSON with snake_case keys (Jackson SNAKE_CASE strategy):
  /pipeline  {"image_url": "https://...", "condition": "foggy|rainy|low-light|clear|auto"}
  /enrol     {"name": "Alice", "image_url": "https://..."}

HuggingFace Space env vars (Settings → Variables and secrets):
  HF_MODEL_REPO    your HF model repo, e.g. "ibmuhd557/cv-thesis-models"
  HF_TOKEN         HF read token (only needed if repo is private)
  INTERNAL_TOKEN   must match Spring Boot INFERENCE_TOKEN
  PROJECT_DIR      override model cache path (default /app/models)
"""
import base64, os, time, uuid
from typing import Optional

import cv2
import numpy as np
import requests as _requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="CV Thesis Inference API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── global model handles ──────────────────────────────────────────────────────
detector      = None
detector_fmt  = None
face_app      = None
enhance_zero  = None   # Zero-DCE++ (low-light)
enhance_ffa   = None   # FFA-Net     (fog)

_gallery: dict[str, dict] = {}   # embedding_id → {name, embedding}

INTERNAL_TOKEN = os.environ.get("INTERNALTOKEN", "thesissecret2026")
HF_REPO        = "IbProgrammmer/cv-thesis-models"
HF_TOKEN       = os.environ.get("HFTOKEN", "")
MODELS         = "/tmp/models"   # /tmp is always writable by any user

# ── HF Hub model manifest ─────────────────────────────────────────────────────
# filename in HF repo → local path under MODELS/
HF_MODELS = {
    # Detection (pick the best available at startup)
    "yolov8n_best.onnx":           "yolov8n_best.onnx",
    "yolov8n_outdoor_aug_best.pt": "yolov8n_outdoor_aug_best.pt",
    "yolov8n_baseline_best.pt":    "yolov8n_baseline_best.pt",
    "rtdetr_outdoor_aug_best.pt":  "rtdetr_outdoor_aug_best.pt",
    "yolov8n_int8.onnx":           "yolov8n_int8.onnx",
    # Enhancement
    "zero_dce_pp.pth":             "zero_dce_pp.pth",
    "ffa_net_outdoor.pk":          "ffa_net_outdoor.pk",
    # Restormer is already on HF Hub at deepinv/Restormer — downloaded separately
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _download(url: str) -> np.ndarray:
    resp = _requests.get(url, timeout=20)
    resp.raise_for_status()
    arr = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("imdecode returned None")
    return img


def _xyxy_to_xywh(coords) -> dict:
    x1, y1, x2, y2 = [float(v) for v in coords]
    return {"x": round(x1, 1), "y": round(y1, 1),
            "w": round(x2 - x1, 1), "h": round(y2 - y1, 1)}


def _to_data_uri(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _clahe(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _match(embedding: np.ndarray, threshold: float = 0.4):
    if not _gallery:
        return "unknown", "unknown", 0.0
    q = embedding / (np.linalg.norm(embedding) + 1e-9)
    best_id, best_name, best_sim = "unknown", "unknown", 0.0
    for eid, entry in _gallery.items():
        ref = entry["embedding"]
        sim = float(np.dot(q, ref / (np.linalg.norm(ref) + 1e-9)))
        if sim > best_sim:
            best_sim, best_id, best_name = sim, eid, entry["name"]
    if best_sim < threshold:
        return "unknown", "unknown", round(best_sim, 4)
    return best_name, best_id, round(best_sim, 4)


# ── model download from HF Hub ────────────────────────────────────────────────

def _pull_from_hub():
    """Download all models from HF Hub into MODELS dir on first boot."""
    if not HF_REPO:
        print("[startup] HF_MODEL_REPO not set — using pre-baked or pretrained models only")
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[startup] huggingface_hub not installed — skipping Hub download")
        return

    os.makedirs(MODELS, exist_ok=True)
    token = HF_TOKEN or None
    for hf_filename, local_name in HF_MODELS.items():
        dest = os.path.join(MODELS, local_name)
        if os.path.exists(dest):
            print(f"[hub] cached   {local_name}")
            continue
        try:
            hf_hub_download(
                repo_id=HF_REPO, filename=hf_filename,
                token=token, local_dir=MODELS,
            )
            # hf_hub_download saves with the hf_filename; rename if different
            downloaded = os.path.join(MODELS, hf_filename)
            if downloaded != dest and os.path.exists(downloaded):
                os.rename(downloaded, dest)
            print(f"[hub] downloaded {local_name}  ({os.path.getsize(dest)//1024} KB)")
        except Exception as e:
            print(f"[hub] skip {hf_filename}: {e}")

    # Restormer: already on public HF Hub at deepinv/Restormer
    rest_dest = os.path.join(MODELS, "restormer_deraining.pth")
    if not os.path.exists(rest_dest):
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(
                repo_id="deepinv/Restormer",
                filename="deraining.pth",
                local_dir=MODELS,
            )
            os.rename(p, rest_dest)
            print(f"[hub] downloaded restormer_deraining.pth  ({os.path.getsize(rest_dest)//1024} KB)")
        except Exception as e:
            print(f"[hub] Restormer skip: {e}")


# ── enhancement loaders ───────────────────────────────────────────────────────

def _load_zero_dce(weights_path: str):
    """Load Zero-DCE++ for low-light enhancement. Requires torch."""
    try:
        import torch
        import torch.nn as nn

        class _DCENet(nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = nn.ReLU(inplace=True)
                n = 32
                self.e_conv1 = nn.Conv2d(3, n, 3, 1, 1, bias=True)
                self.e_conv2 = nn.Conv2d(n, n, 3, 1, 1, bias=True)
                self.e_conv3 = nn.Conv2d(n, n, 3, 1, 1, bias=True)
                self.e_conv4 = nn.Conv2d(n, n, 3, 1, 1, bias=True)
                self.e_conv5 = nn.Conv2d(n * 2, n, 3, 1, 1, bias=True)
                self.e_conv6 = nn.Conv2d(n * 2, n, 3, 1, 1, bias=True)
                self.e_conv7 = nn.Conv2d(n * 2, 24, 3, 1, 1, bias=True)

            def forward(self, x):
                x1 = self.relu(self.e_conv1(x))
                x2 = self.relu(self.e_conv2(x1))
                x3 = self.relu(self.e_conv3(x2))
                x4 = self.relu(self.e_conv4(x3))
                x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
                x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
                x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
                r = torch.split(x_r, 3, dim=1)
                out = x
                for ri in r:
                    out = out + ri * (1 - out)
                return out

        net = _DCENet()
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        net.load_state_dict(state, strict=False)
        net.eval()
        print(f"[startup] Zero-DCE++ loaded: {weights_path}")
        return net
    except Exception as e:
        print(f"[startup] Zero-DCE++ not loaded ({e}) — using CLAHE fallback")
        return None


def _load_ffa(weights_path: str):
    """Load FFA-Net for dehazing. Requires torch."""
    try:
        import torch
        import pickle
        with open(weights_path, "rb") as f:
            net = pickle.load(f)
        net.eval()
        print(f"[startup] FFA-Net loaded: {weights_path}")
        return net
    except Exception as e:
        print(f"[startup] FFA-Net not loaded ({e}) — using CLAHE fallback")
        return None


def _enhance(img_bgr: np.ndarray, condition: str) -> tuple[np.ndarray, str]:
    """Route enhancement by weather condition. Returns (enhanced_bgr, route_label)."""
    try:
        import torch

        if condition in ("low-light",) and enhance_zero is not None:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
            with torch.no_grad():
                out = enhance_zero(t).squeeze(0).permute(1, 2, 0).numpy()
            return cv2.cvtColor((out * 255).clip(0, 255).astype(np.uint8),
                                cv2.COLOR_RGB2BGR), "low_light:zero_dce++"

        if condition in ("foggy",) and enhance_ffa is not None:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
            with torch.no_grad():
                out = enhance_ffa(t).squeeze(0).permute(1, 2, 0).numpy()
            return cv2.cvtColor((out * 255).clip(0, 255).astype(np.uint8),
                                cv2.COLOR_RGB2BGR), "fog:ffa_net"

    except ImportError:
        pass  # torch not installed — fall through to CLAHE

    # CLAHE fallback for all conditions (also used when condition="clear" or "auto")
    return _clahe(img_bgr), f"{condition}:clahe"


# ── startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global detector, detector_fmt, face_app, enhance_zero, enhance_ffa

    _pull_from_hub()

    # ── detector (prefer ONNX, fallback to .pt, fallback to pretrained) ──────
    try:
        from ultralytics import YOLO
        candidates = [
            (f"{MODELS}/yolov8n_best.onnx",           "onnx"),
            (f"{MODELS}/yolov8n_int8.onnx",            "onnx_int8"),
            (f"{MODELS}/yolov8n_outdoor_aug_best.pt",  "pytorch_aug"),
            (f"{MODELS}/yolov8n_baseline_best.pt",     "pytorch_baseline"),
            (f"{MODELS}/rtdetr_outdoor_aug_best.pt",   "rtdetr"),
        ]
        for path, fmt in candidates:
            if os.path.exists(path):
                detector = YOLO(path)
                detector_fmt = fmt
                print(f"[startup] Detector: {os.path.basename(path)} [{fmt}]")
                break
        if detector is None:
            # pretrained fallback — YOLO auto-downloads yolov8n.pt on first call
            detector = YOLO("yolov8n.pt")
            detector_fmt = "pytorch_pretrained"
            print("[startup] Detector: yolov8n.pt [pytorch_pretrained] (auto-downloaded)")
    except Exception as e:
        print(f"[startup] Detector load failed: {e}")

    # ── face analyzer (buffalo_l auto-downloads from InsightFace CDN) ─────────
    try:
        from insightface.app import FaceAnalysis
        face_app = FaceAnalysis(name="buffalo_l",
                                providers=["CPUExecutionProvider"])
        face_app.prepare(ctx_id=-1, det_size=(640, 640))
        print("[startup] Face analyzer: SCRFD-10GF + ArcFace w600k_r50 (CPU)")
    except Exception as e:
        print(f"[startup] Face analyzer load failed: {e}")

    # ── enhancement models (optional — requires torch) ────────────────────────
    zdce_path = f"{MODELS}/zero_dce_pp.pth"
    if os.path.exists(zdce_path):
        enhance_zero = _load_zero_dce(zdce_path)

    ffa_path = f"{MODELS}/ffa_net_outdoor.pk"
    if os.path.exists(ffa_path):
        enhance_ffa = _load_ffa(ffa_path)

    if enhance_zero is None and enhance_ffa is None:
        print("[startup] No enhancement models loaded — CLAHE used for all conditions")


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.post("/pipeline")
async def pipeline(body: dict,
                   x_internal_token: Optional[str] = Header(None)):
    t_total = time.time()
    image_url = body.get("image_url")
    condition = body.get("condition", "auto")
    if not image_url:
        raise HTTPException(status_code=400, detail="image_url is required")

    try:
        img = _download(image_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot download image: {e}")
    h, w = img.shape[:2]

    t0 = time.time()
    enhanced, enh_route = _enhance(img, condition)
    enh_ms = (time.time() - t0) * 1000

    t0 = time.time()
    detections = []
    if detector:
        for r in detector(enhanced, verbose=False):
            for box in r.boxes:
                detections.append({
                    "class":      r.names[int(box.cls)],
                    "confidence": round(float(box.conf), 4),
                    "bbox":       _xyxy_to_xywh(box.xyxy[0].tolist()),
                })
    det_ms = (time.time() - t0) * 1000

    t0 = time.time()
    recognitions = []
    if face_app:
        for face in face_app.get(enhanced):
            name, eid, conf = _match(face.embedding)
            recognitions.append({
                "identity":    name,
                "identity_id": eid,
                "confidence":  conf,
                "bbox":        _xyxy_to_xywh(face.bbox.tolist()),
            })
    rec_ms = (time.time() - t0) * 1000
    total_ms = (time.time() - t_total) * 1000

    return {
        "detections":         detections,
        "recognitions":       recognitions,
        "enhanced_image_url": _to_data_uri(enhanced),
        "enhancement_route":  enh_route,
        "condition":          condition,
        "latency_ms": {
            "enhancement": round(enh_ms, 1),
            "detection":   round(det_ms, 1),
            "recognition": round(rec_ms, 1),
            "total":       round(total_ms, 1),
        },
        "image_width":  w,
        "image_height": h,
    }


@app.post("/enrol")
async def enrol(body: dict,
                x_internal_token: Optional[str] = Header(None)):
    if face_app is None:
        raise HTTPException(status_code=503, detail="Face analyzer not loaded")
    name      = body.get("name")
    image_url = body.get("image_url")
    if not name or not image_url:
        raise HTTPException(status_code=400, detail="name and image_url are required")
    try:
        img = _download(image_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot download image: {e}")
    faces = face_app.get(img)
    if not faces:
        raise HTTPException(status_code=422, detail="No face detected in enrolment image")
    emb = faces[0].embedding.astype(np.float32)
    emb /= np.linalg.norm(emb) + 1e-9
    eid = str(uuid.uuid4())
    _gallery[eid] = {"name": name, "embedding": emb}
    print(f"[enrol] {name} → {eid}  (gallery: {len(_gallery)})")
    return {"embedding_id": eid}


@app.delete("/enrol/{embedding_id}")
async def delete_enrol(embedding_id: str,
                       x_internal_token: Optional[str] = Header(None)):
    _gallery.pop(embedding_id, None)
    return {"status": "deleted", "embedding_id": embedding_id}


@app.get("/health")
async def health():
    return {
        "status":           "ok",
        "detector":         detector is not None,
        "detector_format":  detector_fmt,
        "face_app":         face_app is not None,
        "enhance_zero_dce": enhance_zero is not None,
        "enhance_ffa_net":  enhance_ffa is not None,
        "gallery_size":     len(_gallery),
    }
