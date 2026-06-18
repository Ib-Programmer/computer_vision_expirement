"""
Outdoor Detection & Face Recognition REST API — HuggingFace Spaces Edition
Matches the Spring Boot InferenceClient contract exactly.

Endpoints:
  POST /pipeline              download → enhance → detect → recognize
  POST /enrol                 register a named face identity (in-memory)
  DELETE /enrol/{id}          remove a registered identity
  GET  /health                service status

Spring Boot sends JSON with snake_case keys (Jackson SNAKE_CASE strategy):
  /pipeline  {"image_url": "https://...", "condition": "foggy|rainy|low-light|clear|auto"}
  /enrol     {"name": "Alice", "image_url": "https://..."}

HuggingFace Space env vars (Settings → Variables and secrets):
  INTERNAL_TOKEN   must match Spring Boot INFERENCE_TOKEN
  PROJECT_DIR      path to model weights (default /app/models)
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

detector     = None
detector_fmt = None
face_app     = None
_gallery: dict[str, dict] = {}   # embedding_id → {name, embedding}

INTERNAL_TOKEN = os.environ.get("INTERNAL_TOKEN", "dev-only-internal-token")

_COND_ROUTE = {
    "foggy":     "fog:restormer",
    "rainy":     "rain:restormer",
    "low-light": "low_light:zerodce",
    "clear":     "clear:none",
    "auto":      "auto:clahe",
}

# ── helpers ──────────────────────────────────────────────────────────────────

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

# ── startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global detector, detector_fmt, face_app
    MODELS = os.environ.get("PROJECT_DIR", "/app/models")

    try:
        from ultralytics import YOLO
        for path, fmt in [
            (f"{MODELS}/phase5/yolov8n_best.onnx",                   "onnx"),
            (f"{MODELS}/phase3/yolov8n_outdoor_aug/weights/best.pt", "pytorch_fp32"),
            (f"{MODELS}/phase3/yolov8n_baseline/weights/best.pt",    "pytorch_fp32"),
            ("yolov8n.pt",                                             "pytorch_pretrained"),
        ]:
            if os.path.exists(path):
                detector = YOLO(path); detector_fmt = fmt
                print(f"[startup] Detector: {path} [{fmt}]"); break
    except Exception as e:
        print(f"[startup] Detector load failed: {e}")

    try:
        from insightface.app import FaceAnalysis
        # buffalo_l is auto-downloaded from insightface CDN on first run
        face_app = FaceAnalysis(name="buffalo_l",
                                providers=["CPUExecutionProvider"])
        face_app.prepare(ctx_id=-1, det_size=(640, 640))
        print("[startup] Face analyzer: SCRFD-10GF + ArcFace (CPU)")
    except Exception as e:
        print(f"[startup] Face analyzer load failed: {e}")

# ── endpoints ────────────────────────────────────────────────────────────────

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
    enhanced = _clahe(img)
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
        "enhancement_route":  _COND_ROUTE.get(condition, "auto:clahe"),
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
        "status":          "ok",
        "detector":        detector is not None,
        "detector_format": detector_fmt,
        "face_app":        face_app is not None,
        "gallery_size":    len(_gallery),
    }
