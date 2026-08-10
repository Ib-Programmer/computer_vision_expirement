# Project Implementation Scope

**Title:** Design and Implementation of an Object Recognition System for Complex Outdoor Scenes
**Chinese Title:** 复杂户外场景目标识别系统的设计与实现
**Student:** Muhammad Bashir Dantani (SL24225008) — USTC Software Engineering (085405)
**Supervisors:** Lou Wenqi (internal) · Yan Yu (practice)
**Expected Defence:** December 2026

---

## System Overview

An end-to-end cloud-based vision system that accepts user-uploaded images or videos, runs a three-stage deep-learning pipeline (enhancement → detection → recognition), and returns annotated results through a web interface and REST API.

```
User Upload
    │
    ▼
[Image Enhancement]          Zero-DCE++ (low-light) · FFA-Net (fog) · CLAHE (fallback)
    │
    ▼
[Object Detection]           YOLOv8n ONNX  conf=0.45  iou=0.45
    │
    ▼
[Face Recognition]           InsightFace buffalo_l (RetinaFace + ArcFace w600k_r50)
                             FAISS cosine similarity gallery  threshold=0.4
    │
    ▼
Annotated Result  ──►  Spring Boot API  ──►  Next.js Frontend
```

---

## Repository Structure

| Repo / Folder | Role |
|---|---|
| `computer_vision/` | Experiments, notebooks, HF Space source (`deploy/`) |
| `computer_vision_api/` | Spring Boot REST backend |
| `computer_vision_web_app/` | Next.js frontend |

---

## 1. Inference Server (`computer_vision/deploy/`)

**Deployment:** Docker container on Hugging Face Spaces (`IbProgrammmer/cv_thesis`) · Port 7860 · FastAPI + Uvicorn

| Component | Detail |
|---|---|
| `app.py` | FastAPI app — single `/predict` endpoint |
| `Dockerfile` | `python:3.10-slim`, non-root user `appuser`, port 7860 |
| `requirements.txt` | ultralytics, insightface, onnxruntime, torch, faiss-cpu, fastapi, opencv |

### Pipeline stages

**Stage 1 — Image Enhancement**
- `Zero-DCE++` — low-light enhancement (primary, GPU/CPU)
- `FFA-Net` — dehazing (optional, loaded from HF model repo `IbProgrammmer/cv-thesis-models`)
- `CLAHE` — OpenCV fallback, always available, no GPU needed
- Selection: request field `condition` (`lowlight` / `fog` / `auto` / `clear`)

**Stage 2 — Object Detection**
- Model: `YOLOv8n` exported to ONNX
- Inference: `ultralytics` ONNX runtime
- Thresholds: `conf=0.45`, `iou=0.45` (raised from default 0.25 to cut false positives)
- Output: bounding boxes, class labels, confidence scores

**Stage 3 — Face Recognition**
- Detector: `RetinaFace` (via InsightFace `buffalo_l`)
- Embedder: `ArcFace w600k_r50` (512-dim embedding)
- Gallery: FAISS `IndexFlatIP` cosine similarity, threshold `0.4`
- Supports: face registration + real-time identity matching across multiple identities

---

## 2. Backend API (`computer_vision_api/`)

**Stack:** Java 17 · Spring Boot 3 · PostgreSQL · MinIO · JWT · Maven

### Modules

| Package | Responsibility |
|---|---|
| `auth` | User registration, login, JWT issuance & validation |
| `task` | Task CRUD, async processing queue, status lifecycle |
| `identity` | Face identity registration, gallery management |
| `inference` | HTTP client calling HF Space `/predict` endpoint |
| `storage` | MinIO presigned URL generation for media assets |
| `realtime` | (SSE / WebSocket) real-time stream endpoint |
| `config` | Security config, CORS, async executor, DB config |

### Key design decisions

- **Async race condition fix:** `TransactionSynchronizationManager.registerSynchronization().afterCommit()` in `TaskService.create()` — defers async dispatch until DB transaction commits, preventing tasks stuck at `QUEUED`.
- **Image resize before encoding:** `TaskProcessor` caps images at `MAX_SIDE=1280` using `BufferedImage`/`Graphics2D` before Base64 encoding, preventing HF Space payload overflows.
- **Task status lifecycle:** `QUEUED → PROCESSING → DONE / FAILED`

### Database (PostgreSQL)

| Table | Purpose |
|---|---|
| `users` | UUID PK, username, password_hash (bcrypt), role, created_at |
| `tasks` | UUID PK, user_id FK, file_path, model_used, status, timestamps |
| `results` | UUID PK, task_id FK, objects_json, faces_json, output_path, latency_ms |
| `identities` | Face gallery entries linked to users |

---

## 3. Web Frontend (`computer_vision_web_app/`)

**Stack:** Next.js 14 (App Router) · TypeScript · Tailwind CSS · Zustand · React Query

### Pages

| Route | Page |
|---|---|
| `/dashboard` | Stats overview, recent tasks, refresh button |
| `/upload` | Image/video upload form, model/condition selection |
| `/tasks` | Task list with status polling |
| `/tasks/[id]` | Task detail + annotated result view |
| `/realtime` | Live camera stream inference |
| `/identities` | Face identity gallery management |
| `/profile` | User profile |
| `/settings` | App settings |

### Key components

| Component | Role |
|---|---|
| `task-notifier.tsx` | Background poller (2 s active / 8 s idle) — fires toast + notification on task completion |
| `notification-bell.tsx` | Bell icon with unread badge, dropdown panel, mark-read / clear |
| `topnav.tsx` | Sticky nav with tab pills, theme toggle, notification bell, user dropdown |
| `bbox-overlay.tsx` | Renders detection bounding boxes + face labels over result image |

### State management

| Store | Key | Contents |
|---|---|---|
| `auth-store.ts` | `cv-app-auth` | JWT token, user object (Zustand persist) |
| `notification-store.ts` | `cv-app-notifications` | Last 50 notifications, read state (Zustand persist) |
| `settings-store.ts` | — | App-level settings |

### API layer

- `lib/api.ts` — Axios instance, JWT interceptor, typed request/response wrappers
- Next.js rewrite proxy: `/api/*` → `http://localhost:8080/api/*` (dev)

---

## 4. Experiments (`computer_vision/notebooks/` & `scripts/`)

All experiments run on **Google Colab T4 GPU** (free tier / Colab Pro $10 budget).

| Phase | Experiment |
|---|---|
| Phase 1 | Enhancement baseline — SSIM / PSNR / NIQE on ExDark & ACDC |
| Phase 2 | Object detection benchmark — YOLOv8n/s, RT-DETR on COCO, RTTS, Foggy Cityscapes |
| Phase 3 | Face recognition — ArcFace on LFW, WiderFace (TAR@FAR, AP) |
| Phase 4 | ONNX export + runtime benchmark (latency, FPS, memory) |
| Phase 5 | TensorRT FP16 quantisation — latency vs accuracy trade-off |
| Phase 6 | End-to-end pipeline latency (enhancement + detection + recognition) |

**Planned model upgrade:** YOLOv8n → YOLOv8s for improved detection accuracy in thesis experiments.

**Datasets:**
- Enhancement: ExDark, ACDC, RTTS, Foggy Cityscapes
- Detection: COCO val2017, RTTS, Foggy Cityscapes
- Face: LFW, WiderFace

---

## 5. Bugs Fixed During Implementation

| Bug | Root Cause | Fix |
|---|---|---|
| Tasks stuck at `QUEUED` forever | `@Async` thread started before `@Transactional` committed | `afterCommit()` callback via `TransactionSynchronizationManager` |
| HF Space inference failures | Base64 payload too large for high-res images | Server-side resize to `MAX_SIDE=1280` in `TaskProcessor` |
| YOLO false positives | Default `conf=0.25` too permissive for outdoor scenes | Raised to `conf=0.45, iou=0.45` in `app.py` |
| Toast messages hidden | Positioned `bottom-4`, stacked wrong order | Moved to `top-4`, reversed render order |
| Dashboard refresh broken | Missing `identitiesQ.refetch()` call | Added refetch + spinning icon + disabled state |

---

## 6. Deployment Topology

```
Browser
  │  HTTPS
  ▼
Next.js (localhost:3000)
  │  /api/* rewrite
  ▼
Spring Boot API (localhost:8080)
  │  HTTP POST /predict  (Base64 image)
  ▼
HF Space — FastAPI (IbProgrammmer/cv_thesis, port 7860)
  │
  ├── Zero-DCE++ / FFA-Net / CLAHE
  ├── YOLOv8n ONNX
  └── InsightFace buffalo_l + FAISS gallery

Spring Boot also talks to:
  ├── PostgreSQL  (user/task/result data)
  └── MinIO       (media files, presigned URLs)
```

---

## 7. Pending Work (Aug – Dec 2026)

- [ ] Upgrade detection model to YOLOv8s; export ONNX; redeploy to HF Space
- [ ] Run Phase 1–3 comparative experiments; populate thesis Chapter 3 tables
- [ ] TensorRT FP16 benchmarks (Phase 5)
- [ ] End-to-end latency benchmarks (Phase 6)
- [ ] Write thesis Chapters 3 & 4
