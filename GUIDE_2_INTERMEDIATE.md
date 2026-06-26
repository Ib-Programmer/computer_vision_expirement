# Guide 2 — Intermediate
### Technical Implementation of the CV Thesis Pipeline

*Audience: computer science student or software engineer who knows Python and basic machine learning. Read GUIDE_1_INTRO.md first.*

---

## 1. How Image Enhancement Works

### CLAHE (Contrast Limited Adaptive Histogram Equalisation)

A histogram of an image counts how many pixels exist at each brightness level (0 = black, 255 = white). A dark image has most pixels bunched at the low end. **Histogram equalisation** redistributes those pixels to span the full range, brightening the image.

CLAHE improves on this in two ways:
1. **Adaptive**: the image is divided into small tiles (8×8 by default) and equalisation is applied independently per tile. This prevents a bright sky from influencing the contrast adjustment on a dark road.
2. **Clip limit** (clipLimit=2.0): prevents over-amplification of noise by capping how much any histogram bin can be stretched.

CLAHE operates in the LAB colour space (Lightness, A-channel, B-channel) — only the L channel is equalised, leaving colour hue and saturation unchanged.

```
BGR image → convert to LAB → equalise L channel → convert back to BGR
```

Runtime: < 5 ms. No GPU required. No training.

---

### Zero-DCE++ Architecture

Zero-DCE++ maps each pixel's brightness to a new brightness through an iterative curve. The network outputs 24 channels split into 8 groups of 3 — each group is one iteration of a curve map applied to RGB.

The encoder-decoder:

```
Input (3ch)
  → e1: Conv(3→32, 3×3)       ReLU
  → e2: Conv(32→32, 3×3)      ReLU
  → e3: Conv(32→32, 3×3)      ReLU
  → e4: Conv(32→32, 3×3)      ReLU
  → e5: Conv(64→32, 3×3)      ReLU  [cat(e3, e4)]
  → e6: Conv(64→32, 3×3)      ReLU  [cat(e2, e5)]
  → e7: Conv(64→24, 3×3)      Tanh  [cat(e1, e6)]
```

The 24-channel output is split into 8 curve maps (3 channels each). Each map is applied iteratively:

```python
out = x
for r in curve_maps:   # 8 iterations
    out = out + r * (out - out * out)
```

This formula (`x + r*(x - x²)`) is a smooth, bounded enhancement curve. When r > 0 it brightens; r < 0 darkens. The model learns the r values from image statistics — no reference image (no paired training data) is needed.

Total parameters: **10,561**. This is intentionally tiny — the original paper targets mobile and edge devices.

---

### FFA-Net: Feature Fusion Attention for Dehazing

FFA-Net uses the **atmospheric scattering model** to understand fog:

```
I(x) = J(x) · t(x) + A · (1 - t(x))
```

Where:
- `I(x)` = observed hazy pixel
- `J(x)` = clean scene radiance (what we want)
- `t(x)` = transmission map (how much light reaches camera)
- `A` = global atmospheric light (the white glow)

FFA-Net learns to estimate `t(x)` and `A` implicitly through its attention mechanism, then inverts the equation to recover `J(x)`.

Architecture highlights:
- **Channel Attention**: learns which feature channels contain fog-relevant information
- **Pixel Attention**: learns which spatial locations have thick vs thin fog
- **Feature Fusion**: concatenates multi-scale features to handle fog at different depths

Runtime: **4,226 ms** on T4 — too slow for live video, used for single-image analysis only.

---

## 2. YOLOv8n: Object Detection Architecture

### Anchor-Free Detection

Traditional detectors (Faster R-CNN, YOLOv5) pre-define anchor boxes — fixed shapes that the model adjusts to fit each object. YOLOv8 is anchor-free: it directly predicts the centre, width, and height of each bounding box without anchors.

### Network Structure

```
Input (416×416×3)
  ↓
Backbone (C2f modules + SPPF)  — extracts features at multiple scales
  ↓
Neck (FPN + PAN)               — fuses features from different scales
  ↓
Head (3 scale outputs)         — predicts boxes at 52×52, 26×26, 13×13 grids
  ↓
NMS (non-maximum suppression)  — removes duplicate overlapping boxes
  ↓
Output: list of (class, confidence, x, y, w, h)
```

**C2f** (Cross-Stage Partial with 2 bottleneck blocks) — the main building block. Splits the feature map, runs some through bottleneck layers, then concatenates. Reduces parameters while keeping accuracy.

**SPPF** (Spatial Pyramid Pooling Fast) — applies max-pooling at multiple kernel sizes and concatenates, giving the network a large receptive field without extra layers.

**FPN + PAN** — Feature Pyramid Network passes features top-down (large → small scale); Path Aggregation Network passes them bottom-up. Together they give strong features at every scale, important for detecting both distant-small and close-large objects.

### Training Configuration

| Parameter | Value | Reasoning |
|---|---|---|
| imgsz | 416 | Fits batch=32 in T4's 16 GB VRAM; smaller than default 640 |
| batch | 32 | Maximises GPU utilisation |
| epochs | 25 | Compute budget exhausted; model had not converged |
| fraction | 0.07 | 7% of BDD100K = ~4,900 training images |
| conf threshold | 0.45 | Above default 0.25; reduces phantom detections in fog |
| iou threshold | 0.45 | IoU overlap threshold for NMS |
| optimizer | AdamW, lr=0.000714 | Auto-tuned by Ultralytics from lr0=0.01 |

### Loss Function

YOLOv8 trains with three losses added together:

- **Box loss (CIoU)**: penalises incorrect box centre, size, and aspect ratio
- **Classification loss (BCE)**: penalises wrong class predictions
- **Distribution Focal Loss (DFL)**: a regression loss on the box boundary distribution

---

## 3. Face Recognition: InsightFace Buffalo_L

### SCRFD-10GF: Face Detection

SCRFD (Sample and Computation Redistribution for Face Detection) is a single-stage face detector with 10 GFLOPs of compute. It predicts face bounding boxes at three scales simultaneously (8×, 16×, 32× stride), handling faces from very small (16×16 pixels) to large (640×640 pixels) in one forward pass.

Detection input is resized to 640×640. Output: list of (bounding box, detection confidence score).

### ArcFace w600k_r50: Face Embedding

ArcFace takes a 112×112 cropped and aligned face image and outputs a **512-dimensional embedding vector**.

The backbone is ResNet-50 (r50), trained on 600,000 identities (`w600k`) — approximately 5 million images.

**Why 512 dimensions?** The embedding space needs to be large enough that faces of different people are genuinely separated and faces of the same person cluster tightly, even across lighting, pose, and expression changes.

**Normalisation**: every embedding is L2-normalised to unit length before storage and comparison. This converts the dot product search into a cosine similarity search.

```
similarity(A, B) = A · B   (dot product of unit vectors = cosine of angle between them)
```

Two photos of the same person: similarity typically 0.6–0.9.
Two different people: similarity typically 0.0–0.3.
Threshold used: **0.4** (above → recognised; below → Unknown).

---

## 4. FAISS Gallery Search

FAISS (Facebook AI Similarity Search) stores the enrolled face embeddings as a matrix and searches it with a single matrix multiplication.

**Index type**: `IndexFlatIP` — Flat (exact search, no approximation) Inner Product (dot product, which equals cosine similarity for normalised vectors).

```python
index = faiss.IndexFlatIP(512)    # 512-dimensional space
index.add(embeddings_matrix)      # shape: (N_enrolled, 512)

# At query time:
D, I = index.search(query, k=1)   # D = similarity scores, I = indices
if D[0][0] > 0.4:
    identity = gallery_labels[I[0][0]]
```

Search complexity: O(N × 512) multiplications — linear in gallery size. For 100 enrolled embeddings, this is 51,200 multiplications, completing in under 1 ms on CPU.

---

## 5. Evaluation Metrics

### mAP@0.5 (Mean Average Precision)

For each class (e.g. "car"):
1. Rank all detections by confidence, highest first.
2. For each detection, compute IoU with the nearest ground-truth box.
3. If IoU ≥ 0.5, it is a True Positive; otherwise a False Positive.
4. Build a precision-recall curve by going through the ranked list.
5. Average Precision (AP) = area under the precision-recall curve.

mAP@0.5 = mean of AP across all classes.

mAP@0.5:0.95 = mean of mAP at IoU thresholds 0.50, 0.55, 0.60, ..., 0.95. Stricter — a box must overlap the ground truth by at least 95% to count at the top threshold.

**IoU (Intersection over Union)**:
```
IoU = area(predicted ∩ ground_truth) / area(predicted ∪ ground_truth)
```

### PSNR (Peak Signal-to-Noise Ratio)

Measures pixel-level fidelity between a restored image and the clean reference:

```
PSNR = 10 × log10(255² / MSE)
```

Where MSE (Mean Squared Error) is the average of squared per-pixel differences. Higher PSNR = better reconstruction. Measured in decibels (dB). Typical values: 20–40 dB.

Our results: Zero-DCE++ 12.11 dB, Restormer 15.15 dB — lower than literature because we evaluated on a different domain (synthetic Albumentations degradation on RTTS, not the LOL/SOTS datasets used in the original papers).

### SSIM (Structural Similarity Index)

SSIM measures perceptual similarity by comparing luminance, contrast, and structure jointly, rather than per-pixel. Range: 0–1. Our results: Zero-DCE++ 0.5420, Restormer 0.6252.

### NIQE (No-Reference Image Quality Estimator)

NIQE evaluates image quality without a reference image. It fits a statistical model (multivariate Gaussian) to natural-scene patches and measures how far the enhanced image deviates from natural image statistics. Lower NIQE = more natural-looking. Zero-DCE++ scores better than Restormer on NIQE (3.94 vs 4.23) despite lower PSNR, because it produces perceptually more natural output on our dataset.

### FAR and FRR (Face Recognition)

- **FAR** (False Accept Rate): fraction of impostor pairs accepted as genuine
- **FRR** (False Reject Rate): fraction of genuine pairs rejected as impostors
- **Threshold**: the cosine similarity cutoff that balances FAR and FRR
- LFW evaluation threshold: 0.160 (calibrated on development set)
- Deployment threshold: 0.4 (calibrated for gallery search, not pairs verification)

---

## 6. Optimization Techniques

### ONNX Export

ONNX (Open Neural Network Exchange) is a common format for exporting models from PyTorch into a runtime-agnostic representation. The graph is then run by ONNX Runtime, which can use CPU, CUDA, or TensorRT as its backend (called Execution Providers).

ONNX GPU (CUDA EP) was slower than PyTorch FP32 for YOLOv8n because: for small models (8.1 GFLOPs), the overhead of ONNX Runtime's buffer allocation and provider scheduling exceeds the compute savings. PyTorch's CUDA kernels are tightly optimised for T4 hardware. Result: 118.4 ms vs 88.8 ms (0.75×).

### ONNX INT8 Quantisation

Dynamic quantisation converts FP32 weights to INT8 (8-bit integers) at export time. This reduces model size 3.7× (6.2 MB → 3.4 MB). However, ONNX dynamic quantisation does not support GPU execution — it runs on CPU only. On CPU, parallelism is insufficient to compensate for the T4's speed, resulting in 168.3 ms vs 88.8 ms (0.53×, actually slower).

INT8 is suitable for edge deployment (Raspberry Pi, mobile) where no GPU is available.

### Structured Pruning

Removes entire channels (output filters) from convolutional layers based on their L1-norm importance. 30% of channels removed → fewer GPU operations → 1.50× speedup (59.3 ms vs 88.8 ms). The pruned model requires fine-tuning to recover accuracy — this was not done in Phase 5 due to compute budget.

### TensorRT FP16

TRT converts the model to FP16 (half-precision) and fuses operators (e.g. conv + batch norm + ReLU become one kernel call). Expected speedup: 1.7–2.0× over FP32. The engine build failed in Colab due to a TensorRT version mismatch with the pre-installed runtime. Expected latency from NVIDIA benchmarks: ~45–55 ms.

---

## 7. Phase-by-Phase Technical Summary

### Phase 2 Enhancement Results

| Model | PSNR (dB) | SSIM | NIQE | Latency |
|---|---|---|---|---|
| Zero-DCE++ | 12.11 | 0.5420 | 3.94 | 39.2 ms |
| Restormer | 15.15 | 0.6252 | 4.23 | 4,226 ms |
| CLAHE | — | — | — | < 5 ms |

Zero-DCE++ deployed for low-light; FFA-Net deployed for foggy; CLAHE for clear and rainy.

### Phase 3 Detection Results

YOLOv8n with outdoor augmentation, BDD100K 7%, 25 epochs:
- mAP@0.5 = **0.112**, mAP@0.5:0.95 = 0.063
- Raw image average confidence: 0.852 vs enhanced (Zero-DCE++) 0.834 — enhancement marginally reduced detection confidence on this subset.
- RT-DETR-L trained for 15 epochs achieved mAP@0.5 = 0.195, but at 8.2 ms/image (inference-only) it is slower and was not selected for the deployed pipeline.

### Phase 4 Recognition Results

LFW pairs test accuracy: **98.61%** at threshold 0.160 (FAR=0.67%, FRR=2.17%).

Robustness table (N=300 queries per condition, gallery=300 identities):

| Condition | Top-1 Acc | Detect Failures |
|---|---|---|
| Clean | 98.67% | 0 |
| Fog raw | 95.33% | 1 |
| Fog + FFA-Net | 93.67% | 6 |
| Low-light raw | 95.67% | 6 |
| Low-light + Zero-DCE++ | 94.67% | 9 |
| Rain raw | 97.67% | 2 |
| Rain + CLAHE | 98.33% | 1 |

Detection failure count is the primary driver of accuracy drop under enhancement — more failed face detections mean fewer gallery searches attempted.

### Phase 5 Optimisation Results

| Format | Latency | FPS | Size | mAP | Speedup |
|---|---|---|---|---|---|
| PyTorch FP32 | 88.8 ms | 11.3 | 6.2 MB | 0.112 | 1.00× |
| ONNX GPU | 118.4 ms | 8.4 | 12.3 MB | 0.112 | 0.75× |
| ONNX INT8 (CPU) | 168.3 ms | 5.9 | 3.4 MB | N/A | 0.53× |
| Pruned 30% | 59.3 ms | 16.9 | 6.2 MB | N/A | 1.50× |
| TRT FP16 | — | — | — | — | failed |

### Phase 6 End-to-End Results

| Condition | Total Mean | Bottleneck Stage |
|---|---|---|
| Clear | 111.3 ms | Detection (80%) |
| Rainy | 111.3 ms | Detection (80%) |
| Low-light | 146.3 ms | Enhancement 27% + Detection 61% |
| Foggy | 4,333 ms | Enhancement (FFA-Net, 97.5%) |

Sustained throughput: **~9 QPS** stable from burst size 1 to 20 (single T4 GPU worker).

---

## 8. The System Stack

```
User browser
    ↕ HTTPS
Next.js web app (frontend)
    ↕ HTTP REST
Spring Boot backend (Java)
    ├── PostgreSQL  (task metadata, user accounts)
    ├── MinIO       (image/video storage, S3-compatible)
    └── ↕ HTTPS → HuggingFace Spaces inference API
                        ↓
                   FastAPI (Python)
                        ↓
                   DetectionPipeline
                    ├── Enhancement (CLAHE / Zero-DCE++ / FFA-Net)
                    ├── YOLOv8n detection
                    └── SCRFD + ArcFace + FAISS recognition
```

The inference API returns a JSON payload containing:
- `detections`: list of `{class, confidence, bbox: {x, y, w, h}}`
- `recognitions`: list of `{identity, similarity, bbox}`
- `enhanced_image_url`: base64-encoded JPEG of the enhanced frame
- `latency_ms`: `{enhancement, detection, recognition, total}`

---

*Continue to GUIDE_3_ADVANCED.md for research-level analysis of the results and architecture decisions.*
*Thesis: "Outdoor Object Detection and Face Recognition Under Adverse Weather Conditions"*
*Author: Muhammad Bashir Dantani — USTC Master's Thesis — Defense: December 2026*
