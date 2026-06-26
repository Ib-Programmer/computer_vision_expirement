# Complete Beginner's Guide to the CV Thesis Project
### Outdoor Object Detection & Face Recognition System

---

## Table of Contents

1. [What Is a Computer?](#1-what-is-a-computer)
2. [What Is an Image?](#2-what-is-an-image)
3. [What Is Computer Vision?](#3-what-is-computer-vision)
4. [How Neural Networks Learn](#4-how-neural-networks-learn)
5. [Image Enhancement — Making Bad Images Better](#5-image-enhancement)
6. [Object Detection — Finding Things in Images](#6-object-detection)
7. [Face Recognition — Who Is That?](#7-face-recognition)
8. [The Full Pipeline — How It All Connects](#8-the-full-pipeline)
9. [Every Parameter Explained](#9-every-parameter-explained)
10. [System Architecture — The Big Picture](#10-system-architecture)
11. [Algorithms Deep Dive](#11-algorithms-deep-dive)
12. [Evaluation Metrics Explained](#12-evaluation-metrics-explained)
13. [External Resources](#13-external-resources)

---

## 1. What Is a Computer?

A computer is a machine that follows instructions very fast. At its core it only understands two things: **0 and 1** (off and on, like a light switch). Every number, letter, image, and video is stored as a sequence of 0s and 1s called **binary**.

### CPU vs GPU

| | CPU (Central Processing Unit) | GPU (Graphics Processing Unit) |
|---|---|---|
| Cores | 4–32 powerful cores | Thousands of tiny cores |
| Good at | One complex task at a time | Millions of small tasks at once |
| Use in AI | Loading data, managing program | Running neural network math |
| Example | Intel i7, AMD Ryzen | NVIDIA T4, RTX 3090 |

> **Why GPUs matter for AI**: Training a neural network requires multiplying millions of numbers simultaneously. A GPU with 3,000 cores can do 3,000 multiplications at the same time, making it 100× faster than a CPU for this task.

---

## 2. What Is an Image?

### Pixels

An image is a grid of tiny squares called **pixels** (picture elements). Each pixel holds a color value.

A 640×480 image = 640 columns × 480 rows = **307,200 pixels**.

### Color Channels

Each pixel stores color as three numbers: **Red, Green, Blue (RGB)**.

```
Pixel at position (100, 200):
  Red   = 255   (maximum red)
  Green = 128   (half green)
  Blue  = 0     (no blue)
  → This pixel is orange
```

Each channel value goes from **0** (none of that color) to **255** (maximum). So one pixel = 3 numbers, each 0–255.

### Why 0–255?

Computers store each channel in **8 bits** (1 byte). 8 bits can represent 2⁸ = **256** different values (0 to 255).

### Image Formats

| Format | What it does | Use case |
|---|---|---|
| JPEG | Compresses by discarding some detail | Photos (smaller file) |
| PNG | Compresses without losing any detail | Screenshots, logos |
| WebP | Modern format, smaller than JPEG | Web images |

### BGR vs RGB

OpenCV (the image library used in this project) stores images as **BGR** (Blue, Green, Red) instead of RGB. This is a historical quirk — always convert when displaying.

```python
# OpenCV reads as BGR
img = cv2.imread("photo.jpg")        # BGR

# Convert to RGB for display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

---

## 3. What Is Computer Vision?

Computer vision (CV) teaches computers to **understand images and videos** the way humans do — recognizing objects, faces, scenes, and motion.

### Tasks in Computer Vision

| Task | What it does | Example |
|---|---|---|
| **Classification** | "What is in this image?" | Cat / Dog / Car |
| **Detection** | "Where are the objects?" | Draw boxes around each car |
| **Segmentation** | "Which pixels belong to what?" | Color every pixel of every car |
| **Recognition** | "Who/what specifically is this?" | "That face is Ibrahim" |
| **Enhancement** | "Make this image better" | Remove fog, brighten dark images |

This project uses **Detection** + **Recognition** + **Enhancement** together.

---

## 4. How Neural Networks Learn

### The Brain Analogy

A neural network is loosely inspired by the human brain. Your brain has ~86 billion neurons connected to each other. When you see a cat, signals travel through chains of neurons and eventually your brain says "cat."

A neural network does the same thing with math.

### Layers

A neural network is made of **layers**. Each layer takes numbers in, does some math, and passes numbers out.

```
Input Image (pixels)
       ↓
  Layer 1 (finds edges)
       ↓
  Layer 2 (finds shapes)
       ↓
  Layer 3 (finds parts: wheels, windows)
       ↓
  Layer 4 (finds objects: car)
       ↓
Output: "Car at position (120, 80)"
```

### Weights — The "Memory" of a Network

Each connection between neurons has a **weight** — a number that says "how important is this connection?" Training a neural network means finding the right weights.

### Training (Learning)

1. Show the network 1,000 images of cats labeled "cat"
2. The network guesses randomly at first: "That's a dog" (wrong)
3. Calculate the **error** (how wrong was it?)
4. Adjust all weights slightly to make the network less wrong
5. Repeat millions of times until the network is accurate

This adjustment process is called **backpropagation** and uses an algorithm called **gradient descent**.

### Inference (Using the Trained Network)

Once trained, you **freeze** the weights and run new images through — this is called **inference**. Our project only does inference (the networks are pre-trained).

### Parameters vs Hyperparameters

| Term | Meaning | Who sets it | Example |
|---|---|---|---|
| **Parameter** | A weight inside the network | Learned during training | Weight = 0.73 |
| **Hyperparameter** | A setting that controls training/inference | You set it | Learning rate, confidence threshold |

---

## 5. Image Enhancement

### Why Enhancement Matters

Cameras struggle in **fog**, **rain**, **darkness**, and **haze**. Object detectors trained on clear images perform poorly on degraded images. Enhancement preprocesses the image to recover detail before detection.

### 5.1 CLAHE — Contrast Limited Adaptive Histogram Equalization

**What it does**: Improves local contrast in an image.

**Simple explanation**: Imagine you're looking at a dark photo. The dark areas are all squished into a narrow range of values (0–50). CLAHE spreads those values out (0–255) so you can see detail in the shadows.

**Why "Adaptive"?**: Instead of applying the same enhancement to the whole image, CLAHE divides the image into small tiles (8×8 by default) and enhances each tile independently. This means a bright sky won't wash out a dark road.

**Why "Contrast Limited"?**: Without the limit, noise gets amplified. The `clipLimit=2.0` parameter caps how much amplification can happen in any one tile.

**How it works step by step**:
1. Convert image from BGR to LAB color space (L = lightness, A/B = color)
2. Apply CLAHE only to the L (lightness) channel
3. Convert back to BGR

```python
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l_enhanced = clahe.apply(l)
result = cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)
```

**Parameters used in this project**:
- `clipLimit=2.0` — maximum amplification factor per tile
- `tileGridSize=(8,8)` — divide image into 8×8 = 64 tiles

**When CLAHE is used**: For **clear** and **rainy** conditions where the image just needs contrast boost.

**Speed**: ~2–5 ms per image (very fast, no GPU needed).

---

### 5.2 Zero-DCE++ — Zero-Reference Deep Curve Estimation

**What it does**: Brightens low-light images without needing paired (dark/bright) training data.

**The problem with dark images**: A camera in darkness captures very few photons. The image sensor adds **noise** (random dots). Simply multiplying pixel values makes the noise bigger too.

**Zero-DCE++ approach — "Curve Estimation"**:

Instead of directly predicting what a bright image should look like, Zero-DCE++ learns **tone curves** — mathematical functions that map each dark pixel value to a brighter value, similar to how photographers adjust "curves" in Photoshop.

The network predicts a set of curve parameters (α values) for each pixel. These curves are applied iteratively:

```
Enhanced = Original + α × Original × (1 - Original)
```

Applied 8 times, this S-shaped curve lifts the dark regions significantly while keeping bright regions from blowing out (overexposing).

**Why "Zero-Reference"?**: Most enhancement networks need pairs of (dark image, bright image) to train. Zero-DCE++ trains using only dark images by defining **self-supervised loss functions**:
- **Spatial consistency loss**: neighboring pixels shouldn't change too differently
- **Exposure control loss**: try to bring the average brightness toward 0.6
- **Color constancy loss**: don't shift colors while brightening
- **Illumination smoothness loss**: make the enhancement map smooth, not patchy

**Architecture**: A lightweight CNN (Convolutional Neural Network) with ~79K parameters — very small and fast.

**When used**: For **low-light** conditions.

**Speed**: ~30–80 ms per image on GPU.

---

### 5.3 FFA-Net — Feature Fusion Attention Network

**What it does**: Removes haze and fog from outdoor images.

**Why fog is hard**: Fog scatters light between the camera and the subject. A foggy image pixel contains:
```
I(x) = J(x) × t(x) + A × (1 - t(x))
```
Where:
- `I(x)` = what the camera sees (foggy)
- `J(x)` = what the scene actually looks like (clear)
- `t(x)` = transmission map (how much light gets through)
- `A` = airlight (the color of the fog/haze)

Dehazing means estimating `t(x)` and `A` to recover `J(x)`.

**FFA-Net's approach**:

FFA-Net uses two key ideas:

1. **Channel Attention**: Different color channels (R, G, B) are affected differently by haze (blue channel suffers most). FFA-Net learns to weight each channel differently.

2. **Pixel Attention**: Not all pixels are equally foggy. Areas far from the camera are foggier. FFA-Net learns per-pixel importance weights.

The combination of both is called **Feature and Fusion Attention (FFA)**.

**Architecture**: Multiple Feature Attention (FA) blocks stacked together, each containing channel attention + pixel attention layers.

**When used**: For **foggy** conditions using the OTS (Outdoor Training Set) pretrained weights.

**Speed**: ~200–500 ms per image on GPU (heavier than CLAHE/Zero-DCE++).

---

### Enhancement Routing Logic

The pipeline automatically picks the right enhancer:

```
condition = "auto"?
    → Run CLAHE, measure image brightness
    → brightness < 0.3?  → Low-light → use Zero-DCE++
    → high haze level?   → Foggy     → use FFA-Net
    → else               → Clear     → keep CLAHE result

condition = "low-light" → Zero-DCE++
condition = "foggy"     → FFA-Net
condition = "rainy"     → CLAHE
condition = "clear"     → CLAHE
```

---

## 6. Object Detection

### What Is Object Detection?

Detection = **Classification** (what?) + **Localization** (where?).

The output is a list of **bounding boxes**, each with:
- A class label ("person", "car", "bicycle")
- A confidence score (0.0–1.0, how sure the model is)
- Box coordinates (x, y, width, height)

### 6.1 How YOLO Works

**YOLO** = You Only Look Once. Before YOLO (2016), detectors needed two passes:
1. Propose regions that might contain objects
2. Classify each proposed region

YOLO does both in **one forward pass** through the network, making it much faster.

**Grid-based detection**:

YOLO divides the image into a grid (e.g., 20×20 = 400 cells). Each cell is responsible for detecting objects whose **center** falls in that cell.

```
┌───┬───┬───┬───┬───┐
│   │   │   │   │   │
├───┼───┼───┼───┼───┤
│   │ ● │   │   │   │  ← cell detects the car (center is here)
├───┼───┼───┼───┼───┤
│   │   │   │   │   │
└───┴───┴───┴───┴───┘
```

Each cell predicts:
- Multiple **anchor boxes** (pre-defined box shapes)
- For each anchor: (x_offset, y_offset, width, height, confidence, class probabilities)

### 6.2 Anchor Boxes

Anchor boxes are pre-defined shapes that represent common object sizes. For example:
- Tall narrow box → good for pedestrians
- Wide short box → good for cars
- Square box → good for faces

The network predicts **offsets** from these anchors rather than absolute coordinates. This makes learning easier.

### 6.3 YOLOv8 Architecture

YOLOv8 (2023, by Ultralytics) is the 8th generation of YOLO. Key components:

**Backbone (feature extraction)**:
- Takes the raw image
- Applies convolutional layers to extract features
- Uses **CSP (Cross Stage Partial)** connections to reuse features efficiently

**Neck (feature pyramid)**:
- Combines features from different scales
- Small features map → good at detecting big objects
- Large features map → good at detecting small objects
- Uses **PANet (Path Aggregation Network)** to connect scales

**Head (prediction)**:
- For each scale, predicts boxes + classes
- Uses **anchor-free** detection (YOLOv8 doesn't use anchors — it predicts box centers directly)

### 6.4 Model Sizes

| Model | Parameters | mAP (COCO) | Speed (T4 GPU) |
|---|---|---|---|
| YOLOv8n | 3.2M | 37.3 | 1.77 ms |
| YOLOv8s | 11.2M | 44.9 | 2.98 ms |
| YOLOv8m | 25.9M | 50.2 | 5.09 ms |
| YOLOv8l | 43.7M | 52.9 | 7.96 ms |
| YOLOv8x | 68.2M | 53.9 | 12.81 ms |

**This project uses YOLOv8n** — the smallest and fastest. Good for real-time on limited hardware.

### 6.5 ONNX Format

The model is exported to **ONNX** (Open Neural Network Exchange) format. Benefits:
- Runs on any hardware without PyTorch installed
- Can be optimized with TensorRT for even faster inference
- Smaller deployment package

### 6.6 Non-Maximum Suppression (NMS)

After YOLO predicts boxes, many overlapping boxes exist for the same object. NMS removes duplicates:

1. Sort boxes by confidence score (highest first)
2. Keep the highest-confidence box
3. Remove any box that **overlaps too much** with the kept box (overlap measured by IoU)
4. Repeat until no boxes remain

**IoU (Intersection over Union)** measures overlap:
```
        Area of Overlap
IoU = ─────────────────────
        Area of Union
```
- IoU = 0.0 → no overlap at all
- IoU = 1.0 → boxes are identical
- IoU = 0.5 → boxes overlap by half

---

## 7. Face Recognition

### Detection vs Recognition

| | Face Detection | Face Recognition |
|---|---|---|
| Question | "Is there a face? Where?" | "Whose face is this?" |
| Output | Bounding boxes | Identity ("Ibrahim") |
| Analogy | Finding all people in a crowd | Identifying each person by name |

### 7.1 RetinaFace — Face Detection

RetinaFace (2020) is a face detector that works at multiple scales, detecting:
- The face bounding box
- 5 facial landmarks (left eye, right eye, nose, left mouth corner, right mouth corner)

The landmarks are used to **align** the face before recognition — rotating and scaling it to a standard frontal position.

**Why alignment matters**: ArcFace was trained on aligned faces. Unaligned faces (tilted, profile views) get significantly worse recognition accuracy.

### 7.2 ArcFace — Face Recognition

ArcFace converts a face image into a **512-dimensional vector** (a list of 512 numbers called an **embedding**). The key property:

- Same person → embeddings point in the **same direction** in 512D space
- Different people → embeddings point in **different directions**

**How training works (ArcFace loss)**:

ArcFace adds an **angular margin** of 0.5 radians between different people's embeddings during training. This forces the network to learn more discriminative features — embeddings of the same person cluster very tightly, embeddings of different people are pushed far apart.

**Cosine similarity**:

To compare two face embeddings:
```
similarity = (A · B) / (|A| × |B|)
```
- similarity = 1.0 → same direction → very likely same person
- similarity = 0.0 → perpendicular → unrelated
- similarity = -1.0 → opposite direction → maximally different

**Threshold used in this project**: `0.4`
- similarity > 0.4 → recognized as known person
- similarity ≤ 0.4 → "Unknown"

### 7.3 buffalo_l — The Model Pack

`buffalo_l` is InsightFace's large model package containing:

| Component | Model | Task |
|---|---|---|
| Face detector | SCRFD-10GF | Detect faces + landmarks |
| Face recognizer | w600k_r50 | ArcFace with ResNet-50 backbone |
| Gender/age | GendAge | Optional demographic info |

**w600k_r50** means: trained on **600,000** identities using **ResNet-50** backbone.

### 7.4 FAISS — Fast Similarity Search

**The problem**: You have 100 enrolled people, each with a 512D embedding. For each new face, you need to find the closest match. Comparing one vector to 100 is fast. But comparing to 10,000 enrolled people, checking each of 512 numbers → slow.

**FAISS** (Facebook AI Similarity Search) solves this. It indexes all embeddings and uses clever math (approximate nearest neighbor search) to find the closest match in microseconds even with millions of entries.

**Index type used**: `IndexFlatIP` (Inner Product / cosine similarity)

```python
index = faiss.IndexFlatIP(512)   # 512-dimensional vectors
index.add(embeddings)             # add all enrolled faces
D, I = index.search(query, k=1)  # find 1 nearest neighbor
# D = similarity scores, I = indices of matches
```

---

## 8. The Full Pipeline

### 8.1 Data Flow

```
User uploads image/video
         │
         ▼
   Spring Boot API
   (Java backend)
         │
         │ 1. Save to MinIO (object storage)
         │ 2. Create task in PostgreSQL
         │ 3. Trigger async processing
         ▼
   Task Processor
         │
         │ Download image from MinIO
         │ Encode as base64
         ▼
   HuggingFace Space
   (Python / FastAPI)
         │
    ┌────┴────┐
    │         │
    ▼         ▼
Enhancement  Object
(CLAHE /    Detection
Zero-DCE++ / (YOLOv8n
FFA-Net)     ONNX)
    │         │
    └────┬────┘
         │
         ▼
   Face Recognition
   (RetinaFace +
    ArcFace + FAISS)
         │
         ▼
   Return JSON result
         │
         ▼
   Spring Boot stores
   result in PostgreSQL
         │
         ▼
   Frontend polls for
   result and displays it
```

### 8.2 What Happens Inside HuggingFace Space

**For images**:
1. Receive base64-encoded image
2. Decode to numpy array
3. Run enhancement (pick model based on condition)
4. Run YOLOv8n ONNX → get bounding boxes
5. Run RetinaFace → get face boxes + landmarks
6. Align each face → run ArcFace → get 512D embedding
7. Search FAISS gallery → find closest match
8. Return JSON with all detections, recognitions, latencies

**For videos**:
1. Receive base64-encoded video
2. Decode → save to temp file
3. Open with OpenCV VideoCapture
4. Loop through frames (every 4th frame gets full inference)
5. Apply last detected boxes to intermediate frames (smooth output)
6. Draw bounding boxes on all frames as JPEG files
7. Run ffmpeg to assemble JPEG frames → H264 MP4 (with original audio)
8. Return base64-encoded annotated video

### 8.3 Why Each Technology Was Chosen

| Technology | Why this one |
|---|---|
| YOLOv8n | Fastest YOLO, fits T4 GPU in under 2ms |
| ArcFace w600k_r50 | State-of-the-art on LFW (99.83%), free |
| FAISS | Sub-millisecond search at any gallery size |
| CLAHE | Zero cost (CPU, no model needed), instant |
| Zero-DCE++ | Only 79K params, runs on CPU if needed |
| FFA-Net | Best outdoor dehazing on SOTS benchmark |
| Spring Boot | Mature Java framework, handles async well |
| MinIO | S3-compatible, self-hosted, free |
| HuggingFace Spaces | Free GPU hosting (T4), Docker support |
| PostgreSQL | Robust, JSONB for flexible result storage |
| Next.js | React framework with server-side rendering |

---

## 9. Every Parameter Explained

### 9.1 Object Detection Parameters

#### `conf = 0.45` (Confidence Threshold)

**What it is**: Minimum confidence score for a detection to be kept.

**Range**: 0.0 – 1.0

**What happens at different values**:

| conf | Effect |
|---|---|
| 0.1 | Keep almost everything → many false positives (phantom objects) |
| 0.25 | Default YOLO value → balanced but noisy |
| **0.45** | **Our value** → cleaner results, fewer false detections |
| 0.8 | Only very confident detections → might miss real objects |

**Why 0.45**: The default 0.25 was producing phantom detections in fog and low-light conditions. Testing showed 0.45 eliminates most false positives while keeping real detections. This is a good balance for outdoor scenes with variable quality.

**Formula**: The model outputs a raw score `p`. This becomes the confidence:
```
confidence = objectness_score × class_probability
```
Only boxes where `confidence > 0.45` are kept.

---

#### `iou = 0.45` (IoU Threshold for NMS)

**What it is**: How much overlap is allowed before two boxes are considered duplicates.

**Range**: 0.0 – 1.0

**What happens at different values**:

| iou | Effect |
|---|---|
| 0.1 | Very aggressive removal → might merge nearby objects |
| 0.45 | **Our value** → standard, works well for most scenes |
| 0.7 | Lenient → keeps many overlapping boxes |

**Why 0.45**: This is the standard recommended value. It removes clear duplicates while keeping genuinely separate objects that happen to be close together.

---

### 9.2 Face Recognition Parameters

#### `cosine_threshold = 0.4`

**What it is**: Minimum similarity score for a face to be recognized as a known person.

**Range**: -1.0 – 1.0 (cosine similarity)

**What happens at different values**:

| threshold | Effect |
|---|---|
| 0.2 | Too lenient → wrong people get matched |
| **0.4** | **Our value** → good balance |
| 0.6 | Too strict → enrolled people might not be recognized |
| 0.8 | Almost never matches → useless |

**Why 0.4**: ArcFace embeddings are very discriminative. A similarity of 0.4 is already high in 512D space. Testing on LFW shows this threshold gives TAR (True Accept Rate) > 99% at FAR (False Accept Rate) < 0.1%.

---

#### `det_size = (640, 640)`

**What it is**: The size of the image fed into RetinaFace for face detection.

**Why 640×640**: This is the standard input size for RetinaFace. Larger = slower but detects smaller faces. Smaller = faster but misses tiny faces. 640 is the sweet spot.

---

### 9.3 Enhancement Parameters

#### CLAHE `clipLimit = 2.0`

**What it is**: Maximum amplification per tile. Values above this are "clipped" (redistributed).

**Range**: typically 1.0 – 5.0

| clipLimit | Effect |
|---|---|
| 1.0 | No clipping → same as regular histogram equalization |
| **2.0** | **Our value** → gentle enhancement, minimal noise |
| 5.0 | Strong enhancement, but amplifies noise significantly |

---

#### CLAHE `tileGridSize = (8, 8)`

**What it is**: Divides the image into an 8×8 grid of tiles. Each tile gets independent histogram equalization.

**Effect**: Smaller tiles = more local adaptation but can look patchy. 8×8 is standard.

---

#### Video `SAMPLE_EVERY = 4`

**What it is**: Run inference on every 4th frame; apply the last detected boxes to frames in between.

**Why not every frame**: At 30fps, running inference on every frame would need the pipeline to complete in 33ms. Face recognition alone takes ~100ms. By sampling every 4th frame (effectively ~7.5 inference calls per second), we get smooth annotated video without waiting forever.

**Trade-off**: Fast-moving objects may have stale boxes for 3 frames (~100ms). Acceptable for thesis demo purposes.

---

#### Video `MAX_VIDEO_SECONDS = 60`

**What it is**: Hard limit — stop processing after 60 seconds of video.

**Why**: A 10-minute video at SAMPLE_EVERY=4 would take ~30 minutes to process on T4. This keeps the HuggingFace Space responsive.

---

### 9.4 System Parameters

#### `read-timeout = PT300S` (5 minutes)

**What it is**: How long the Spring Boot backend waits for HuggingFace Space to respond.

**ISO 8601 duration**: `PT300S` = Period of Time = 300 Seconds.

**Why 5 minutes**: A 60-second video processes ~450 frames. Each frame takes ~0.65s on T4. Total ≈ 5 minutes. Setting the timeout to 5 minutes prevents premature failures.

---

#### `upload-url-ttl = PT15M` (15 minutes)

**What it is**: How long the presigned upload URL is valid.

**Why 15 minutes**: Users have 15 minutes to upload their file after requesting the URL. Shorter = more secure. Longer = more user-friendly. 15 minutes is standard.

---

#### `image-url-ttl = PT24H` (24 hours)

**What it is**: How long the presigned download URL for results is valid.

**Why 24 hours**: Users might want to come back and view their results the next day. 24 hours balances security and convenience.

---

#### `max-bytes = 209715200` (200 MB)

**What it is**: Maximum file size allowed for upload.

**Calculation**: 200 × 1024 × 1024 = 209,715,200 bytes = 200 MB.

**Why 200 MB**: Reasonable limit for video files up to ~60 seconds at good quality. Prevents abuse of storage.

---

### 9.5 Training Parameters (Phase 3)

#### `epochs = 50`

**What it is**: Number of times the training algorithm sees the entire dataset.

**Why 50**: Enough for the model to converge on outdoor datasets without overfitting. With data augmentation, each epoch looks slightly different, so 50 passes provide diversity.

#### `batch = 16`

**What it is**: Number of images processed together before updating weights.

**Why 16**: Fits in T4's 16GB VRAM with YOLOv8n. Larger batches = more stable gradients but more memory.

#### `imgsz = 640`

**What it is**: All images are resized to 640×640 for training.

**Why 640**: YOLOv8's default. Larger = better accuracy but slower and more memory. 640 is the optimal balance.

#### `lr0 = 0.01` (Initial Learning Rate)

**What it is**: How big a step to take when updating weights.

**Analogy**: Imagine walking downhill (toward minimum error) blindfolded. Learning rate = step size.
- Too large (0.1) → overshoot the valley
- Too small (0.0001) → takes forever
- **0.01** → standard starting point for YOLO

---

## 10. System Architecture

### 10.1 Component Map

```
┌─────────────────────────────────────────────────────────────┐
│                        User's Browser                        │
│                       (Next.js Frontend)                     │
│  Upload page → Task status → Results page with video player  │
└─────────────────┬───────────────────────────────────────────┘
                  │ HTTPS REST API
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    Spring Boot API (Java)                     │
│  Port 8080                                                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐   │
│  │Auth (JWT)│ │  Tasks   │ │Identities│ │Task Processor│   │
│  │/auth/**  │ │/api/tasks│ │/api/ident│ │   @Async     │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────┬───────┘   │
└─────────────────────────────────────────────────┼───────────┘
          │                                        │
          ▼                                        ▼
┌──────────────────┐                  ┌──────────────────────┐
│   PostgreSQL DB  │                  │  HuggingFace Space   │
│   (Port 5432)    │                  │  (Docker / T4 GPU)   │
│                  │                  │  FastAPI + Python    │
│  tables:         │                  │  /pipeline           │
│  - users         │                  │  /pipeline_video     │
│  - tasks         │                  │  /enrol              │
│  - identities    │                  │  /health             │
│  - face_photos   │                  └──────────────────────┘
└──────────────────┘
          │
          ▼
┌──────────────────┐
│     MinIO        │
│  (Port 9000)     │
│  S3-compatible   │
│  object storage  │
│                  │
│  bucket:         │
│  cv-uploads/     │
│  ├── uploads/    │
│  └── enhanced/   │
└──────────────────┘
```

### 10.2 Authentication Flow (JWT)

**JWT** = JSON Web Token. A token is a signed string proving who you are.

```
1. User sends email + password → POST /auth/login
2. Backend checks password hash (bcrypt)
3. Backend creates JWT: {userId, email, expiry} signed with secret key
4. Returns token to frontend
5. Frontend stores token in memory
6. Every API request includes: Authorization: Bearer <token>
7. Backend verifies signature → knows who the user is
```

**Why stateless**: The server doesn't store sessions. The token itself contains all needed info. This scales to multiple servers easily.

### 10.3 Async Task Processing

Tasks are processed asynchronously because inference takes 1–300 seconds — too long to wait for an HTTP response.

```
POST /api/tasks
    │
    ├── Save task to DB (status: QUEUED)
    ├── Register afterCommit() hook
    └── Return immediately → 200 OK

[DB transaction commits]
    │
    └── afterCommit() fires
            │
            └── @Async → new thread
                    │
                    ├── Status: ENHANCING (20%)
                    ├── Status: DETECTING (50%)
                    ├── Status: RECOGNIZING (80%)
                    ├── Call HuggingFace Space
                    └── Status: DONE (100%)

Frontend polls GET /api/tasks/{id} every 1.5s
    → Sees DONE
    → Redirects to results page
```

---

## 11. Algorithms Deep Dive

### 11.1 Convolutional Neural Networks (CNNs)

A **convolution** slides a small filter (e.g., 3×3) over the image and computes dot products. Different filters detect different features:

```
Edge detector filter:       Blur filter:
 -1  -1  -1               1/9  1/9  1/9
 -1   8  -1               1/9  1/9  1/9
 -1  -1  -1               1/9  1/9  1/9
```

In a CNN:
- Early layers learn low-level features (edges, colors)
- Middle layers learn mid-level features (textures, shapes)
- Late layers learn high-level features (objects, faces)

### 11.2 ResNet — Residual Networks

The backbone of ArcFace (`r50` = ResNet-50) uses **skip connections** (residual connections):

```
Input → [Conv Layer] → [Conv Layer] → + → Output
  └──────────────────────────────────┘
           (skip connection)
```

**Why skip connections**: Deep networks (50+ layers) used to suffer from **vanishing gradients** — gradients become too small to update early layers. Skip connections let gradients flow directly, enabling training of very deep networks.

ResNet-50 has 50 layers and ~25 million parameters.

### 11.3 Attention Mechanisms

Used in FFA-Net. **Attention** lets the network focus on the most important parts of the image.

**Channel attention**: "Which color channel matters most for this image?"
**Pixel attention**: "Which pixels are most important?"

```
Feature Map → Squeeze (global average pooling)
           → Excitation (small neural network)
           → Scale (multiply back)
           → Attended Feature Map
```

### 11.4 Vector Similarity

**Why cosine similarity instead of Euclidean distance?**

Euclidean distance measures absolute position. Two faces could be different just because one photo is brighter (larger embedding magnitude).

Cosine similarity ignores magnitude — it only cares about **direction**. Two embeddings in the same direction = same person regardless of photo brightness.

```
Euclidean: distance = √(Σ(a_i - b_i)²)   ← sensitive to magnitude
Cosine:    similarity = (a·b)/(|a|×|b|)   ← only cares about direction
```

---

## 12. Evaluation Metrics Explained

### 12.1 Detection Metrics

#### mAP — Mean Average Precision

The standard metric for object detectors. Here's how it's computed:

**Step 1: Precision and Recall**

For each class (e.g., "car"):
```
Precision = True Positives / (True Positives + False Positives)
           = "Of all boxes the model predicted as cars, what fraction were real cars?"

Recall    = True Positives / (True Positives + False Negatives)
           = "Of all real cars in the images, what fraction did the model find?"
```

**Step 2: Precision-Recall Curve**

By varying the confidence threshold (0.0 to 1.0), you get different precision/recall trade-offs. Plot these → a curve.

**Step 3: Average Precision (AP)**

AP = area under the precision-recall curve (0.0 to 1.0).
AP = 1.0 = perfect detector.

**Step 4: Mean AP (mAP)**

Average AP across all classes.

`mAP@0.5` = mAP using IoU threshold 0.5 to decide if a detection is correct.
`mAP@0.5:0.95` = average mAP over IoU thresholds 0.5, 0.55, 0.6, ..., 0.95.

#### FPS — Frames Per Second

```
FPS = 1000 / latency_ms
```
Real-time video = 30 FPS = 33ms per frame budget.

---

### 12.2 Face Recognition Metrics

#### TAR@FAR — True Accept Rate at given False Accept Rate

**TAR** (True Accept Rate) = fraction of genuine pairs correctly recognized as same person.
**FAR** (False Accept Rate) = fraction of impostor pairs incorrectly accepted as same person.

By varying the cosine threshold, you get different TAR/FAR trade-offs.

**TAR@FAR=0.001** means: at a false accept rate of 0.1%, what is the true accept rate?
- ArcFace w600k_r50 achieves **99.83% TAR @ FAR=0.001** on LFW.

#### AP — Average Precision (for face detection)

Same as object detection AP, applied to face bounding boxes on WiderFace dataset.

---

### 12.3 Enhancement Metrics

#### PSNR — Peak Signal-to-Noise Ratio

Measures how similar the enhanced image is to the ground-truth clear image.

```
PSNR = 10 × log₁₀(MAX² / MSE)
```
- MAX = 255 (max pixel value)
- MSE = mean squared error between images
- Higher is better. > 30 dB is generally good.

#### SSIM — Structural Similarity Index

Measures structural similarity between images, accounting for luminance, contrast, and structure. Range: 0 to 1. Higher is better.

#### NIQE — Naturalness Image Quality Evaluator

Used when no ground-truth clear image is available (blind quality assessment). Measures how "natural" the image looks. Lower is better.

---

## 13. External Resources

### Papers (with free PDF links)

| Paper | What it introduces | Link |
|---|---|---|
| YOLOv8 | Fast object detection | ultralytics.com/yolov8 |
| ArcFace (2019) | Face recognition loss | arXiv:1801.07698 |
| RetinaFace (2020) | Face detection + landmarks | arXiv:1905.00641 |
| FFA-Net (2020) | Image dehazing | arXiv:1911.07559 |
| Zero-DCE++ (2021) | Low-light enhancement | arXiv:2103.00860 |
| RT-DETR (2023) | Real-time detection transformer | arXiv:2304.08069 |
| FAISS (2017) | Efficient similarity search | arXiv:1702.08734 |

### Datasets

| Dataset | What it contains | How to get it |
|---|---|---|
| COCO | 118K images, 80 classes | cocodataset.org |
| RTTS | 4,322 real hazy outdoor images | Kaggle: tuncnguyn/rtts-dataset |
| Foggy Cityscapes | 3,975 synthetic foggy images | Kaggle: yessicatuteja/foggy-cityscapes-image-dataset |
| LFW | 13,233 face images, 5,749 people | sklearn.datasets.fetch_lfw_people |
| WiderFace | 32,203 images, 393,703 faces | wider-challenge.github.io |
| BDD100K | 100K driving images | bdd-data.berkeley.edu |
| LOL | 500 low/normal light pairs | HuggingFace: geekyrakshit/LoL-Dataset |

### Tools & Libraries

| Tool | Purpose | Website |
|---|---|---|
| PyTorch | Deep learning framework | pytorch.org |
| Ultralytics | YOLOv8 implementation | ultralytics.com |
| InsightFace | RetinaFace + ArcFace | insightface.ai |
| OpenCV | Image/video processing | opencv.org |
| FAISS | Vector similarity search | github.com/facebookresearch/faiss |
| FastAPI | Python REST API | fastapi.tiangolo.com |
| Spring Boot | Java REST API | spring.io/projects/spring-boot |
| Next.js | React web framework | nextjs.org |
| MinIO | S3-compatible storage | min.io |
| HuggingFace | Model hosting + GPU spaces | huggingface.co |
| ONNX Runtime | Fast model inference | onnxruntime.ai |

### Video Tutorials (YouTube)

| Topic | Search term |
|---|---|
| How neural networks work | "3Blue1Brown neural networks" |
| YOLO explained | "Computerphile YOLO object detection" |
| Face recognition explained | "Siraj Raval face recognition" |
| Convolutional networks | "Andrej Karpathy CNN lecture Stanford" |
| Transformer attention | "3Blue1Brown attention mechanism" |

### Interactive Learning

- **fast.ai** — Free practical deep learning course (fastai.com)
- **CS231n** — Stanford's CNN course, free lectures on YouTube
- **Papers With Code** — Every paper with code and benchmark results (paperswithcode.com)
- **Roboflow** — Dataset tools + YOLO training tutorials (roboflow.com)
- **HuggingFace Learn** — NLP and CV tutorials (huggingface.co/learn)

---

## Glossary

| Term | Definition |
|---|---|
| **API** | Application Programming Interface — a way for software to talk to other software |
| **Async** | Asynchronous — doing work in the background without blocking |
| **Backbone** | The feature-extraction part of a neural network |
| **Base64** | Encoding binary data (images) as text characters for transmission |
| **Batch** | A group of images processed together |
| **Bounding box** | A rectangle drawn around a detected object |
| **Confidence** | How sure the model is about a prediction (0–1) |
| **Convolution** | Sliding a filter over an image to detect patterns |
| **Cosine similarity** | Measure of angle between two vectors (0 = perpendicular, 1 = same) |
| **Docker** | A tool that packages software with all its dependencies into a container |
| **Embedding** | A compact numerical representation of something (face, word, image) |
| **Epoch** | One complete pass through the training dataset |
| **FAR** | False Accept Rate — how often impostors are wrongly accepted |
| **FPS** | Frames per second — how many images processed per second |
| **GPU** | Graphics Processing Unit — fast parallel processor for AI |
| **Gradient** | Direction and magnitude of error slope, used to update weights |
| **Inference** | Using a trained model to make predictions on new data |
| **IoU** | Intersection over Union — overlap ratio between two boxes |
| **JWT** | JSON Web Token — signed token proving user identity |
| **Latency** | Time taken to process one request |
| **mAP** | Mean Average Precision — standard detection benchmark metric |
| **MinIO** | Open-source S3-compatible object storage |
| **NMS** | Non-Maximum Suppression — removes duplicate detection boxes |
| **ONNX** | Open Neural Network Exchange — portable model format |
| **Pixel** | Smallest unit of an image; stores RGB color values |
| **Presigned URL** | Temporary URL granting access to a private file |
| **PSNR** | Peak Signal-to-Noise Ratio — image quality metric |
| **ResNet** | Residual Network — deep CNN with skip connections |
| **REST** | Representational State Transfer — standard web API style |
| **TAR** | True Accept Rate — how often genuine pairs are correctly matched |
| **Token** | A piece of data (JWT token, Kaggle token, HuggingFace token) |
| **Weight** | A learned number inside a neural network |

---

*Document generated for the USTC Master's Thesis: "Outdoor Object Detection and Face Recognition Under Adverse Weather Conditions"*
*Author: Muhammad Bashir Dantani — Defense: December 2026*
