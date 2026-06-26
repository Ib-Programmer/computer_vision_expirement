# Guide 1 — Introduction
### Outdoor Detection and Face Recognition Under Adverse Weather Conditions

*Audience: no technical background required. Read this before any other document.*

---

## What Problem This Thesis Solves

Outdoor surveillance cameras are used to monitor roads, public spaces, and building entrances. They work well on a clear day. In fog, rain, or at night, the image quality drops and two things go wrong:

1. Objects — cars, people, bicycles — become hard to spot automatically.
2. Faces become hard to match against a list of known individuals.

This thesis builds and tests a software system that takes a degraded outdoor image, improves it, finds the objects in it, and identifies any faces against an enrolled gallery. All three steps run automatically as a single pipeline.

---

## The Three-Stage Pipeline

Every image that enters the system passes through three stages in sequence.

```
Stage 1         Stage 2          Stage 3
─────────       ───────────      ──────────────────────
Enhancement  →  Detection    →   Face Recognition
Improve the     Find objects     Match faces against
image quality   and draw boxes   a known gallery
```

**Stage 1 — Enhancement**: The image is processed to remove fog, raise low-light visibility, or reduce rain noise. The system chooses which technique to use based on the weather condition label attached to the image.

**Stage 2 — Detection**: A neural network scans the enhanced image and draws a box around every object it recognises — cars, people, trucks, motorcycles, and so on. Each box comes with a confidence score (how sure the model is).

**Stage 3 — Face Recognition**: For each face found inside the image, the system computes a numerical fingerprint (called an embedding) and searches a stored list of enrolled faces to find the closest match. If the match is close enough, the person is identified by name. Otherwise they are labelled Unknown.

---

## What Happens in Each Weather Condition

| Condition | What It Looks Like | How the System Handles It |
|---|---|---|
| Clear | Normal daytime outdoor scene | Light local contrast enhancement (CLAHE) |
| Rainy | Grey film over everything, streaks | CLAHE to remove the flat grey overlay |
| Low-light | Dark image, little colour | Zero-DCE++ neural network to brighten |
| Foggy | White haze, washed-out objects | FFA-Net neural network to remove haze |

---

## The Models and What They Do

### Enhancement

**CLAHE** (Contrast Limited Adaptive Histogram Equalisation) — a classical algorithm, not a neural network. It divides the image into small tiles and stretches the contrast in each tile independently. Fast (under 5 milliseconds), no training required.

**Zero-DCE++** — a very small neural network (about 10,500 trainable values, compared to millions in typical networks). It learns a curve that maps each pixel's brightness to a new brightness value, brightening dark areas while keeping bright areas natural. Takes about 39 milliseconds.

**FFA-Net** — a larger network trained specifically for removing outdoor haze. It models how light scatters through fog and reverses that effect. Takes about 4,226 milliseconds (over 4 seconds), which makes it too slow for live video but suitable for single image analysis.

### Detection

**YOLOv8n** — a neural network trained to find objects in images in a single pass. "n" stands for nano — the smallest and fastest version. Trained on 4,900 outdoor driving images from the BDD100K dataset to recognise 10 classes: car, truck, bus, person, rider, bicycle, motorcycle, traffic light, traffic sign, train.

### Face Recognition

**SCRFD** — finds where faces are in an image (a face detector, not a recogniser).

**ArcFace** — converts a detected face into a 512-number vector (its embedding). Two photos of the same person will produce similar vectors; photos of different people will produce different vectors.

**FAISS** — a fast search library. It holds all the enrolled face embeddings and, when given a new embedding, finds the closest match in the gallery almost instantly (under 1 millisecond for hundreds of entries).

---

## What the Numbers Mean

### Detection Accuracy — mAP@0.5

mAP@0.5 is a single number between 0 and 1 that summarises how accurately the detector finds objects. 1.0 means it found every object with a perfectly placed box. 0.0 means it found nothing.

Our result: **0.112** — trained on 7% of the available training data due to time and cost constraints. The same model trained on all the data is expected to reach about 0.37.

### Face Recognition Accuracy

Measured on a standard test — the Labeled Faces in the Wild (LFW) dataset:

- **98.61%** accuracy on 288 pairs (correctly judging whether two photos show the same person)
- When searching a gallery of 300 known identities, the system retrieves the correct person **98.67%** of the time

### Speed — Milliseconds and FPS

Latency is measured in milliseconds (ms). 1,000 ms = 1 second.

| Condition | Total time per image |
|---|---|
| Clear | ~111 ms |
| Low-light | ~146 ms |
| Rainy | ~111 ms |
| Foggy | ~4,333 ms |

For clear, rainy, and low-light: the system processes roughly 7–9 images per second on a T4 cloud GPU.

---

## The Most Important Finding

The experiment revealed something unexpected: **applying image enhancement to foggy and low-light images made face recognition less accurate**, not more.

- Fog without enhancement: **95.33%** face recognition accuracy
- Fog with FFA-Net enhancement: **93.67%**

The reason: FFA-Net is trained to make the whole scene look good to human eyes. A face occupies a tiny fraction of the image. The process that makes fog disappear can also blur the fine facial textures that the recognition system relies on.

Rain was the exception: CLAHE improved rain recognition from 97.67% → 98.33% because it makes a minimal, conservative change to the image that removes the rain overlay without disturbing face features.

---

## The Deployed System

The pipeline does not just run in a research notebook. It is deployed as a live web service:

- **Inference API** runs on HuggingFace Spaces (cloud GPU), accepts an image and returns all detections, recognised identities, and latency breakdown in JSON.
- **Backend server** (Spring Boot, Java) receives upload requests from users, calls the inference API, and stores results.
- **Web application** (Next.js) lets users upload images or videos, select a weather condition, and view annotated results with bounding boxes and face labels.

---

## Summary of Results by Phase

| Phase | What Was Done | Key Result |
|---|---|---|
| 2 — Enhancement | Benchmarked three image enhancers | Zero-DCE++ chosen: 108× faster than FFA-Net with acceptable quality |
| 3 — Detection | Trained YOLOv8n on outdoor driving data | mAP@0.5 = 0.112 on 7% of BDD100K |
| 4 — Recognition | Tested face recognition under degradation | 98.61% LFW accuracy; enhancement hurt foggy/low-light recognition |
| 5 — Optimization | Tested ONNX, INT8, pruning, TRT FP16 | Pruning gave 1.50× speedup; TRT FP16 environment failed |
| 6 — Deployment | Integrated pipeline measured end-to-end | ~111 ms for clear/rainy, ~146 ms for low-light, ~4,333 ms for foggy |

---

*Continue to GUIDE_2_INTERMEDIATE.md for the technical implementation details.*
*Thesis: "Outdoor Object Detection and Face Recognition Under Adverse Weather Conditions"*
*Author: Muhammad Bashir Dantani — USTC Master's Thesis — Defense: December 2026*
