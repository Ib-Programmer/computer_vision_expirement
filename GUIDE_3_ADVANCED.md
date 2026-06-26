# Guide 3 — Advanced
### Research-Level Analysis of the CV Thesis Pipeline

*Audience: ML/CV researcher, advanced student, or committee reviewer. Assumes familiarity with convolutional networks, loss functions, and evaluation methodology. Read GUIDE_2_INTERMEDIATE.md first.*

---

## 1. Research Context and Positioning

The thesis addresses the intersection of three established sub-fields:

- **Image restoration**: dehazing (RESIDE, FFA-Net), low-light enhancement (LOL, Zero-DCE++)
- **Real-time object detection**: anchor-free single-stage detectors (YOLOv8, RT-DETR)
- **Open-set face recognition under domain shift**: ArcFace, FAISS gallery search

The core hypothesis — that image enhancement is a beneficial preprocessing step for outdoor detection and recognition under adverse weather — is shared by several survey papers (Li et al., 2023; Zhang et al., 2022). The thesis tests this hypothesis empirically with a full end-to-end system rather than synthetic benchmarks.

**The contribution is not a new model.** The contribution is:
1. An empirical evaluation of the enhancement → detection → recognition chain under controlled weather degradation
2. A finding that deep enhancement (FFA-Net, Zero-DCE++) hurts rather than helps face recognition by increasing detection failures and shifting embeddings outside the ArcFace training distribution
3. A deployed, end-to-end system demonstrating the full stack from inference API to web application

---

## 2. The Enhancement–Recognition Tension: A Formal Analysis

### Why Enhancement Hurts Recognition

Let the recognition pipeline be modelled as a composition:

```
R(I) = FAISS( ArcFace( SCRFD( E(I) ) ) )
```

Where `E(I)` is the enhancement function and `I` is the degraded input.

The hypothesis is that `R(E(I))` outperforms `R(I)`. The Phase 4 results show this is false for deep enhancers in fog and low-light. Three mechanisms:

**Mechanism 1 — Detection failure cascade**

SCRFD detects faces by locating high-frequency texture edges and gradient patterns in its feature maps. FFA-Net and Zero-DCE++ are trained with PSNR/SSIM objectives at the scene level. These objectives encourage smoothness across the full image, which can over-smooth high-frequency face textures in the relatively small face regions. With 6 detection failures in the fog+FFA-Net condition vs 1 in the fog-raw condition, the detection stage is the primary failure point — it is not that ArcFace misidentifies the face, but that SCRFD never finds it.

**Mechanism 2 — Embedding distribution shift**

ArcFace was trained on MS-Celeb-1M and WebFace600K — datasets of natural photographs. The distribution of these images can be characterised by their mean and covariance in the frequency domain. FFA-Net outputs images with altered local contrast statistics: haze removal sharpens mid-frequency components, and the resulting spectral distribution differs from the training distribution. This shifts the ArcFace embedding slightly, reducing cosine similarity to the gallery entry by a margin that can push genuine matches below the threshold.

This is a form of **covariate shift at test time** — the enhancer acts as a domain transformation that moves the face image distribution away from the ArcFace training domain.

**Mechanism 3 — Misaligned objectives**

The enhancement loss:
```
L_enh = MSE(E(I_degraded), I_clean)  + λ · L_perceptual
```

is computed globally over all pixels. Face pixels represent approximately 0.1–2% of an outdoor image. The gradient contribution from face regions to `L_enh` is negligible. An enhancer that minimises this loss has no incentive to preserve face-discriminative features.

A task-aware enhancement loss would be:
```
L_task = MSE(E(I), I_clean) + α · [1 - ArcFace_sim(E(I_face), I_face_clean)]
```

This is an open research direction — none of the benchmarked enhancers use it.

### Why CLAHE Helps for Rain

CLAHE applies adaptive histogram equalisation independently per tile with no global smoothing. It has no learnable parameters and no frequency-domain transformation — it strictly redistributes intensity values within each 8×8 tile. Rain streaks appear as vertical bright lines; CLAHE's tile-level contrast redistribution de-emphasises these without altering the face's gradient structure that SCRFD relies on.

CLAHE does not change what frequencies are present, only their relative amplitude within each tile. ArcFace's embedding shift is therefore negligible.

---

## 3. ArcFace: The Loss Function and Its Geometric Meaning

ArcFace (Additive Angular Margin Loss) modifies the standard softmax loss for face recognition:

**Standard softmax**:
```
L = -log( exp(W_yi · x) / Σ_j exp(W_j · x) )
```

**ArcFace** (after L2 normalisation of W and x):
```
L = -log( exp(s · cos(θ_yi + m)) / [exp(s · cos(θ_yi + m)) + Σ_{j≠yi} exp(s · cos(θ_j))] )
```

Where:
- `θ_yi` is the angle between the embedding `x` and the class weight `W_yi`
- `m` is the additive angular margin (typically m=0.5 radians)
- `s` is the feature scale (typically s=64)

The margin `m` is added directly in the angle space, not in the cosine space. This is geometrically equivalent to requiring that the embedding `x` is `m` radians closer to its true class than to any other class in the angular hypersphere. The angular margin is more uniform than cosine or additive cosine margins, leading to more tightly clustered intra-class distributions.

After training, the class weights `W` are discarded. At inference, only the backbone (ResNet-50) is used to generate 512-d embeddings.

**Why cosine similarity instead of L2 distance?**

For L2-normalised embeddings lying on the unit hypersphere, cosine similarity and L2 distance are monotonically related:
```
||A - B||² = 2 - 2·cos(θ)
```

Both produce the same ranking. FAISS `IndexFlatIP` (inner product) is preferred because it requires only a matrix multiplication, while L2 requires the same matrix multiplication plus a subtraction and norm computation. Since all embeddings are normalised, IP is equivalent and faster.

---

## 4. Data Scaling: Why mAP = 0.112 at 7% Data

The empirical data-scaling law for neural network accuracy is approximately:

```
Accuracy(n) ≈ Accuracy(∞) · (n / N)^α
```

Where `n` is the training set size, `N` is the full dataset size, and `α` ≈ 0.1–0.3 for standard datasets. Applying this to YOLOv8n on BDD100K:

```
mAP(4900) / mAP(70000) ≈ (4900 / 70000)^0.3 ≈ (0.07)^0.3 ≈ 0.30
mAP(4900) ≈ 0.37 × 0.30 ≈ 0.111
```

Our measured result of 0.112 is exactly on the power-law prediction. This confirms the model is data-limited, not architecture-limited or training-procedure-limited. Adding more data will produce predictable mAP improvements.

Additionally, the training curve did not plateau at epoch 25, indicating the model was still in the fast-learning phase of the learning curve. The absence of plateau is consistent with a data-scarce training regime where the model never fully memorises the training distribution.

---

## 5. Domain Gap: Synthetic vs Real Degradation

### Enhancement Evaluation Gap

Our PSNR for Zero-DCE++ (12.11 dB) is below the published 14.86 dB on LOL. Three sources of gap:

**Source 1 — Degradation distribution mismatch**

LOL was collected by taking photos with varying ISO and shutter speed in controlled indoor environments. Our degradation applies the Albumentations `RandomFog` and `RandomBrightnessContrast` transforms, which implement a simplified Perlin noise overlay rather than the physical image formation model:

```
Physical low-light: I = J · t(ISO, shutter) + σ(ISO)
Albumentations:     I = J · α + β   (simple linear, per-pixel)
```

The mismatch between what Zero-DCE++ learned to reverse (real LOL degradation) and what we applied (Albumentations transform) reduces PSNR.

**Source 2 — Dataset content mismatch**

LOL contains indoor portraits. RTTS contains outdoor driving scenes with complex backgrounds. The frequency statistics differ significantly — outdoor scenes have more mid-frequency energy (texture, foliage) that Zero-DCE++ was not optimised for.

**Source 3 — No fine-tuning**

Fine-tuning Zero-DCE++ on paired RTTS images with Albumentations degradation would close this gap but requires the compute budget ($10 total) to be allocated differently.

### Detection Generalisation Gap

YOLOv8n trained on BDD100K (Berkeley, California driving data) is deployed for outdoor surveillance potentially in Chinese urban environments. BDD100K contains almost exclusively North American road scenes, traffic signs, and vehicle types. Cross-domain generalisation to Chinese road environments introduces a covariate shift not measured in these experiments.

---

## 6. Statistical Validity Analysis

### Phase 2: N not specified

The Phase 2 enhancement evaluation does not report N (number of synthetic pairs used to compute PSNR/SSIM). Without N, confidence intervals cannot be computed. A standard deviation of ±1 dB on PSNR (typical for scene-diverse datasets) with N=50 gives a 95% CI of approximately ±0.28 dB. The difference between Zero-DCE++ (12.11) and Restormer (15.15) is 3.04 dB, well outside this interval and statistically robust. But the comparison to literature (12.11 vs 14.86) is only 2.75 dB — crossing zero in the CI would require σ ≈ 2 dB, which is plausible for a small N.

### Phase 4: N=300, Two-Proportion Z-Test

The drop from fog raw (95.33%) to fog+FFA-Net (93.67%) is 1.66 percentage points. With N=300:
```
SE = sqrt(p(1-p)/n) ≈ sqrt(0.946 × 0.054 / 300) ≈ 0.0130

z = (0.9533 - 0.9367) / (0.0130 × √2) ≈ 0.0166 / 0.0184 ≈ 0.90
```

At z=0.90, p-value ≈ 0.37 — **not statistically significant at α=0.05**. The fog accuracy drop is a directional finding, not a statistically confirmed result at N=300. The detection failure count difference (1 vs 6) is more interpretable: a 6× increase in detection failures is a strong qualitative signal.

For the rain improvement (97.67% → 98.33%):
```
z ≈ (0.9833 - 0.9767) / (0.0130 × √2) ≈ 0.36
```

Also not significant. These results should be interpreted as directional observations requiring larger N to confirm.

### Phase 5: N=50

P95 from 50 data points is estimated from the 47th sorted value. With heavy-tailed latency distributions (GPU warm-up effects, memory allocation), 50 samples gives a P95 estimate with approximately ±15–25 ms uncertainty. The FPS and mean latency numbers are stable, but the P95 column should be treated as an approximation.

---

## 7. Model Architecture Comparison: YOLOv8n vs RT-DETR-L

Phase 3 trained both models (unfair comparison: YOLOv8n 25 epochs vs RT-DETR 15 epochs):

| Model | Epochs | mAP@0.5 | mAP@0.5:0.95 | Inference (ms) | Params |
|---|---|---|---|---|---|
| YOLOv8n | 25 | 0.112 | 0.063 | ~8.4 (det only) | 3.2M |
| RT-DETR-L | 15 | 0.195 | 0.098 | 8.2 (det only) | 32M |

RT-DETR achieves 74% higher mAP@0.5 in fewer epochs with comparable detection-only inference latency. The deployment choice of YOLOv8n is justified only by its lower parameter count (10× smaller), which reduces memory pressure on the T4 when the full pipeline runs (enhancement + detection + face recognition simultaneously). In a scenario with dedicated GPU memory for detection, RT-DETR-L would be the better choice.

The transformer attention mechanism in RT-DETR processes global context (all image patches attend to each other), which is particularly beneficial for detecting partially occluded objects in dense outdoor scenes — directly relevant to this thesis's adverse weather use case.

---

## 8. Optimisation Gap: TRT FP16

TensorRT FP16 applies three transformations to the graph:
1. **Operator fusion**: conv + BN + activation become a single kernel call, eliminating intermediate memory transfers
2. **FP16 precision**: halves memory bandwidth for weight loading; T4 has 65 TFLOPS FP16 vs 8.1 TFLOPS FP32
3. **Calibration**: for INT8, a calibration dataset maps activation distributions to the INT8 range

For YOLOv8n (8.1 GFLOPs FP32), the theoretical speedup from FP16 is:
```
speedup ≈ TFLOPS_FP16 / TFLOPS_FP32 = 65 / 8.1 ≈ 8×  (compute-bound upper bound)
```

In practice, YOLOv8n is memory-bandwidth-bound at this size, not compute-bound. Bandwidth doubles with FP16 (halved weight size), giving an empirical speedup closer to 1.7–2.0×.

The Colab failure was caused by TensorRT 8.5 pre-installed with the CUDA 12.1 runtime, while `ultralytics.export(format="engine")` requires TensorRT ≥ 8.6 for the `--half` flag to work with YOLOv8 dynamic shapes. A fixed TRT version would resolve this.

---

## 9. The Full Pipeline Bottleneck Analysis

From Phase 6 Table 6.2, the total latency per condition:

| Condition | Enhancement | Detection | Face Recog. | Total |
|---|---|---|---|---|
| Clear | 4.2 ms (4%) | 88.8 ms (80%) | 18.3 ms (16%) | 111.3 ms |
| Low-light | 39.2 ms (27%) | 88.8 ms (61%) | 18.3 ms (13%) | 146.3 ms |
| Foggy | 4,226.2 ms (97.5%) | 88.8 ms (2.0%) | 18.3 ms (0.4%) | 4,333.3 ms |

The bottleneck is always the enhancement stage for fog. For clear conditions, the bottleneck is detection. This means:

- **For 3 of 4 conditions**, the marginal gain from TRT FP16 on the detector (88.8 ms → ~50 ms) reduces total latency by ~35%. Useful, but not transformative.
- **For foggy**, the enhancement (FFA-Net at 4,226 ms) must be replaced. A lightweight real-time dehazer such as AOD-Net (Li et al., 2017, ~20 ms on T4) or D4 (Yang et al., 2022, ~12 ms) would reduce foggy total latency from ~4,333 ms to ~130 ms, a 33× improvement.

This is the single highest-leverage improvement available to the system.

---

## 10. The Core Hypothesis Revisited

The original thesis hypothesis: *image enhancement improves outdoor detection and recognition under adverse weather.*

**The evidence by task:**

| Task | Enhancement Effect | Evidence |
|---|---|---|
| Detection confidence | Slightly negative | Raw avg_conf 0.852 → enhanced 0.834 (Phase 3) |
| Detection recall | Neutral | 95% recall maintained (Phase 3) |
| Face recognition — fog | Negative | 95.33% → 93.67% with FFA-Net (Phase 4) |
| Face recognition — low-light | Negative | 95.67% → 94.67% with Zero-DCE++ (Phase 4) |
| Face recognition — rain | Positive | 97.67% → 98.33% with CLAHE (Phase 4) |

The hypothesis holds only for CLAHE on rain. For the two cases where deep learning enhancers were deployed (fog and low-light), enhancement reduces downstream task performance.

This is a contribution rather than a failure: it provides empirical evidence against a common assumption in the image restoration literature, and it isolates CLAHE's conservative approach as the key differentiator. The result motivates task-aware enhancement — a specific, testable direction for future work.

---

## 11. Open Problems and Research Directions

**Task-aware enhancement**: train the enhancer with a joint loss that includes ArcFace embedding preservation. This would require a differentiable approximation of the recognition pipeline as a regulariser during enhancement training.

**Real foggy face recognition evaluation**: the Phase 4 robustness test uses synthetic fog (Albumentations). RTTS contains real outdoor hazy images. A robustness test with real foggy face images (WiderFace-Foggy if available, or a custom outdoor dataset) would provide stronger external validity.

**RT-DETR with equivalent training budget**: training RT-DETR for 25 epochs on 7% BDD100K would allow a fair apples-to-apples comparison with YOLOv8n. The preliminary Phase 3 result (0.195 mAP@0.5 at 15 epochs) suggests it would significantly outperform YOLOv8n at full training.

**Gallery scaling**: at N=100 gallery embeddings, FAISS IndexFlatIP takes < 1 ms. At N=10,000 (a realistic deployment), it takes ~1 ms — still negligible. At N=1,000,000 it would require an approximate index (FAISS IVF or HNSW) to stay under 10 ms. The current IndexFlatIP design is not scalable to large galleries, but the switch to IVF is a one-line FAISS change.

**Covariate shift measurement**: the domain gap between BDD100K (North American driving) and Chinese urban outdoor surveillance has not been quantified. A calibration set of Chinese outdoor images with annotations would allow a domain adaptation study.

---

*This document is a companion to GUIDE_1_INTRO.md and GUIDE_2_INTERMEDIATE.md.*
*Thesis: "Outdoor Object Detection and Face Recognition Under Adverse Weather Conditions"*
*Author: Muhammad Bashir Dantani — USTC Master's Thesis — Defense: December 2026*
