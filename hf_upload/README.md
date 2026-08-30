---
license: apache-2.0
tags:
  - image-classification
  - ai-generated-image-detection
  - deepfake-detection
pipeline_tag: image-classification
---

# Robust AI-Generated Image Detector — TikTok TechJam 2026 (PS5)

Binary real-vs-AI image classifier optimised for **cross-generator
generalisation under redistribution**. Scored metric: `0.5·AUC(clean) +
0.5·AUC(robust)` over the 14 required transform conditions.

`1` = AI-generated, `0` = authentic.

## Architecture

```
Community Forensics ViT-S/16 @224   (frozen after warm-start)  → base logit
  + frozen OpenAI CLIP ViT-B/16 branch  → LayerNorm → Linear(768→256) → head → residual Δ
                                logit = base + Δ
```

- ~108M parameters at inference (22M ViT + 86M frozen CLIP), ~200K trainable.
- The CLIP branch is trained only as a small fusion head on **precomputed
  frozen features**, then trained with a feature-jitter augmentation that
  simulates the CLIP-embedding shift measured under degradation.
- No per-branch auxiliary loss (that pressure makes a shallow branch
  memorise training-generator spectra and invert on unseen ones).

## Results — organiser composition (WildFake pixel-diffusion vs COCO, resolution-matched)

Six generator families — ADM, DALL·E, DDIM, DDPM, Imagen, VQDM — with **zero
representation in training**.

| | Final Score | AUC(clean) | AUC(robust) |
|---|---|---|---|
| **This model** | **0.9326** | 0.9548 | 0.9105 |
| ViT-S only (no CLIP branch) | 0.9129 | 0.9288 | 0.8970 |

Per-generator clean AUC: ADM 0.89 · DDPM 0.93 · DDIM 0.95 · Imagen 0.97 ·
DALL·E 0.99 · VQDM 1.00. On DRAGON (8 unseen latent-diffusion generators):
Final Score 0.996.

## Usage

```python
import torch
from detector.model import load_checkpoint          # from the TechJam repo
model, meta = load_checkpoint("best.pt", device="cuda")   # "xpu" / "cpu"
model.eval()
# preprocess: resize short-edge 256 (bilinear), center-crop 224, ImageNet norm
prob_ai = torch.sigmoid(model(image_tensor))
pred = int(prob_ai >= meta["threshold"])   # threshold 0.215, see below
```

The frozen CLIP encoder weights are **not** in this checkpoint (kept at
~88 MB); `load_checkpoint` rebuilds them from `openai/clip-vit-base-patch16`.

## Operating threshold

The Final Score is threshold-free. For a hard label, the shipped threshold
**0.215** was calibrated on withheld *latent-diffusion* generators
(`minmax_fpfn` rule). Pixel-diffusion fakes score systematically lower, so
this threshold under-flags them on the organiser set (FNR ~23% clean vs
~2% on DRAGON). Re-calibrate per deployment — see `detector/calibrate.py`.

## Limitations

- Sensor noise (σ0.10) is the weakest condition (AUC 0.86).
- ADM (2021 pixel-diffusion) is the hardest family; the published ceiling
  for methods that do not train on it is ~0.82.
- Modern flow/DiT generators (SD3, Flux, Firefly v4) are the frontier where
  every open detector collapses — not in this evaluation.
- Whole-image classification only; localised edits are out of scope.

## Training data (public only)

SID_Set (full-synthetic), DRAGON (17 diffusion generators; 8 held out),
Community-Forensics-Small (latent-diffusion + GAN + LAION/ImageNet/CelebA/COCO
reals). Perceptual-hash deduplicated against the evaluation set. No
test-label training.

Full code, pipeline, and every iteration:
https://github.com/KIREETI001/TikTok-Tech-Jam-2026
