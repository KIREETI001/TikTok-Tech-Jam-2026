# Robust AI-Generated Image Detection

> TikTok TechJam 2026 · Problem Statement 5
> **Status: numbers below are iteration 4; iteration 6 (frozen-CLIP branch) result pending — placeholders marked ⟨iter6⟩.**

## Inspiration

A detector that scores 99% on its own test set and 65% on next month's
generator is worse than useless — it ships false confidence. The briefing
deck says it plainly: *"clean accuracy is the #1 way to fool yourself,"* and
*"there is no silver bullet."* We took that literally. Every decision in this
project was driven by one number: ROC-AUC on generator families the model
has **never seen**, measured after realistic redistribution (JPEG, blur,
resize, noise, crop).

## What it does

Binary real-vs-AI image classification, built for the scored metric:

**Final Score = 0.5 · AUC(clean) + 0.5 · AUC(robust)**, where AUC(robust) is
the mean ROC-AUC over the 14 required transform conditions.

| Held-out benchmark | Final Score | Clean AUC | Robust AUC |
|---|---|---|---|
| Organiser composition (WildFake pixel-diffusion + COCO, resolution-matched) | **⟨iter6⟩** / 0.9126 (iter4) | ⟨iter6⟩ / 0.9286 | ⟨iter6⟩ / 0.8965 |
| DRAGON — 8 unseen latent-diffusion generators | 0.9959 | 0.9972 | 0.9945 |
| SID_Set full-synthetic (in-domain) | 0.997 | 0.9985 | 0.9963 |

The organiser benchmark's six generator families (ADM, DALL·E, DDIM, DDPM,
Imagen, VQDM) have **zero representation in training** — it is a true
cross-generator test, not a held-out split.

## How we built it

### Architecture — small, frozen-backbone, additive branches

```
Community Forensics ViT-S/16  (21.7M params, frozen after stage 1)  → base logit
  + frozen OpenAI CLIP ViT-B/16 branch      → zero-init residual Δ₁
  + SAFE DWT high-frequency branch          → zero-init residual Δ₂
                                    logit = base + Δ₁ + Δ₂
```

- **Purpose-built backbone.** Community Forensics ViT-S is a detector trained
  across 20+ generator families; it ranks #1 of 23 on its own benchmark
  out-of-the-box. 21.7M parameters — 90× under the 2B limit.
- **Frozen semantic branch, not fine-tuned.** We measured (and a parallel
  pipeline independently measured) that a *fine-tuned* backbone loses ~0.20
  AUC from seen to unseen generators, while a *frozen* large ViT loses only
  ~0.09. Fine-tuning on your training generators teaches their specific
  quirks. A frozen CLIP that never saw your data cannot overfit to it.
- **Zero-initialised residual fusion.** Each branch's fusion head starts at
  exactly zero, so at step 0 the model is numerically identical to the
  proven ViT and can only *add* a learned correction. This is deliberately
  **not** a per-branch auxiliary loss — a parallel pipeline's FFT branch,
  forced to be independently label-predictive, memorised training-generator
  spectra and scored 0.457 (inverted) on unseen generators. Ours is trained
  by the final BCE only.
- **SAFE high-frequency branch** (KDD 2025): input is the DWT diagonal
  detail sub-band, not RGB — a *local* frequency statistic that survives
  crops, where a global FFT does not.

### Training data — public only, content-matched

- **SID_Set** — real + fully-synthetic (we drop the "tampered" class: it's
  localised photo editing, a different task, explicitly out of scope).
- **DRAGON** — 17 modern diffusion generators in training; 8 held out.
- **Community-Forensics-Small** — many latent-diffusion + GAN families, with
  real images from LAION / ImageNet / CelebA / COCO. Perceptual-hash
  deduplicated against the evaluation set.
- 61,465 images; each epoch draws a fresh class-balanced 20–24k.

### The augmentation *is* the robustness contribution

Per the deck: *"write your own transform script — this IS your robustness
contribution."* Training applies the six evaluated transform families across
continuous ranges, plus SAFE's RandomMask and RandomRotation, and
crop-from-native (never resize — resizing low-pass-filters the artifact
away). Heavy augmentation lowered clean training accuracy and held clean
AUC — the trade the deck says to make.

### Calibration — on held-out generators, never the test set

The scored metric is threshold-free, but a deployable detector needs an
operating point. We fit it with a 5-way split: `train / val / genval (one
withheld generator, fits the threshold) / calval (a different withheld
generator, picks the rule) / holdout (reported, never an input)`. This took
recall on unseen fakes from ~25% to ~98% with zero change to AUC.

## Iteration log — what actually moved the number

| Change | Effect on organiser Final Score |
|---|---|
| Baseline ViT, SID_Set with tampered images | held-out 0.936 |
| **Drop tampered images** | in-domain 0.94 → 0.999 |
| **+ 17 DRAGON generators + held-out calibration** | unseen FNR 18.8% → 1.4% |
| **+ Community-Forensics-Small (data diversity)** | 0.804 → **0.9126** |
| **+ frozen CLIP branch + SAFE DWT + SAFE aug** | 0.9126 → **⟨iter6⟩** |

Two of the biggest levers were data, not architecture — matching the deck's
"augmentation + data alignment > architecture tricks."

## Trade-offs (the deck asks for this explicitly)

- **Robustness vs clean accuracy.** Heavy augmentation cost a little clean
  training accuracy; clean AUC held and robust AUC rose. Worth it.
- **Generalisation vs specialisation.** We hold six generator families fully
  out of training. A detector tuned on them would score higher on this
  benchmark and break on the next generator. We optimised for the break.
- **Single threshold vs generator spread.** Pixel-diffusion fakes get
  systematically lower AI-scores than latent-diffusion fakes; no single
  cutoff is optimal for both. We report the threshold-free score as the
  headline and ship the held-out calibration recipe.
- **Model size vs a bigger ensemble.** ~108M inference params (ViT-S 22M + frozen CLIP-B 86M), 609K trainable. Runs on CPU in ~1s/image — the demo
  works on a laptop. The deck: *"a 2-branch ensemble may win 1% but cost you
  the demo. Ship what runs."*

## Known limitations

- **Sensor noise is the weakest condition** (organiser noise σ0.10 AUC ~0.85):
  additive noise most directly overwrites the high-frequency evidence
  detectors lean on.
- **ADM** (2021 ImageNet pixel-diffusion) is the hardest single family
  (clean AUC ~0.82) — it is the furthest from anything in training.
- **Localised edits** (a real photo with one AI-generated region) are not
  detected — whole-image classification only, per the deck's scope.
- Camera-RAW photographs are the real-image class most at risk of false
  positives; we add a RAISE-1k diagnostic slice.

## Built with

PyTorch 2.6 (Intel Arc XPU — no CUDA), timm, pytorch-wavelets, scikit-learn.
Backbones: `OwensLab/commfor-model-224`, `openai/clip-vit-base-patch16`.
All training on a single laptop iGPU.
