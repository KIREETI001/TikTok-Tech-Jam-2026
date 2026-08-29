# Robust AI-Generated Image Detection

TikTok TechJam 2026 · Problem Statement 5

## Inspiration

A detector that scores 99% on its own test set and 65% on next month's
generator is worse than useless — it ships false confidence. The briefing
deck says it plainly: *"clean accuracy is the #1 way to fool yourself"* and
*"there is no silver bullet."* We took that literally. Every decision was
driven by one number: **ROC-AUC on generator families the model has never
seen, measured after realistic redistribution** (JPEG, blur, resize, noise,
crop).

## What it does

Binary real-vs-AI image classification, built for the scored metric:

**Final Score = 0.5·AUC(clean) + 0.5·AUC(robust)** — threshold-free, where
AUC(robust) is the mean over the 14 required transform conditions.

| Held-out benchmark | Final Score | Clean AUC | Robust AUC |
|---|---|---|---|
| **Organiser composition** (WildFake pixel-diffusion + COCO, resolution-matched) | **0.927** | 0.953 | 0.900 |
| DRAGON — 8 unseen latent-diffusion generators | 0.996 | 0.997 | 0.995 |
| SID_Set full-synthetic (in-domain) | 0.997 | 0.999 | 0.996 |

The organiser composition's six families — ADM, DALL·E, DDIM, DDPM, Imagen,
VQDM — have **zero representation in training**. This is a true
cross-generator test, not a held-out split.

## How we built it

### Architecture — small, frozen backbones, additive residual

```
Community Forensics ViT-S/16   21.7M params   (frozen after warm-start)  → base logit
  + frozen OpenAI CLIP ViT-B/16 branch   → LayerNorm → Linear → zero-init residual Δ
                                logit = base + Δ
```

~108M parameters at inference (22M ViT + 86M frozen CLIP), **609K trainable**
— 18× under the 2B limit, runs on a laptop CPU at ~1 s/image.

- **Purpose-built backbone.** Community Forensics ViT-S is a detector trained
  across 20+ generator families; an independent 2026 benchmark ranks it #1
  of 23 open detectors (75% mean / 82% median accuracy, most stable).
- **Frozen semantic branch, not fine-tuned.** We measured — and a parallel
  pipeline independently measured — that a *fine-tuned* backbone loses ~0.20
  AUC from seen to unseen generators while a *frozen* large ViT loses only
  ~0.09. Fine-tuning teaches your training generators' quirks; a frozen CLIP
  that never saw your data can't overfit to it.
- **Zero-init residual fusion.** The fusion head starts at zero — at step 0
  the model is numerically identical to the proven ViT and can only *add* a
  learned correction. This is deliberately **not** a per-branch auxiliary
  loss: a parallel pipeline's frequency branch, forced to be independently
  label-predictive, memorised training-generator spectra and scored 0.457
  (inverted) on unseen generators. Ours is trained by the final BCE only.
- **Trained on precomputed frozen features.** The CLIP forward is too slow
  to run live in the training loop on an Intel Arc iGPU, so we precompute
  the ViT logit + CLIP embedding once per image and train only the ~200K
  fusion head on the cached vectors — seconds per config, dozens of
  experiments where one live run would have taken hours.

### What the CLIP branch bought us

| generator (clean AUC) | ViT only | + CLIP | Δ |
|---|---|---|---|
| **ADM** (2021 pixel-diffusion, hardest) | 0.817 | **0.895** | **+0.078** |
| Imagen | 0.952 | 0.976 | +0.024 |
| DDIM | 0.917 | 0.944 | +0.027 |
| DDPM | 0.899 | 0.916 | +0.017 |
| DALL·E / VQDM | 0.99 | 0.99 | ~0 |

Exactly the pattern the literature predicts: semantic features bridge the
GAN↔diffusion gap that low-level artifact features can't. We also built and
tested a hand-crafted frequency branch (NPR radial spectrum) — **it did not
help** (0.905 vs 0.908), confirming that frequency features don't transfer to
pixel-space diffusion, which by construction has no sharp spectral peaks.

### Training data — public only, content-matched

- **SID_Set** — real + fully-synthetic (we drop "tampered": localised photo
  editing, a different task, out of scope per the deck).
- **DRAGON** — 17 diffusion generators in training; 8 held out.
- **Community-Forensics-Small** — many latent-diffusion + GAN families;
  reals from LAION / ImageNet / CelebA / COCO. Perceptual-hash deduplicated
  against the evaluation set.
- 61,465 images, fresh class-balanced draw each epoch.

A shortcut probe (gradient-boosted trees on global pixel stats and DCT
high-frequency energy) confirms the corpus is **not** format-poisoned —
pixel-stats recoverability 60%, DCT-HF 52%, both below the ~65% "off-task"
line. The "+11 pp from removing JPEG-vs-PNG bias" (Fake or JPEG?, ECCV 2024)
doesn't apply because both classes are re-saved as JPEG.

### The augmentation *is* the robustness contribution

Per the deck: *"write your own transform script — this IS your robustness
contribution."* Training applies the six evaluated transform families across
continuous ranges, plus SAFE's RandomMask/Rotation and crop-from-native
(never resize — resizing low-pass-filters the artifact away). Noise at
training time has a measured mechanism: it suppresses semantic content while
exposing generation artifacts.

### Calibration — on held-out generators, never the test set

The metric is threshold-free, but a deployable detector needs an operating
point. We fit it with a 5-way split: `train / val / genval (one withheld
generator — fits the threshold) / calval (a different withheld generator —
picks the rule) / holdout (reported, never an input)`. This took recall on
unseen fakes from ~25% to ~98% with zero change to AUC. Calibrated threshold:
0.240, genval FPR 1.4% / FNR 1.3%, calval 1.4% / 2.2%.

## Iteration log — what actually moved the number

| Change | Organiser Final Score |
|---|---|
| Baseline ViT, SID_Set incl. tampered images | held-out 0.936 |
| **Drop tampered images** | in-domain 0.94 → 0.999 |
| **+ 17 DRAGON generators + held-out calibration** | unseen FNR 18.8% → 1.4% |
| **+ Community-Forensics-Small (data diversity)** | 0.717 → 0.804 → **0.913** |
| **+ frozen CLIP-B semantic branch** | 0.913 → **0.927** |

Two of the three biggest levers were **data, not architecture** — matching
the field consensus (a 2026 benchmark found 20–60% accuracy variance within
*identical architectures*, purely from training data).

## Trade-offs (the deck asks for this explicitly)

- **Robustness vs clean accuracy.** Heavy augmentation cost some clean
  training accuracy; clean AUC held and robust AUC rose. Worth it.
- **Generalisation vs specialisation.** We hold six generator families fully
  out of training. A detector tuned on them scores higher here and breaks on
  the next generator. We optimised for the break.
- **Semantic gain vs robust transfer.** The CLIP branch adds +0.025 clean
  AUC but only +0.003 robust — CLIP embeddings shift under degradation
  (cos-sim drops to ~0.8 under heavy JPEG/noise). Training the fusion on
  degraded-view embeddings recovers most of it.
- **Single threshold vs generator spread.** Pixel-diffusion fakes get lower
  AI-scores than latent-diffusion fakes; no one cutoff is optimal for both.
  We report the threshold-free score and ship the calibration recipe.
- **Model size vs a bigger ensemble.** ~108M inference params. Runs on a
  laptop CPU. The deck: *"a 2-branch ensemble may win 1% but cost you the
  demo. Ship what runs."*

## Known limitations

- **Sensor noise is the weakest condition** (organiser noise σ0.10 AUC 0.84):
  additive noise most directly overwrites the high-frequency evidence.
- **ADM** remains the hardest single family (0.90) — the furthest from
  anything in training; the literature ceiling here is ~0.82 for methods that
  don't train on it, so 0.90 is strong.
- **Modern flow/DiT generators** (SD3, Flux, Firefly v4) — not in the
  organiser set, and the class where every open detector collapses
  (Community Forensics drops to 35–42%). Named honestly as the frontier.
- **Localised edits** (a real photo with one AI region) — out of scope,
  whole-image classification only.

## Built with

PyTorch 2.6 (Intel Arc XPU — no CUDA), timm, scikit-learn. Backbones:
`OwensLab/commfor-model-224`, `openai/clip-vit-base-patch16`. All training on
a single laptop iGPU.
