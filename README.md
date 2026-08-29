# Robust AI-Generated Image Detection — TikTok TechJam 2026 (PS5)

Binary real-vs-AI image classification, optimised for the scored metric:

**Final Score = 0.5 · AUC(clean) + 0.5 · AUC(robust)** — ROC-AUC, threshold-free,
`AUC(robust)` = mean over the 14 required transform conditions.

`1` = AI-generated, `0` = authentic.

---

## Results (held-out, cross-generator)

Full methodology and every iteration in [`experiments.md`](experiments.md);
tabulated in [`TRAINING_REPORT.docx`](TRAINING_REPORT.docx).

| Benchmark | Final Score | Clean AUC | Robust AUC | Clean FPR / FNR |
|---|---|---|---|---|
| **Organiser composition** (WildFake pixel-diffusion + COCO, resolution-matched) | **_pending iter6_** / 0.9126 | 0.9286 | 0.8965 | 5.9% / 24.8% |
| DRAGON — 8 unseen latent-diffusion generators | 0.9959 | 0.9972 | 0.9945 | 2.1% / 2.4% |
| SID_Set full-synthetic (in-domain) | 0.997 | 0.9985 | 0.9963 | 2.1% / 0.9% |

The organiser composition's six generator families (ADM, DALL·E, DDIM, DDPM,
Imagen, VQDM) have **zero representation in training** — it is a genuine
cross-generator test, not a held-out split of the training distribution.

Per-generator clean AUC on the organiser set: ADM 0.82 · DDPM 0.90 ·
DDIM 0.92 · Imagen 0.95 · DALL·E 0.99 · VQDM 0.99.

---

## Model

```
Community Forensics ViT-S/16 @224   21,666,049 params   (frozen after warm-start)
  └─ base logit
  + frozen OpenAI CLIP ViT-B/16 branch    → LayerNorm → Linear → zero-init residual head
  + SAFE DWT high-frequency branch         → conv stem → zero-init residual head   (optional)
  ────────────────────────────────────────────────────────────────────
  logit = base + Σ branch corrections
```

- **Backbone**: `OwensLab/commfor-model-224` (timm `vit_small_patch16_224.augreg_in21k_ft_in1k`).
  A purpose-built AI-image detector, #1 of 23 on its own benchmark out-of-the-box.
- **Semantic branch**: `openai/clip-vit-base-patch16` vision tower, **frozen** (ViT-L/14 was ~6x slower on the Arc iGPU for a marginal gain).
  A fine-tuned backbone loses ~0.20 AUC seen→unseen; a frozen large ViT loses
  ~0.09 — freezing is what makes it generalise.
- **Frequency branch** (SAFE, KDD 2025): input is the DWT `bior1.3` diagonal
  detail sub-band, not RGB — a *local* frequency statistic that survives crops.
- **Zero-init residual fusion**: each branch head starts at 0, so the model
  begins identical to the proven ViT and only ever adds a correction. No
  per-branch auxiliary loss (that pressure makes a shallow branch memorise
  training-generator spectra and invert on unseen ones).
- Trainable parameters: ~0.3–0.6M. Total inference: ~108M (ViT-S 22M + frozen CLIP-B 86M) — ~18x under the 2B limit.
  Runs on CPU at ~1 s/image.

---

## Environment

Trained on a single laptop **Intel Arc iGPU (XPU)** — no CUDA.

```bash
python -m venv .venv                       # from a 3.12 interpreter
.venv/Scripts/python -m pip install -r requirements-xpu.txt
```

`requirements-xpu.txt` pins `torch==2.6.0+xpu` / `torchvision==0.21.0+xpu`
(2.13+xpu hangs on the first oneDNN GEMM with the Nov-2024 Arc driver).
For an NVIDIA box, use `requirements-cuda.txt` and pass `--device cuda`.

Set before every run:

```bash
export SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=<cache>/sycl \
       HF_HOME=<cache>/hf HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

The first run downloads the pinned Community Forensics and CLIP checkpoints.

---

## Reproduce

```bash
# 1. materialise the public training corpus (SID_Set + DRAGON + Community-Forensics-Small)
python pipeline.py materialize-sid-set --split train  --shards 30 --output <data>/sid_train --max-size 448
python scratchpad/materialize_dragon.py     # 17 training generators + 8 held out
python scratchpad/materialize_cf_small.py   # latent-diffusion + GAN + diverse reals
python scratchpad/build_iter4.py            # assemble + perceptual-hash dedup vs eval set

# 2. one iteration: train -> calibrate on held-out generators -> evaluate
bash run_iteration.sh <name>                # full: internal + fs + dragon + organiser + montages
bash run_fast.sh <name>                     # lean: organiser + dragon only

# 3. predict a folder (required deliverable format)
python pipeline.py predict --input <folder> --output predictions.json --checkpoint runs/<name>/best.pt
```

`config.yaml` holds the knobs. Key ones:

| Key | Meaning |
|---|---|
| `model_type` | `vit` (backbone only) or `hybrid` (+ branches) |
| `branch_kind` | `clip`, `wavelet`, `fft` — comma-separate for multiple |
| `vit_checkpoint` | warm-start the backbone from a prior run |
| `crop_from_native` | skip `Resize`, `RandomCrop(224)` from native pixels (SAFE) |
| `safe_augment` | `RandomRotation(180)` + `RandomMask` |
| `threshold` | `auto` = use the held-out-generator-calibrated value |

---

## Threshold calibration

The scored metric is threshold-free; a deployable detector still needs an
operating point. Fit it on **held-out generators, never the test set**:

| Split | Role |
|---|---|
| `train` | fit weights |
| `val` | watch for training failure only |
| `genval` | one withheld generator — fits the threshold value |
| `calval` | a different withheld generator — picks the rule (`minmax_fpfn`) |
| `holdout` | reported; never an input to any decision |

```bash
python -m detector.calibrate runs/<name>/best.pt \
  --genval <dir> --calval <dir> --rule minmax_fpfn --apply
```

This took recall on unseen fakes from ~25% to ~98% with no change to AUC.

---

## Evaluation matrix (the 14 required conditions + clean)

JPEG q90/70/50/30 · Gaussian blur σ0.5/1.0/2.0 · resize 0.5×/0.25× then up ·
Gaussian noise σ0.02/0.05/0.10 · brightness/contrast/saturation ±20% ·
center-crop 80% then resize back.

Each row reports accuracy, F1, ROC-AUC, FPR, FNR, and the clean-to-transformed
gap ([`detector/evaluation.py`](detector/evaluation.py)). Representative FP/FN
with the model's probability are written to `errors.csv` and montaged into
[`ERROR_ANALYSIS.md`](ERROR_ANALYSIS.md).

---

## Outputs (`runs/<name>/`)

```
manifest.csv     selected images, labels, split, SHA-256
best.pt          best validation ROC-AUC checkpoint (frozen CLIP weights excluded)
training.csv     per-epoch loss + clean validation metrics
metrics.csv      clean + every transform condition
summary.json     mean/worst transformed performance, error-rate goal block
errors.csv       representative FP/FN per condition
```

Prediction JSON uses only the required fields:

```json
[{"image_path": "nested/example.jpg", "pred": 1}]
```

A sibling `predictions.scores.csv` keeps `probability_ai` separate from the
minimal organiser JSON.

---

## Limitations

- **Sensor noise** is the weakest condition (organiser noise σ0.10 AUC ~0.85)
  — additive noise most directly overwrites the high-frequency evidence.
- **ADM** (2021 ImageNet pixel-diffusion) is the hardest single family — the
  furthest from anything in training.
- **Localised edits** (real photo + one AI region) are out of scope — this is
  whole-image classification only.
- A single fixed threshold cannot be optimal across both pixel-space and
  latent diffusion at once; we report the threshold-free score and ship the
  calibration recipe.
- Hackathon prototype, not a production moderation system.
- The evaluation-only benchmark must never be used for training or threshold
  selection (enforced in code: `detector/data.py:assert_not_eval_only`).
