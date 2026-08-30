# Robust AI-Generated Image Detection — TikTok TechJam 2026 (PS5)

Binary real-vs-AI image classification, optimised for the scored metric:

**Final Score = 0.5 · AUC(clean) + 0.5 · AUC(robust)** — ROC-AUC, threshold-free,
`AUC(robust)` = mean over the 14 required transform conditions.

`1` = AI-generated, `0` = authentic.

---

## Results (held-out, cross-generator)

| Benchmark | Final Score | Clean AUC | Robust AUC | Clean FPR / FNR |
|---|---|---|---|---|
| **Organiser composition** (WildFake pixel-diffusion + COCO, resolution-matched) | **0.933** | 0.955 | 0.911 | 4.1% / 23.3% * |
| DRAGON — 8 unseen latent-diffusion generators | 0.996 | 0.997 | 0.995 | 2.1% / 2.4% |
| SID_Set full-synthetic (in-domain) | 0.997 | 0.999 | 0.996 | 2.1% / 0.9% |

The organiser composition's six generator families (ADM, DALL·E, DDIM, DDPM,
Imagen, VQDM) have **zero representation in training** — a genuine
cross-generator test, not a held-out split.

Per-generator clean AUC: ADM 0.89 · DDPM 0.93 · DDIM 0.95 · Imagen 0.97 ·
DALL·E 0.99 · VQDM 1.00 (ViT-only, no CLIP branch: ADM 0.82).

\* Final Score is threshold-free. The FPR/FNR are at the operating threshold
0.215, calibrated on withheld *latent-diffusion* generators — pixel-diffusion
fakes score systematically lower, so one fixed cutoff under-flags them (the
same threshold is 2.1% / 2.4% on DRAGON). The calibration script lets an
operator retune per deployment.

---

## Model

```
Community Forensics ViT-S/16 @224   21,666,049 params   (frozen after warm-start)  → base logit
  + frozen OpenAI CLIP ViT-B/16 branch   → LayerNorm → Linear(768→256) → near-zero-init residual Δ
                                logit = base + Δ
```

- **Backbone**: `OwensLab/commfor-model-224` (timm
  `vit_small_patch16_224.augreg_in21k_ft_in1k`) — a purpose-built AI-image
  detector, #1 of 23 open detectors on an independent 2026 benchmark.
- **Semantic branch**: `openai/clip-vit-base-patch16` vision tower, **frozen**.
  A fine-tuned backbone loses ~0.20 AUC seen→unseen; a frozen large ViT loses
  ~0.09 — freezing is what makes it generalise across generator families.
- **Near-zero-init residual fusion**: the branch head starts ≈0, so the model
  begins identical to the proven ViT and only ever adds a correction. No
  per-branch auxiliary loss (that pressure makes a shallow branch memorise
  training-generator spectra and invert on unseen ones).
- Trainable: ~200K (the fusion head). Inference: ~108M (22M ViT + 86M frozen
  CLIP-B) — ~18× under the 2B limit. Runs on CPU at ~1 s/image.

The CLIP fusion head was trained on precomputed frozen features (the CLIP
forward is too slow to run live in the training loop on an Intel Arc iGPU).
That training path and the corpus-build scripts are in the maintainer's
`local_reference/`; the shipped model is `runs/iter7/best.pt`.

---

## Repository layout

```
detector/           the model, data, transforms, training, evaluation, calibration
pipeline.py         ingest · train · evaluate · predict · smoke · materialize-sid-set
config.yaml         paths and training knobs
run_iteration.sh    one loop: train → calibrate on held-out generators → evaluate
run_fast.sh         lean variant (organiser + DRAGON only)
finalize.sh         full 15-condition eval + error montages on a finished checkpoint
scripts/            eval-only tooling (WildFake stream-eval, shortcut probe, matched-set builder)
webapp/             local FastAPI demo — single image · robustness grid · batch
space/              Hugging Face Docker Space packaging (space/deploy.sh)
serve_demo.ps1      one command → a public Cloudflare-tunnel URL for the demo
hf_upload/          push the checkpoint to a HF model repo
requirements-*.txt  xpu (dev), cuda (fallback)
```

---

## Environment

Developed on a single laptop **Intel Arc iGPU (XPU)** — no CUDA.

```bash
python -m venv .venv                       # from a 3.12 interpreter
.venv/Scripts/python -m pip install -r requirements-xpu.txt
```

`requirements-xpu.txt` pins `torch==2.6.0+xpu` / `torchvision==0.21.0+xpu`
(2.13+xpu hangs on the first oneDNN GEMM with the Nov-2024 Arc driver). On an
NVIDIA box use `requirements-cuda.txt` and pass `--device cuda`; on CPU-only,
`--device cpu`.

Set before each run:

```bash
export SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=<cache>/sycl \
       HF_HOME=<cache>/hf HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

The first run downloads the pinned Community Forensics and CLIP checkpoints.

---

## Run the pipeline

```bash
# self-contained regression check (synthetic data, no dataset needed)
python pipeline.py smoke --device xpu

# train a detector (reads config.yaml) → calibrate → evaluate
bash run_iteration.sh <name>

# calibrate an operating point on held-out generators (never the test set)
python -m detector.calibrate runs/<name>/best.pt \
  --genval <dir> --calval <dir> --rule minmax_fpfn --apply

# required directory-to-JSON inference
python pipeline.py predict --input <folder> --output predictions.json \
  --checkpoint runs/<name>/best.pt
```

`config.yaml` knobs: `model_type` (`vit` / `hybrid`), `branch_kind`
(`clip,wavelet,fft` comma list), `vit_checkpoint` (warm-start),
`crop_from_native`, `safe_augment`, `threshold` (`auto` = use the calibrated
value).

Prediction JSON uses only the required fields:

```json
[{"image_path": "nested/example.jpg", "pred": 1}]
```

A sibling `predictions.scores.csv` keeps `probability_ai` separate.

---

## Host the demo

**Local:**

```bash
pip install fastapi uvicorn python-multipart
DETECTOR_CHECKPOINT=runs/iter7/best.pt uvicorn webapp.server:app --port 8000
```

**Public URL (free, temporary — for a judging window):**

```powershell
powershell -ExecutionPolicy Bypass -File serve_demo.ps1
```

Prints a `https://<...>.trycloudflare.com` URL; Ctrl+C stops it.

**Hugging Face Docker Space (permanent, needs HF PRO):**

```bash
bash space/deploy.sh <user>/<space-name>
```

**Model weights:**

```bash
python hf_upload/upload.py --repo <user>/<repo-name>
```

---

## Threshold calibration — the 5-way split

| split | role |
|---|---|
| `train` | fit weights |
| `val` | watch for training failure only |
| `genval` | one withheld generator — fits the threshold value |
| `calval` | a different withheld generator — picks the rule (`minmax_fpfn`) |
| `holdout` | reported; never an input to any decision |

Calibrating on held-out generators (not `val`) took recall on unseen fakes
from ~25% to ~98% with no change to AUC.

---

## Evaluation matrix (14 conditions + clean)

JPEG q90/70/50/30 · Gaussian blur σ0.5/1.0/2.0 · resize 0.5×/0.25× then up ·
Gaussian noise σ0.02/0.05/0.10 · brightness/contrast/saturation ±20% ·
centre-crop 80% then resize back.

Per-condition accuracy, F1, ROC-AUC, FPR, FNR and the clean-to-transformed
gap are written to `runs/<name>/metrics.csv` and `summary.json`;
representative FP/FN to `errors.csv`. `bash finalize.sh <ckpt> <tag>`
regenerates the full table + montages for a finished checkpoint.

---

## Limitations

- **Sensor noise** (σ0.10) is the weakest condition (AUC ~0.86) — additive
  noise most directly overwrites the high-frequency evidence.
- **ADM** (2021 pixel-diffusion) is the hardest family; the published ceiling
  for methods that don't train on it is ~0.82.
- **Modern flow/DiT generators** (SD3, Flux, Firefly) — the frontier where
  every open detector collapses; not in this evaluation.
- **Localised edits** (a real photo with one AI region) are out of scope —
  whole-image classification only.
- A single fixed threshold can't be optimal across pixel-space and latent
  diffusion at once; the score is threshold-free and the recipe is shipped.
- Hackathon prototype, not a production moderation system.
- Evaluation-only sets are never used for training or threshold selection
  (enforced: `detector/data.py:assert_not_eval_only`).
