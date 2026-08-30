# Robust AI-Generated Image Detection

**TikTok TechJam 2026 · Problem Statement 5** — build a prototype that tells
AI-generated images from authentic photographs, and *keeps working* after the
compression, blur, resizing and noise every image picks up in circulation.

> **Result: Final Score `0.933`** on the organisers' benchmark composition
> (WildFake vs COCO, resolution-matched), measured on **six generator
> families with zero representation in training**.

| | |
|---|---|
| **Live demo** | run `serve_demo.ps1` → a public `https://<...>.trycloudflare.com` URL (see [Try it](#try-it-2-minutes)) |
| **Model weights** | `runs/iter7/best.pt` — 108M params, Apache-2.0 → HuggingFace via `hf_upload/upload.py` |
| **Demo video** | *(link in the Devpost submission)* |
| **Robustness table** | [below](#robustness-the-15-condition-matrix) · full: [`docs/ERROR_ANALYSIS.md`](docs/ERROR_ANALYSIS.md) |
| **Error-analysis note** | [`docs/ERROR_ANALYSIS.md`](docs/ERROR_ANALYSIS.md) + FP/FN montages [below](#error-analysis) |

`pred = 1` → AI-generated · `pred = 0` → authentic.

---

## The task and how it is scored

From the briefing deck:

- **Primary metric: ROC-AUC** — threshold-free, robust to class imbalance.
  *"Clean accuracy is the #1 way to fool yourself."*
- **Final Score = 0.5 · AUC(clean) + 0.5 · AUC(robust)**, where AUC(robust)
  is the mean ROC-AUC over the **14 required transform conditions**.
- **Cross-generator test** — evaluate on generators *not* in training. That
  is the real generalisation test, and every number in this README is
  measured that way.
- No hard FPR/FNR bar is set by the organisers; the scored quantity is the
  Final Score.

### The 14 required transforms (deck slide 11)

| family | settings |
|---|---|
| JPEG compression | quality 90, 70, 50, 30 |
| Gaussian blur | σ = 0.5, 1.0, 2.0 |
| Resize | 0.5× / 0.25× then upscale back |
| Gaussian noise | σ = 0.02, 0.05, 0.10 |
| Colour jitter | brightness / contrast / saturation ±20% |
| Centre crop | 80% then resize back |

Implemented once in [`detector/transforms.py`](detector/transforms.py) and
used identically for training augmentation and evaluation.

---

## Results

All held-out. The organiser composition is the reporting number.

| Benchmark | Final Score | AUC(clean) | AUC(robust) |
|---|---|---|---|
| **Organiser composition** — WildFake pixel-diffusion (ADM, DALL·E, DDIM, DDPM, Imagen, VQDM) vs COCO, resolution-matched | **0.933** | 0.955 | 0.911 |
| DRAGON — 8 unseen latent-diffusion generators (Flux, SDXL-Turbo, SD3, Kolors, …) | 0.996 | 0.997 | 0.995 |
| SID_Set full-synthetic — in-domain | 0.997 | 0.999 | 0.996 |

**Per-generator ROC-AUC on the organiser set** (all six are pixel-space
diffusion, none in training):

| | ADM | DDPM | DDIM | Imagen | DALL·E | VQDM |
|---|---|---|---|---|---|---|
| clean | 0.89 | 0.93 | 0.95 | 0.97 | 0.99 | 1.00 |
| mean-robust | 0.85 | 0.86 | 0.89 | 0.93 | 0.96 | 0.97 |
| *ViT only, no CLIP branch* | *0.82* | *0.90* | *0.92* | *0.95* | *0.99* | *0.99* |

The frozen CLIP branch lifts the hardest family (ADM) from **0.82 → 0.89** —
above the published ceiling of ~0.82 for methods that do not train on ADM.

### Robustness — the 15-condition matrix

Operating threshold 0.215 (calibrated on withheld generators). The Final
Score is threshold-free; FPR/FNR are shown for completeness.

| Condition | ROC-AUC | FPR | FNR |
|---|---|---|---|
| clean | 0.955 | 0.04 | 0.23 |
| jpeg q90 / q70 / q50 / q30 | 0.94 / 0.92 / 0.89 / 0.85 | 0.04 / 0.05 / 0.06 / 0.07 | 0.27 / 0.34 / 0.42 / 0.48 |
| blur σ0.5 / σ1 / σ2 | 0.96 / 0.94 / 0.87 | 0.03 / 0.07 / 0.44 | 0.25 / 0.21 / 0.08 |
| resize 0.5× / 0.25× | 0.95 / 0.89 | 0.10 / 0.36 | 0.14 / 0.10 |
| noise σ0.02 / σ0.05 / σ0.10 | 0.91 / 0.89 / 0.86 | 0.07 / 0.03 / 0.03 | 0.35 / 0.57 / 0.68 |
| colour jitter ±20% | 0.95 | 0.05 | 0.21 |
| centre crop 80% | 0.93 | 0.13 | 0.18 |

**Weakest condition: sensor noise** (σ0.10, AUC 0.86) — additive noise most
directly overwrites the high-frequency evidence detectors lean on. Regenerate
this table for any checkpoint with `bash finalize.sh <ckpt> <tag>`.

### Error analysis

![highest-confidence false positives](docs/errors_FP.png)
*Authentic COCO photos the detector was surest were AI — all high-contrast,
low-noise studio shots whose statistics resemble the "too clean" look of a
generator.*

![highest-confidence false negatives](docs/errors_FN.png)
*AI images (all ADM/DDPM — 2021-era pixel diffusion) the detector was surest
were real. ADM's iterative denoising leaves almost no spectral fingerprint,
and the content is plausible ImageNet fauna/objects.*

Full note, per-condition FPR/FNR, and the file lists: [`docs/ERROR_ANALYSIS.md`](docs/ERROR_ANALYSIS.md).

---

## How it works

```
Community Forensics ViT-S/16 @224   21,666,049 params   (frozen after warm-start)  → base logit
  + frozen OpenAI CLIP ViT-B/16 branch   → LayerNorm → Linear(768→256) → near-zero-init residual Δ
                                logit = base + Δ
```

- **Backbone** — `OwensLab/commfor-model-224` (timm
  `vit_small_patch16_224.augreg_in21k_ft_in1k`). A purpose-built AI-image
  detector; an independent 2026 benchmark ranks it #1 of 23 open detectors.
- **Semantic branch** — `openai/clip-vit-base-patch16` vision tower,
  **frozen**. We measured (and a parallel team independently measured) that a
  *fine-tuned* backbone loses ~0.20 AUC seen→unseen while a *frozen* large ViT
  loses only ~0.09 — freezing is what makes it generalise across generator
  families. Semantic features don't depend on the artifact type, so they
  bridge the GAN↔diffusion gap that low-level artifact features cannot.
- **Near-zero-init residual fusion** — the branch head starts ≈0, so the
  model begins numerically identical to the proven ViT and can only *add* a
  correction. Deliberately **not** a per-branch auxiliary loss: that pressure
  makes a shallow branch memorise training-generator spectra and *invert* on
  unseen ones (a parallel team's frequency branch scored 0.457 — worse than
  chance — that way).
- **Trainable: ~200K** (the fusion head only). **Inference: ~108M** (22M ViT
  + 86M frozen CLIP-B) — 18× under the 2B limit. Runs on a laptop CPU at
  ~1 s/image.

The CLIP branch was trained as a fusion head on *precomputed* frozen features
(the CLIP forward is too slow to run live in the training loop on the Intel
Arc iGPU this was built on), then trained with a feature-jitter augmentation
that simulates the CLIP-embedding shift measured under degradation. That
training path and the corpus-build scripts are archived with the maintainer;
the shipped model is `runs/iter7/best.pt`.

### What we tried that did **not** work (measured, not guessed)

- A **SAFE-style DWT frequency branch** and a **hand-crafted NPR radial-
  spectrum branch** — neither transferred to pixel-space diffusion, which by
  construction has no sharp spectral peaks. Both dropped.
- **CLIP ViT-L/14** — ~6 s/step on the iGPU, hung repeatedly. CLIP-B's
  +0.014 Final Score is banked; L/14 would run on an NVIDIA box.

---

## Trade-offs (the deck asks for this explicitly)

- **Robustness vs clean accuracy** — heavy augmentation cost a little clean
  training accuracy; clean AUC held and robust AUC rose. Worth it.
- **Generalisation vs specialisation** — we hold six generator families
  *fully* out of training. A detector tuned on them scores higher on this
  benchmark and breaks on the next generator. We optimised for the break.
- **Complexity vs feasibility** — ~108M params, runs on a laptop CPU, one
  checkpoint, one `uvicorn` command. The deck: *"a 2-branch ensemble may win
  1% but cost you the demo — ship what runs."*
- **One threshold vs the generator spread** — pixel-diffusion fakes score
  systematically lower than latent-diffusion fakes, so no single cutoff is
  optimal for both (FNR on the organiser set is ~23% at the DRAGON-calibrated
  threshold vs ~2% on DRAGON itself). The scored metric is threshold-free;
  `detector/calibrate.py` lets an operator refit per deployment.

There is **no silver bullet** — modern flow/DiT generators (SD3, Flux,
Firefly) are the frontier where every open detector still collapses, and they
are named honestly in [Limitations](#limitations).

---

## Rules compliance (deck slide 14)

| Rule | This submission |
|---|---|
| Model < 2B parameters | **107,680,386** (18× under) |
| Public pretrained backbones only | `OwensLab/commfor-model-224`, `openai/clip-vit-base-patch16` — both public |
| Custom code MIT/Apache | Apache-2.0 (this repo) |
| Public/licensed data only, no test-label training | SID_Set · DRAGON · Community-Forensics-Small. `detector/data.py:assert_not_eval_only` **hard-fails** if an `eval_only_*` path reaches the training or calibration code |
| Augmentation scripts included | [`detector/transforms.py`](detector/transforms.py) |
| No "directly replicate an existing model" | novel fusion of two public backbones + a feature-jitter calibration step; negative results documented |
| Winning teams open-source everything | pipeline, hyperparameters (`config.yaml`), eval code, weights — all here / on HF |
| Submission: repo + run script + Devpost + demo video | this repo, `run_iteration.sh`, Devpost, YouTube |

**Scope** (slide 17): image-level binary detection only — no video/audio, no
production deployment, no localisation. Consistent with this project.

---

## Try it (2 minutes)

**1 — set up the environment** (Python 3.11 or 3.12):

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate
pip install -r requirements-cuda.txt          # NVIDIA / CPU
# or, on the Intel Arc box it was built on:  pip install -r requirements-xpu.txt
```

**2 — predict a folder of images** (the required deliverable format):

```bash
python pipeline.py predict \
  --input demo_images \
  --output predictions.json \
  --checkpoint runs/iter7/best.pt \
  --device auto
```

`predictions.json` → `[{"image_path": "...", "pred": 0|1}]`.
`predictions.scores.csv` → the calibrated `probability_ai` alongside.

> `runs/iter7/best.pt` is ~88 MB and **not in git** (large binary). Download
> it from the HuggingFace model repo linked at the top, or run
> `python hf_upload/upload.py` to publish your own copy. The frozen CLIP
> weights are rebuilt from the pinned public checkpoint on load, so the file
> stays small.

**3 — the interactive demo:**

```bash
pip install fastapi uvicorn python-multipart
DETECTOR_CHECKPOINT=runs/iter7/best.pt uvicorn webapp.server:app --port 8000
# open http://localhost:8000
```

Three views: **single image** (verdict + confidence band), **robustness
grid** (that image re-scored under all 15 conditions), **batch** (drop a
folder → sorted table + CSV, up to 200 images).

**4 — a public URL for a judging session** (free, no account, temporary):

```powershell
powershell -ExecutionPolicy Bypass -File serve_demo.ps1
```

Prints a `https://<...>.trycloudflare.com` link; **Ctrl+C** stops it.

**Verify the smoke test** (synthetic data, no dataset needed):

```bash
python pipeline.py smoke --device auto
```

---

## Reproduce the training

```bash
export TTJ_DATA=/path/to/materialised/data          # see below
export PYTHON=python  DEVICE=auto

bash run_iteration.sh iter4      # train the ViT baseline → calibrate → evaluate
```

`run_iteration.sh` reads [`config.yaml`](config.yaml), trains
[`detector/`](detector/), calibrates the threshold on **withheld generators**
(never the test set — 5-way split: `train / val / genval / calval / holdout`),
and evaluates every held-out set. The CLIP fusion head (iteration 7, the
shipped model) is trained on top with scripts archived alongside the
maintainer's notes.

**Data** — all public: `nebula/SID_Set` (drop the `tampered` class — localised
editing, out of scope), `lesc-unifi/dragon` (17 generators in training, 8
held out), `OwensLab/CommunityForensics-Small` (latent-diffusion + GAN +
LAION/ImageNet/CelebA/COCO reals), all perceptual-hash deduplicated against
the eval set. `python pipeline.py materialize-sid-set --help` and
`scripts/build_matched_eval.py` build the pieces.

---

## Repository layout

```
detector/            model · data · transforms · training · evaluation · calibration
pipeline.py          ingest · train · evaluate · predict · smoke · materialize-sid-set
config.yaml           training knobs
run_iteration.sh      one loop: train → calibrate on held-out generators → evaluate
run_fast.sh           lean variant (organiser + DRAGON only)
finalize.sh           full 15-condition eval + FP/FN montages for a finished checkpoint
scripts/              eval-only tooling (WildFake stream-eval, shortcut probe, matched-set builder)
webapp/               local FastAPI demo — single · robustness · batch
space/                Hugging Face Docker Space packaging (space/deploy.sh)
serve_demo.ps1        one command → a public Cloudflare-tunnel URL
hf_upload/            push the checkpoint to a HF model repo
docs/                 error-analysis note + FP/FN montages
requirements-*.txt    cuda (portable), xpu (the dev box)
```

Env vars the scripts honour: `PYTHON`, `TTJ_DATA`, `DEVICE`, `HF_HOME`,
`SYCL_CACHE_DIR`, `DETECTOR_CHECKPOINT`, `DETECTOR_DEVICE`.

---

## Threshold calibration — the 5-way split

| split | role |
|---|---|
| `train` | fit weights |
| `val` | watch for training failure only — **never** an operating-point input |
| `genval` | one withheld generator — fits the threshold value |
| `calval` | a *different* withheld generator — picks the rule (`minmax_fpfn`) |
| `holdout` | the reported metric — never an input to any decision |

```bash
python -m detector.calibrate runs/<name>/best.pt \
  --genval <dir> --calval <dir> --rule minmax_fpfn --apply
```

Calibrating on held-out generators instead of `val` took recall on unseen
fakes from ~25% to ~98% with **no change to AUC** — it fixes the operating
point, not the model.

---

## Limitations

- **Sensor noise** (σ0.10) is the weakest condition (AUC 0.86).
- **ADM** (2021 pixel-diffusion) is the hardest family; ~0.82 is the
  published ceiling for methods that don't train on it, we reach 0.89.
- **Modern flow / DiT generators** (SD3, Flux, Firefly) — the frontier where
  every open detector still collapses; not in this evaluation.
- **Localised edits** (a real photo with one AI-generated region) are out of
  scope — whole-image classification only.
- One fixed threshold cannot be optimal across pixel-space and latent
  diffusion at once — the score is threshold-free and the recipe is shipped.
- This is a hackathon prototype, not a production moderation system.

---

## License

Apache-2.0. Backbones are used under their own public licenses.
