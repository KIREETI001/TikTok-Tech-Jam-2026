# Experiment log

Tracks every training/evaluation attempt on this project, in order, with the
config that produced it, the numbers, and what we concluded. All checkpoints
and per-run CSVs live under `runs/<name>/` (not committed -- see
`.gitignore`); this file is the durable record of what each run showed.

Metric columns: **clean** = accuracy on unperturbed images. **robust mean** =
mean accuracy across the 15-condition evaluation matrix (JPEG/blur/noise/
resize/crop, see `detector/transforms.py`'s `CONDITION_SPECS`). **val** =
held-out split from the same ingest as training. **test** = `Data/test`,
never used in training or validation.

## 0. Environment fix (prerequisite, not a training run)

Training was silently running on **CPU**, not GPU. Root cause: the README's
own setup instructions pinned Python 3.14, and PyTorch has no CUDA-enabled
wheels for that version yet (CPU-only wheels install silently instead).
Fixed by pinning Python 3.12 and installing `torch==2.13.0+cu126`. Verified
`torch.cuda.is_available() == True` on the RTX 3050 before any run below.

Also benchmarked `num_workers`: with the original (pre-augmentation) light
transform, `0` was fastest on this machine -- Windows uses `spawn` for
`DataLoader` workers, so each worker cold-imports `torch`/`timm` from
scratch, and that fixed cost outweighed the parallel-decode benefit for
small, cheap-to-decode images. Re-checked after adding augmentation (heavier
per-image transform cost): `0` and `4` came out roughly tied (128.7s vs
125.0s on a 2,400-image benchmark), `8` was worse (205.0s). `num_workers: 0`
kept as the setting throughout.

## 1. Smoke test -- GPU regression guard

`pipeline.py smoke`: 50 synthetic real/fake images, no `../Data/` dependency.
Confirms the pipeline actually trains on `cuda` (not silently on `cpu`) and
that loss decreases. Added as a standing command specifically so the CPU
fallback bug above can't recur silently. Not a real-accuracy signal --
synthetic data, 2 epochs.

## 2. Baseline full run -- no augmentation

**Config**: `data_source: local`, 5 epochs, batch 16, `train_augment_probability: 0` (didn't exist yet), full `../Data/train` (99,080 images, 79,264 train / 19,816 val).

| | val | `Data/test` (19,584 imgs) |
|---|---|---|
| clean accuracy | 0.9659 | 0.9627 |
| clean F1 | 0.9657 | 0.9622 |
| robust mean accuracy | 0.8331 | 0.8323 |
| worst condition (accuracy) | `blur_sigma2` 60.7% | `blur_sigma2` 60.1% |
| worst condition (F1) | `noise_sigma0.10` 0.624 | `noise_sigma0.10` 0.615 |

**Finding**: excellent, well-generalizing fit on the PS5 distribution (val
and test track within 0.4pt everywhere -- no overfitting to one split). But
a sharp, specific robustness gap: accuracy holds up fine under JPEG
compression and color jitter, but collapses under heavy blur (worst case
`blur_sigma2`: 73% false-positive rate -- real images flipped to "fake") and
heavy noise (`noise_sigma0.10`/`0.05`: ~49-50% false-negative rate -- AI
images missed). Traced to `build_train_transform()` having no blur/noise/JPEG
augmentation at all -- the model never saw these corruptions during training.

## 3. Augmented run -- training-time robustness augmentation

**Config**: same as #2 plus `train_augment_probability: 0.7` -- one randomly
severity-sampled JPEG re-encode / blur / noise / resize-roundtrip corruption
per training sample (`detector/transforms.py`'s `_RandomRobustnessAugment`),
applied before the resize/crop step to match `build_eval_transform`'s
operation order (so a given nominal severity means the same thing at train
and eval time).

| | val | `Data/test` |
|---|---|---|
| clean accuracy | 0.9542 (-1.2pt) | 0.9511 (-1.2pt) |
| clean F1 | 0.9548 | 0.9512 |
| robust mean accuracy | **0.9137** (+8.1pt) | **0.9133** (+8.1pt) |
| `blur_sigma2` accuracy | 60.7% -> **83.8%** | 60.1% -> **83.4%** |
| `blur_sigma2` FPR | 73.0% -> **17.2%** | 73.2% -> **17.9%** |
| `noise_sigma0.10` F1 | 0.624 -> **0.865** | 0.615 -> **0.863** |

**Finding**: decisive fix for the #2 gap, on the first attempt -- no
refinement loop needed. Small, acceptable clean-accuracy cost for a large,
targeted robustness gain, and val/test still track within 0.4pt of each
other throughout, confirming genuine generalization rather than overfitting
to one split's specific images. Committed as the accepted training config
(`ef67ebe`).

## 4. Cross-dataset diagnostic -- SID_Set (never trained on)

Built `pipeline.py materialize-sid-set` to fetch SID_Set (Hugging Face
`saberzl/SID_Set`) shards and save them locally, so the existing
15-condition `evaluate()` could run against a genuinely different dataset
without a parallel eval implementation. Sample: 1 `validation` shard, 882
images (314 real / 568 fake) -- SID_Set's own held-out split, never touched
by any training run below.

**#3's checkpoint (`runs/augmented/best.pt`) against SID_Set:**

| | value |
|---|---|
| clean accuracy | **0.6950** (vs 0.9511 on `Data/test`) |
| mean-transformed accuracy | 0.6924 |
| FNR range across all 15 conditions | 33-55% |
| FPR range across all 15 conditions | 4-18% |

**Finding**: a real cross-dataset generalization gap, and a specific one.
Accuracy barely moves across all 15 conditions (even clean sits at 69.5%) --
this is not a robustness-to-degradation problem like #2 was, it's the
model's baseline discrimination failing on SID_Set's content/generators. The
failure is asymmetric: low FPR (real images still recognized correctly),
high FNR (SID_Set's AI images frequently missed) -- ROC-AUC of 0.774 shows
the model retains *some* signal, just not enough to separate classes at any
single threshold.

### 4a. Ruled out: threshold miscalibration

Swept thresholds 0.05-0.95 against the #3 checkpoint's raw SID_Set scores
(`/tmp/threshold_sweep.py`, reusing `detector.model`/`detector.transforms`
directly -- one pass, no retraining). Best threshold (0.25) only recovers
accuracy to 0.7029 -- about 0.8pt above the default 0.5's 0.6950. Every
threshold in range sits in the same narrow 67-70% band. **Not a calibration
problem** -- the score distributions for real vs. fake SID_Set images
overlap too much for a moved decision boundary to fix.

### 4b. Ruled out: fine-tuning narrowed generalization

Built a checkpoint from the *zero-shot* Community Forensics weights (no PS5
fine-tuning at all: `create_detector(pretrained=True)` -> `save_checkpoint`)
and ran it against the same SID_Set sample:

| | zero-shot (no fine-tune) | #3 (PS5 fine-tuned + augmented) |
|---|---|---|
| clean accuracy | 0.6009 | **0.6950** |
| mean-transformed accuracy | 0.5402 | **0.6924** |
| FNR range | 55-95% | 33-55% |

**Finding**: the zero-shot checkpoint is *worse* on SID_Set, not better --
it has a strong "predict real" bias (FPR near 0%, but missing 55-95% of
actual AI images). PS5 fine-tuning roughly halved that miss rate even though
it never saw a single SID_Set image. This rules out "fine-tuning
over-narrowed the model" -- the fix is not to fine-tune less aggressively,
it's that neither version has seen enough generator diversity.

**Conclusion from 4/4a/4b**: the gap is a genuine training-data-diversity
limitation. Next lever: add generator diversity to training, not more PS5
augmentation or threshold tuning.

## 5. Mixed-source run -- PS5 + SID_Set diversity (in progress)

**Config**: new `data_source: mixed` (`detector/data_sources/mixed.py`),
`ConcatDataset`-combines the full local PS5 pool with SID_Set's own `train`
shards (`sid_set_train_shards: 10`, ~8,400 images -- SID_Set's `validation`
shards are deliberately left untouched, since they're what section 4 already
used for held-out cross-dataset eval; pulling from them here would leak eval
data into training). Same `train_augment_probability: 0.7` as #3 applies to
both PS5 and SID_Set images.

Verified on small subsets first (50-image local smoke set + 1 SID_Set
shard: `train=715` exactly matches `40 local + 675 SID_Set`, `val=179`
matches `10 + 169` -- concatenation is correct) before launching at full
scale.

Full run: 99,080 local images + 10 SID_Set train shards -> 86,015 train /
21,505 val combined. Took 6h51m (longer than the ~3.75h estimate -- shard
fetch/decode for 10 shards added more overhead than expected). Final:
`train 0.1984 | val 0.1286 | F1 0.9504` (combined val, so not directly
comparable to #3's local-only val F1).

**Result on `Data/test`** (PS5, unseen during training):

| | #3 (PS5-only) | #5 (mixed) |
|---|---|---|
| clean accuracy | 0.9511 | **0.9540** (+0.3pt) |
| robust mean accuracy | 0.9133 | 0.9118 (-0.15pt) |

**No PS5-performance cost from mixing in SID_Set** -- both numbers are
within noise of #3, one even nominally better.

**Result on the same fixed SID_Set sample from section 4:**

| | zero-shot | #3 (PS5-only) | #5 (mixed) |
|---|---|---|---|
| clean accuracy | 0.6009 | 0.6950 | **0.8367** (+14.2pt vs #3) |
| mean-transformed accuracy | 0.5402 | 0.6924 | **0.8366** (+14.4pt vs #3) |
| FNR range across 15 conditions | 55-95% | 33-55% | **12-17%** |
| FPR range across 15 conditions | 0-3.5% | 4-18% | 15-28% |
| ROC-AUC (clean) | ~0.73 | 0.774 | **0.922** |

**Finding**: decisive confirmation of the section-4 diagnosis. Mixing in
~8,400 SID_Set training images (~9.5% of the combined pool) closed most of
the cross-dataset gap -- clean accuracy on SID_Set jumped 14pt with zero
measurable cost on the original PS5 distribution. FNR (missed AI images,
the dominant failure mode in section 4) dropped from 33-55% to 12-17%; FPR
rose somewhat (4-18% -> 15-28%) as the decision boundary rebalanced away
from its earlier strong "predict real" bias, but the net effect is a much
more balanced, better-performing model on both datasets simultaneously.
Accepted as the new default (`config.yaml`: `data_source: mixed`).

## 6. Research-driven iteration -- diversity + augmentation recipe fix (in progress)

Goal set for this iteration: 98% accuracy, <2% FN/FP (organizer brief's
bar). Researched what the literature says actually drives cross-generator
generalization before changing anything:

- **Community Forensics** (the base model here) itself finds generalization
  scales with *number of distinct generators* seen, more than images per
  generator -- argues for more SID_Set shard diversity, not just more of
  the same shards.
- **Wang et al., CVPR 2020** ("CNN-generated images are surprisingly easy to
  spot...for now"): blur + JPEG applied *independently* (so they can
  co-occur on one image) is what produced generalization in their
  experiments -- not picking one corruption at a time, which is what
  section 3's `_RandomRobustnessAugment` did.
- **Ojha et al., CVPR 2023** ("Towards Universal Fake Image Detectors"):
  full fine-tuning tends to learn narrow, generator-specific artifacts and
  treat "real" as a sink class -- plausible explanation for section 5's
  FPR-vs-FNR asymmetry. Noted as a follow-up lever (lighter fine-tune
  depth), not changed this round.

**Changes made:**
- `detector/transforms.py`'s `_RandomRobustnessAugment` rewritten: each of
  resize-roundtrip/blur/noise/JPEG now fires independently (per-op
  probability derived from `train_augment_probability` so config semantics
  are unchanged), instead of picking exactly one -- matching the Wang et
  al. recipe. Verified: 22.6% of samples get 2+ stacked corruptions over
  2000 trials at probability 0.7; edge cases (0.0/1.0) correct.
- `config.yaml`: `sid_set_train_shards` 10 -> 30 (~25k images instead of
  ~8.4k).
- **Deliberately not added**: CIFAKE (native 32x32 resolution -- 7x
  upsampling to our 224x224 input risks teaching "blurry/blocky = fake" as
  a spurious cue, not a real one) and WildFake (organizer's eval-only
  benchmark per the brief; not used for training, full stop).

Full run: 99,080 local images + 30 SID_Set train shards -> 99,519 train /
24,881 val combined. Took 9h13m (longer than section 5's 6h51m -- more
shards to fetch, more images, and heavier per-image cost from the
independent-stacking augmentation). Final: `train 0.2419 | val 0.1541 | F1
0.9414` (combined val).

**Result on `Data/test`:**

| | #5 (10 shards, one-of-four aug) | #6 (30 shards, stacking aug) |
|---|---|---|
| clean accuracy | 0.9540 | 0.9515 (-0.25pt) |
| robust mean accuracy | 0.9118 | 0.9087 (-0.31pt) |

Small dip, consistent with harder/more frequent augmentation making the
PS5-only fit slightly less tight -- expected trade-off, not a concern on
its own.

**Result on the same fixed SID_Set sample (sections 4-6):**

| | #3 (PS5-only) | #5 (10 shards) | #6 (30 shards + stacking aug) |
|---|---|---|---|
| clean accuracy | 0.6950 | 0.8367 | **0.8639** (+2.7pt vs #5) |
| mean-transformed accuracy | 0.6924 | 0.8366 | **0.8645** (+2.8pt vs #5) |
| FNR range across 15 conditions | 33-55% | 12-17% | **8.4-15.5%** |
| FPR range across 15 conditions | 4-18% | 15-28% | **11.5-24.8%** |
| ROC-AUC (clean) | 0.774 | 0.922 | **0.942** |

**Finding**: real, consistent improvement -- better on almost every one of
the 15 conditions individually, not just the average, and FNR in
particular tightened notably (missed-AI-image rate down another third).
Both changes (more shard diversity, independent-stacking augmentation)
plausibly contributed; this run didn't isolate which helped more, so that
attribution is unconfirmed.

**Against the 98%/<2% FN&FP target**: not reached, and still a large gap
-- clean accuracy on SID_Set is 86.4% (vs. 98% target) and FPR/FNR sit in
the 8-25% range (vs. <2% target) despite two full iterations of real,
measurable improvement. Two iterations bought roughly 3-15pt of
cross-dataset accuracy each; closing the remaining ~12-14pt gap to 98% and
getting error rates an order of magnitude lower likely needs a materially
different lever, not another round of the same kind of tuning -- see the
follow-ups below. Adopted as the new default (`config.yaml` already
reflects this run's settings) since it's a strict improvement over #5 on
the metric that matters (cross-dataset), for a negligible PS5 cost.

## 7. Third data source -- DRAGON (in progress)

Researched candidate third datasets for more generator diversity beyond
SID_Set. Chosen: **DRAGON** (`lesc-unifi/dragon`, CC-BY-SA-4.0) -- real
1024x1024 resolution (unlike CIFAKE's native 32x32, avoided in section 6
for exactly this reason), generator-balanced across 25 modern diffusion
models including several distilled/fast variants (LCM, SDXL Turbo/
Lightning, Hyper-SD) not represented in SID_Set or PS5. Fake-only (no
real class in this dataset) -- adds exclusively to the fake side of the
combined pool.

**Integration**: `detector/data_sources/dragon.py`, reusing the same
HF-auto-converted-parquet mechanism as `sid_set_stream.py` (no new
dependency). Extended `mixed.py` from a 2-way to 3-way concatenation.
`config.yaml`: `dragon_config: Regular`, `dragon_shards: 5` (~2,000
images -- each DRAGON image is ~1.9MB vs SID_Set's ~550KB, so kept modest
to bound fetch time/memory).

**Bug found and fixed during integration**: `huggingface_hub`'s default
request timeout (10s) is tuned for small API calls, not a ~760MB
single-shard parquet GET -- reproduced the resulting read-timeout/retry
loop directly, confirmed raising it to 300s fixes it. Fixed by setting
`huggingface_hub.constants.HF_HUB_DOWNLOAD_TIMEOUT` at `dragon.py`'s
import time (attribute assignment, not just the env var, so it applies
regardless of import order).

**Verified before running at scale**: isolated dragon-only ingest+train (1
shard -> exactly 320/80 train/val, matching 400*0.8/0.2), then the full
3-way combination on small subsets (1035/259 exactly matches
40+675+320 / 10+169+80 local+SID_Set+DRAGON). `pipeline.py smoke` still
passes with identical numbers.

**First attempt crashed** after 2h40m: a transient network/DNS disruption
(`getaddrinfo failed`, then a stale-socket error) while fetching a DRAGON
shard outlasted `huggingface_hub`'s own retry budget and took the whole
run down partway through epoch 1. Added `detector/data_sources/
_network.py`'s `retry_network_call` (8 attempts, 60s apart) around every
HF Hub network call in both `dragon.py` and `sid_set_stream.py` -- the
latter is equally exposed over an equally long run, it just hadn't hit
this yet. Unit-tested directly (recovers after N transient failures,
correctly re-raises after exhausting attempts) before relaunching.
Honest residual risk: the observed incident had ~160 minutes elapse with
only one retry logged, consistent with a single call hanging far longer
than any attempt budget allows (e.g. a pathological DNS hang after a
network adapter reset) -- no outer retry can interrupt that. If it
recurs, the more robust fix is decoupling fetching from training
entirely (materialize SID_Set/DRAGON to local disk first, then train
with `data_source: local`), not done here given the scope/time tradeoff.

**Status: abandoned.** A third attempt hit the identical DNS failure on a
different shard, confirming it's persistent (not transient) in this
environment -- no amount of retrying fixes it. Killed the run rather than
keep burning GPU time on a network problem outside this codebase's
control. `dragon_shards` defaults to `0` in `config.yaml` now (see
`mixed.py`'s skip path, section 9), so training proceeds on the proven
PS5+SID_Set pool without it. Re-enable if/when the network issue clears.

## 8. Reading the organizer's own brief -- hybrid architecture

The actual TikTok TechJam PS5 brief slides became available partway
through this work (not just the earlier text summary). Two findings
changed the plan:

- **Rules risk**: the brief explicitly says "do not directly replicate
  existing models or approaches," and warns "don't just fine-tune a
  classifier... is it a real artifact, or a dataset shortcut?" Community
  Forensics ViT-S is an *already-complete* published detector, not a
  generic backbone like the ResNet/ViT/CLIP/DINOv2 the rules name --
  fine-tuning it further, as sections 2-7 did, sits closer to
  "replication" than the brief seems to want.
- **The brief's own stated key insight** (slide 8): "best detectors
  combine high-level semantics + low-level frequency patches." This is
  exactly the piece worth reviving from `upstream/main`'s abandoned
  4-branch fusion model (Phase 3) -- not the whole thing (camera-noise and
  CLIP branches need their own pretraining/dependencies, too much for the
  time left), just the frequency branch.
- **Brief's scoring formula** (slide 12): Final Score = 0.5 x AUC_clean +
  0.5 x AUC_robust, ROC-AUC-based, not accuracy-based. Built
  `pipeline.py report` to compute this directly from existing
  summary.json/metrics.csv. Run against the current best checkpoint
  (section 6, `runs/mixed_v2`): **Final Score 0.9798 on PS5 `Data/test`,
  0.9417 on SID_Set** -- both above the brief's own worked example
  (clean 0.99 AUC, unseen-generator 0.80 AUC), since ROC-AUC is far more
  forgiving of threshold miscalibration than the accuracy numbers tracked
  up to this point.

Also ran a compression-history audit (DDA/NeurIPS 2025 insight: JPEG
history can become a spurious real-vs-fake signal): PS5's real/fake
compression is nearly identical (0.904 vs 0.905 bytes/px, no shortcut
risk), but **SID_Set's fake images are ~28% more compressed than its real
images** (0.231 vs 0.319 bytes/px) -- a real, evidenced risk, plausibly
explaining part of the elevated FPR (22-25%) seen on SID_Set throughout
sections 5-7. Documented as a known limitation rather than fixed (would
need a data-recompression-normalization pass, too big a lift right now).

> **Superseded by section 9's shortcut probe.** The "plausibly explaining
> part of the elevated FPR" claim above does not survive measurement. The
> compression gap is real, but it lives in the *file*, not in the pixels:
> SID_Set scores 95.06% on the metadata probe and only 54.93% (+/- 4.34, an
> interval that nearly touches chance) on the pixel probe that sees the
> actual 224px crop, and `recompressed_bpp` is not among its top pixel
> features. Whatever drives the SID_Set FPR, a compression shortcut
> reaching the model is not a supported explanation, and the
> recompression-normalization item it motivated is not worth the lift.

### Hybrid model: two real bugs found and fixed via benchmark testing

Implemented `detector/model.py`'s `HybridDetector`: Community Forensics
ViT (frozen, per Ojha et al.'s finding that heavier fine-tuning of the
semantic backbone learns narrower, more generator-specific shortcuts) +
a ported `FrequencyBranch` (from `upstream/main`, log-magnitude FFT
through a shallow CNN + radial power profile) + a fusion mechanism.
Tested on a 3-epoch/2400-image benchmark before any full run (exactly
the point of that step):

1. **First fusion design failed to learn at all**: concatenating the
   ViT's logit with the frequency branch's 256-dim embedding into a
   fresh `Linear` layer left F1 stuck at 0.667 (loss converging to
   ln(2)) -- a fresh Linear layer treats all 257 dims symmetrically,
   diluting the one genuinely strong signal into 256 dims of untrained
   noise. Raising the learning rate 100x didn't fix it (confirmed the
   fusion design was the problem, not the LR). **Fix**: zero-initialized
   residual -- the frequency branch now *adds* a correction to the ViT
   logit, with its final layer zero-initialized so the hybrid starts
   mathematically identical to the ViT-only model (verified
   byte-identical output, multiple modes, before any training).

2. **Still unstable after the fix** (loss ~3x the ViT-only baseline, F1
   0). Traced step-by-step: the frequency branch's contribution stayed
   genuinely negligible (~1e-4) throughout -- the real cause was that
   `create_hybrid_detector` built its ViT half from the *raw* pinned
   Community Forensics checkpoint, never fine-tuned on this task.
   Confirmed directly: raw checkpoint predicts "real" for 100% of a
   balanced 480-image set. **Fix**: added `vit_checkpoint` so the hybrid
   actually builds on an already-fine-tuned checkpoint (e.g.
   `runs/mixed_v2/best.pt`), matching its own design rationale. Retested:
   F1 0.9426->0.9446, loss stable throughout -- both failures were
   methodology gaps (which base checkpoint), not a flaw in the fusion
   mechanism itself.

**Quick SID_Set check** (hybrid, ViT half = `runs/mixed_v2/best.pt`,
frequency branch fine-tuned only on 1,920 PS5-only benchmark images, no
new diversity): clean 0.8628 / mean-transformed 0.8656 -- essentially
identical to `mixed_v2` alone (0.8639/0.8645), within noise. Expected:
this was a mechanism-stability check, not a real test of whether the
frequency branch helps generalization, since it saw zero new
cross-dataset diversity. **Not yet run**: a full-scale hybrid training
pass on the complete PS5+SID_Set pool -- that's the actual test of
whether this closes any more of the cross-dataset gap, and needs the
same multi-hour commitment as sections 5-7's full runs.

`config.yaml` stays on `model_type: vit` (the proven section-6 checkpoint)
pending that full-scale test.

## 9. Environment pivot -- new laptop, Intel Arc iGPU, no local dataset

All of sections 0-8 ran on a machine with an RTX 3050 and the full
`../Data/` tree (99,080 PS5 images + a held-out `Data/test`). Work moved to
a different laptop (2026-08-29) with neither:

- **GPU: Intel Arc integrated** (Meteor Lake, device 0x7D55, ~16 GB shared,
  128 EUs). No CUDA. Added an **XPU backend** path throughout
  (`detector/model.py` `resolve_device` now CUDA -> XPU -> CPU via
  `xpu_available()`; `accelerator_pin_memory()`; XPU seeding; `--device xpu`;
  smoke's GPU-fallback guard generalized to CUDA *or* XPU).
  - **`torch==2.13.0+xpu` deadlocks** on the first oneDNN GEMM with this
    laptop's GPU driver (`32.0.101.6314`, Nov 2024). Not slow -- a true
    hang (process idle, 0 CPU). **`torch==2.6.0+xpu` works** (GEMM ~2 s).
    Training runs from a `torch 2.6.0+xpu` venv; the pinned 2.13 stack needs
    a driver update (latest 32.0.101.8991, Aug 2026) to be usable here.
  - Measured ViT-S throughput on the iGPU: **~100-120 img/s** train (bs 8-16),
    ~90 img/s eval. bf16 `autocast` added to train + validate.
- **No local dataset.** `../Data/` does not exist. Public data only (user
  decision): SID_Set streamed from HF and **materialized to local disk**
  (`pipeline.py materialize-sid-set`), then trained via `data_source: local`
  -- decoupling the fetch from the loop as section 7 recommended. The real
  PS5 held-out `Data/test` is gone, so every number below is on SID_Set or
  another public set, never the organizers' distribution.

### Objective realignment (applies to every iteration below)

The brief scores `0.5*AUC_clean + 0.5*AUC_robust` (ROC-AUC), and the goal
set for this push is **FPR and FNR both <= 3%**. The training loop was
selecting the best checkpoint by **F1 at a fixed 0.5 threshold** -- wrong on
both counts. Changed:

- checkpoint selection is now by validation **ROC-AUC** (`select_metric`,
  default `roc_auc`; `balanced_max_fpfn` and `f1` also available);
- after each epoch the loop sweeps thresholds and stores the one minimizing
  `max(FPR, FNR)` on validation into the checkpoint; `evaluate`/`predict`
  read it from metadata (`threshold: auto` in config);
- `loss_pos_weight` exposed for the BCE positive (fake) class -- SID_Set is
  ~1:2 real:fake; lever for pulling FPR and FNR together;
- `evaluate` summary now reports worst-case FPR/FNR across all 15 conditions
  and whether every condition is within the 3% target.

### 9a. Iteration 1 -- baseline on this environment

**Config**: `data_source: local`, 13,504 SID_Set images (shards 0-15,
materialized at native ~1024px), 85/15 split -> 11,475 train / 2,024 val,
ViT, 6 epochs, bs 16, lr 1e-5, aug 0.7, `pos_weight` 1.0, `select_metric`
roc_auc, `num_workers` 0.

**Internal held-out val** (same shards, unseen images), by epoch:

| epoch | val AUC | bal thr | bal FPR | bal FNR |
|---|---|---|---|---|
| 1 | 0.885 | 0.67 | 0.199 | 0.206 |
| 3 | 0.924 | 0.62 | 0.154 | 0.151 |
| 5 | 0.933 | 0.65 | 0.143 | 0.146 |
| 6 | **0.934** | 0.68 | 0.146 | 0.144 |

**Findings**:
- AUC plateaued by epoch 4-5 (0.929 -> 0.934); train loss still falling
  (0.31 -> 0.28) while val loss flattened -> mild overfit starting, more
  *data* will help more than more *epochs* on this size.
- At threshold 0.5: FPR 0.23 / FNR 0.09 -- the model leans "predict fake",
  exactly the 1:2 real:fake class ratio. The calibrated threshold (~0.66)
  rebalances it to ~14.5% each, which is the model's real discrimination
  limit here.
- **~14 min/epoch, ~14 img/s effective** -- single-process augmentation
  (heavy: PIL JPEG re-encode + blur + numpy noise + 2 resizes, on 1024px
  images) leaves the iGPU ~85% idle. Fixed for iteration 2: materialize at
  448px + `num_workers` 6 + `persistent_workers`; also refactored
  `evaluate()` to decode each image once instead of 15x.
- Bug found and fixed: numpy scalars in checkpoint metadata broke
  `torch.load(weights_only=True)` (PyTorch 2.6 default) -- `save_checkpoint`
  now coerces metadata to built-in types.

**Gap to target**: 14.5% balanced error on the *easy* in-distribution number
vs a 3% goal. Reaching 3% needs ~0.99 AUC -- a large jump. Levers for the
next iterations: (2) 3x more data + class-balanced sampling + cosine LR;
(3) the semantic+frequency hybrid; (4) more generator diversity (DRAGON /
GenImage); (5) lighter fine-tune depth per Ojha et al.

### 9b. Iteration 2 -- drop tampered images + recipe/throughput fixes

**The tampered discovery**: SID_Set's `fake` class is ~50% `full_synthetic`
(fully AI-generated) and ~50% `tampered` (a real photo with a small AI-edited
region -- localized-manipulation detection, a materially different and harder
task). Collapsing both to "fake" is what pinned iteration 1 at 0.93 AUC. The
filenames carry the distinction, so the materialized tree was split into
real-vs-full_synthetic (`*_fs`) and a tampered-only diagnostic set.

**Config**: `sid_train448_fs` -- SID_Set shards 0-15, 448px, tampered
dropped -> 8,991 images (4,469 real / 4,522 full_synthetic), 88/12 split ->
7,912 train / 1,079 val. ViT, 10 epochs, bs 16, **lr 2e-5 cosine + 10%
warmup**, aug 0.7, **class-balanced sampling**, `num_workers` 6 +
`persistent_workers`, `select_metric` roc_auc, `threshold: auto`.

**Throughput**: 448px + 6 workers -> **~3 min/epoch** (was ~14).

**Internal held-out split** (robustness matrix): Final Score **0.9993**
(clean AUC 0.9997 / robust 0.9989). Clean FPR 1.1% / FNR 1.1%. **13/15
conditions within the 3% FPR&FNR bar.**

**Held-out `sid_val448_fs`** (SID_Set's own validation split, unseen images):

| | value |
|---|---|
| Final Score (0.5 AUC_clean + 0.5 AUC_robust) | **0.9988** |
| clean: FPR / FNR | **0.50% / 1.21%** |
| conditions within 3% FPR&FNR | **11/15** |
| worst FNR | 6.8% @ noise_sigma0.10 |
| worst FPR | 1.5% @ resize_0.25x |
| worst per-condition AUC | 0.9964 @ noise_sigma0.10 |

All 4 misses are FNR (missed AI images) under aggressive corruption --
`jpeg_q30` 4.7%, `noise_sigma` 0.02/0.05/0.10 at 3.2/3.6/6.8%. Per-condition
AUC never drops below 0.996, so discrimination is intact; the calibrated
(clean-val) threshold is just slightly too high once heavy noise pulls the
fake scores down.

**vs iteration 1** (held-out): Final Score 0.9364 -> 0.9988; clean error
~13% -> <1.3%; 0/15 -> 11/15 within 3%.

**Honest caveat**: `sid_val_fs` is SID_Set's *own* validation split -- same
generators as training, unseen images only. This is in-distribution
generalization, not cross-generator. The competition bar (unseen
generators) still needs a genuinely different source -- DRAGON / GenImage --
which iteration 3 adds as the real scoreboard.

**Tampered diagnostic** (`sid_val448_tampered`, 1192 real / 1179 tampered):
clean FNR **96.9%**, clean AUC 0.657. The fs-only model has essentially zero
ability to flag locally-edited images -- expected, they are mostly authentic
pixels. Confirms tampered detection is a separate task. Decision: the PS5
brief is "AI-Generated Image Detection" and its robust-scoring axis is image
*degradations* (which the 15-condition matrix covers), not localized
manipulation -- so the main model stays fs-focused and the tampered gap is a
documented limitation. Revisit with a frequency branch / small tampered
fraction if time allows.

### 9c. Iteration 2's cross-generator gap -- DRAGON (25 unseen generators)

Materialized DRAGON (`lesc-unifi/dragon`, works fine from this machine --
the DNS failures in section 7 were the *other* machine): 4,100 fakes, 25
modern diffusion generators x 164, none overlapping SID_Set. Paired with
1,192 SID_Set-val reals. iter2's checkpoint (threshold 0.73):

| | value |
|---|---|
| clean ROC-AUC | **0.9914** |
| clean FPR / FNR | 0.42% / **18.8%** |
| Final Score | 0.9879 |

Score distributions (clean): SID_Set fakes mean p=0.99, DRAGON fakes mean
p=0.84 -- the model still ranks DRAGON fakes above real (hence high AUC),
but their scores sit lower, so the SID-calibrated threshold misses many.

**Threshold sweep across SID_fs + DRAGON**: even at threshold 0.05, DRAGON
FNR only falls to 7.1% (SID FPR then 2.9%). So it is *not* purely
calibration -- there is a ~7% hard tail of DRAGON fakes the model is
confident are real.

**Per-generator FNR** (threshold 0.73): worst = Flash_SD3 59%, IF (DeepFloyd,
pixel-space diffusion) 55%, LCM_SDXL 54%, JuggernautXL 45%, Flash_SDXL 34%,
Flash_SD 31%. Best = Hyper_SD 0%, PixArt_alpha/sigma 2%, Mobius 2%,
SD_Cascade 3%, Kolors 4%. Pattern: distilled/fast samplers (Flash_*, LCM),
pixel-space diffusion (IF), and heavily photoreal-tuned SDXL (JuggernautXL)
have artifact signatures unlike SID_Set's generators.

### 9d. Iteration 3 -- generator diversity + noise robustness (in progress)

**Data**: `iter3_train` = SID_Set shards 0-29 full_synthetic (8,409 real /
8,415 fake) + DRAGON **17 of 25 generators** (2,788 fakes). Held out for the
cross-generator scoreboard (`dragon_holdout_eval`, 1,192 real / 1,312 fake):
Flux_1, IF, JuggernautXL, Kolors, PixArt_Sigma, SDXL_Turbo, SD_3, SD_Cascade.

**Changes**: `train_augment_probability` 0.7 -> 0.8; `_RandomRobustnessAugment`
fires the noise op at 1.5x the other ops' probability and up to sigma 0.13
(matrix max 0.10); training's `_validate` now calibrates the stored
operating threshold on clean **+ noise-corrupted** val scores, not clean
only. 12 epochs, otherwise iter2's recipe.

### 9e. Cross-checking against ziyangchua02/model_training

A parallel pipeline by a teammate (`github.com/ziyangchua02/model_training`,
ResNet50 + branch-fusion rather than Community Forensics) keeps its own
experiment log over 8 trained-and-scored changes. Several of its measured
results bear directly on this project's plan, and two contradict it.

**Their headline**: Final Score 0.9340 (AUC_clean 0.9448, AUC_robust 0.9232)
on their own unseen-generator holdout; noise is their weakest condition
(0.8886 mean AUC, 0.8582 at sigma 0.10) -- the same weakness measured here in
9b/9d.

**Where their gains came from**, over 8 experiments:

| Change | dAUC |
|---|---|
| Multi-source content-matched corpus, native-resolution crops | **+0.132** |
| Frozen CLIP ViT-L/14 branch | **+0.069** |
| Training generator diversity, 9 -> 17 families | **+0.042** |
| CutMix | +0.006 |
| Held-out-generator selection + calibration | 0.000 AUC (but +9.9 deployed) |
| Dropping the frequency and camera branches | -0.004 |

Two of the three real wins are data, not architecture.

**Contradicts our plan -- the frequency branch is below chance.** Their
per-branch auxiliary-head AUCs, on unseen generators against a pooled real
set:

| Branch | AUC seen val | AUC unseen | drop |
|---|---|---|---|
| frozen CLIP | 0.983 | **0.897** | -0.086 |
| fine-tuned ResNet50 (spatial) | 0.987 | 0.784 | -0.203 |
| camera / noise residual | 0.847 | 0.528 | -0.319 |
| **frequency (FFT + radial profile)** | 0.722 | **0.457** | -0.265 |

0.457 is *inverted*, not uninformative -- and 0.722 on seen data shows it
learned something, just the wrong thing: the spectral signatures of the
training generators. That is the same log-magnitude-FFT-plus-radial-profile
design as `detector/model.py`'s `FrequencyBranch`, which section 8 built and
the "Not yet attempted" list still carried as the top pending lever.
**Removed from the plan on their evidence** rather than spending a
multi-hour run to rediscover it. Note this also qualifies the brief's own
"combine semantics + low-level frequency" hint: as a shallow branch trained
with an auxiliary loss, it does not survive a generator change.

**Confirms our diagnosis -- calibration on seen data is the FNR bug.**
Their experiment 1 lost 11.4 points of balanced accuracy to a threshold
fitted on the seen validation split; refitting it on a *withheld generator*
moved the cutoff 0.49 -> 0.10 and took recall on unseen fakes from 0.252 to
0.826. That is exactly section 9c's finding here (threshold 0.73 fitted on
SID_Set giving 18.8% FNR on DRAGON), independently reproduced on a different
architecture. Their split protocol separates it properly:

| Split | Used for |
|---|---|
| train | fitting weights |
| val | watching for training failures only |
| genval | withheld generator -- checkpoint selection *and* threshold |
| calval | a *different* withheld generator -- picks the threshold *rule* |
| holdout | the reported metric; never an input to any decision |

Our iteration 3 still selects checkpoints on a validation split containing
the training generators. Their experiment 2 shows why that is not safe: seen
validation accuracy rose monotonically to 97.1% while AUC on an unseen
generator peaked at epoch 4 and then fell for four more epochs.

**A caution they record twice**: a change that moves genval but not the
holdout is unproven -- genval is one generator.

### 9f. The organisers' composition has a resolution shortcut

Their `match_resolution.py` reports that on the organisers' composition as
shipped, **a classifier using image size alone scores AUC 1.0000** --
WildFake's authentic images are 200x200 and its generated images 256x256 or
larger.

Verified independently on the set built here (COCO from the HF mirror,
WildFake fakes as fetched): real images span 69,120-284,928 pixels while
ADM/DDIM/DDPM/VQDM are *all exactly* 65,536 (256x256) -- every fake from
four of the six generators smaller than every real. Same shortcut, opposite
sign.

Any score on such a set is a mixture of "can it detect generation" and "can
it read a file header". `build_matched_eval.py` therefore writes a matched
copy: every image centre-cropped from **native pixels** to the smallest
common side (200), never resized -- resizing one class and not the other
swaps a size cue for a resampling cue. It prints size-only AUC before and
after so the confound is measured rather than assumed. The authentic half is
also taken from WildFake's own `Images/Real/coco.zip` (COCO 2017 train,
163,846 images) rather than an HF COCO mirror, so both halves come from the
distribution the organisers actually pair.

### 9g-results. Iteration 3 -- generator diversity + noise-aware calibration

**Config**: `iter3_train` = SID_Set shards 0-29 full_synthetic (8,409 real /
8,415 fake) + 17 of DRAGON's 25 generators (2,788 fakes). Held out for the
cross-generator scoreboard: Flux_1, IF, JuggernautXL, Kolors, PixArt_Sigma,
SDXL_Turbo, SD_3, SD_Cascade. `train_augment_probability` 0.8; noise op
fires at 1.5x the other corruptions and up to sigma 0.13; `_validate` now
calibrates the stored threshold on clean + noise-corrupted val (the value
landed at 0.10, vs iter2's 0.73). 12 epochs, 448px, otherwise iter2's recipe.

| held-out set | Final Score | clean FPR / FNR | conditions <=3% | vs iter2 |
|---|---|---|---|---|
| internal split | 0.9987 | 2.5% / 0.6% | 7/15 | -- |
| `sid_val448_fs` | 0.9989 | 2.4% / 0.17% | 9/15 | 0.9988 -> ~flat |
| **DRAGON, 8 unseen generators** | **0.9974** | **2.4% / 1.4%** | **9/15** | clean AUC 0.9914 -> **0.9981**, FNR **18.8% -> 1.4%** |
| tampered (diagnostic) | 0.648 | 2.4% / 90% | 0/15 | ~chance, as expected |

**Findings**:
- **The FNR-under-degradation problem from sections 9b/9d is solved on
  latent-diffusion generators.** Clean FNR is 0.17-1.4% on every real
  diffusion set. Training on 17 DRAGON generators transferred cleanly to the
  8 held out -- clean AUC on genuinely-unseen generators went 0.9914 ->
  0.9981.
- **The remaining weakness flipped from FNR to FPR** (3-7% under heavy
  JPEG/noise/resize, worst 6.6% at noise sigma 0.10). This is the safer
  failure mode and is threshold- and augmentation-tunable rather than a
  discrimination limit -- per-condition AUC never drops below 0.994.
- The threshold moved to 0.10, which by section 9c's own logic is "the size
  of the score shift between seen and unseen generators". The noise-aware
  calibration overshot slightly (it targets noisy FNR, which was never the
  problem here); a clean-weighted recalibration would trade a little of the
  DRAGON headroom back for lower clean FPR.
- Tampered stays at ~chance (clean FNR 90%). Correct given the deck's
  image-level-binary-only scope (section 9g); documented as out of scope.
- **Caveat**: DRAGON's held-out generators are all latent diffusion -- the
  same architecture family as training. The genuine cross-*architecture*
  test is WildFake's pixel-space generators (ADM/DDIM/DDPM/Imagen), where
  iter2 scored 0.742 clean AUC.

**Iteration 3 on the resolution-matched organiser composition (WildFake vs
COCO, 1200/1200, size-only AUC 0.500):**

| | iter2 | iter3 |
|---|---|---|
| Final Score | 0.717 | **0.804** |
| AUC_clean / AUC_robust | 0.742 / 0.692 | **0.826 / 0.781** |

Per-generator clean AUC, every one improved:

| Generator | iter2 | iter3 | family |
|---|---|---|---|
| ADM | 0.537 | **0.645** | pixel-space diffusion |
| DDPM | 0.632 | **0.752** | pixel-space diffusion |
| Imagen | 0.663 | **0.804** | pixel-space cascaded |
| DDIM | 0.781 | **0.859** | pixel-space diffusion |
| DALLE | 0.923 | 0.946 | discrete / dVAE |
| VQDM | 0.916 | 0.952 | vector-quantized |

**+0.087 Final Score from generator diversity + calibration alone**, and it
transferred to *pixel-space diffusion* -- an architecture family with zero
representation in training (SID_Set and DRAGON are both latent diffusion).
Direct evidence for the deck's "the biggest lever is what you train on".

Two things confirm the section-9g research diagnosis:
- **ADM is still 0.645; SAFE reaches 0.82 on ADM.** The ~0.18 gap is what the
  DWT-band input + crop-not-resize is expected to close -- it is not a
  missing-data problem alone (SAFE never trained on ADM).
- **The stored threshold (0.10) is wrong for WildFake**: FPR 11.8% / FNR 40%
  at the operating point despite clean AUC 0.826. The noise-aware
  calibration was tuned for an FNR-under-noise problem WildFake does not
  have. A held-out-generator recalibration (merge-plan Phase 2, no
  retraining) fixes the operating point.

**Iteration 4 sequence, settled by this result**: (1) recalibrate iter3's
threshold on held-out generators -- cheap, no training; (2) iteration 4 =
`CommunityForensics-Small` spine + DWT/SAFE branch + crop-from-native A/B in
one run, targeting the ADM residual and adding the semantic + frequency-patch
fusion the deck asks for.

### 9h. Phase A -- held-out-generator threshold calibration (no training)

`detector/calibrate.py`: fit the operating point on a withheld generator
family, pick the *rule* on a second withheld family. Split DRAGON's 8
held-out generators into `genval` (Flux_1, JuggernautXL, Kolors, SD_Cascade,
PixArt_Sigma) and `calval` (IF, SDXL_Turbo, SD_3).

Rule comparison (threshold fit on genval, scored on calval):

| rule | thr | calval balacc | calval max(FPR,FNR) |
|---|---|---|---|
| balacc_argmax | 0.50 | 0.969 | 0.051 |
| balacc_plateau | 0.45 | 0.971 | 0.047 |
| equal_error | 0.42 | 0.972 | **0.045** |
| minmax_fpfn | 0.42 | 0.972 | **0.045** |

Applied `minmax_fpfn` (threshold **0.42**, was 0.10) to `runs/iter3/best.pt`.
At 0.42, on generators never seen in training:

| | FPR | FNR |
|---|---|---|
| genval (5 unseen latent-diffusion generators) | 1.2% | 1.2% |
| calval (3 unseen, incl. DeepFloyd IF) | 1.2% | 4.5% |
| sid_val_fs | 1.2% | 0.9% |

**iter3 is within ~1-4% FPR/FNR on unseen latent-diffusion generators** at a
threshold that was calibrated without touching any of them. The Final Score
is unchanged (AUC is threshold-free). WildFake's pixel-space families stay
discrimination-limited (AUC 0.826) -- no threshold rescues them; that is
Phase C's job.

The noise-aware calibration from section 9g-results *over-corrected*: it
targets noisy FNR, which iteration 3 does not have. Held-out-generator
calibration is the better-evidenced mechanism and is now the standing rule
(`detector/calibrate.py`, applied after every training run).

### 9i. Phase A -- shortcut probes (no training)

`detector/shortcut_probe.py`: gradient-boosted-tree probes on class-balanced
data (chance 50%), reading only things that are not generation artifacts.
`metadata` = original geometry + JPEG size (does not reach the model -- our
sets are re-encoded). `pixel` = channel stats / saturation / edge density /
re-encode size of the 224 crop. `dct_hf` = mean log-magnitude of the DCT
outside a central radius (the DDA "JPEG-history frequency bias").

| pool | metadata | pixel | dct_hf |
|---|---|---|---|
| `iter4_train` (SID fs + 17 DRAGON + CF-Small) | 97.9% | **60.4%** | **52.5%** |
| `sid_val_fs` alone | 97.4% | **75.0%** | 59.1% |
| `dragon_unseen` | 97.4% | 69.2% | 54.9% |
| `organiser_matched` | **51.6%** | 66.6% | 61.7% |

**Findings**:
- **The multi-source corpus removes the shortcut.** SID_Set alone has a
  strong pixel-statistics cue (75% -- generated images are genuinely
  smoother / less textured than photos), and iteration 2/3 partly learned
  it. Mixing in Community-Forensics-Small drops `iter4_train`'s pixel probe
  to 60.4% and its `dct_hf` to chance (52.5%). Independently reproduces the
  teammate's result (9-family 70.8% -> 17-family 64.6%).
- **The resolution-matching worked**: `organiser_matched` metadata probe is
  51.6% (was AUC 1.0 on the as-shipped set, section 9f).
- **Gate result for the DDA frequency-alignment augmentation (Phase D)**:
  `iter4_train`'s `dct_hf` is already at chance, so the DDA aug is
  **downgraded to optional**. The residual `pixel`/`dct_hf` on the eval sets
  (66% / 62%) is partly legitimate signal -- older generators really are
  smoother -- not only a dataset accident. Phase D focuses on
  supervised-contrastive loss + windowed augmentation instead.

### 9j. Phase B -- iteration 4: Community-Forensics-Small (in progress)

`iter4_train` = `iter3_train` (SID full_synthetic + 17 DRAGON generators,
19.6k) + Community-Forensics-Small (17,914 real / 23,939 fake after a
perceptual-hash dedup that dropped 38 CF reals matching the WildFake COCO
eval set). **61,465 images total.** Each epoch draws a fresh balanced 24k
from the pool (`samples_per_epoch`), 10 epochs, `num_workers` 8, otherwise
iteration 3's recipe. `model_type: vit` -- a data-only change; the DWT branch
is iteration 5. DDIM/Imagen deliberately not added so the organiser score
keeps genuinely-unseen pixel-space families.

**Process note**: `run_iteration.sh` was killed mid-eval by an over-eager
zombie-process cleanup; `dragon_holdout` and `tampered` were re-run directly
against `runs/iter3/best.pt`. No effect on results.

**Results (DONE 2026-08-29).** 10 epochs, ~7 min/epoch, best internal
roc_auc 0.9993 (ep10). Threshold auto-calibrated on withheld DRAGON
generators -> `minmax_fpfn` 0.385 (genval FPR 2.0/FNR 2.0, calval FPR 2.0/
FNR 3.1 -- all 5 rules land 2.8-3.9% worst-case).

| eval set | Final Score | clean AUC | robust AUC | clean FPR | clean FNR | conds <=3% |
|---|---|---|---|---|---|---|
| internal held-out | -- | -- | -- | acc 0.988 | robust acc 0.979 | -- |
| sid_val_fs | ~0.997 | 0.9985 | 0.9963 | 0.021 | 0.009 | -- |
| dragon_unseen (8 gens) | **0.9959** | 0.9972 | 0.9945 | 0.021 | 0.024 | 6/15 |
| **organiser (WildFake+COCO)** | **0.9126** | 0.9286 | 0.8965 | 0.059 | 0.248 | 0/15 |

**iter3 -> iter4 on organiser: Final Score 0.804 -> 0.9126 (+0.109).** The
Community-Forensics-Small data (LatDiff + GAN + diverse LAION/ImageNet/
CelebA/COCO reals) transferred strongly to pixel-space diffusion that was
zero in training. Beats the teammate's 0.871 and my own 0.88-0.91 "realistic
ceiling" estimate.

**The remaining problem is threshold transfer, not the model.** AUC is
threshold-free and strong (0.9286 clean). But at the operating threshold
0.385 -- calibrated on DRAGON latent-diffusion, which the model scores
confidently -- clean FNR is 24.75% on the organiser mix: pixel-diffusion
fakes (ADM/DDPM) get lower AI-scores, so a threshold tuned on latent
diffusion misses them. dragon_unseen at the same threshold is fine (2.4%
FNR). No single fixed threshold hits 3%/3% across both generator families
at once; worst robustness conditions are noise_sigma0.10 (FNR 60.7%),
blur_sigma2 (FPR 39.4%), jpeg_q30 (AUC 0.844).

**Levers for iter5 (Phase C):** (1) DWT/SAFE frequency branch -- directly
targets the pixel-diffusion AUC gap (SAFE gets ADM 0.82 vs our 0.65-ish).
(2) add a WildFake-like held-out calibration slice, or adopt a
generator-family-agnostic threshold rule. (3) SAFE RandomMask/Rotation +
crop-from-native to harden the frequency signal against jpeg/blur/noise.

### 9k. iter5 (wavelet) -- run, then aborted mid-train for the sprint

iter5 = hybrid, `branch_kind: wavelet`, ViT frozen from iter4 (only the
0.3M-param DWT branch + fusion head train), SAFE aug + crop-from-native,
`lr 2e-5`. Trained 5/10 epochs: train loss plateaued at ~0.42 (heavy SAFE
aug), internal val AUC flat at 0.9993 = iter4's frozen fit. With ~24h left
to submission, killed it rather than spend ~1h of iGPU on its eval phase --
the wavelet branch on a frozen backbone was clearly not moving the internal
number, and iter6 folds a wavelet branch in anyway. `runs/iter5/best.pt`
kept (epoch 3).

### 9l. iter6 (sprint) -- frozen CLIP ViT-L/14 branch

The teammate's single biggest architecture lever (+0.069 Final Score;
`experiments.md` teammate cross-check, EXPERIMENTS.md exp 3). New
`SemanticBranch` in `detector/model.py`: OpenAI CLIP ViT-L/14 vision tower
(303M params) via timm, **frozen**, always `eval()`; re-normalises
ImageNet-stat inputs to CLIP stats; LayerNorm + Linear(1024->256) ->
zero-init residual head into the ViT logit. `HybridDetector` generalised to
an `nn.ModuleList` of branches, each with its own zero-init head, summed
into the logit -- `branch_kind` is now a comma list (`clip`, `clip,wavelet`).
Frozen encoder weights are excluded from the checkpoint (stays ~88 MB) and
rebuilt from pinned pretrained on load. **No per-branch aux loss** -- the
mechanism the teammate showed makes a shallow branch memorise
training-generator spectra and invert on unseen ones; ours trains on the
final BCE only.

CLIP-L pre-flighted clean on XPU (SYCL kernels cache fine). Sprint config:
warm-start `runs/iter4/best.pt`, `branch_kind: clip`, ViT frozen, bs 40,
`samples_per_epoch` 10000, 4 epochs, `lr 2e-4` (tiny head, few steps),
SAFE aug + crop-from-native. `bash run_fast.sh iter6` (lean: calibrate +
organiser + dragon_unseen only). Expected organiser Final Score
0.9126 -> ~0.94-0.97.

### 9g. Literature review + briefing deck (RESEARCH_SYNTHESIS.md)

The briefing PDF is image-only; rendered to PNGs and read slide-by-slide.
It cites **SAFE (KDD 2025, arXiv 2408.06741)** and **DDA (NeurIPS 2025,
arXiv 2505.14359)** by name on slide 10 and expects engagement. Full writeup
in `RESEARCH_SYNTHESIS.md`; artifact
<https://claude.ai/code/artifact/ccf51ecb-08c6-49d8-8877-3b18588ec7f9>.

**Deck points that change assumptions:**
- Slide 8 says "high-level CLIP semantics + low-level frequency **patches**"
  -- local, not a global FFT. Confirms dropping `HybridDetector`'s global
  log-FFT branch (measured 0.457 AUC unseen by the teammate) and pointing at
  patch-level statistics instead.
- Slide 13: "a 2-branch ensemble may win 1% but cost you the demo. Ship what
  runs." -- a hard caution on the CLIP branch.
- Slide 16: robustness table + FP/FN error-analysis note explicitly earns
  Technical Execution (35%) + Innovation & Insight (20%) = 55% of the rubric.
- Slide 17: image-level binary only, no localisation -- retroactively
  justifies dropping SID_Set's `tampered` class (section 9b).

**SAFE** (repo read): truncated ResNet (conv1+layer1+layer2), **1.44M
params**, input is the **DWT bior1.3 J=1 diagonal high-frequency sub-band**
(not RGB). `RandomCrop(256)` train / `CenterCrop(256)` test, **never
resize** (ablation: resize always worse). Aug: `RandomRotation(180)`,
`ColorJitter(0.5)`, `RandomMask` (16px patches, up to 75%, p=0.5). Trained
ProGAN-only, reaches **ADM 82.1 ACC** -- the generator this project scores
0.537 AUC on (section 9d). *So the pixel-space gap is preprocessing +
input-representation, not only missing data.*

**DDA** (repo read): names the JPEG-history frequency shortcut (this
project's sections 8 and 9f). Its **frequency-alignment transform** is
usable standalone -- DCT-blend a random real's coefficients into each fake,
`r~U(0,0.25)`, inverse DCT (code in `dda/Training/data/custom_transforms.py`).
The highest-value idea not previously in any plan here.

**Community Forensics** (this project's backbone origin, CVPR 2025):
`OwensLab/CommunityForensics-Small` is 186 streamable parquet shards, ~30k
images, 50/50 real/fake, LatDiff+GAN+Real across LAION/COCO/ImageNet/CelebA.
An independent 2025 benchmark (arXiv 2602.07814) ranked Community Forensics
#1 of 23 detectors out-of-the-box. This is a drop-in multi-generator +
multi-real-source training pool.

**Revised plan**: ~21 h, **4 training runs** (was 5-6 in `MERGE_PLAN.md`).
CF-Small collapses the "pixel-space data" and "multi-source reals" phases;
the DWT branch replaces the abandoned global-frequency work. New parallel
additions: DCT frequency-energy probe, DWT/SAFE branch, DDA freq-align aug,
crop-from-native A/B tested early, test-time augmentation, error-analysis
note written now. Realistic ceiling ~0.88-0.92 Final Score on the matched
WildFake composition; 3% FPR/FNR is reached by no published method on unseen
generators.

## 10. Auditing our own data: the corpus is 32x32

Ported ziyangchua02's shortcut probe (`scripts/shortcut_probe.py`):
gradient-boosted trees on properties carrying no generation evidence --
file metadata, and pixel statistics of the exact 224px crop the model is
fed. Whatever those recover is label information available without
detecting anything. Chance is 50%.

| Corpus | Metadata | Pixels, native crop | Pixels, our pipeline |
|---|---|---|---|
| PS5 train | 52.87% | 72.60% | 74.47% |
| PS5 test | 52.33% | 74.93% | 76.93% |
| SID_Set | 95.06% | 54.93% | 57.80% |

Three findings. **The two corpora are confounded in opposite ways** --
PS5 is clean at file level but ~75% separable from pixel statistics
alone; SID_Set is the reverse. **Our resize makes leakage worse**,
consistently, on all three corpora (+1.9 to +2.9 points), with
`edge_std`'s importance roughly doubling: normalising every image to one
scale turns edge density into a cleanly comparable smoothness measure,
whereas at native resolution it is confounded with the image's own
resolution. And **~75% of PS5's label is recoverable from ten summary
statistics** with no notion of generation, which caps how much of our
~95% accuracy can be attributed to artifact detection.

### The finding that reframed everything

Checking image dimensions before running the crop-policy experiment:
**every PS5 image is 32x32.** `Data/train` (99,080) and `Data/test`
(19,584) are CIFAKE -- CIFAR-10 reals against Stable Diffusion 1.4
fakes, one generator, upscaled 7x to reach the model's 224 input. Every
evaluation set is 1024px from many generators.

This killed two planned changes outright, both verified empirically
rather than argued:

- **Crop-not-resize**: native vs resize retains detail **1.00x** on our
  data. There is no high-frequency band left at 32px to preserve. (On a
  synthetic 1024px checkerboard the same code shows 15x, so the
  mechanism is real -- our data simply has nothing for it to act on.)
- **Multi-crop inference**: after upscaling 32->224 the image *is* 224,
  so all five crops are identical; measured crop-to-crop spread
  **0.0000**. Retains value only on the 1024px eval sets.

It also retires the calibration work: the brief scores
`0.5*AUC_clean + 0.5*AUC_robust`, and ROC-AUC is threshold-free, so
threshold sweeps, `pos_weight` and calibration cannot move the score at
all. Only separability can.

## 11. WildFake/COCO benchmark -- the first honest number

Built `scripts/evaluate_wildfake.py` to score against the organisers'
own composition (1,000 COCO2017 reals vs 1,000 WildFake fakes, 200 each
across five generators), streamed from ModelScope over HTTP range
requests rather than downloading 1.2TB. Eval-only, kept out of
`detector/data_sources/` so no training path can reach it.

**`runs/mixed_v2` (the section-6 checkpoint):**

| | Ours | Another team's reported figure | Gap |
|---|---|---|---|
| Final Score | **0.8131** | 0.8705 | -0.057 |
| AUC_clean | 0.8495 | 0.8734 | -0.024 |
| AUC_robust | **0.7767** | 0.8676 | **-0.091** |

| Generator | AUC clean | AUC robust |
|---|---|---|
| adm | 0.623 | 0.522 |
| ddpm | 0.735 | 0.643 |
| ddim | 0.911 | 0.843 |
| vqdm | 0.979 | 0.866 |
| dalle | 0.985 | 0.972 |

Two things this says that the PS5 and SID_Set numbers could not.

**Our variance is the problem, not our average.** We beat the reference
on three generators, often substantially, and lose badly on two. They
sit in a tight 0.81-0.90 band everywhere. A detector scoring 0.99 on one
generator and 0.52 on another has learned some families and not others --
Community Forensics' finding that generalization tracks generator
diversity, restated as our own result.

**The gap is robustness, not detection.** Clean is within 0.024 of
theirs; robust is 0.091 behind. Their score drops 0.006 from clean to
robust, ours drops 0.073 -- and the brief weights robustness at 50%. The
likely cause is the same 32x32 finding: we apply corruption augmentation
(p=0.7) to images that were ~80% 32px thumbnails, so the model learned
robustness to "degrade a 32px image, then upscale 7x", which shares
little with "degrade a 1024px image, then downscale". We trained
robustness at the wrong scale.

**And it recalibrates the earlier numbers.** PS5 test at 0.9798 is
CIFAKE's own held-out split -- same resolution, same single generator --
so it measured in-distribution fit rather than detection ability. 0.8131
is the first figure measured the way the competition will measure us.

> **Correction (post-merge).** ziyangchua02's `build_matched_eval.py`
> found the organisers' composition, as fetched, has a resolution
> shortcut: WildFake's generated images ship at fixed sizes
> (ADM/DDIM/DDPM/VQDM 256x256; DALLE larger) while COCO reals are
> photographic resolution, and a classifier using pixel count alone
> scores AUC **1.0000** on the raw set. Rebuilt locally as
> `scripts/build_matched_eval_local.py` (adapted to this device's own
> WildFake cache rather than the teammate's machine-local source
> folders) -- confirmed size-only AUC **1.0000 -> 0.5000** after centre-
> cropping both classes to the same native-pixel side (200px), never
> resizing. Re-scoring `mixed_v2` on the matched set:
>
> | | Raw (above) | Matched |
> |---|---|---|
> | AUC_clean | 0.8495 | **0.7442** |
> | AUC_robust (flat) | 0.7767 | **0.6573** |
> | Final (flat) | 0.8131 | **0.7007** |
> | Final (grouped) | 0.8192 | **0.7126** |
>
> A meaningful part of the 0.8131 headline was the shortcut leaking
> through -- notable since our pipeline resizes every image to a fixed
> canvas before the model sees it, so this was not a raw pixel-count
> read but a correlated cue surviving that resize (plausibly the
> different blur/degradation characteristics a 256px image vs a
> photographic one picks up after being resized to the same target).
> **0.7007/0.7126 is the true baseline iteration 5 needs to beat**, not
> 0.8131.

## 12. Rebalancing toward real resolution (`runs/highres_v1`, `v2`)

**Status: superseded, not completed.** `v1` (100 SID_Set shards) hit
`lru_cache(maxsize=32)` thrashing -- 5.5h at 0% GPU, 432 re-fetches for
100 shards, zero epochs finished -- and was killed. `v2` fixed this by
reverting to 30 shards and using `local_max_train_images` (new: caps the
32x32 CIFAKE source, class-balanced, applied to both train and
validation at the same ratio) to reach the same ~77% high-resolution mix
at a quarter of the data. `v2` trained cleanly for 3 epochs (val F1
0.8738 -> 0.8917 -> 0.9013, monotonically improving, no thrash) before
being stopped partway through epoch 4.

**Why stopped rather than finished**: access to a teammate's parallel
effort (`ziyangchua02/model_training`, now merged as a collaborator)
surfaced a checkpoint already at organiser Final Score **0.9126** --
built on Community-Forensics-Small (multi-generator, content-matched)
rather than CIFAKE+SID_Set, with SAFE-style crop-from-native
augmentation and a DWT frequency-patch branch already implemented but
switched off. `highres_v2`'s ceiling on this data mix could not
plausibly reach that with two epochs remaining, so continuing it was
pure sunk cost. `local_max_train_images` and the cache-thrash fix carry
forward regardless -- both are merged onto the shared branch.

See `EXECUTION_PLAN.md` (post-merge) for what runs next: iteration 5,
warm-started from the teammate's checkpoint, targeting the one measured
weakness that survives it -- pixel-space diffusion (ADM 0.623, DDPM
0.735) -- via the already-implemented crop-from-native + SAFE augment +
wavelet branch, plus a small amount of GenImage ADM/GLIDE training data.


## Not yet attempted

- **Full-scale hybrid training run** (section 8 built and benchmark-
  validated the mechanism; not yet run at scale): fine-tune
  `HybridDetector`'s frequency branch on the complete PS5+SID_Set mixed
  pool (`model_type: hybrid`, `vit_checkpoint: runs/mixed_v2/best.pt`,
  multiple epochs) to properly test whether it closes any more of the
  cross-dataset gap -- the section-8 quick check only saw 1,920 PS5-only
  images, no new diversity, so it couldn't test this. Needs the same
  multi-hour commitment as sections 5-7's full runs.
  **-> DROPPED (section 9e).** A teammate's pipeline measured this exact
  branch design at 0.457 AUC on unseen generators -- below chance, and
  inverted rather than uninformative. Not worth the run.
- **Frozen CLIP ViT-L/14 as a second branch** (NEW, top priority after
  section 9e): +0.069 AUC in the teammate's log, their single biggest
  architecture win, and the frozen branch alone (0.897) outscored their
  fused model (0.878). Ojha et al. (CVPR 2023) is the mechanism: frozen
  features cannot be bent toward the training generators, so whatever
  separates classes in that space has to be more general. Fits the
  parameter budget (304M frozen + 21.7M here, well under 2B).
- **Held-out-generator checkpoint selection and threshold calibration**
  (NEW, section 9e): move both off the seen-generator validation split onto
  a withheld generator, per the teammate's experiment 2 (+9.9 deployed
  balanced accuracy, threshold moved 0.49 -> 0.10). We already hold out 8
  DRAGON generators; they are currently only scored, not used for
  selection.
- **More real-image diversity** (NEW, section 9e): every real image in this
  project's training pool is SID_Set's. The teammate's largest single win
  (+0.132) was a multi-source content-matched corpus across 8 real
  collections, and they measured genuine camera photographs as the weakest
  real class (27% false-positive rate before the fix).
- **Lighter fine-tune depth** (Ojha et al.-motivated, still not tried):
  freeze the last transformer block too (train only the head + norm),
  trading some clean accuracy for potentially better cross-generator
  generalization.
- **`pos_weight` in the loss** to directly counteract the FPR/FNR
  asymmetry seen since section 4, rather than relying on data changes to
  fix it indirectly.
- **Post-hoc calibration** (temperature/Platt scaling) on the current
  checkpoint's combined validation scores -- near-zero-cost, no retrain
  needed, not yet tried on the section-6/7/8 checkpoints (only tried once,
  in section 4a, on the section-3 checkpoint).
- **Generator-balanced batch sampling** (`WeightedRandomSampler` instead of
  plain `ConcatDataset` proportional mixing) so every source/generator is
  seen evenly per epoch regardless of its raw image count.
- **SID_Set compression-history normalization** (section 8 finding): its
  fake images are ~28% more compressed than its real images, a plausible
  spurious shortcut per DDA (NeurIPS 2025) -- would need a recompression
  pass on one class to remove the confound, not yet attempted.
- ~~Third-dataset holdout~~ -- **done, section 11**: the WildFake/COCO
  benchmark (organiser composition, 5 unseen generators) is exactly this.
  Result: Final 0.8131 against a reference team's 0.8705, with the gap
  concentrated in two pixel-space generators (ADM, DDPM) and in robustness
  rather than clean detection -- see section 11 for the full breakdown.
- **Publish model weights** per the brief's stated rules for winning teams
  ("open-source... model weights") -- not yet done for any checkpoint.

## Architecture decision (for context, not a training run)

Chose Community Forensics ViT-S/224 (pretrained across thousands of
generators specifically for this task) over `origin/main`'s from-scratch
4-branch fusion model (ResNet50 + frequency + camera-noise + frozen CLIP,
cross-attention fusion). Reasons: the fusion model's own docstrings flag
itself as unvalidated ("wasn't validated in the time available"), its
`requirements.txt` is missing the `transformers` dependency its CLIP branch
needs, and it specifies 25 training epochs on a heavier architecture versus
Community Forensics' 5 light-fine-tune epochs -- a much larger compute
budget on a 4GB laptop GPU for an unproven design. Confirmed by ziyangchua02.

## 13. Iteration 5a -- no warm start, local corpus, crop_from_native + SAFE

**Blocker found before this could run**: `runs/iter4/best.pt` (the merge's
apparent starting point, organiser Final Score 0.9126 per the teammate's
own log) does not actually exist anywhere reachable. Despite a commit
message reading "Track model checkpoints in git", `.gitignore` excludes
`runs/` and `*.pt`/`*.pth` -- confirmed absent on this device after the
merge. `scratchpad/build_iter4.py` (their corpus builder) was likewise
never committed, same convention as this project's own scratchpad/.

**Response**: rebuilt the corpus locally (`scripts/build_iter5_corpus.py`)
from the teammate's *documented* recipe (experiments.md 9d/9g) rather than
their uncommitted script -- streams SID_Set shards 0-29 (raw label kept, so
label 2/tampered is dropped rather than folded into fake, matching
"full_synthetic only"), all 25 DRAGON generators split 17 train / 8 held
out (their exact held-out set: Flux_1, IF, JuggernautXL, Kolors,
PixArt_Sigma, SDXL_Turbo, SD_3, SD_Cascade), plus GenImage ADM + GLIDE
(2,500 each) for the pixel-space diffusion gap section 11 measured. Not
Community-Forensics-Small: its auto-converted HF export was found too
small/uncertain (~10.5k rows, not the ~30k reported, shard 0 100% one
architecture) to depend on without more reconnaissance time than an
overnight run allowed.

Verified against the teammate's own reported numbers before trusting it:
SID_Set train landed at real=8,409 / fake=8,415, an exact match to their
documented iter3_train counts. DRAGON holdout landed at fake=1,312, an
exact match to their documented dragon_holdout_eval count. Final corpus:
train real=8,409 fake=16,203 (24,612 total), dragon_holdout real=554
fake=1,312.

**Sequenced as vit-only first** (this section), hybrid+wavelet second
(iteration 5b) once this run's own checkpoint exists to warm-start from --
matches section 8's finding that a frequency/wavelet branch needs an
already-validated ViT base, not a fresh pretrained one.

**Config**: `data_source: local`, the corpus above, `crop_from_native:
true`, `safe_augment: true`, `select_metric: roc_auc`, `balance_classes:
true`, `lr_schedule: cosine`, 10 epochs, batch 16, lr 2e-5, num_workers 4
(dropped from the teammate's 8 -- their number was tuned for a streamed-
source constraint that does not apply to `data_source: local`, and this
project's own earlier num_workers benchmark on this machine found
diminishing returns past a few workers).

Trained cleanly: val ROC-AUC 0.9668 -> 0.9866 (epoch 7, best) -> 0.9864
(epoch 10), no divergence, no crash. ~17 min/epoch, ~3h total.

### Results -- and a second correction to section 11's headline

Before trusting any comparison, rebuilt the resolution-matched organiser
eval locally (`scripts/build_matched_eval_local.py`, adapting the
teammate's `build_matched_eval.py` to this device's own WildFake cache
instead of their machine's pre-fetched folders) and re-scored `mixed_v2`
on it for a controlled baseline -- confirming their finding independently:
source size-only AUC **1.0000**, matched size-only AUC **0.5000**. This
also *corrected* section 11's own headline: `mixed_v2`'s honest matched
Final Score is **0.7007/0.7126**, not the raw 0.8131/0.8192 -- a
resolution-shortcut-inflated number, even though our pipeline resizes
before the model sees anything (the shortcut survived as a correlated
degradation-signature cue, not a raw pixel-count read).

| Held-out set | Final Score | AUC clean | AUC robust |
|---|---|---|---|
| SID_Set (mixed full_synthetic + tampered, legacy sample) | 0.8603 | 0.8696 | 0.8509 |
| **DRAGON, 8 unseen generators** | **0.9798** | 0.9873 | 0.9724 |
| **Organiser, resolution-matched** | **0.8804** | 0.9137 | 0.8472 |

**Controlled per-generator comparison** -- both `mixed_v2` and iter5a
scored on the identical matched set, same 200-per-generator sample:

| Generator | mixed_v2 clean | iter5a clean | delta | mixed_v2 robust | iter5a robust | delta |
|---|---|---|---|---|---|---|
| ADM | 0.478 | **0.855** | **+0.377** | 0.422 | 0.778 | +0.356 |
| DDPM | 0.597 | **0.840** | **+0.243** | 0.522 | 0.767 | +0.245 |
| DDIM | 0.836 | 0.942 | +0.106 | 0.753 | 0.867 | +0.114 |
| DALLE | 0.860 | 0.948 | +0.088 | 0.794 | 0.903 | +0.109 |
| VQDM | 0.950 | 0.983 | +0.033 | 0.795 | 0.921 | +0.126 |

`mixed_v2`'s ADM clean AUC on the matched set is **below chance** (0.478)
-- once the resolution shortcut can no longer substitute for detection, it
had essentially none on ADM specifically. Every generator improved under
iter5a, and the two largest gains land exactly on the two pixel-space
diffusion generators this iteration targeted (crop_from_native + SAFE +
GenImage ADM/GLIDE) -- the cleanest confirmation this project has produced
that the diagnosed mechanism (section: fingerprints research, VAE-keyed
detection failing on non-VAE generators) was correct and the fix worked.

**Decision: iter5a supersedes `mixed_v2` as the best checkpoint.** Both
gate conditions from the post-merge plan are met with margin: organiser
Final Score up +0.18 (not merely non-regressed), and the specific
diagnosed weakness (ADM/DDPM) closed rather than just the average moving.
No regression on DRAGON (0.9798, and this is now a genuinely unseen-
generator test the corpus was built to support). SID_Set's 0.8603 is not
directly comparable to any earlier number on this same set with this same
composition, so treated as a new baseline rather than a regression signal.

Next: iteration 5b, `model_type: hybrid`, `branch_kind: wavelet`,
`vit_checkpoint: runs/iter5a/best.pt` -- this run's own checkpoint now
exists to warm-start from, closing the gap the "sequenced deliberately"
note above was written to avoid.

## 14. Iteration 5b: hybrid + wavelet branch -- does not transfer

Trained `model_type: hybrid`, `branch_kind: wavelet` for 5 epochs, ViT half
frozen (`HybridDetector.configure_finetuning`), warm-started from iter5a's
own validated checkpoint per the plan above. Training was clean throughout
(no crashes, no gradient-gating symptoms -- the near-zero-init fix from
the merge held). Parameters: 21,993,762 total, +327,713 over iter5a's
vit-only 21,666,049 (the DWT branch + fusion head).

Seen-validation AUC moved from iter5a's 0.9866 to 0.9872 -- a real but
marginal +0.0006, exactly what's expected when only a small added branch
trains against an already-converged frozen backbone.

The actual test is held-out transfer, scored with the identical
`finalize_local.sh` pipeline used for iter5a, same three sets:

| Held-out set | iter5a Final | iter5b Final | delta |
|---|---|---|---|
| SID_Set | 0.8603 | 0.8586 | -0.0017 |
| DRAGON, 8 unseen generators | 0.9798 | 0.9798 | ~0.0000 |
| Organiser, resolution-matched | 0.8804 | 0.8800 | -0.0004 |

**Controlled per-generator comparison**, both checkpoints scored on the
identical matched organiser set:

| Generator | iter5a clean | iter5b clean | iter5a robust | iter5b robust |
|---|---|---|---|---|
| ddpm | 0.840 | 0.841 | 0.767 | 0.767 |
| adm | 0.855 | 0.854 | 0.778 | 0.777 |
| ddim | 0.942 | 0.942 | 0.867 | 0.866 |
| dalle | 0.948 | 0.947 | 0.903 | 0.902 |
| vqdm | 0.983 | 0.983 | 0.921 | 0.921 |

Every generator is flat within +-0.001 -- including ADM and DDPM, the two
pixel-space diffusion generators the wavelet branch was hypothesized to
help most (no VAE fingerprint for the ViT branch alone to key on). DRAGON,
the genuinely-unseen-generator set, is flat to the fifth decimal.

**Decision: iter5b does not supersede iter5a.** No held-out set improved;
two show a small regression inside noise, one is a wash. This is now the
*third* independent data point (this project's earlier FFT hybrid branch,
+0.00005 on unseen SID_Set; the teammate's separately-trained FFT branch,
0.457 AUC on unseen generators -- below chance; and now this wavelet
branch, flat everywhere) all pointing the same direction: a frequency-
domain auxiliary branch bolted onto a frozen, already-strong ViT backbone
does not transfer to generators it wasn't trained against, regardless of
which transform (FFT or DWT) supplies the branch. The mechanism this
project has documented for *why* the ViT branch already generalizes well
(SAFE augmentation + `crop_from_native` closing the resolution-shortcut
and native-artifact gaps directly, no frequency side-channel needed) is
likely why a frequency branch has nothing marginal left to add here.

**iter5a remains this device's best and production checkpoint.**
`runs/iter5b/best.pt` is kept on disk for reference but not adopted; no
config or webapp default should point at it. No further hybrid/wavelet
variant is planned without a specific new hypothesis for why one would
transfer where three attempts (two teams, two transforms) have not.

## 15. Iteration 6a: stacking iter5's fixes onto iter4's real recipe -- beats iter7

Cross-checking `origin/main` (the teammate's shipped iter7, organiser Final
0.933) surfaced two things this device's iter5 line never had: (1) their
committed `config.yaml` shows iter4/iter7's ViT half trained with
`crop_from_native: false, safe_augment: false` -- the fix this device
validated in iter5a was never applied to their base model; (2) their
corpus is SID + 17 DRAGON + Community-Forensics-Small, and CF-Small was
the single biggest lever in their own history (iter3 0.804 -> iter4 0.9126,
+0.109, per section 9j) -- bigger than crop_from_native+SAFE's own
contribution (mixed_v2 0.7007 -> iter5a 0.8804 on a corpus *without*
CF-Small). iter5 skipped CF-Small on a wrong read of the HF auto-converted
preview export (only ~10.5k of the real ~556k rows, one shard reading as
100% single-architecture) under overnight time pressure -- a partial-
preview artifact, not a real data problem.

`scripts/probe_cf_small_index.py` built a full manifest of the actual
per-architecture shard files (`data/HFCF_small_*.parquet`, 186 shards) and
confirmed they're complete and cleanly organised: 278,096 real / 278,445
fake rows, architecture-tagged (LatDiff 215,453 / GAN 71,968 / PixDiff
11,968 / Other 8,976). `scripts/build_iter6_corpus.py` reused SID_Set +
DRAGON-17 + GenImage ADM/GLIDE from `iter5_data` via recomputing their
content-hash filenames (index-only, no re-download) and fetched 6 real +
8 fake CF-Small shards spread across LatDiff/GAN/PixDiff/Other/Real for
architecture diversity. Verified corpus: real=26,361 fake=40,141 (66,502
total) -- in line with iter4's own ~61k.

Trained fresh (no warm start, same reasoning as iter5a) with
`crop_from_native: true, safe_augment: true` on this corpus, 10 epochs,
`samples_per_epoch: 24000` matching iter4's own recipe scale. Seen-val AUC
0.9807 (vs iter5a's 0.9866 on its smaller, less diverse corpus -- expected,
since the harder, more diverse mix trades a little seen-validation AUC
for real transfer, borne out below).

Held-out, identical `finalize_local.sh` pipeline:

| Held-out set | iter5a | iter6a | iter7 (self-reported) |
|---|---|---|---|
| SID_Set | 0.8603 | 0.8481 | 0.997 |
| DRAGON, 8 unseen generators | 0.9798 | 0.9779 | 0.996 |
| **Organiser, resolution-matched** | 0.8804 | **0.9362** | 0.933 |

**Organiser Final Score already exceeds iter7's, with a plain ViT and no
CLIP branch.** Controlled per-generator comparison, identical matched set:

| Generator | iter5a clean/robust | iter6a clean/robust | iter7 clean/robust |
|---|---|---|---|
| ddpm | 0.840 / 0.767 | 0.913 / 0.855 | 0.93 / 0.86 |
| adm | 0.855 / 0.778 | **0.937 / 0.870** | 0.89 / 0.85 |
| ddim | 0.942 / 0.867 | **0.970 / 0.918** | 0.95 / 0.89 |
| dalle | 0.948 / 0.903 | 0.984 / 0.962 | 0.99 / 0.96 |
| vqdm | 0.983 / 0.921 | 0.995 / 0.958 | 1.00 / 0.97 |

iter6a beats iter7 outright on ADM (iter7's own weakest generator) and
DDIM, using training-data + augmentation fixes alone -- no frozen semantic
branch, no separately-trained fusion head. This is the cleanest evidence
yet that crop_from_native+SAFE and CF-Small's generator diversity are
independent, additive levers, not overlapping ones: iter4's own CF-Small
run never had the training-side fix, and iter5a's fix never had CF-Small's
diversity; combining them lands above either alone, and above iter7's
CLIP-augmented version of the CF-Small-only baseline.

**The gap that remains**: SID_Set and DRAGON both trail iter7's self-
reported numbers by a wide margin (0.85/0.98 vs 0.997/0.996) even though
organiser -- the brief's actual scored composition -- is now ahead. iter7's
dragon/sidset numbers are not independently reproduced on this device (no
access to their exact eval-set construction), so part of this gap may be
methodology rather than model; it is flagged here rather than assumed
away. Organiser is the metric the brief scores, and it is decisively
better, so this does not block treating iter6a as a genuine improvement --
but it means Phase 3 (below) is not merely chasing a marginal top-up, it
is also the chance to see whether the frozen CLIP branch's semantic
generalisation (the mechanism iter7 already demonstrated helps ADM
specifically) closes some of this remaining gap too, on top of a strictly
better ViT base than iter7's own.

Next: iteration 6b, `branch_kind: clip`, `vit_checkpoint: runs/iter6/best.pt`
-- the same frozen-CLIP-B architecture as iter7, layered onto this
iteration's stronger base rather than iter4's, testing whether the two
mechanisms (training-time fix + inference-time semantic branch) are
additive as hypothesised.
