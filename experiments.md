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

**Status: running** (`runs/mixed_v3`). Will evaluate against `Data/test`
and the same fixed SID_Set sample used since section 4. *This section
will be updated with results.*

**Known caveat going in**: DRAGON being fake-only skews the combined
pool's class balance somewhat toward fake (roughly 1.1-1.15:1 fake:real
by rough estimate) -- not compensated for yet (e.g. via `pos_weight` in
the loss); worth checking if this run's FPR moves in an unexpected
direction.

## Not yet attempted

- **Frequency/noise-residual auxiliary signal** (deferred, bigger lift): the
  cross-dataset gap in section 4 is exactly the kind of thing pixel-content
  models struggle with across generators; a lightweight frequency-domain
  signal alongside the Community Forensics logit (not a full rebuild of
  `origin/main`'s 4-branch fusion model -- that was assessed and explicitly
  not adopted, see the architecture-decision note below) is the next lever
  if section 7's changes don't fully close the gap.
- **Lighter fine-tune depth** (Ojha et al.-motivated, still not tried):
  freeze the last transformer block too (train only the head + norm).
- **`pos_weight` in the loss** to directly counteract the FPR/FNR
  asymmetry seen since section 4, rather than relying on data changes to
  fix it indirectly.
- **Post-hoc calibration** (temperature/Platt scaling) on the current
  checkpoint's combined validation scores -- near-zero-cost, no retrain
  needed, not yet tried on the section-6/7 checkpoints (only tried once,
  in section 4a, on the section-3 checkpoint).
- **Generator-balanced batch sampling** (`WeightedRandomSampler` instead of
  plain `ConcatDataset` proportional mixing) so every source/generator is
  seen evenly per epoch regardless of its raw image count.
- **Lighter fine-tune depth** (Ojha et al.-motivated): freeze the last
  transformer block too (train only the head + norm), trading some clean
  accuracy for potentially better cross-generator generalization.
- **Third-dataset holdout**: sections 4-6 only check generalization to one
  additional dataset (SID_Set). A truly "any dataset" claim would want at
  least one more, still-unseen source (GenImage is a strong real-resolution
  candidate) to confirm the fix isn't just overfitting to two datasets
  instead of one.

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
