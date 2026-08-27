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

**Status: running.** Launched against the full local pool (99,080 images)
plus 10 SID_Set train shards. Will evaluate the resulting checkpoint against
both `Data/test` (compare to #3's 0.9511/0.9133) and the same fixed SID_Set
sample from section 4 (compare to #3's 0.6950/0.6924 and the zero-shot
0.6009/0.5402) once training finishes. *This section will be updated with
results.*

## Not yet attempted

- **Frequency/noise-residual auxiliary signal** (deferred, bigger lift): the
  cross-dataset gap in section 4 is exactly the kind of thing pixel-content
  models struggle with across generators; a lightweight frequency-domain
  signal alongside the Community Forensics logit (not a full rebuild of
  `origin/main`'s 4-branch fusion model -- that was assessed and explicitly
  not adopted, see the architecture-decision note below) is the next lever
  if section 5's data-diversity fix doesn't fully close the gap.
- **Third-dataset holdout**: sections 4/5 only check generalization to one
  additional dataset (SID_Set). A truly "any dataset" claim would want at
  least one more, still-unseen source to confirm the mixed-training fix
  isn't just overfitting to two datasets instead of one.

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
