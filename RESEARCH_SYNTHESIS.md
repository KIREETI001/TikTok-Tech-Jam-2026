# Open-source research review and revised plan

A survey of published work on robust cross-generator AI-image detection, a
close reading of the briefing deck, and what both mean for `MERGE_PLAN.md`.
Written so the team does not need to loop through experiments the field has
already run.

---

## Part 1 -- The briefing deck, read closely

The deck (17 slides) is more prescriptive than the one-line summary suggested.
Points that change assumptions:

### 1a. It names two specific papers and expects us to have read them

- **Slide 10, "SAFE insight (KDD 2025)"**: *crop instead of down-sample to
  preserve high-freq artifacts; ColorJitter + RandomRotation kill
  colour/semantic shortcuts.*
- **Slide 10, "DDA insight (NeurIPS 2025)"**: *watch out for frequency bias --
  JPEG in your real images can become a spurious signal. Align pixel +
  frequency.*

Both are dissected in Part 2. The judges will recognise whether a submission
engaged with them.

### 1b. "Frequency patches", not a global spectrum

Slide 8's key insight: *best detectors combine high-level CLIP semantics +
low-level frequency **patches**.* Not a whole-image FFT. This matters --
PatchCraft (Part 2) and the SAFE random-mask both operate on local patches,
and the global log-FFT branch in `detector/model.py`'s `HybridDetector` is
precisely the thing the teammate's log measured at 0.457 AUC. The deck is
pointing at *patch-level local statistics*, which is a different and
better-supported idea.

### 1c. "Do not directly replicate existing models or approaches" (slide 14)

Combined with slide 8's explicit endorsement of CLIP+frequency, the reading
is: **use published backbones (CLIP, ViT, DINOv2 are named on slide 14 as
explicitly allowed), but the pipeline -- augmentation, fusion, calibration,
data composition -- must be our own.** Fine-tuning Community Forensics
further is closer to "replicating an approach" than assembling our own
CLIP+patch-frequency pipeline is. This is an argument *for* the CLIP branch
and *against* leaning entirely on the Community Forensics checkpoint.

### 1d. The augmentation grid is fixed and disclosed (slide 11)

JPEG q = {90,70,50,30}; blur sigma = {0.5,1.0,2.0}; resize {0.5x, 0.25x};
noise sigma = {0.02,0.05,0.10}; colour jitter +/-20%; centre crop 80%. Our
`detector/transforms.py` matrix already matches this exactly. The deck says
*apply these randomly during training* and *required transforms are tested at
evaluation* -- so training on continuous ranges that span these points (not
the points themselves) is the intended move.

### 1e. Scoring and judging

- **Final Score = 0.50 x AUC_clean + 0.50 x AUC_robust**, ROC-AUC, threshold-free.
- Slide 12's worked example: clean AUC 0.99, unseen-generator AUC 0.80.
- Judging: Technical Execution 35, Innovation & Insight 20, Impact 20,
  Feasibility 15, Presentation 10. **A compact robustness table + an
  error-analysis note with FP/FN examples is explicitly called out as
  directly earning Technical Execution and Insight points.** That is a
  deliverable, not a nice-to-have.
- Slide 13: *complexity vs feasibility -- a 2-branch ensemble may win 1% but
  cost you the demo. Ship what runs.* An explicit caution against the CLIP
  branch if it jeopardises a working submission.
- Slide 15 rabbit holes to avoid: *training from scratch, chasing SOTA,
  over-engineering the UI, datasets you can't finish.*

### 1f. Datasets

WildFake, CIFAKE, SID_Set are *recommended, not required* ("Not limited to
these datasets"). **CIFAKE is 32x32** -- confirmed not useful, matches the
teammate's and our own reasoning. Scope is **image-level binary only** -- no
localisation. This retroactively justifies dropping SID_Set's `tampered`
class (section 9b): localised editing is out of scope by the deck's own
words.

---

## Part 2 -- Published work, and what each result saves us

### SAFE -- Improving Synthetic Image Detection Towards Generalization (KDD 2025)

`arxiv.org/abs/2408.06741`, code at `github.com/Ouxiang-Li/SAFE`.
Cited by name on slide 10. Read the actual repo:

- **Architecture**: a *truncated* ResNet -- `conv1` + `layer1` + `layer2`
  only, then global pool + linear. **1.44M parameters.** The input is not
  the RGB image: it is the **DWT high-frequency sub-bands** (`bior1.3`
  wavelet, `J=1`, symmetric mode), i.e. `DWTForward` then take `Yh[0][:,:,2]`
  (the diagonal detail band), resized back to input size.
- **Preprocessing**: `RandomCrop(256)` at train, `CenterCrop(256)` at test.
  **No resize, ever.** Their ablation (Fig 4) shows bilinear/nearest resize
  at *either* train or test is strictly worse than crop/crop.
- **Augmentations**: `RandomRotation(180)`, `ColorJitter(0.5,0.5,0.5)`,
  `HorizontalFlip`, and **RandomMask** -- zero out random `16x16` patches up
  to 75% of the image, applied with p=0.5. Ablation (Fig 5): RandomMask +
  ColorJitter each add ~2-9 points ACC; rotation adds ~2.5.
- **Training**: AdamW, lr 5e-3, wd 0.01, batch 32, **20 epochs**, 1 warmup +
  cosine, trained on **ProGAN only** (plus matched real).
- **Results** (ACC/AP, mean over 26 generators, one-model): **95.6 / 99.3**,
  beating FatFormer (76.9/86.8, 577M params) and UniFD/Ojha (81.0/94.1,
  428M). On GenImage cross-architecture: **ADM 82.1, Glide 96.3, VQDM 96.3,
  Midjourney 95.3, BigGAN 97.8.**

**What this saves us**: SAFE's ADM number (82.1) against our 0.537 is the
clearest evidence that the fix for our pixel-space gap is not "add ADM
training data" alone -- it is *DWT input + crop-not-resize + local masking*.
SAFE never trained on ADM (ProGAN only) and still reached 82.1. This is a
concrete, tiny (1.44M param), MIT-licensed second branch that directly
implements slide 8's "low-level frequency patches" and slide 10's
"crop-not-downsample". Far more defensible than the global-FFT branch.

### DDA -- Dual Data Alignment (NeurIPS 2025 Spotlight)

`arxiv.org/abs/2505.14359`, code at `github.com/roy-ch/Dual-Data-Alignment`.
Cited by name on slide 10. Read the repo:

- **The bias**: detectors trained on (real repo A vs fake repo B) learn that
  real images have *less* high-frequency content -- because real images in
  these corpora have been through real-world JPEG, and the fakes (often
  VAE-decoded or freshly saved PNG) have not. The detector then flags any
  high-frequency image as fake and any compressed image as real. This is
  exactly section 8's compression-history finding and section 9f's
  resolution finding, given a name.
- **The frequency-alignment operation** (from `Training/data/custom_transforms.py`):
  for a real/fake pair, take the 2D DCT of each (per channel, optionally
  per-patch), blend `mixed = r * real_dct + (1-r) * fake_dct` with
  `r ~ U(0, R)`, inverse DCT. This produces a "fake" image whose frequency
  statistics are pulled toward the real one, so the detector cannot use the
  frequency gap as a shortcut.
- **The pixel-alignment operation**: mixup `x = r * real + (1-r) * fake_freq`,
  `r ~ U(0, R_pixel)`, applied with probability `P_pixel`.
- **Backbone**: DINOv2 ViT + LoRA (rank 8). Their EvalGEN result: **94.0%**
  average accuracy, +14.8 over second best; in-the-wild WildRF 95.1,
  SynthWildx 84.0, Chameleon 74.3.

**What this saves us**: DDA's full method needs paired real/fake images and
VAE reconstruction -- not applicable to our streamed corpus. **But the
frequency-alignment transform is a standalone augmentation we can apply to
any real/fake pair we already have** (DCT-blend a random real into each fake
during training). It costs one DCT per image and directly neutralises the
shortcut both the deck and our own audits flagged. This is the single
highest-value idea in this document that is *not* in the current plan.

### Community Forensics (CVPR 2025) -- our own backbone's origin

`arxiv.org/abs/2411.04125`. 2.7M images from **4,803 generators**. The paper's
central result: detection generalisation scales with the *number of distinct
generators* seen at training, more than with images-per-generator or model
capacity.

- **`OwensLab/CommunityForensics-Small`** is on HF as **186 streamable
  parquet shards, ~30k images, 50/50 real/fake**, spanning `LatDiff` + `GAN`
  + `Real` (LAION, COCO, ImageNet, CelebA and more). Same streaming path as
  SID_Set.
- An independent 2025 benchmark ("How well are open-sourced detectors
  out-of-the-box", `arxiv.org/abs/2602.07814`) ranked **Community Forensics
  #1 of 23** detectors for out-of-the-box generalisation (78.0% mean,
  rank-stddev 1.27 -- the only stable one).

**What this saves us**: (a) validates the backbone choice with an external
number; (b) CF-Small is a ready-made multi-generator, multi-real-source,
pre-balanced training pool -- it is Phase 1 and Phase 3 of the merge plan in
a single 30k-image stream, and it is the exact data our backbone was built
for. Adding it is lower-risk than assembling `bitmind/*` piecemeal.

### The generalisation-via-frozen-features line (Ojha -> C2P-CLIP -> RINE)

- **UniFD / Ojha (CVPR 2023)**: linear probe on frozen CLIP Vi-L/14. The
  origin of "frozen beats fine-tuned out of distribution". 81.0 mean ACC.
- **C2P-CLIP (2025)**: inject a learned "this is a real/fake photo" prompt
  into CLIP's text path, fine-tune. Tops most 2025 benchmarks (ForenSynths
  97.6). Fine-tunes CLIP -> heavier, and closer to "replicating an approach".
- **RINE (ECCV 2024)**: uses *intermediate* CLIP layers, not just the final
  embedding -- cheap, strong (ForenSynthsCh 87.2).
- **AIDE (ICLR 2025)**: CLIP semantic expert + a DCT-based "low-frequency /
  high-frequency" patch expert, fused. This *is* slide 8's architecture, done
  by a published paper: +3.5 on AIGCDetectBenchmark, +4.6 on GenImage.

**What this saves us**: the frozen-CLIP branch in the merge plan is
well-supported, but the *specific* pairing the deck wants -- CLIP semantics +
DCT/wavelet frequency *patches* -- is AIDE and SAFE, not a global-FFT branch.
If we build a fusion, it should be `frozen CLIP (or our ViT) + SAFE-style DWT
patch branch`, which is defensible as "informed by AIDE/SAFE" rather than
"replicating" either.

### NPR (CVPR 2024) and PatchCraft (2023) -- the "simple local" line

- **NPR**: upsampling in every GAN/diffusion decoder creates fixed local
  pixel correlations. NPR = `x - interpolate(interpolate(x, 0.5), 2x)`, a
  one-line operation, fed to a small CNN. **92.2 mean ACC over 28 models.**
  SAFE's repo includes `_preprocess_NPR` as an option.
- **PatchCraft**: split image into rich-texture and poor-texture patches,
  take the inter-pixel-correlation contrast. Generators fail hardest at
  synthesising realistic rich texture.

**What this saves us**: NPR is essentially free (no params, one interp
round-trip) and could be a *third* cheap input channel alongside DWT. But the
empirical-study paper (`arxiv.org/abs/2511.02791`) found frequency/NPR-family
methods *unstable* across distributions (FreqNet: 91% on one set, 1.6% on
another). Use NPR/DWT as an *auxiliary* signal fused with a semantic branch,
never alone -- which is what the deck says too.

### Robustness-specific findings

- The empirical study and multiple papers agree: **noise is the hardest
  degradation** (it overwrites exactly the high-frequency evidence).
  Confirmed in our own numbers and the teammate's.
- **Test-time augmentation** (multi-crop, flip, average logits) is a
  consistent small AUC gain across the literature and is threshold-free, so
  it lifts the exact Final-Score quantity. Not in the current plan.
- SAFE's robustness figures (their Fig 9/10) show DWT-input + crop training
  holds up markedly better under blur and JPEG than RGB-input baselines --
  the frequency-domain input is itself a robustness mechanism, because the
  wavelet detail band is less content-dependent.

---

## Part 3 -- Theoretical evaluation of MERGE_PLAN.md

| Merge-plan phase | Verdict after research | Change |
|---|---|---|
| 0 -- Instrument (shortcut probe, size-only AUC, freeze holdout) | **Confirmed essential.** DDA and the deck both make dataset bias the central risk. | Keep, expand: add a **DCT high-frequency-energy probe** (mean |DCT| above a radius, real vs fake) -- this is the specific bias DDA names and the one our audits keep hinting at. |
| 1 -- Pixel-space generator families (ADM/GLIDE/BigGAN...) | **Partially misdiagnosed.** SAFE reaches ADM 82.1 having *never trained on ADM*. The gap is preprocessing (resize vs crop) + input representation (RGB vs DWT), not only missing data. | Re-scope: still add the data, but **pair it with the SAFE-style DWT branch and crop-not-resize**, or the data alone under-delivers. |
| 2 -- Genval calibration off seen data | **Strongly confirmed** by DDA's and the teammate's independent reproduction. | Keep as-is. |
| 3 -- Multi-source content-matched reals | **Confirmed, and there's a shortcut**: `CommunityForensics-Small` is this, pre-built, 30k images, our backbone's own training distribution. | Replace the piecemeal `bitmind/*` fetch with CF-Small as the spine; add specific real sources only to fill gaps. |
| 4 -- Frozen CLIP branch | **Confirmed but re-pointed.** The deck wants CLIP + frequency *patches*. Pair frozen CLIP (or our ViT) with the **SAFE DWT branch**, not a global FFT. Watch slide 13's feasibility caution. | Merge phases 4 and the (dropped) frequency idea into one **"semantic + DWT-patch fusion"** phase, explicitly credited to AIDE/SAFE. |
| 5 -- Augmentation + SupCon | **Confirmed and upgraded.** Add SAFE's **RandomMask** (16px patches, up to 75%, p=0.5) and **RandomRotation(180)**. Add **DDA frequency-alignment** as an augmentation on real/fake pairs. Add **TTA** at inference. | Expand. |
| 6 -- Native-resolution crops A/B | **Promoted from "highest risk, do last" to "core".** SAFE's entire thesis and ablation is that resize destroys the signal; the deck's slide 10 says the same. Our Community Forensics ViT-S/224 pretraining is the only reason for caution -- resolve by testing crop-from-native at 224 against the current resize path early, not last. | Move to Phase 1-adjacent. |

### Net assessment

The merge plan is directionally right but has two structural gaps the
research fills:

1. **It treats the pixel-space gap as a data problem.** SAFE shows it is
   substantially a *preprocessing + input-representation* problem. Adding ADM
   images to a resize-based RGB pipeline will help less than expected.
2. **It has no answer to the frequency/JPEG shortcut** beyond "measure it in
   Phase 0". DDA gives a concrete, cheap, applicable fix (DCT-blend
   augmentation) that the current plan omits entirely.

Neither requires a new training run beyond what the plan already budgets --
both fold into existing phases.

---

## Part 4 -- Additions to run in parallel (no extra critical-path runs)

These attach to training runs already in the plan or need no training:

### P-A. DWT / SAFE-style auxiliary branch  (fold into Phase 1 & 4)
Add `pytorch_wavelets`; compute the `bior1.3` J=1 diagonal detail band of the
224 crop as a 3-channel input to a **small** conv stem (SAFE's is `conv1 +
layer1 + layer2`, 1.44M params -- trivial on the Arc iGPU). Fuse its logit
into ours as a zero-init residual (the pattern already in `HybridDetector`,
reused with a branch that actually generalises). Credit: AIDE (ICLR 2025),
SAFE (KDD 2025).

### P-B. DDA frequency-alignment augmentation  (fold into Phase 5, or earlier)
During training, with probability ~0.1, replace a fake image's high-frequency
DCT coefficients with those of a random real image from the same batch
(per-channel, full-image DCT, blend ratio `U(0, 0.25)`). Forces the model off
the "high-freq = fake" shortcut. ~1 DCT per affected image. Code adaptable
from `dda/Training/data/custom_transforms.py`. Credit: DDA (NeurIPS 2025).

### P-C. Crop-from-native at 224, A/B against resize  (needs 1 run, but it's Phase 1's run)
Materialise a small eval + train slice at native resolution; change the
train/eval transform from `Resize(256)->Crop(224)` to `RandomCrop(224)` /
`CenterCrop(224)` from native pixels. Compare organiser AUC. SAFE's ablation
predicts a clear win; our ViT-S/224 pretraining is the risk. Resolve early.

### P-D. Community-Forensics-Small as the training spine  (fold into Phase 1 or 3)
Add a `data_source` for `OwensLab/CommunityForensics-Small` (186 parquet
shards, streamable). 30k images, 50/50, `LatDiff`+`GAN`+`Real` across many
sources. This is the multi-generator + multi-real-source corpus of Phases 1
and 3 in one stream, and it is the distribution our backbone was trained on.

### P-E. Test-time augmentation at inference  (no training)
Average sigmoid over {centre crop, 4 corner crops, h-flip} -- SAFE and the
5-crop protocol in the teammate's `data_utils.prepare_crops`. Threshold-free,
so it lifts Final Score directly. ~5x inference cost, still seconds per image.

### P-F. NPR channel  (fold into P-A's branch, optional)
Add `x - interp(interp(x, .5), 2x)` as an extra input plane to the DWT branch.
Free. Credit: NPR (CVPR 2024). Only if P-A lands and there's time.

### P-G. The error-analysis + robustness-table deliverable  (no training)
Slide 16 says this directly earns points. `detector/evaluation.py` already
writes `errors.csv` (representative FP/FN per condition) and `metrics.csv`.
Build the one-page note now, keep it updated per iteration: robustness table,
6 FP + 6 FN thumbnails with the model's probability, one paragraph on the
noise weakness and the frequency-shortcut mitigation. This is 20% of the
score (Innovation & Insight) partly gated on a document we can write today.

### P-H. Model-weights + reproducibility release  (no training)
Slide 14: winning teams open-source weights, pipeline, hyperparameters, eval
code. `experiments.md` + `config.yaml` + `run_iteration.sh` already cover
most of it. Add: push the best `.pt` to HF, a `predict.py` smoke path, and a
`requirements.txt` that pins the `+xpu` reality *and* a CUDA fallback.

---

## Part 5 -- Revised phase order

```
Phase 0   Instrument (+ DCT freq-energy probe)            2h   no GPU
Phase 1   Pixel-space data + CF-Small  + DWT branch (P-A) 6h   1 run   <- merged
          + crop-from-native A/B (P-C) rides this run
Phase 2   Genval / calval calibration                    1h   no GPU  (rescores P1)
Phase 3   DDA freq-align aug (P-B) + SAFE mask/rotation   4h   1 run
          + multi-source reals to fill CF-Small gaps
Phase 4   Semantic + DWT-patch fusion, tuned (P-A cont.)  4h   1 run
Phase 5   SupCon loss + TTA (P-E) + motion blur          4h   1 run
--- parallel, no critical path ---
P-G  error-analysis + robustness note        write now, update each iteration
P-H  reproducibility / weights release        assemble now
P-F  NPR channel                               only if P-A lands
```

~21 hours, 4 training runs (down from the merge plan's 5-6), because CF-Small
collapses Phases 1 and 3 and the DWT branch replaces the abandoned frequency
work rather than adding to it.

### Realistic outcome

The literature ceiling for *published* cross-generator methods on hard
in-the-wild sets (Chameleon) is ~74% accuracy (DDA). On the organisers'
WildFake composition, resolution-matched, a Final Score of **0.88-0.92** is a
strong, defensible result. The 3% FPR/FNR target is not reached by any
published method on unseen generators; the honest framing for the submission
is the deck's own words -- *"this remains an open question... no silver
bullet"* -- backed by our robustness table and error analysis showing exactly
where and why the model fails.
