# Merging ziyangchua02/model_training into this pipeline

An execution plan for porting every high-value finding from the parallel
pipeline. Source: `github.com/ziyangchua02/model_training` (ResNet50 +
branch fusion, 8 trained-and-scored experiments). Read alongside
`experiments.md` sections 9e/9f.

## Why this is worth doing

On the organisers' composition (WildFake vs COCO, resolution-matched), the
two pipelines currently sit:

| | this pipeline (iter2) | theirs |
|---|---|---|
| Final Score | **0.7170** | **0.8705** |
| AUC_clean | 0.7421 | 0.8734 |
| AUC_robust | 0.6919 | 0.8676 |

The gap is not uniform, which is what makes it actionable:

| Generator | ours | theirs | delta | family |
|---|---|---|---|---|
| ADM | 0.537 | 0.900 | **-0.363** | pixel-space diffusion |
| DDPM | 0.632 | 0.835 | **-0.203** | pixel-space diffusion |
| Imagen | 0.663 | n/a | -- | pixel-space cascaded |
| DDIM | 0.781 | 0.814 | -0.033 | pixel-space diffusion |
| VQDM | 0.916 | 0.886 | **+0.030** | vector-quantized |
| DALLE | 0.923 | 0.864 | **+0.059** | discrete / dVAE |

We are *ahead* on the two non-pixel-space generators and far behind on the
four pixel-space ones. Our entire training pool is latent diffusion
(SID_Set is SD-family, DRAGON is modern latent diffusion). The model has
never seen a pixel-space denoiser. That is a missing family, not a general
weakness -- and it is the single largest recoverable gap.

## Rules of engagement

Taken from their log, where each was learned the hard way:

1. **One substantive change per training run.** Their experiment 4 predicted
   an improvement, got a null result, and the reason was only legible
   because nothing else moved.
2. **A change that moves the selection split but not the holdout is
   unproven.** They record this warning three times. genval is one
   generator.
3. **The holdout never informs a decision** -- not checkpoint selection, not
   the threshold, not the choice of threshold *rule*. Ranking rules by
   holdout performance is hyperparameter tuning on the test set.
4. **Record null and negative results.** A change that improves seen
   accuracy and leaves held-out AUC flat is not an improvement.
5. **WildFake and COCO stay evaluation-only** (`EVAL_ONLY_DATASETS.md`,
   enforced by `detector/data.py`'s `assert_not_eval_only`).

## Do NOT port

Measured negatives -- porting these would cost runs to rediscover:

| Component | Their measurement | Verdict |
|---|---|---|
| `FrequencyBranch` (FFT + radial profile) | 0.457 AUC unseen (0.722 seen) | **Inverted**, below chance. Already dropped from our plan; `detector/model.py`'s `HybridDetector` should be marked dead rather than deleted, with the number in the docstring. |
| `CameraBranch` (noise residual) | 0.528 AUC unseen | Noise, despite self-supervised pretraining. |
| Fixed-false-positive-rate threshold rules | calval ranking came out *inverted* against the holdout | The real side shifts distribution as much as the fake side. |
| Auxiliary loss on a weak branch | mechanism for the 0.457 | Deep supervision on an underpowered branch is pressure to overfit. |

---

## Phase 0 -- Instrument before changing anything

**Cost**: ~2 h, no training.
**Why first**: their log repeatedly shows a change moving one split and not
another. Without a leakage measurement and a frozen reporting holdout we
cannot distinguish a real gain from a shortcut.

| Task | Detail |
|---|---|
| Port `shortcut_probe.py` | Two gradient-boosted probes: source metadata (width/height/aspect/format) and pixel statistics (channel mean/std, saturation, edge density, re-encode size) of the exact 224 crop. Chance is 50% after class balancing. Above ~65% on the pixel-stats probe the corpus is teaching something other than the task. |
| Run it on every pool we have | `iter3_train`, `sid_val448_fs`, `dragon_holdout_eval`, `eval_only_organisers_matched`. |
| Size-only AUC everywhere | `build_matched_eval.py` already computes it. Extend to SID and DRAGON sets -- we have only checked the organiser one (1.0000 -> 0.5000 after matching). |
| Freeze the reporting holdout | Declare `eval_only_organisers_matched` + 8 held-out DRAGON generators as report-only, in writing, and never select on them. |
| Baseline iter3 | Score `runs/iter3/best.pt` on the organiser set for a like-for-like starting point. |

**Exit criterion**: a table of leakage numbers per pool, and a written
statement of which split is allowed to influence which decision.

---

## Phase 1 -- Pixel-space generator families

**Cost**: ~2 h fetch, ~2.5 h train.
**Expected**: the largest single gain available. Their comparable change
(generator diversity 9 -> 17 families) was +0.042 AUC; ours targets a family
that is *entirely* absent rather than merely thin, against generators where
we currently score 0.537.

| Task | Detail |
|---|---|
| Fetch pixel-space fakes | `bitmind/GenImage_ADM` (46 shards), `_GLIDE` (21), `_BigGAN` (8), `_VQDM` (32), `_Wukong` (157), `_Midjourney` (402). All confirmed reachable, parquet, same streaming path as SID_Set. Target ~2,000 each. |
| Materialise to disk | Reuse `pipeline.py materialize-*` pattern: fetch decoupled from training (section 7's lesson), `--max-size` capped, RAM-safe per-shard `cache_clear`. |
| Keep an honest unseen set | Do **not** chase every organiser generator. Hold DDIM and Imagen out of training entirely so the organiser score retains genuinely-unseen families. ADM/VQDM entering training makes those rows "seen" -- report them separately and say so. |
| Retrain | iter3 recipe unchanged otherwise. |

**Validate**: organiser per-generator AUC on ADM/DDPM (now seen-family) and
on DDIM/Imagen (still unseen). The second pair is the honest number.

**Risk**: GenImage ADM is ImageNet-conditioned; WildFake ADM may be
LSUN-conditioned. Same architecture, different content distribution -- so a
gain here is partly content, partly family. The DDIM/Imagen rows control for
that.

---

## Phase 2 -- Selection and calibration off seen data

**Cost**: ~1 h code. **Rides on Phase 1's checkpoint -- no extra training.**
**Why now**: our worst symptom is not AUC, it is that FNR at the deployed
threshold is 81% clean and 99.8% under noise. Their experiment 1 -> 2 fixed
exactly this: threshold moved 0.49 -> 0.10, unseen-fake recall 0.252 ->
0.826, +9.9 balanced accuracy, with **no AUC change**.

| Task | Detail |
|---|---|
| Promote DRAGON holdout to `genval` | 8 withheld generators already exist. Use them for checkpoint selection (`select_metric`) and threshold fitting -- not the internal SID split. |
| Add a `calval` | A second withheld family (e.g. GLIDE, held out of Phase 1) picks the threshold *rule*, so the rule choice never touches the holdout. |
| Port `calibrate.py`'s rules | `balacc_argmax` (their default), `balacc_plateau`, `equal_error`, `median_midpoint`. Skip the fixed-FPR family entirely -- measured inverted. |
| Replace our noise-mixed calibration | Our current `_validate(calibrate_on_noise=...)` mixes clean and noisy scores at 2:1 on the *seen* split. Held-out-generator calibration is the better-evidenced mechanism; keep the noise mix only if it still helps on top. |

**Validate**: FNR and FPR at the deployed threshold on the organiser set and
DRAGON holdout. AUC should not move -- if it does, something else changed.

**Note**: because this changes selection and thresholding rather than
weights, both old and new calibration can be scored against the *same*
checkpoint. That keeps rule 1 intact.

---

## Phase 3 -- Multi-source, content-matched real images

**Cost**: ~3 h fetch, ~2.5 h train.
**Expected**: their single biggest win, +0.132 AUC. Every real image in our
training pool comes from SID_Set. Theirs spans 8 collections, and they
measured genuine camera photographs as the *weakest* real class (27%
false-positive rate before the fix, 2.4% after).

| Task | Detail |
|---|---|
| Add real collections | ImageNet, Open Images, FFHQ, CelebA-HQ, AFHQ, Caltech-256, DTD -- all on HF via `bitmind/*` or equivalents. |
| **Exclude COCO** | WildFake's authentic half is COCO **train2017** (163,846 images, verified). Training on COCO would contaminate the organiser benchmark's real side. This is the one source we must refuse despite them using it. |
| Content-pair | Match real content to fake content: ImageNet-real against ImageNet-conditioned GenImage fakes, faces against face generators, etc. Unpaired content is a shortcut -- the classifier learns "cat photo = real" rather than an artifact. |
| Deduplicate | SHA-256 exact + perceptual dhash near-duplicate, within and across splits. Report overlap with every eval set explicitly. |

**Validate**: shortcut-probe pixel-stats number should **drop**; FPR on
non-SID real sources; organiser AUC.

---

## Phase 4 -- Frozen CLIP ViT-L/14 branch

**Cost**: ~1 h code, ~3 h train (measure CLIP forward cost on the Arc iGPU
first -- 304M frozen params is heavy; may need batch 8 and bf16).
**Expected**: their biggest architecture win, +0.069 AUC. Their frozen
branch *alone* scored 0.897 against 0.878 for the fused model.

| Task | Detail |
|---|---|
| Add `ClipFeatureBranch` | `CLIPVisionModelWithProjection`, `openai/clip-vit-large-patch14`, all params frozen, `.eval()`. Vision tower only -- the text tower is dead weight. |
| Fix the normalization | Our pipeline normalizes with ImageNet stats; CLIP expects its own. Undo one, apply the other, or the frozen tower reads out-of-distribution inputs -- "the quiet way to make a frozen backbone useless". |
| Fuse as a zero-init residual | We already have this pattern working from section 8's `HybridDetector`: the new branch adds a correction to the proven logit, final layer zero-initialised so training starts mathematically identical to the current model. |
| Check the parameter budget | 304M frozen + 21.7M ours = ~326M against the brief's 2B cap. Fine. |
| Report per-branch AUC | If CLIP alone beats the fusion (as it did for them), simplify rather than keep the fusion. |

**Risk**: this is the one change that meaningfully raises inference cost on
a 128-EU iGPU. Benchmark before committing to a full run.

---

## Phase 5 -- Augmentation and contrastive robustness

**Cost**: ~2 h code, ~2.5 h train.
**Target**: noise is the weakest condition for *both* pipelines (their
robust-AUC 0.8886 mean, 0.8582 at sigma 0.10; ours similar).

| Task | Detail |
|---|---|
| Port `transforms_lib.py` wholesale | One definition of each degradation, shared by training, the evaluation grid and diagnostics. Their rationale is a live-hazard argument, not tidiness: `_jpeg` existed in three places and any drift between train and eval would silently invalidate every number. Ours has the same duplication risk between `transforms.py`'s augment and eval paths. |
| `WindowedAugment` | Degrade a 320px window, *then* crop to 224. Evaluation degrades the whole image and then crops, so applying JPEG or a 4x rescale directly to a 224 window is a different operation. Ours degrades before `Resize(256)/RandomCrop(224)` -- closer than theirs was, but the window size still differs from eval. |
| `CompetitionAugment` | Six effects sampled from continuous ranges that *contain* the grid points rather than sitting on them, 1-3 composed per view in redistribution order: geometry, colour, blur, noise, compression last. |
| Add motion blur | Not in the grid, but a camera pan produces it constantly and a detector that has never seen a directional smear reads that smoothness as a generator artifact. |
| Supervised contrastive loss | Two independently-degraded views of each image pulled to the same embedding. They credit this for degrading *symmetrically* -- recall falling with everything else rather than rising under downscaling, which means a deployment seeing resized images does not need a different threshold. Weight 0.2. |

**Validate**: `AUC_robust`, and specifically the noise rows.

**Skip**: CutMix (+0.006, inside noise) unless everything else is done.

---

## Phase 6 -- Native-resolution crops (A/B, highest risk)

**Cost**: ~3 h. **Do last.**
**Why last**: part of their +0.132, and the mechanism is real -- resizing to
224 low-pass filters away the high-frequency evidence detection depends on,
and a non-aspect-preserving resize encodes aspect ratio as squash
distortion. But it is also the change most likely to backfire here.

| Consideration | Detail |
|---|---|
| Our backbone disagrees | Community Forensics ViT-S/224 was pretrained with short-edge-256 resize + centre crop. Feeding it native crops is a train/inference preprocessing mismatch against its own pretraining -- their row 0b shows exactly what that costs. |
| It changes the measurement | Our 15-condition matrix resizes after degradation. Switching to native crops changes the eval protocol, so iter1-3 numbers stop being comparable. Must be run as an isolated A/B with the matrix held fixed. |
| 5-crop averaging | Their `prepare_crops`: centre plus four corners, scores averaged. A single 224 window onto a 1024px image sees 5% of it. Cheap accuracy at 5x inference cost. |

**Adopt only if** organiser AUC improves. Otherwise record the null result
and keep the resize.

---

## Sequencing and critical path

```
Phase 0  instrument            2h   ────────┐
Phase 1  pixel-space fakes     4.5h ────────┼──> retrain
Phase 2  genval calibration    1h   ────────┘    (scores same checkpoint)
Phase 3  multi-source reals    5.5h ─────────────> retrain
Phase 4  frozen CLIP           4h   ─────────────> retrain
Phase 5  augmentation + supcon 4.5h ─────────────> retrain
Phase 6  native crops A/B      3h   ─────────────> retrain (adopt or reject)
```

Roughly **25 hours of work and ~5 training runs**. Phases 1-3 are the
evidence-backed majority of the available gain; 4-6 are refinement.

Phases 0 and 2 are pure instrumentation and cost no training time -- they
can interleave with any run. Phases 1 and 3 are both data changes and must
not be merged into one run, or a null result will be unattributable.

## What success looks like

| Metric | Now (iter2) | After 1-3 | After 4-6 |
|---|---|---|---|
| Organiser Final Score | 0.7170 | ~0.85 | ~0.90 |
| Organiser clean FNR @ threshold | 0.808 | <0.15 | <0.10 |
| ADM / DDPM AUC | 0.537 / 0.632 | >0.85 | >0.90 |
| Unseen-family AUC (DDIM, Imagen) | 0.781 / 0.663 | >0.80 | >0.85 |
| Shortcut-probe pixel-stats | unmeasured | <65% | <65% |

The 3% FPR/FNR target remains a research-grade bar. Their best pipeline sits
at 0.9340 Final Score on their own holdout and does not reach it either.
Closing to ~0.90 on the organisers' composition is the realistic goal; the
honest statement is that neither pipeline is separable enough for a 3%
operating point on unseen generators.
