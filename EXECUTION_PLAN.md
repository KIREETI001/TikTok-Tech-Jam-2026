# Execution plan

The decision, consolidated from `MERGE_PLAN.md` (porting the teammate's
findings), `RESEARCH_SYNTHESIS.md` (literature + briefing deck), and
iteration 3's measured results (`experiments.md` section 9g-results).

Those two are analysis. This is what we do.

---

## Where we are

| Held-out set | iter 3 Final Score | clean FPR / FNR |
|---|---|---|
| SID_Set full-synthetic | 0.9989 | 2.4% / 0.2% |
| DRAGON, 8 unseen generators | 0.9974 | 2.4% / 1.4% |
| **WildFake vs COCO (resolution-matched)** | **0.804** | 11.8% / 40% * |
| tampered (out of scope) | 0.648 | -- |

\* the FPR/FNR are at a threshold calibrated for a problem WildFake does not
have -- see Task 3.

**The gap to the teammate's 0.871 is one generator family.** Per-generator
clean AUC on WildFake:

| | ADM | DDPM | Imagen | DDIM | DALLE | VQDM |
|---|---|---|---|---|---|---|
| iter 3 | **0.645** | **0.752** | 0.804 | 0.859 | 0.946 | 0.952 |
| SAFE (KDD 2025) | 0.821 | 0.835 | -- | 0.814 | -- | 0.963 |

We are strong on the discrete / VQ generators and weak on pixel-space
diffusion (ADM/DDPM/DDIM/Imagen), which has **zero representation in our
training pool** (SID_Set and DRAGON are both latent diffusion).

**What iteration 3 proved**: generator diversity is the dominant lever
(+0.087 Final Score, zero architecture change, transferred to an unseen
family). What it did not fix: ADM is still 0.645, and the model now
over-flags real images under degradation.

---

## Phase A -- now, in parallel, no GPU

Four tasks. A-1 and A-4 gate later decisions; A-2 and A-3 are deliverables
the rubric rewards directly (slide 16: robustness table + error analysis =
part of 55% of the score).

### A-1. Test-time augmentation on iter 3  (~30 min)
Average the sigmoid over {centre crop, 4 corner crops, h-flip} at inference.
Threshold-free, so any gain lands directly in Final Score. SAFE and the
teammate both use 5-crop. Measure on the WildFake matched set.
- **Gate**: if it adds >= 0.01 AUC, it becomes the default eval path for
  every run below. Expected +0.01 to +0.03.

### A-2. Error-analysis + robustness note  (~2 h)
`detector/evaluation.py` already writes `errors.csv` (top FP/FN per
condition) and `metrics.csv`. Assemble a one-page note: the 15-condition
table for each held-out set, 6 FP + 6 FN thumbnails with the model's
probability, one paragraph on the noise weakness and one on the
frequency-shortcut question. Living document -- update every iteration.

### A-3. Shortcut probe + DCT frequency-energy probe  (~2 h)
Port the teammate's `shortcut_probe.py`: two gradient-boosted probes
(source metadata; pixel statistics of the 224 crop). Add a DCT
high-frequency-energy probe (mean |DCT| above a radius, real vs fake) --
the specific bias DDA names. Run on `iter3_train`, `sid_val448_fs`,
`dragon_holdout_eval`, `eval_only_organisers_matched`.
- **Gate**: pixel-stats probe above ~65% means the corpus teaches a
  non-artifact shortcut, and Phase C's DDA frequency-alignment augmentation
  moves up the priority list. Below 65%, it is optional.

### A-4. Held-out-generator calibration framework  (~2 h code)
Implement the two-split rule: `genval` (checkpoint selection + threshold)
= DRAGON's 8 held-out generators; `calval` (picks the threshold *rule*)
= one further family held out of Phase B. Port SAFE / teammate rules:
`balacc_argmax`, `balacc_plateau`, `median_midpoint`, `equal_error`. Skip
the fixed-FPR family (measured inverted by the teammate).
- This does **not** change Final Score (AUC is threshold-free). It fixes the
  reported FPR/FNR table and the progress toward the <=3% goal, which
  iteration 3 broke (FPR 11.8% / FNR 40% on WildFake at threshold 0.10).
- Re-score iter 3 with the chosen rule for a clean like-for-like baseline.

**A-5. In the background while A-1..A-4 run**: materialise Phase B's data
(next section). Network-bound, no CPU contention with the above.

---

## Phase B -- iteration 4: generator + real-source diversity

**One training run. The highest-confidence gain available.** Iteration 3
already demonstrated this lever; Phase B widens it to the family we are
weakest on and to real-image sources beyond SID_Set.

### Data
| Add | Source | Count | Why |
|---|---|---|---|
| Community-Forensics-Small | `OwensLab/CommunityForensics-Small` (186 parquet shards, streamable) | ~12k, 50/50 | Our backbone's own training distribution: many latent-diffusion + GAN generators, real images from LAION / ImageNet / CelebA. Ranked #1 of 23 detectors out-of-the-box (arXiv 2602.07814). |
| Pixel-space diffusion fakes | `bitmind/GenImage_ADM`, `_GLIDE`, `_BigGAN` | ~2k each | Directly attacks the WildFake families we score 0.645-0.75 on. ADM and GLIDE become seen-family. |

Keep `iter3_train` (SID full-synthetic + 17 DRAGON generators) as the spine.
New total ~35k, class-balanced sampler.

**Contamination guard**: Community-Forensics-Small's real sources include
COCO, and WildFake's authentic half *is* COCO train2017. Drop CF-Small's
COCO real records, or dedup by perceptual hash against
`eval_only_wfcoco_native`. Enforce and report.

### Preprocessing A/B (rides this run)
Run a second short variant with `RandomCrop(224)` from native pixels instead
of `Resize(256) -> RandomCrop(224)`. SAFE's central claim, and slide 10's:
resize low-pass-filters away the evidence. The risk is our ViT-S/224
backbone was pretrained with short-edge-256 resize, so this is a
train/inference mismatch against its own pretraining.
- **Gate**: adopt native crops only if organiser AUC improves. Otherwise
  record the null result and keep the resize -- do not carry it forward.

### Recipe
Otherwise iteration 3's: 448px materialisation, `train_augment_probability`
0.8, cosine LR, 10-12 epochs. Threshold from A-4's rule.

### Expected
ADM 0.645 -> ~0.80, DDPM 0.752 -> ~0.85. Organiser Final Score 0.804 ->
**~0.85**. Hold DDIM and Imagen out of training so the organiser score
keeps genuinely-unseen families; label ADM/GLIDE rows "seen family".

---

## Phase C -- iteration 5: semantic + frequency-patch fusion

**One training run.** This is the briefing deck's stated key insight
(slide 8: "high-level CLIP semantics + low-level frequency **patches**") and
the piece iteration 4's data cannot deliver on its own -- SAFE reaches
ADM 0.82 *without* training on ADM, purely from the input representation.

### The DWT branch
Add `FrequencyPatchBranch`, adapted from SAFE (`github.com/Ouxiang-Li/SAFE`,
MIT):
- Input: `bior1.3`, J=1 diagonal high-frequency wavelet sub-band of the
  224 crop (via `pytorch_wavelets`), **not** the RGB image.
- Body: a truncated conv stem (`conv1` + two residual stages), ~1.4M
  parameters. Trivial on the Arc iGPU.
- Fusion: zero-initialised residual into the ViT logit -- the pattern
  already working in `detector/model.py`'s `HybridDetector`, reused with a
  branch that actually generalises (the old global-FFT branch measured
  0.457 AUC unseen and is retired).

### Augmentation additions (SAFE)
`RandomRotation(180)`, and `RandomMask` -- zero out random 16px patches, up
to 75% of the image, p=0.5. SAFE's ablation credits each with 2-9 points.

### Expected
Closes the residual pixel-space gap. Organiser Final Score ~0.85 -> **~0.88**.

### Fallback
If the DWT branch does not move ADM, the mismatch is architectural (our ViT
vs SAFE's ResNet stem). Only then consider a frozen CLIP branch (teammate's
+0.069) -- accepting slide 13's feasibility caution (304M frozen params,
real inference cost on a 128-EU iGPU) and slide 14's "do not replicate
approaches" (the DWT branch is a cleaner answer to the same slide-8 insight).

---

## Phase D -- iteration 6: robustness

**One training run.** iteration 3's `AUC_robust` (0.781) trails `AUC_clean`
(0.826) by 0.045 on WildFake, and the new failure is the model getting
trigger-happy under noise (FPR 3-7%). Both are addressed by the same idea.

| Add | Detail | Source |
|---|---|---|
| Supervised contrastive loss | Two independently-degraded views of each image pulled to the same embedding, weight 0.2. Credited for making a model degrade *symmetrically* rather than over-flag. | teammate `losses.py`, Khosla 2020 |
| Windowed augmentation | Degrade a 320px window, *then* crop to 224 -- because evaluation degrades the whole image then crops. | teammate `train.py` |
| DDA frequency-alignment aug | With p~0.1, DCT-blend a random real's high-frequency coefficients into a fake (`r ~ U(0, 0.25)`). Neutralises "high-freq = fake". | DDA `custom_transforms.py` |
| Motion blur | Not in the grid, but a camera pan produces it constantly. | teammate `transforms_lib.py` |

### Expected
`AUC_robust` closes toward `AUC_clean`. Organiser Final Score ~0.88 ->
**~0.89-0.91**.

---

## Sequence and budget

```
Phase A   now, parallel, no GPU                    ~1 day wall-clock
  A-1 TTA measure          30m
  A-2 error-analysis note   2h    <- rubric
  A-3 shortcut probes       2h    <- gate
  A-4 calibration framework 2h    <- fixes iter3's operating point
  A-5 materialise Phase B data (background)

Phase B   iter 4  data diversity + crop A/B        ~3h train + eval
Phase C   iter 5  DWT / SAFE fusion branch         ~3h train + eval
Phase D   iter 6  SupCon + windowed aug + DDA      ~3h train + eval

--- assemble throughout ---
  best .pt -> HuggingFace
  run_iteration.sh + config.yaml + requirements (xpu + cuda fallback)
  robustness table + error analysis (from A-2, updated per iter)
  Devpost writeup incl. the trade-offs discussion (slide 13)
  2-4 min demo video
```

**~4 training runs, ~1 day of parallel prep, ~2-3 days total.**

---

## Decision gates

| After | If | Then |
|---|---|---|
| A-1 | TTA gain < 0.01 | drop it, don't complicate the eval path |
| A-3 | pixel-stats probe < 65% | DDA freq-align aug in Phase D is optional |
| B | crop-from-native worse than resize | revert, record null, keep resize |
| B | Final Score < 0.82 | data lever is spent -- go straight to Phase C |
| C | DWT branch doesn't move ADM | architectural mismatch -- evaluate frozen CLIP branch, accept the feasibility cost |
| any | demo day close and a checkpoint is unstable | freeze the last good `.pt`, ship it (slide 13) |

---

## Honest outcome statement

The published ceiling for cross-generator detection on hard in-the-wild sets
is ~74% accuracy (DDA on Chameleon). The teammate's best pipeline reaches
0.9340 Final Score on its *own* holdout. A realistic target for this
pipeline on the resolution-matched organiser composition is **0.88-0.91**.

The 3% FPR/FNR bar is reached by no published method on unseen generators.
The submission should say so, in the briefing deck's own words -- *"this
remains an open question... there is no silver bullet"* -- and back it with
the robustness table and error analysis showing exactly where and why the
model fails. That honesty is worth Innovation & Insight points; a
suspiciously clean 3% claim on unseen generators is not credible and judges
who wrote slide 13 will know it.
