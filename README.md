# AI-vs-Real Image Detector

A binary image classifier that predicts whether an image is AI-generated or
a real photograph, built as a four-branch fusion model (spatial, frequency,
camera-authenticity, and CLIP-based concept branches) rather than a single
CNN.

## Dataset

`dataset/` is a pre-split tree:

```
dataset/
  train/
    FAKE/   50,000 images
    REAL/   50,000 images
  test/
    FAKE/   10,000 images
    REAL/   10,000 images
```

This is the CIFAKE dataset: 32x32 CIFAR-style images (planes, cats, frogs,
etc.), half real CIFAR-10 photos, half Stable-Diffusion-generated
equivalents. `data_utils.py` further splits the `train/` folder 90/10 into
an actual training set and a validation set; `test/` is used as-is as the
held-out test set (see "Data pipeline" below).

Class folder names don't have to be exactly `FAKE`/`REAL` -- `data_utils.py`
resolves which folder is the AI class by matching folder names containing
"ai", "fake", "synthetic", or "generated", and which is real by matching
"real". If it can't find exactly one of each, it raises an error rather
than guessing, so a folder-naming mistake fails loudly instead of silently
training a backwards model.

## Files

### Core library

- **`data_utils.py`** -- everything that must agree between training,
  evaluation, and prediction: which folder maps to which label
  (`resolve_class_indices`), the stratified train/val split
  (`get_splits`), and the persisted split manifest (`split_manifest.csv`,
  generated) that keeps train/val membership stable across separate runs
  even if the dataset folder changes later. Also defines the `TransformSubset`
  / `TwoViewTransformSubset` dataset wrappers used everywhere else.

- **`model.py`** -- the model architecture itself. Defines four branches
  and the fusion module that combines them:
  - `SpatialBranch` -- a ResNet50 backbone (ImageNet-pretrained) for
    general visual/semantic features.
  - `FrequencyBranch` -- 2D FFT magnitude spectrum + radial power profile
    through a shallow CNN, targeting GAN/diffusion upsampling artifacts and
    unnatural frequency-spectrum decay.
  - `CameraBranch` -- a noise-residual (image minus a blurred version of
    itself) through a small CNN, meant to pick up on camera-sensor-noise-like
    statistics. Its encoder can be initialized from
    `camera_branch_pretrained.pth` (see below) instead of random weights.
  - `VLMConceptBranch` -- CLIP zero-shot similarity against a bank of
    artifact-related text prompts ("distorted hands", "garbled text", etc).
    **Currently disabled** (`ENABLE_VLM_BRANCH = False` in train.py/
    validate.py/predict.py) because this dataset's 32x32 generic-object
    thumbnails don't contain any of the things those prompts describe --
    the code is intact and ready to re-enable for photographic data.
  - `FusionClassifier` -- combines the branches via cross-attention (a set
    of learned "fusion query" tokens attend over the branch embeddings),
    with a per-branch auxiliary classifier head each (for deep supervision
    and per-branch ablation reporting) and a final evidential-uncertainty
    output head.

- **`losses.py`** -- `evidential_loss` (the primary training loss: treats
  the model's output as Dirichlet evidence rather than softmax logits, so
  it can express calibrated "I don't know" uncertainty instead of always
  looking confident), `evidence_to_probs_and_uncertainty` (converts that
  evidence into a probability + uncertainty score at inference time), and
  `supervised_contrastive_loss` (pulls same-label images' embeddings
  together across two independently-augmented views, training the model to
  be invariant to JPEG/blur/resize/crop degradation).

- **`anomaly.py`** -- `ConvAutoencoder`, a small convolutional autoencoder
  meant to be trained on real images only. Its reconstruction error is a
  second, independent "does this look like a real photo" signal that
  doesn't require having seen any fake image at all -- useful in principle
  against future AI generators the classifier has never seen.

### Scripts you run, in order

1. **`pretrain_camera_branch.py`** -- self-supervised pretraining for
   `CameraBranch`, using only real images (no labels needed). Pretext task:
   given two crops, predict whether they came from the same source image or
   two different ones. Saves `camera_branch_pretrained.pth`, which
   `train.py` loads automatically if present.
   *(Caveat: with this dataset's 32x32 images, the 96px crop size is larger
   than the source image, so every "crop" ends up being the whole image
   resized up -- the task degenerates into "are these two images identical"
   rather than a real noise-statistics signal. Not harmful (the branch gets
   fully fine-tuned during the real training run afterward regardless), but
   don't read much into its ~100% pretext accuracy.)*

2. **`train_anomaly.py`** -- trains the real-image-only `ConvAutoencoder`
   from `anomaly.py`. Saves `anomaly_autoencoder_best.pth` and
   `anomaly_threshold.txt` (the 95th-percentile reconstruction error on real
   validation images, used later to flag "this doesn't look like a typical
   real photo").

3. **`train.py`** -- the main event. Trains the full `FusionClassifier`
   (see "How training works" below). Saves `ai_detector_best.pth` (best
   checkpoint by validation macro-F1) and `training_curves.png`
   (loss/accuracy curves).

4. **`validate.py`** -- evaluates the saved checkpoint on the held-out
   `dataset/test/` split: accuracy/precision/recall/F1/confusion matrix,
   per-branch ablation accuracy, a robustness sweep (JPEG/blur/resize/
   noise/crop at fixed levels), an anomaly-score sanity check, and
   (optionally, if you add a `dataset/../unseen_generators/<name>/` folder)
   a generalization test against AI generators not seen during training.
   Writes `test_metrics.csv` and `robustness_report.csv`.

5. **`predict.py`** -- classify one new image:
   `python predict.py path/to/image.jpg [--verbose]`. Prints AI/Real
   probabilities, the model's uncertainty score, and (if
   `anomaly_autoencoder_best.pth` exists) an anomaly score. `--verbose`
   also shows each branch's individual prediction before fusion.

### Other files

- **`requirements.txt`** -- pinned Python dependencies.
- **`split_manifest.csv`** (generated) -- the persisted train/val split;
  see `data_utils.py` above. Delete it if you want a freshly randomized
  split (e.g. after meaningfully changing the dataset).
- **`ai_detector_best.pth`, `camera_branch_pretrained.pth`,
  `anomaly_autoencoder_best.pth`, `anomaly_threshold.txt`,
  `training_curves.png`, `test_metrics.csv`, `robustness_report.csv`**
  (all generated) -- outputs of the scripts above.
- **`*.log`** (generated) -- captured stdout from background training runs.
- **`test.py`, `test.JPG`, `IMG_9453.JPG`** -- an older, single-branch
  prediction script and two sample images from an earlier version of this
  project (before the multi-branch rewrite). `test.py` is superseded by
  `predict.py`.

## How training works (`train.py`)

**Two-stage transfer learning**, same idea for the spatial branch as a
standard fine-tuning recipe, extended to coordinate the other branches:

- **Stage 1** (5 epochs): the `SpatialBranch`'s ResNet50 backbone is
  entirely frozen. Only the frequency branch, camera branch, fusion module,
  and auxiliary heads train (plus `SpatialBranch`'s own final projection
  layer). This lets the new branches and fusion mechanism find reasonable
  weights before the pretrained backbone's weights start moving at all.
- **Stage 2** (up to 20 epochs, early-stopped after 5 epochs without
  validation-F1 improvement): `SpatialBranch.layer3`/`layer4` (the last two
  ResNet50 blocks) are unfrozen and fine-tuned at a much lower learning rate
  than everything else, so the pretrained ImageNet features get nudged
  toward AI-detection-relevant texture cues rather than overwritten.

A custom `set_train_mode()` on the model makes sure only the
*currently*-trainable submodules go into PyTorch's `train()` mode --
otherwise frozen layers' BatchNorm statistics would keep drifting during
forward passes even with `requires_grad=False`, which is a real bug this
project hit once already with an earlier single-branch version.

**Per training step**, each image gets passed through the model twice --
two independently-augmented views (JPEG recompression, blur, noise, resize,
crop, color jitter, flip/rotate) -- and three loss terms are combined:

1. **Evidential loss** on the fused prediction (primary loss) -- see
   `losses.py`.
2. **Auxiliary cross-entropy loss** on each branch's own prediction (deep
   supervision), weighted at 0.3x, so no branch can get ignored by fusion
   and never receive a useful gradient.
3. **Supervised contrastive loss** between the two augmented views' spatial
   embeddings, weighted at 0.2x, training the representation to be
   invariant to exactly the kind of post-processing (recompression,
   resizing) that breaks fragile detectors on real-world images.

The best checkpoint (by validation macro-F1) is saved to
`ai_detector_best.pth` after every epoch that improves on the previous
best.

## What's deliberately not implemented

From the original research proposal this project is based on, a few pieces
were left out because the current dataset can't support them honestly:

- No domain-adversarial training against generator identity, and no
  continual-learning/EWC wiring -- both need per-generator labels (e.g.
  "this FAKE image came from Midjourney vs. Stable Diffusion"), which this
  dataset doesn't have.
- No adversarial-attacker training loop (a module that perturbs known fakes
  to try to evade the detector) -- real future work, left out as an
  unvalidated addition rather than shipping something untested.
- The VLM/CLIP branch is implemented and tested, but disabled by default
  for this dataset (see `model.py` above) since its artifact-concept
  prompts don't apply to 32x32 generic-object thumbnails.
