"""Two-stage transfer-learning pipeline for the multi-branch AI-vs-real
image detector (spatial + frequency + camera-authenticity + CLIP concept
branches, fused via cross-attention -- see model.py).

Recommended run order:
  1. python pretrain_camera_branch.py   (self-supervised, real images only)
  2. python train_anomaly.py            (self-supervised, real images only)
  3. python train.py                    (this file)
  4. python validate.py
  5. python predict.py path/to/image.jpg

Steps 1-2 are optional but recommended -- train.py will run without them
(camera branch starts from random init, and predict.py just won't have an
anomaly score to report), printing a note either way.

Produces: ai_detector_best.pth      (best checkpoint)
          training_curves.png       (loss + accuracy curves)

------------------------------------------------------------------------
Honest scope notes -- what's in this file vs. the full research proposal:

- No domain-adversarial training against generator identity: the dataset
  has category subfolders (animals/nature/people/city/food) but no
  per-generator labels (which images came from Midjourney vs. Stable
  Diffusion vs. ...), so there's nothing to train a domain-adversarial
  branch against yet. If you later organize AI images by generator, that
  becomes addable.
- No continual-learning/EWC wiring: same reason -- that machinery needs
  multiple generator-labeled batches arriving over time to be meaningful,
  and would otherwise just be dead code.
- No explicit adversarial-attacker training loop (a module that perturbs
  known fakes to try to evade the detector, per the research proposal's
  section 3): this is real future work, left out here because it's a
  significant additional training-stability risk to get right and wasn't
  validated in the time available -- flagging it rather than shipping an
  unvalidated version of it.
- The held-out-*generator* evaluation protocol from the proposal is
  supported (see validate.py's run_unseen_generator_test), but requires
  you to supply a dataset/../unseen_generators/<name>/ folder -- there's
  no way to fabricate that data.
------------------------------------------------------------------------
"""

import io
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.optim as optim
from PIL import Image, ImageFilter

from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from data_utils import (
    SEED,
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_splits,
    TwoViewTransformSubset,
    TransformSubset,
    resolve_class_indices,
)
from model import FusionClassifier
from losses import evidential_loss, evidence_to_probs_and_uncertainty, supervised_contrastive_loss

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Disabled for the current dataset (CIFAKE-style 32x32 thumbnails of generic
# CIFAR objects -- planes, cats, frogs, etc, not photos of people). Branch
# D's artifact-concept bank ("distorted hands", "unnatural shadow",
# "garbled text") checks for things that don't exist in this content at all,
# so it would just be inert dead weight here. Flip back to True for
# higher-resolution, photographic data where those concepts are meaningful
# (requires `pip install transformers` + network access to fetch CLIP once).
ENABLE_VLM_BRANCH = False
BEST_MODEL_PATH = "ai_detector_best.pth"
CAMERA_PRETRAINED_PATH = "camera_branch_pretrained.pth"

BATCH_SIZE = 32

STAGE1_EPOCHS = 5
STAGE1_LR = 1e-3

STAGE2_EPOCHS = 20
HEAD_LR = 1e-4
BACKBONE_LR = 1e-5

EARLY_STOPPING_PATIENCE = 5

AUX_LOSS_WEIGHT = 0.3     # per-branch deep-supervision loss weight
SUPCON_WEIGHT = 0.2       # contrastive robustness loss weight
EVIDENTIAL_ANNEALING_EPOCHS = 10

# Set to an integer to train on only a small, class-balanced subset of the
# training set (fast iteration / debugging a code change without waiting
# through a full ~2 hour run on all 90,000 images). Set to None to use the
# full training set for a real run. Does not affect the size of the
# validation set.
DEBUG_MAX_TRAIN_IMAGES = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("GPU available:", torch.cuda.is_available())
print("Using device:", DEVICE)
print("VLM branch enabled:", ENABLE_VLM_BRANCH)


# ---------------------------------------------------------------------------
# Augmentations (unchanged from the single-branch model -- see comments
# there for the rationale; reproduced here since train.py owns the
# transform definitions)
# ---------------------------------------------------------------------------


class RandomJPEGCompression:
    def __init__(self, quality_range=(30, 95), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class RandomDownscaleUpscale:
    def __init__(self, scale_range=(0.3, 0.9), p=0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        scale = random.uniform(*self.scale_range)
        w, h = img.size
        small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


class RandomGaussianBlur:
    def __init__(self, radius_range=(0.0, 2.0), p=0.3):
        self.radius_range = radius_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        return img.filter(ImageFilter.GaussianBlur(random.uniform(*self.radius_range)))


class RandomCenterCrop:
    def __init__(self, crop_frac_range=(0.7, 0.95), p=0.3):
        self.crop_frac_range = crop_frac_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        frac = random.uniform(*self.crop_frac_range)
        new_w, new_h = int(w * frac), int(h * frac)
        left, top = (w - new_w) // 2, (h - new_h) // 2
        cropped = img.crop((left, top, left + new_w, top + new_h))
        return cropped.resize((w, h), Image.BILINEAR)


class RandomGaussianNoise:
    def __init__(self, sigma_range=(0.02, 0.10), p=0.3):
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        sigma = random.uniform(*self.sigma_range)
        arr = np.asarray(img).astype(np.float32) / 255.0
        noise = np.random.normal(0, sigma, arr.shape)
        noisy = np.clip(arr + noise, 0.0, 1.0) * 255.0
        return Image.fromarray(noisy.astype(np.uint8))


random_color_jitter = transforms.RandomApply(
    [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)], p=0.3
)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    RandomCenterCrop(p=0.3),
    RandomDownscaleUpscale(p=0.4),
    random_color_jitter,
    RandomGaussianBlur(p=0.3),
    RandomGaussianNoise(p=0.3),
    RandomJPEGCompression(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Must match validate.py / predict.py exactly.
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

train_subset, val_subset, test_subset, class_to_idx, class_counts = get_splits()
ai_idx, real_idx = resolve_class_indices(class_to_idx)
print(f"class_to_idx={class_to_idx}  AI_IDX={ai_idx}  REAL_IDX={real_idx}")

if DEBUG_MAX_TRAIN_IMAGES is not None:
    # Group train_subset's own positions by class, then take an equal share
    # of each class -- a plain first-N slice could easily land on a
    # class-skewed handful of images depending on how the split manifest
    # happened to shuffle them.
    positions_by_class = {}
    for pos, idx in enumerate(train_subset.indices):
        label = train_subset.dataset.samples[idx][1]
        positions_by_class.setdefault(label, []).append(pos)

    per_class = DEBUG_MAX_TRAIN_IMAGES // len(positions_by_class)
    rng = random.Random(SEED)
    selected_positions = []
    for label, positions in positions_by_class.items():
        positions = positions[:]
        rng.shuffle(positions)
        selected_positions.extend(positions[:per_class])
    rng.shuffle(selected_positions)

    train_subset = Subset(train_subset, selected_positions)
    print(f"DEBUG_MAX_TRAIN_IMAGES set -- truncated training set to {len(train_subset)} images (class-balanced)")

# Two views per training image (independently augmented) for the
# supervised contrastive robustness loss; a single deterministic view for
# validation.
train_dataset = TwoViewTransformSubset(train_subset, train_transform)
val_dataset = TransformSubset(val_subset, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model = FusionClassifier(enable_vlm_branch=ENABLE_VLM_BRANCH).to(DEVICE)

if os.path.exists(CAMERA_PRETRAINED_PATH):
    model.camera.load_state_dict(torch.load(CAMERA_PRETRAINED_PATH, map_location=DEVICE))
    print("Loaded self-supervised camera-branch pretraining from", CAMERA_PRETRAINED_PATH)
else:
    print(
        f"NOTE: no camera-branch pretraining found at {CAMERA_PRETRAINED_PATH} -- "
        "training that branch from random init. Run pretrain_camera_branch.py "
        "first for a better-initialized camera branch."
    )

n_params = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {n_params / 1e6:.1f}M  (trainable: {n_trainable / 1e6:.1f}M, rest is frozen CLIP)")


# ---------------------------------------------------------------------------
# Train / eval loop helpers
# ---------------------------------------------------------------------------


def compute_aux_loss(aux_logits_a, aux_logits_b, labels):
    total = 0.0
    for name in aux_logits_a:
        total = total + torch.nn.functional.cross_entropy(aux_logits_a[name], labels)
        total = total + torch.nn.functional.cross_entropy(aux_logits_b[name], labels)
    return total / (2 * len(aux_logits_a))


def train_epoch(model, loader, optimizer, spatial_unfrozen, epoch, desc):
    model.set_train_mode(True, spatial_unfrozen_submodules=spatial_unfrozen)

    running_loss = 0.0
    all_preds, all_labels = [], []

    for view_a, view_b, labels in tqdm(loader, desc=desc, leave=False):
        view_a, view_b, labels = view_a.to(DEVICE), view_b.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        out_a = model(view_a)
        out_b = model(view_b)

        primary_loss = 0.5 * (
            evidential_loss(out_a["evidence"], labels, epoch, annealing_epochs=EVIDENTIAL_ANNEALING_EPOCHS)
            + evidential_loss(out_b["evidence"], labels, epoch, annealing_epochs=EVIDENTIAL_ANNEALING_EPOCHS)
        )
        aux_loss = compute_aux_loss(out_a["aux_logits"], out_b["aux_logits"], labels)
        supcon_loss = supervised_contrastive_loss(
            out_a["branch_embeds"]["spatial"], out_b["branch_embeds"]["spatial"], labels
        )

        loss = primary_loss + AUX_LOSS_WEIGHT * aux_loss + SUPCON_WEIGHT * supcon_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = torch.argmax(out_a["evidence"], dim=1)  # evidence argmax == alpha argmax == prob argmax
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = running_loss / len(all_labels)
    accuracy = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return {"loss": avg_loss, "accuracy": accuracy}


def evaluate(model, loader, epoch, desc=""):
    model.set_train_mode(False)

    running_loss = 0.0
    all_preds, all_labels, all_uncertainty = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            out = model(images)

            loss = evidential_loss(out["evidence"], labels, epoch, annealing_epochs=EVIDENTIAL_ANNEALING_EPOCHS)
            running_loss += loss.item() * labels.size(0)

            probs, uncertainty = evidence_to_probs_and_uncertainty(out["evidence"])
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_uncertainty.extend(uncertainty.cpu().tolist())

    avg_loss = running_loss / len(all_labels)
    accuracy = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    mean_uncertainty = sum(all_uncertainty) / len(all_uncertainty)

    return {
        "loss": avg_loss, "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1": f1, "mean_uncertainty": mean_uncertainty, "preds": all_preds, "labels": all_labels,
    }


def check_overfitting(train_acc, val_acc, threshold=15.0):
    gap = train_acc - val_acc
    if gap > threshold:
        print(f"  WARNING: train accuracy is {gap:.1f} points above validation accuracy -- possible overfitting.")


history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
best_val_f1 = -1.0
epochs_without_improvement = 0


def maybe_save_checkpoint(val_result, epoch_label):
    global best_val_f1, epochs_without_improvement
    if val_result["f1"] > best_val_f1:
        best_val_f1 = val_result["f1"]
        epochs_without_improvement = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"  New best model saved ({epoch_label}) -- val macro-F1: {val_result['f1']:.3f}")
    else:
        epochs_without_improvement += 1


# ---------------------------------------------------------------------------
# Stage 1: spatial backbone frozen; frequency/camera/vlm-head/fusion trained
# ---------------------------------------------------------------------------
#
# Same rationale as the original single-branch model: the spatial branch's
# backbone already has useful ImageNet-pretrained features, while every
# other branch (frequency, camera, fusion, aux heads) starts from scratch
# or a small pretext-task init and needs to move fast without the frozen
# backbone's large gradients destabilizing it.

print("\n=== Stage 1: spatial backbone frozen, other branches + fusion trained ===")

stage1_optimizer = optim.Adam(model.stage1_trainable_parameters(), lr=STAGE1_LR)

for epoch in range(STAGE1_EPOCHS):
    train_result = train_epoch(
        model, train_loader, stage1_optimizer, spatial_unfrozen=[], epoch=epoch,
        desc=f"Stage1 Epoch {epoch + 1}/{STAGE1_EPOCHS} [train]",
    )
    val_result = evaluate(model, val_loader, epoch, desc=f"Stage1 Epoch {epoch + 1}/{STAGE1_EPOCHS} [val]")

    print(
        f"Stage1 Epoch {epoch + 1}/{STAGE1_EPOCHS} | train loss {train_result['loss']:.4f} "
        f"acc {train_result['accuracy']:.2f}% | val loss {val_result['loss']:.4f} "
        f"acc {val_result['accuracy']:.2f}% f1 {val_result['f1']:.3f} "
        f"mean uncertainty {val_result['mean_uncertainty']:.3f}"
    )
    check_overfitting(train_result["accuracy"], val_result["accuracy"])

    history["train_loss"].append(train_result["loss"])
    history["train_acc"].append(train_result["accuracy"])
    history["val_loss"].append(val_result["loss"])
    history["val_acc"].append(val_result["accuracy"])
    history["val_f1"].append(val_result["f1"])

    maybe_save_checkpoint(val_result, f"stage1 epoch {epoch + 1}")


# ---------------------------------------------------------------------------
# Stage 2: unfreeze spatial layer3/layer4, fine-tune everything together
# ---------------------------------------------------------------------------

print("\n=== Stage 2: fine-tuning spatial layer3/layer4 + everything else ===")

stage2_optimizer = optim.Adam([
    {"params": model.stage1_trainable_parameters(), "lr": HEAD_LR},
    {"params": list(model.spatial.layer3.parameters()) + list(model.spatial.layer4.parameters()), "lr": BACKBONE_LR},
])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(stage2_optimizer, mode="max", factor=0.5, patience=2)

epochs_without_improvement = 0  # stage-1 non-improvement doesn't count toward stage-2 early stopping

for epoch in range(STAGE2_EPOCHS):
    global_epoch = STAGE1_EPOCHS + epoch  # keeps evidential-loss annealing continuous across stages
    train_result = train_epoch(
        model, train_loader, stage2_optimizer, spatial_unfrozen=["layer3", "layer4"], epoch=global_epoch,
        desc=f"Stage2 Epoch {epoch + 1}/{STAGE2_EPOCHS} [train]",
    )
    val_result = evaluate(model, val_loader, global_epoch, desc=f"Stage2 Epoch {epoch + 1}/{STAGE2_EPOCHS} [val]")

    print(
        f"Stage2 Epoch {epoch + 1}/{STAGE2_EPOCHS} | train loss {train_result['loss']:.4f} "
        f"acc {train_result['accuracy']:.2f}% | val loss {val_result['loss']:.4f} "
        f"acc {val_result['accuracy']:.2f}% precision {val_result['precision']:.3f} "
        f"recall {val_result['recall']:.3f} f1 {val_result['f1']:.3f} "
        f"mean uncertainty {val_result['mean_uncertainty']:.3f}"
    )
    check_overfitting(train_result["accuracy"], val_result["accuracy"])

    history["train_loss"].append(train_result["loss"])
    history["train_acc"].append(train_result["accuracy"])
    history["val_loss"].append(val_result["loss"])
    history["val_acc"].append(val_result["accuracy"])
    history["val_f1"].append(val_result["f1"])

    scheduler.step(val_result["f1"])
    maybe_save_checkpoint(val_result, f"stage2 epoch {epoch + 1}")

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping: no val macro-F1 improvement for {EARLY_STOPPING_PATIENCE} epochs.")
        break

print(f"\nBest validation macro-F1: {best_val_f1:.3f}")
print("Best model saved to:", BEST_MODEL_PATH)


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------

epochs_x = range(1, len(history["train_loss"]) + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(epochs_x, history["train_loss"], label="train loss")
axes[0].plot(epochs_x, history["val_loss"], label="val loss")
axes[0].axvline(STAGE1_EPOCHS + 0.5, color="gray", linestyle="--", label="stage 1 -> 2")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].set_title("Loss curve"); axes[0].legend()

axes[1].plot(epochs_x, history["train_acc"], label="train acc")
axes[1].plot(epochs_x, history["val_acc"], label="val acc")
axes[1].axvline(STAGE1_EPOCHS + 0.5, color="gray", linestyle="--", label="stage 1 -> 2")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy (%)"); axes[1].set_title("Accuracy curve"); axes[1].legend()

fig.tight_layout()
fig.savefig("training_curves.png")
print("Saved training curves to training_curves.png")


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

print("\n=== Final report (best checkpoint, validation split) ===")
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
final_val = evaluate(model, val_loader, epoch=STAGE1_EPOCHS + STAGE2_EPOCHS, desc="Final eval")

print(f"Validation accuracy: {final_val['accuracy']:.2f}%")
print(f"Validation precision: {final_val['precision']:.3f}  recall: {final_val['recall']:.3f}  f1: {final_val['f1']:.3f}")
print(f"Mean predictive uncertainty: {final_val['mean_uncertainty']:.3f}")
print("Confusion matrix (rows=true, cols=pred), label order =", sorted(class_to_idx, key=class_to_idx.get))
print(confusion_matrix(final_val["labels"], final_val["preds"]))

print(
    "\nFor held-out test-set metrics, per-branch ablation, robustness "
    "testing, and unseen-generator generalization, run validate.py."
)
