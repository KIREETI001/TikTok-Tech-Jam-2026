"""Shared dataset utilities for the AI-vs-real image detector.

Everything that needs to agree between train.py, validate.py and predict.py
lives here: which folder maps to which label, how the dataset is split, and
how class imbalance is measured. Centralizing this is what prevents the
classic "preprocessing/label mismatch" bug where training and inference
quietly disagree about which index means "AI" and which means "real".
"""

import csv
import os
import random
from collections import Counter

import torch
from torchvision import datasets
from torch.utils.data import Subset

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

# The dataset ships as a pre-split train/test tree:
#   DATASET_PATH/train/<class_name>/*.jpg
#   DATASET_PATH/test/<class_name>/*.jpg
# (CIFAKE-style: train/{FAKE,REAL}, test/{FAKE,REAL}.) We use the provided
# test/ folder as-is for the held-out test set, and carve our own
# train/validation split out of train/ only -- there's no reason to
# re-split test/ ourselves when the dataset already draws that line.
DATASET_PATH = "/home/huythuan-bui/model_training/dataset"
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
TEST_DIR = os.path.join(DATASET_PATH, "test")

# Fraction of the provided train/ folder used for actual training; the
# remainder becomes the validation split.
TRAIN_VAL_SPLIT_FRAC = 0.9

SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BEST_MODEL_PATH = "ai_detector_resnet50_best.pth"

# get_splits() is called independently by train.py and validate.py, in
# possibly separate runs. If it just re-scanned DATASET_PATH and re-computed
# the split fresh every time, any change to the dataset folder between those
# runs (adding, removing, or renaming an image) would silently shift indices
# and could put a training image into what validate.py thinks is the
# held-out test set -- invalidating the reported metrics with no warning.
# Persisting the split to disk the first time it's computed, and reusing it
# on every later call, makes train/val/test membership stable across runs
# regardless of later dataset edits.
SPLIT_MANIFEST_PATH = "split_manifest.csv"


# ---------------------------------------------------------------------------
# Label resolution
# ---------------------------------------------------------------------------
#
# ImageFolder assigns integer labels by sorting folder names alphabetically.
# That means the AI class is NOT always index 0 -- it depends entirely on
# your folder names ("AI" vs "Ai_generated_dataset" vs "ai_images" all sort
# differently relative to "REAL"/"real_dataset"/"real"). Hardcoding "0 = AI"
# anywhere is exactly how a label-reversal bug sneaks in silently: the model
# trains and evaluates fine (labels are self-consistent), but any script
# that prints results assumes the wrong mapping and reports every prediction
# backwards. So we never hardcode it -- we resolve it by name, once, here.


def load_class_to_idx():
    """Returns {class_name: index} by listing the train/ folder's top-level
    class folders, without walking every image file the way a full
    ImageFolder scan does. Cheap enough to call from predict.py on every
    single prediction, just to stay in sync with however train.py resolved
    labels.
    """
    _, class_to_idx = datasets.folder.find_classes(TRAIN_DIR)
    return class_to_idx


# Folder-name substrings recognized as "this is the AI-generated class".
# "fake" covers datasets (like CIFAKE) that label the synthetic class FAKE
# rather than AI; "synthetic"/"generated" cover other common conventions.
_AI_NAME_PATTERNS = ("ai", "fake", "synthetic", "generated")


def resolve_class_indices(class_to_idx):
    """Figures out which integer index is AI-generated and which is real by
    matching folder names, instead of assuming a fixed order. Raises loudly
    if the folder names are ambiguous so a naming mistake fails fast at
    startup instead of silently training a backwards model.
    """
    ai_matches = [name for name in class_to_idx if any(p in name.lower() for p in _AI_NAME_PATTERNS)]
    real_matches = [name for name in class_to_idx if "real" in name.lower()]

    if len(ai_matches) != 1 or len(real_matches) != 1:
        raise ValueError(
            "Could not unambiguously resolve AI/real class names from "
            f"folders {list(class_to_idx.keys())}. Expected exactly one "
            f"folder name containing one of {_AI_NAME_PATTERNS} and one "
            "containing 'real' (case-insensitive). Rename your dataset "
            "folders accordingly."
        )

    ai_idx = class_to_idx[ai_matches[0]]
    real_idx = class_to_idx[real_matches[0]]
    return ai_idx, real_idx


# ---------------------------------------------------------------------------
# Dataset stats / imbalance check
# ---------------------------------------------------------------------------


def print_dataset_stats(train_pool, test_full):
    """Prints per-class image counts (train pool vs. provided test set) and
    flags imbalance. An AI detector trained on a lopsided dataset (e.g. 3x
    more real photos than AI images) learns to lean toward the majority
    class whenever it's unsure, which is a common cause of a model that
    "mostly" works but is systematically biased on the underrepresented
    class.
    """
    train_counts = Counter(label for _, label in train_pool.samples)
    test_counts = Counter(label for _, label in test_full.samples)
    idx_to_class = {idx: name for name, idx in train_pool.class_to_idx.items()}

    print("\nDataset class counts:")
    for idx in sorted(idx_to_class):
        print(
            f"  {idx_to_class[idx]:<12} idx={idx}  "
            f"train_pool n={train_counts.get(idx, 0):>7}  test n={test_counts.get(idx, 0):>7}"
        )

    total = sum(train_counts.values())
    largest = max(train_counts.values())
    smallest = min(train_counts.values())
    ratio = largest / smallest if smallest else float("inf")

    print(f"  Total train-pool images: {total}")
    print(f"  Imbalance ratio (largest/smallest class, train pool): {ratio:.2f}x")
    if ratio >= 1.5:
        print(
            "  WARNING: classes are imbalanced. train.py compensates with "
            "class-weighted loss, but consider collecting more data for the "
            "smaller class if this ratio is large."
        )

    print("\nSample image paths (first 3 per class, train pool):")
    per_class_shown = Counter()
    for path, label in train_pool.samples:
        if per_class_shown[label] < 3:
            print(f"  [{idx_to_class[label]}] {path}")
            per_class_shown[label] += 1

    return train_counts


def compute_class_weights(counts, class_to_idx, device):
    """Inverse-frequency class weights for nn.CrossEntropyLoss, so the loss
    penalizes mistakes on the minority class more heavily instead of letting
    the model coast by mostly predicting the majority class.
    """
    num_classes = len(class_to_idx)
    total = sum(counts.values())
    weights = [total / (num_classes * counts[idx]) for idx in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Stratified train/val splitting (test/ comes from the dataset as-is)
# ---------------------------------------------------------------------------
#
# A plain random_split splits the whole pool without regard to class, which
# is risky with any class imbalance: a random split could land on far more
# or fewer than train_frac of a given class by chance, making per-class
# metrics noisy. Splitting per-class and then concatenating guarantees the
# train and val splits each have the same class ratio as the full pool.


def _compute_stratified_split(dataset, train_frac):
    indices_by_class = {label: [] for label in dataset.class_to_idx.values()}
    for i, (_, label) in enumerate(dataset.samples):
        indices_by_class[label].append(i)

    rng = random.Random(SEED)
    train_indices, val_indices = [], []

    for label, indices in indices_by_class.items():
        indices = indices[:]
        rng.shuffle(indices)
        n_train = int(len(indices) * train_frac)
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices


def _save_split_manifest(dataset, train_indices, val_indices):
    rows = []
    for split_name, indices in (("train", train_indices), ("val", val_indices)):
        for i in indices:
            path, _label = dataset.samples[i]
            rows.append((os.path.relpath(path, TRAIN_DIR), split_name))

    with open(SPLIT_MANIFEST_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["relpath", "split"])
        writer.writerows(rows)


def _load_split_manifest(dataset):
    path_to_index = {
        os.path.relpath(path, TRAIN_DIR): i
        for i, (path, _label) in enumerate(dataset.samples)
    }

    train_indices, val_indices = [], []
    missing = 0
    with open(SPLIT_MANIFEST_PATH, newline="") as f:
        for row in csv.DictReader(f):
            idx = path_to_index.pop(row["relpath"], None)
            if idx is None:
                missing += 1
                continue
            (train_indices if row["split"] == "train" else val_indices).append(idx)

    if missing:
        print(
            f"  NOTE: {missing} images listed in {SPLIT_MANIFEST_PATH} no "
            "longer exist on disk and were skipped."
        )
    if path_to_index:
        print(
            f"  WARNING: {len(path_to_index)} images in {TRAIN_DIR} are not "
            f"in {SPLIT_MANIFEST_PATH} (added since the split was created) "
            "and will not be used. Delete the manifest to recompute the "
            "split including them (this changes train/val membership for "
            "existing images too)."
        )

    return train_indices, val_indices


def get_splits(train_frac=TRAIN_VAL_SPLIT_FRAC):
    train_pool = datasets.ImageFolder(TRAIN_DIR, transform=None)
    test_full = datasets.ImageFolder(TEST_DIR, transform=None)

    if train_pool.class_to_idx != test_full.class_to_idx:
        raise ValueError(
            f"train/ and test/ disagree on class_to_idx: "
            f"{train_pool.class_to_idx} vs {test_full.class_to_idx}. Both "
            "folders must have exactly the same class subfolder names."
        )

    counts = print_dataset_stats(train_pool, test_full)
    ai_idx, real_idx = resolve_class_indices(train_pool.class_to_idx)
    print(f"\nResolved labels -> AI_IDX={ai_idx}  REAL_IDX={real_idx}")

    if os.path.exists(SPLIT_MANIFEST_PATH):
        print(f"Loading existing train/val split from {SPLIT_MANIFEST_PATH}")
        train_indices, val_indices = _load_split_manifest(train_pool)
    else:
        train_indices, val_indices = _compute_stratified_split(train_pool, train_frac)
        _save_split_manifest(train_pool, train_indices, val_indices)
        print(f"Computed new stratified train/val split and saved it to {SPLIT_MANIFEST_PATH}")

    train_subset = Subset(train_pool, train_indices)
    val_subset = Subset(train_pool, val_indices)
    test_subset = test_full  # dataset's own held-out test set, used as-is

    print(
        f"Split sizes -> train={len(train_subset)}  val={len(val_subset)}  "
        f"test={len(test_subset)}"
    )

    return train_subset, val_subset, test_subset, train_pool.class_to_idx, counts


class TransformSubset(torch.utils.data.Dataset):
    """Wraps a Subset so each split (train/val/test) can use its own
    transform, since ImageFolder + Subset otherwise forces every split to
    share one transform.
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        image = image.convert("RGB")
        return self.transform(image), label


class TwoViewTransformSubset(torch.utils.data.Dataset):
    """Like TransformSubset, but applies `transform` twice independently to
    produce two differently-augmented views of the same image. Used for the
    supervised contrastive loss (see losses.py), which needs two views per
    sample. Only useful with a randomized transform -- calling a
    deterministic transform twice would just produce two identical views.
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        image = image.convert("RGB")
        return self.transform(image), self.transform(image), label
