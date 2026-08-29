"""Local-disk data source: the existing ``../Data/train`` ingest path.

Thin wrapper over :mod:`detector.data` so it satisfies the same
``ingest(settings) -> (train_dataset, val_dataset, info)`` contract as every
other data source; no behavior change versus calling
:func:`detector.data.ingest_training_data` directly.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from ..data import ImageDataset, ingest_training_data
from ..transforms import build_eval_transform, build_train_transform


def _balanced_subset(records: list, limit: int, seed: int) -> list:
    """Take at most ``limit`` records, keeping the class balance even.

    Balanced rather than a plain head/random slice: an unbalanced cap would
    change the class prior at the same time as the source ratio, confounding
    the one variable this cap exists to move.
    """

    rng = random.Random(seed)
    by_label: dict[int, list] = {}
    for record in records:
        by_label.setdefault(record.label, []).append(record)

    per_class = max(1, limit // max(len(by_label), 1))
    kept: list = []
    for label in sorted(by_label):
        pool = list(by_label[label])
        rng.shuffle(pool)
        kept.extend(pool[:per_class])
    rng.shuffle(kept)
    return kept


def ingest(settings: dict[str, Any]) -> tuple[Dataset, Dataset, dict[str, Any]]:
    """Load the local labeled folder as train/validation datasets."""

    run_dir = Path(settings["run_dir"])
    manifest = run_dir / "manifest.csv"
    train_records, val_records = ingest_training_data(
        settings["data_dir"],
        manifest,
        validation_fraction=float(settings.get("validation_fraction", 0.2)),
        seed=int(settings.get("seed", 2026)),
    )
    # Optional cap on how many local training images are used, class-balanced.
    #
    # Added after measuring that Data/train is CIFAKE: 99,080 images that are
    # all 32x32 and all from a single generator (Stable Diffusion 1.4), while
    # the evaluation sets are 1024px from many generators. Left uncapped, this
    # source is ~80% of the training pool, so the model spends most of its
    # capacity on 7x-upscaled thumbnails from one generator.
    #
    # Capping it is not throwing away data so much as declining to be
    # dominated by one distribution: SSAFE measured 10K curated images beating
    # a 4M-image baseline, and Community Forensics measured generalization
    # tracking generator *count* rather than image count. Both say a smaller,
    # more representative pool is the better trade.
    # The cap applies to validation too, at the same ratio. Capping only
    # training would leave validation ~54% CIFAKE while training is ~23% --
    # and validation F1 is what selects the checkpoint, so the run would
    # optimise for the 32x32 distribution this cap exists to move away from.
    # That is exactly the selection-metric failure ziyangchua02 documented:
    # picking checkpoints on a split that does not resemble what you are
    # scored on. Keeping one ratio for both keeps the two comparable.
    max_train = settings.get("local_max_train_images")
    if max_train:
        seed = int(settings.get("seed", 2026))
        original_train = len(train_records)
        train_records = _balanced_subset(train_records, int(max_train), seed)
        if original_train:
            keep_ratio = len(train_records) / original_train
            train_records_val_limit = max(2, round(len(val_records) * keep_ratio))
            val_records = _balanced_subset(val_records, train_records_val_limit, seed)

    augment_probability = float(settings.get("train_augment_probability", 0.7))
    crop_policy = str(settings.get("crop_policy", "resize"))
    train_dataset = ImageDataset(
        train_records, build_train_transform(augment_probability, crop_policy=crop_policy)
    )
    val_dataset = ImageDataset(val_records, build_eval_transform(crop_policy=crop_policy))
    info = {
        "train_count": len(train_records),
        "val_count": len(val_records),
        "manifest": manifest,
        "train_records": train_records,
        "val_records": val_records,
    }
    return train_dataset, val_dataset, info
