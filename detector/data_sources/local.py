"""Local-disk data source: the existing ``../Data/train`` ingest path.

Thin wrapper over :mod:`detector.data` so it satisfies the same
``ingest(settings) -> (train_dataset, val_dataset, info)`` contract as every
other data source; no behavior change versus calling
:func:`detector.data.ingest_training_data` directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from ..data import ImageDataset, ingest_training_data
from ..transforms import build_eval_transform, build_train_transform


def ingest(settings: dict[str, Any]) -> tuple[Dataset, Dataset, dict[str, Any]]:
    run_dir = Path(settings["run_dir"])
    manifest = run_dir / "manifest.csv"
    train_records, val_records = ingest_training_data(
        settings["data_dir"],
        manifest,
        validation_fraction=float(settings.get("validation_fraction", 0.2)),
        seed=int(settings.get("seed", 2026)),
    )
    augment_probability = float(settings.get("train_augment_probability", 0.7))
    train_dataset = ImageDataset(train_records, build_train_transform(augment_probability))
    val_dataset = ImageDataset(val_records, build_eval_transform())
    info = {
        "train_count": len(train_records),
        "val_count": len(val_records),
        "manifest": manifest,
        "train_records": train_records,
        "val_records": val_records,
    }
    return train_dataset, val_dataset, info
