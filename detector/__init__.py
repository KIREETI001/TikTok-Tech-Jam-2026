"""Simple real-vs-AI image detection pipeline."""

from .data import (
    ImageDataset,
    ImageRecord,
    ingest_training_data,
    load_labeled_root,
    load_manifest,
    stratified_split,
    write_manifest,
)
from .transforms import (
    EVALUATION_CONDITIONS,
    apply_condition,
    build_eval_transform,
    build_train_transform,
)

__all__ = [
    "EVALUATION_CONDITIONS",
    "ImageDataset",
    "ImageRecord",
    "apply_condition",
    "build_eval_transform",
    "build_train_transform",
    "ingest_training_data",
    "load_labeled_root",
    "load_manifest",
    "stratified_split",
    "write_manifest",
]
