"""Swappable training data sources.

Each module here exposes one function:

    ingest(settings: dict) -> tuple[Dataset, Dataset, dict]

returning ``(train_dataset, val_dataset, info)`` where both datasets already
yield ``(image_tensor, label)`` pairs (transform applied) ready for
:func:`detector.training.train_model_from_datasets`, and ``info`` carries at
least ``{"train_count": int, "val_count": int}`` for logging.

``config.yaml``'s ``data_source`` key selects the module by name via
:func:`get_data_source`.
"""

from __future__ import annotations

from typing import Any, Callable

from torch.utils.data import Dataset

IngestFn = Callable[[dict[str, Any]], tuple[Dataset, Dataset, dict[str, Any]]]


def get_data_source(name: str) -> IngestFn:
    if name == "local":
        from .local import ingest

        return ingest
    if name == "sid_set_stream":
        from .sid_set_stream import ingest

        return ingest
    if name == "mixed":
        from .mixed import ingest

        return ingest
    raise ValueError(f"Unknown data_source {name!r}; choose 'local', 'sid_set_stream', or 'mixed'.")
