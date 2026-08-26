"""HTTP-streamed data loading for the SID_Set dataset (saberzl/SID_Set on
Hugging Face) -- ported from the ``origin/main`` branch's ``sid_set_data.py``
and adapted to this project's ``ingest(settings) -> (Dataset, Dataset, info)``
data-source contract.

Reads parquet shards directly from the Hugging Face Hub over HTTP at
training time via ``HfFileSystem``, instead of downloading shards to local
disk first -- useful on a machine (or CI runner) with no local
``../Data/train`` copy. Every shard used in a run is fetched over the
network once (roughly 10-40s per shard for the image bytes); image bytes are
cached in memory for the rest of the run so that cost is paid once per shard
per run, not once per epoch.

SID_Set's own row schema: img_id, image ({bytes, path}), mask (unused here),
width, height, and label (0 = real, 1 = full_synthetic, 2 = tampered/locally
edited). This project is a binary AI-vs-real classifier, so both AI classes
(1 and 2) collapse to the single FAKE label, matching ``detector.data``'s
polarity (0 = real, 1 = fake).

Requires the optional ``pyarrow`` and ``huggingface-hub`` dependencies
(not needed for the ``local`` data source).
"""

from __future__ import annotations

import csv
import functools
import io
import random
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset, Subset

try:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem
    from PIL import Image
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "The sid_set_stream data source needs 'pyarrow' and 'huggingface-hub' "
        "installed (pip install pyarrow huggingface-hub)."
    ) from exc

REPO_ID = "saberzl/SID_Set"

# Matches detector.data's REAL_FOLDERS/FAKE_FOLDERS polarity: 0 = real, 1 = fake.
CLASS_TO_IDX = {"REAL": 0, "FAKE": 1}
_RAW_LABEL_TO_BINARY = {0: CLASS_TO_IDX["REAL"], 1: CLASS_TO_IDX["FAKE"], 2: CLASS_TO_IDX["FAKE"]}

DEFAULT_TRAIN_SHARDS = 13
DEFAULT_VAL_SHARDS = 1

_HF_FS = HfFileSystem()
_HF_API = HfApi()


def _hf_path(repo_relative_path: str) -> str:
    return f"datasets/{REPO_ID}/{repo_relative_path}"


def _shard_paths(prefix: str, limit: int) -> list[str]:
    all_files = _HF_API.list_repo_files(REPO_ID, repo_type="dataset")
    matches = sorted(f for f in all_files if f.startswith(f"data/{prefix}-") and f.endswith(".parquet"))
    if not matches:
        raise FileNotFoundError(
            f"No '{prefix}' shards found in the {REPO_ID} repo (expected files "
            f"like data/{prefix}-00000-of-00249.parquet)."
        )
    return matches[:limit]


def _load_shard_index(path: str) -> list[tuple[str, int]]:
    """img_id + label only, for building the sample list cheaply -- no image
    bytes fetched here.
    """
    table = pq.read_table(_hf_path(path), columns=["img_id", "label"], filesystem=_HF_FS)
    return list(zip(table.column("img_id").to_pylist(), table.column("label").to_pylist()))


@functools.lru_cache(maxsize=32)
def _load_shard_images(path: str):
    """Full image-bytes column for one shard, fetched once over HTTP and
    cached in memory for the rest of the run.
    """
    print(f"  Fetching image bytes for shard: {path} (first access only, ~10-40s)")
    return pq.read_table(_hf_path(path), columns=["image"], filesystem=_HF_FS)


class SIDSetDataset(Dataset):
    """Binary real-vs-AI dataset streaming directly from ``num_shards`` of
    SID_Set's ``prefix`` parquet shards on the Hugging Face Hub (no local
    copy). ``__getitem__`` returns ``(PIL.Image, label)``.
    """

    def __init__(self, prefix: str, num_shards: int):
        self.class_to_idx = CLASS_TO_IDX
        self.samples: list[tuple[str, int]] = []
        self._locations: list[tuple[str, int]] = []

        shard_paths = _shard_paths(prefix, num_shards)
        for i, path in enumerate(shard_paths):
            print(f"  Fetching {prefix} shard index {i + 1}/{len(shard_paths)}: {path}")
            for row_idx, (img_id, raw_label) in enumerate(_load_shard_index(path)):
                self.samples.append((img_id, _RAW_LABEL_TO_BINARY[raw_label]))
                self._locations.append((path, row_idx))

        if not self.samples:
            raise FileNotFoundError(f"No SID_Set '{prefix}' shards resolved from {REPO_ID}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, row_idx = self._locations[idx]
        image_bytes = _load_shard_images(path).column("image")[row_idx].as_py()["bytes"]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        _img_id, label = self.samples[idx]
        return image, label


class _TransformSubset(Dataset):
    """Wraps a ``Subset`` of :class:`SIDSetDataset` so train/val can each use
    their own transform, and returns the ``(image_tensor, label)`` pairs that
    :func:`detector.training.train_model_from_datasets` expects.
    """

    def __init__(self, subset: Subset, transform) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        image, label = self.subset[idx]
        return self.transform(image), label


def _compute_stratified_split(dataset: SIDSetDataset, train_frac: float, seed: int):
    indices_by_class: dict[int, list[int]] = {label: [] for label in dataset.class_to_idx.values()}
    for i, (_img_id, label) in enumerate(dataset.samples):
        indices_by_class[label].append(i)

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for indices in indices_by_class.values():
        indices = indices[:]
        rng.shuffle(indices)
        n_train = int(len(indices) * train_frac)
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def _save_split_manifest(path: Path, dataset: SIDSetDataset, train_indices, val_indices) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["img_id", "split"])
        for split_name, indices in (("train", train_indices), ("val", val_indices)):
            for i in indices:
                img_id, _label = dataset.samples[i]
                writer.writerow([img_id, split_name])


def _load_split_manifest(path: Path, dataset: SIDSetDataset):
    id_to_index = {img_id: i for i, (img_id, _label) in enumerate(dataset.samples)}
    train_indices: list[int] = []
    val_indices: list[int] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            idx = id_to_index.pop(row["img_id"], None)
            if idx is None:
                continue
            (train_indices if row["split"] == "train" else val_indices).append(idx)
    return train_indices, val_indices


def ingest(settings: dict[str, Any]) -> tuple[Dataset, Dataset, dict[str, Any]]:
    from ..transforms import build_eval_transform, build_train_transform

    num_train_shards = int(settings.get("sid_set_train_shards", DEFAULT_TRAIN_SHARDS))
    # SID_Set also ships separate "validation" shards; unlike origin/main this
    # pipeline has no distinct held-out test concept beyond train/val, so those
    # shards aren't fetched here. Reserved for a future `pipeline.py evaluate`
    # external-benchmark source.
    num_val_shards = int(settings.get("sid_set_val_shards", DEFAULT_VAL_SHARDS))
    validation_fraction = float(settings.get("validation_fraction", 0.2))
    seed = int(settings.get("seed", 2026))

    run_dir = Path(settings["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = run_dir / "sid_set_split_manifest.csv"

    train_pool = SIDSetDataset("train", num_train_shards)

    if split_manifest.exists():
        print(f"Loading existing SID_Set train/val split from {split_manifest}")
        train_indices, val_indices = _load_split_manifest(split_manifest, train_pool)
    else:
        train_indices, val_indices = _compute_stratified_split(
            train_pool, 1.0 - validation_fraction, seed
        )
        _save_split_manifest(split_manifest, train_pool, train_indices, val_indices)
        print(f"Computed new stratified SID_Set split and saved it to {split_manifest}")

    train_dataset = _TransformSubset(Subset(train_pool, train_indices), build_train_transform())
    val_dataset = _TransformSubset(Subset(train_pool, val_indices), build_eval_transform())

    info = {
        "train_count": len(train_indices),
        "val_count": len(val_indices),
        "manifest": split_manifest,
        "num_train_shards": num_train_shards,
        "num_val_shards": num_val_shards,
    }
    return train_dataset, val_dataset, info
