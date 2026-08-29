"""HTTP-streamed data loading for the DRAGON dataset (lesc-unifi/dragon on
Hugging Face, CC-BY-SA-4.0) -- a real-resolution (1024x1024), generator-
balanced source of AI-generated images from 25 modern diffusion models
(SD 1.5/2.1/3, SDXL, Flux, PixArt, Kolors, Kandinsky, and several
distilled/fast variants -- LCM, SDXL Turbo/Lightning, Hyper-SD -- whose
artifacts differ from standard diffusion sampling and aren't represented
in SID_Set or the PS5 dataset). Added as a third training-diversity source
after evaluating the mixed (PS5 + SID_Set) checkpoint and finding room to
improve cross-dataset generalization further; see experiments.md section 7.

DRAGON is fake-only -- every image here is AI-generated (there is no
paired "real" class in this dataset). This mixes in as extra fake-class
diversity; the real class continues to come entirely from PS5/SID_Set via
detector.data_sources.mixed.

Reads Hugging Face's auto-converted parquet export (the same mechanism
sid_set_stream.py uses for SID_Set) rather than the dataset's native
webdataset/TAR format, so no new dependency (e.g. the `datasets` library)
is needed -- just the `pyarrow`/`huggingface-hub` already used elsewhere.
That auto-conversion caps each "partial-train" shard at 400 rows
regardless of which size config (Small/Regular/Large/ExtraLarge) it comes
from; the "Regular" config is used here as an arbitrary but representative
choice. Each DRAGON image is a large PNG (~1.9MB) -- noticeably heavier
than SID_Set's ~550KB/image -- so the shard count here is kept modest by
default to bound fetch time and memory.
"""

from __future__ import annotations

import csv
import functools
import io
import random
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset, Subset

from ._network import retry_network_call

try:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem, constants as _hf_constants
    from PIL import Image
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "The dragon data source needs 'pyarrow' and 'huggingface-hub' "
        "installed (pip install pyarrow huggingface-hub)."
    ) from exc

# huggingface_hub's default request timeout (10s) is tuned for small API
# calls, not a ~760MB single-shard parquet GET -- DRAGON's images are large
# PNGs (~1.9MB each vs SID_Set's ~550KB), so a "partial-train" shard's full
# 400-row fetch routinely needs longer than that and would otherwise fail
# with repeated read-timeout retries (confirmed: reproduced at the 10s
# default, fixed by raising this). Assigning the attribute directly (not
# just setting the HF_HUB_DOWNLOAD_TIMEOUT env var) so it takes effect
# regardless of whether huggingface_hub was already imported elsewhere in
# the process (e.g. by sid_set_stream) before this module loads.
_hf_constants.HF_HUB_DOWNLOAD_TIMEOUT = max(_hf_constants.HF_HUB_DOWNLOAD_TIMEOUT, 300)

REPO_ID = "lesc-unifi/dragon"
PARQUET_REVISION = "refs/convert/parquet"

# DRAGON has no real class; every row is AI-generated (fake=1), matching
# detector.data's REAL_FOLDERS/FAKE_FOLDERS polarity (0=real, 1=fake).
FAKE_LABEL = 1

DEFAULT_CONFIG = "Regular"
DEFAULT_SHARDS = 5  # 5 * 400 rows/shard = 2000 images, ~3.7GB of PNG bytes

_HF_FS = HfFileSystem()
_HF_API = HfApi()


def _hf_path(repo_relative_path: str) -> str:
    return f"datasets/{REPO_ID}@refs%2Fconvert%2Fparquet/{repo_relative_path}"


def _shard_paths(config: str, limit: int) -> list[str]:
    all_files = retry_network_call(
        lambda: _HF_API.list_repo_files(REPO_ID, repo_type="dataset", revision=PARQUET_REVISION),
        description="DRAGON repo file listing",
    )
    prefix = f"{config}/partial-train/"
    matches = sorted(f for f in all_files if f.startswith(prefix) and f.endswith(".parquet"))
    if not matches:
        raise FileNotFoundError(
            f"No '{prefix}' shards found in the {REPO_ID} auto-converted-parquet "
            f"export (expected files like {prefix}0000.parquet)."
        )
    return matches[:limit]


def _load_shard_index(path: str) -> list[str]:
    """model.txt only, for building the sample list cheaply -- no image
    bytes fetched here.
    """
    table = retry_network_call(
        lambda: pq.read_table(_hf_path(path), columns=["model.txt"], filesystem=_HF_FS),
        description=f"DRAGON shard index fetch ({path})",
    )
    return table.column("model.txt").to_pylist()


@functools.lru_cache(maxsize=8)
def _load_shard_images(path: str):
    """Full image-bytes column for one shard, fetched once over HTTP and
    cached in memory for the rest of the run. Smaller cache than
    sid_set_stream's (8 vs 32) since DRAGON's images are ~3.5x larger.

    Retries through retry_network_call: a real incident during a
    multi-hour training run had a transient network/DNS disruption
    outlast huggingface_hub's own (much shorter) retry budget and crash
    the whole run partway through epoch 1.
    """
    print(f"  Fetching image bytes for DRAGON shard: {path} (first access only)")
    return retry_network_call(
        lambda: pq.read_table(_hf_path(path), columns=["png"], filesystem=_HF_FS),
        description=f"DRAGON shard image fetch ({path})",
    )


class DragonDataset(Dataset):
    """Fake-only dataset streaming directly from ``num_shards`` of DRAGON's
    ``config`` parquet shards on the Hugging Face Hub (no local copy).
    ``__getitem__`` returns ``(PIL.Image, label)`` with label always 1.
    """

    def __init__(self, config: str = DEFAULT_CONFIG, num_shards: int = DEFAULT_SHARDS):
        self.samples: list[tuple[str, int]] = []  # (generator_name, label)
        self._locations: list[tuple[str, int]] = []

        shard_paths = _shard_paths(config, num_shards)
        for i, path in enumerate(shard_paths):
            print(f"  Fetching DRAGON shard index {i + 1}/{len(shard_paths)}: {path}")
            for row_idx, model_name in enumerate(_load_shard_index(path)):
                self.samples.append((model_name, FAKE_LABEL))
                self._locations.append((path, row_idx))

        if not self.samples:
            raise FileNotFoundError(f"No DRAGON '{config}' shards resolved from {REPO_ID}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, row_idx = self._locations[idx]
        image_bytes = _load_shard_images(path).column("png")[row_idx].as_py()["bytes"]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        _model_name, label = self.samples[idx]
        return image, label


class _TransformSubset(Dataset):
    """Wraps a ``Subset`` of :class:`DragonDataset` so train/val can each use
    their own transform, and returns the ``(image_tensor, label)`` pairs
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


def _compute_split(dataset: DragonDataset, train_frac: float, seed: int):
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    n_train = int(len(indices) * train_frac)
    return indices[:n_train], indices[n_train:]


def _save_split_manifest(path: Path, dataset: DragonDataset, train_indices, val_indices) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "model", "split"])
        for split_name, indices in (("train", train_indices), ("val", val_indices)):
            for i in indices:
                model_name, _label = dataset.samples[i]
                writer.writerow([i, model_name, split_name])


def ingest(settings: dict[str, Any]) -> tuple[Dataset, Dataset, dict[str, Any]]:
    from ..transforms import build_eval_transform, build_train_transform

    config = str(settings.get("dragon_config", DEFAULT_CONFIG))
    num_shards = int(settings.get("dragon_shards", DEFAULT_SHARDS))
    validation_fraction = float(settings.get("validation_fraction", 0.2))
    seed = int(settings.get("seed", 2026))

    run_dir = Path(settings["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = run_dir / "dragon_split_manifest.csv"

    pool = DragonDataset(config, num_shards)
    train_indices, val_indices = _compute_split(pool, 1.0 - validation_fraction, seed)
    _save_split_manifest(split_manifest, pool, train_indices, val_indices)

    augment_probability = float(settings.get("train_augment_probability", 0.7))
    crop_from_native = bool(settings.get("crop_from_native", False))
    train_dataset = _TransformSubset(
        Subset(pool, train_indices),
        build_train_transform(augment_probability, crop_from_native=crop_from_native),
    )
    val_dataset = _TransformSubset(
        Subset(pool, val_indices), build_eval_transform(crop_from_native=crop_from_native)
    )

    info = {
        "train_count": len(train_indices),
        "val_count": len(val_indices),
        "manifest": split_manifest,
        "config": config,
        "num_shards": num_shards,
    }
    return train_dataset, val_dataset, info
