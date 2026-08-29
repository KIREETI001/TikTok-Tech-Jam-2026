"""Materialize the iteration-5 training corpus and a DRAGON holdout eval set
to local disk, reproducing ziyangchua02's iter3_train recipe (documented in
experiments.md sections 9d/9g) plus GenImage ADM/GLIDE for the pixel-space
diffusion gap this project's own WildFake benchmark found (Final 0.62/0.74
on ADM/DDPM vs 0.91-0.99 on the rest -- see experiments.md section 11).

Their scratchpad/build_iter4.py was never committed (private, machine-local,
same convention as this project's own scratchpad/), so this is a
reconstruction from their documented recipe. Community-Forensics-Small was
found too small/uncertain in its auto-converted HF export (~10.5k rows, not
the ~30k they reported, and shard 0 was 100% one architecture) to depend on
for an overnight run without further reconnaissance -- see the session log.
Everything here streams from sources already proven working in this repo
(SID_Set, DRAGON) or already public and simple (GenImage).

Produces:
    ../iter5_data/train/{real,fake}/          -- training corpus
    ../iter5_data/dragon_holdout/{real,fake}/ -- 8 unseen DRAGON generators
                                                  + a disjoint SID_Set real
                                                  slice, for the cross-
                                                  generator scoreboard

Usage:
    ./.venv/Scripts/python.exe scripts/build_iter5_corpus.py
"""

from __future__ import annotations

import hashlib
import io
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from detector.data_sources._network import retry_network_call  # noqa: E402
from detector.data_sources.dragon import (  # noqa: E402
    _load_shard_images as _dragon_load_images,
    _load_shard_index as _dragon_load_index,
    _shard_paths as _dragon_shard_paths,
)
from detector.data_sources.sid_set_stream import (  # noqa: E402
    _load_shard_images as _sid_load_images,
    _load_shard_index as _sid_load_index,
    _shard_paths as _sid_shard_paths,
)

OUT_ROOT = Path("../iter5_data").resolve()
TRAIN_DIR = OUT_ROOT / "train"
HOLDOUT_DIR = OUT_ROOT / "dragon_holdout"

SID_SET_TRAIN_SHARDS = 30  # matches iter3_train: shards 0-29
SID_SET_HOLDOUT_SHARD_START = 30  # a disjoint shard for the holdout's real half
SID_SET_HOLDOUT_SHARDS = 2

DRAGON_HELD_OUT = {
    "Flux_1", "IF", "JuggernautXL", "Kolors",
    "PixArt_Sigma", "SDXL_Turbo", "SD_3", "SD_Cascade",
}

GENIMAGE_REPOS = {
    "adm": "bitmind/GenImage_ADM",
    "glide": "bitmind/GenImage_glide",
}
GENIMAGE_PER_GENERATOR = 2500


def _save(image_bytes: bytes, dest_dir: Path, key: str) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(key.encode()).hexdigest()[:20] + ".jpg"
    out_path = dest_dir / name
    if out_path.exists():
        return
    with Image.open(io.BytesIO(image_bytes)) as img:
        img.convert("RGB").save(out_path, format="JPEG", quality=95)


def build_sid_set(train_shards: int, holdout_start: int, holdout_shards: int) -> None:
    print(f"\n=== SID_Set: {train_shards} train shards (full_synthetic only) ===")
    paths = _sid_shard_paths("train", train_shards + holdout_start + holdout_shards)
    train_paths = paths[:train_shards]
    holdout_paths = paths[holdout_start : holdout_start + holdout_shards]

    counts = {"real": 0, "fake": 0, "dropped_tampered": 0}
    for i, path in enumerate(train_paths):
        print(f"  [{i + 1}/{len(train_paths)}] {path}")
        index = _sid_load_index(path)
        images = _sid_load_images(path)
        for row_idx, (img_id, raw_label) in enumerate(index):
            if raw_label == 2:  # tampered/locally edited -- out of scope
                counts["dropped_tampered"] += 1
                continue
            image_bytes = images.column("image")[row_idx].as_py()["bytes"]
            cls = "real" if raw_label == 0 else "fake"
            _save(image_bytes, TRAIN_DIR / cls, f"sidset-{path}-{img_id}")
            counts[cls] += 1
    print(f"  train: {counts}")

    holdout_real = 0
    for i, path in enumerate(holdout_paths):
        print(f"  [holdout {i + 1}/{len(holdout_paths)}] {path}")
        index = _sid_load_index(path)
        images = _sid_load_images(path)
        for row_idx, (img_id, raw_label) in enumerate(index):
            if raw_label != 0:  # only need the real class here
                continue
            image_bytes = images.column("image")[row_idx].as_py()["bytes"]
            _save(image_bytes, HOLDOUT_DIR / "real", f"sidset-holdout-{path}-{img_id}")
            holdout_real += 1
    print(f"  dragon_holdout real: {holdout_real}")


def build_dragon() -> None:
    print("\n=== DRAGON: 25 generators, 17 train / 8 held out ===")
    paths = _dragon_shard_paths("Regular", 10)
    counts_train: dict[str, int] = defaultdict(int)
    counts_holdout: dict[str, int] = defaultdict(int)
    for i, path in enumerate(paths):
        print(f"  [{i + 1}/{len(paths)}] {path}")
        names = _dragon_load_index(path)
        images = _dragon_load_images(path)
        for row_idx, model_name in enumerate(names):
            image_bytes = images.column("png")[row_idx].as_py()["bytes"]
            key = f"dragon-{path}-{row_idx}"
            if model_name in DRAGON_HELD_OUT:
                _save(image_bytes, HOLDOUT_DIR / "fake", key)
                counts_holdout[model_name] += 1
            else:
                _save(image_bytes, TRAIN_DIR / "fake", key)
                counts_train[model_name] += 1
    print(f"  train generators: {dict(counts_train)}")
    print(f"  holdout generators: {dict(counts_holdout)}")


def build_genimage() -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem

    fs = HfFileSystem()
    api = HfApi()
    print("\n=== GenImage: ADM + GLIDE (pixel-space diffusion gap) ===")
    for name, repo_id in GENIMAGE_REPOS.items():

        def _list_files() -> list[str]:
            return sorted(
                f for f in api.list_repo_files(repo_id, repo_type="dataset") if f.endswith(".parquet")
            )

        files = retry_network_call(_list_files, description=f"{name} file listing")
        print(f"  {name}: {len(files)} shard(s) available")

        collected = 0
        for path in files:
            if collected >= GENIMAGE_PER_GENERATOR:
                break

            def _read(p=path) -> "pq.Table":
                return pq.read_table(f"datasets/{repo_id}/{p}", columns=["image"], filesystem=fs)

            table = retry_network_call(_read, description=f"{name} shard fetch ({path})")
            col = table.column("image")
            for row_idx in range(len(col)):
                if collected >= GENIMAGE_PER_GENERATOR:
                    break
                image_bytes = col[row_idx].as_py()["bytes"]
                _save(image_bytes, TRAIN_DIR / "fake", f"genimage-{name}-{path}-{row_idx}")
                collected += 1
            print(f"    {path}: {collected}/{GENIMAGE_PER_GENERATOR} collected")


def main() -> int:
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)

    build_sid_set(SID_SET_TRAIN_SHARDS, SID_SET_HOLDOUT_SHARD_START, SID_SET_HOLDOUT_SHARDS)
    build_dragon()
    build_genimage()

    for split_dir in (TRAIN_DIR, HOLDOUT_DIR):
        real_n = len(list((split_dir / "real").glob("*.jpg"))) if (split_dir / "real").exists() else 0
        fake_n = len(list((split_dir / "fake").glob("*.jpg"))) if (split_dir / "fake").exists() else 0
        print(f"\n{split_dir}: real={real_n} fake={fake_n}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
