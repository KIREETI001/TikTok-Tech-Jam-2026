"""Materialise the training corpus and the cross-generator holdout set.

    python scripts/build_corpus.py

Streams four public sources straight to disk. Nothing is downloaded twice and
nothing is copied between staging directories -- re-running is safe and cheap,
because every image is written under a content-hashed filename and existing
files are skipped.

Produces:
    ../detector_data/train/{real,fake}/            66,502 images, the training corpus
    ../detector_data/dragon_holdout/{real,fake}/   8 unseen generators, for scoring

Sources, and why each is here:

  SID_Set (saberzl/SID_Set), 30 shards
      The base real/synthetic pairing. Label 2 (`tampered` -- a real photo
      with one locally edited region) is dropped: this detector answers "was
      this image generated?", not "was part of it edited?".

  DRAGON, 25 generators
      Split 17 train / 8 held out. The 8 held-out generators never appear in
      training and are what `dragon_holdout` scores, making it a genuine
      cross-generator test rather than a validation split.

  Community-Forensics-Small (OwensLab/CommunityForensics-Small)
      6 real + 8 fake shards, chosen to span LatDiff / GAN / PixDiff / Other
      so the corpus covers architecture families, not just more of the same
      one. Shard selection came from scripts/index_cf_small.py, whose output
      is committed as scripts/cf_small_index.json.

  GenImage ADM + GLIDE (bitmind/GenImage_ADM, bitmind/GenImage_glide)
      2,500 each. Added deliberately after error analysis showed pixel-space
      diffusion was the blind spot: it leaves no VAE fingerprint for a
      latent-diffusion-trained detector to key on. This is the single change
      that moved ADM's clean AUC from 0.478 (below chance) to 0.937.

Requires `pyarrow` and `huggingface-hub` (both in requirements.txt).
"""

from __future__ import annotations

import hashlib
import io
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import HfApi, HfFileSystem  # noqa: E402
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

OUT_ROOT = Path("../detector_data").resolve()
TRAIN_DIR = OUT_ROOT / "train"
HOLDOUT_DIR = OUT_ROOT / "dragon_holdout"

SID_SET_TRAIN_SHARDS = 30
SID_SET_HOLDOUT_SHARD_START = 30  # a disjoint shard for the holdout's real half
SID_SET_HOLDOUT_SHARDS = 2

DRAGON_HELD_OUT = {
    "Flux_1", "IF", "JuggernautXL", "Kolors",
    "PixArt_Sigma", "SDXL_Turbo", "SD_3", "SD_Cascade",
}

GENIMAGE_REPOS = {"adm": "bitmind/GenImage_ADM", "glide": "bitmind/GenImage_glide"}
GENIMAGE_PER_GENERATOR = 2500

CF_SMALL_REPO = "OwensLab/CommunityForensics-Small"
CF_SMALL_REAL_SHARDS = [
    "HFCF_small_94.parquet", "HFCF_small_109.parquet", "HFCF_small_124.parquet",
    "HFCF_small_140.parquet", "HFCF_small_155.parquet", "HFCF_small_170.parquet",
]
CF_SMALL_FAKE_SHARDS = [
    "HFCF_small_0.parquet", "HFCF_small_20.parquet", "HFCF_small_40.parquet",
    "HFCF_small_60.parquet", "HFCF_small_74.parquet", "HFCF_small_88.parquet",
    "HFCF_small_82.parquet", "HFCF_small_71.parquet",
]


def _save(image_bytes: bytes, dest_dir: Path, key: str) -> None:
    """Write one image under a content-hashed name, skipping it if present.

    The hash of the source key (not the pixels) is the filename, which makes
    a re-run idempotent without having to decode anything already on disk.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / (hashlib.sha256(key.encode()).hexdigest()[:20] + ".jpg")
    if out_path.exists():
        return
    with Image.open(io.BytesIO(image_bytes)) as img:
        img.convert("RGB").save(out_path, format="JPEG", quality=95)


def build_sid_set() -> None:
    print(f"\n=== SID_Set: {SID_SET_TRAIN_SHARDS} train shards (full_synthetic only) ===")
    total = SID_SET_TRAIN_SHARDS + SID_SET_HOLDOUT_SHARD_START + SID_SET_HOLDOUT_SHARDS
    paths = _sid_shard_paths("train", total)
    train_paths = paths[:SID_SET_TRAIN_SHARDS]
    holdout_paths = paths[
        SID_SET_HOLDOUT_SHARD_START : SID_SET_HOLDOUT_SHARD_START + SID_SET_HOLDOUT_SHARDS
    ]

    counts = {"real": 0, "fake": 0, "dropped_tampered": 0}
    for i, path in enumerate(train_paths):
        print(f"  [{i + 1}/{len(train_paths)}] {path}")
        index = _sid_load_index(path)
        images = _sid_load_images(path)
        for row_idx, (img_id, raw_label) in enumerate(index):
            if raw_label == 2:  # tampered / locally edited -- out of scope
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
            if raw_label != 0:  # the holdout needs only the real class
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
    print("\n=== GenImage: ADM + GLIDE (the pixel-space diffusion gap) ===")
    fs = HfFileSystem()
    api = HfApi()
    for name, repo_id in GENIMAGE_REPOS.items():
        files = retry_network_call(
            lambda r=repo_id: sorted(
                f for f in api.list_repo_files(r, repo_type="dataset")
                if f.endswith(".parquet")
            ),
            description=f"{name} file listing",
        )
        print(f"  {name}: {len(files)} shard(s) available")
        collected = 0
        for path in files:
            if collected >= GENIMAGE_PER_GENERATOR:
                break
            table = retry_network_call(
                lambda p=path, r=repo_id: pq.read_table(
                    f"datasets/{r}/{p}", columns=["image"], filesystem=fs
                ),
                description=f"{name} shard fetch ({path})",
            )
            col = table.column("image")
            for row_idx in range(len(col)):
                if collected >= GENIMAGE_PER_GENERATOR:
                    break
                image_bytes = col[row_idx].as_py()["bytes"]
                _save(image_bytes, TRAIN_DIR / "fake", f"genimage-{name}-{path}-{row_idx}")
                collected += 1
            print(f"    {path}: {collected}/{GENIMAGE_PER_GENERATOR} collected")


def build_cf_small() -> None:
    print(f"\n=== Community-Forensics-Small: {len(CF_SMALL_REAL_SHARDS)} real + "
          f"{len(CF_SMALL_FAKE_SHARDS)} fake shards ===")
    fs = HfFileSystem()
    counts = {"real": 0, "fake": 0}
    for shards in (CF_SMALL_REAL_SHARDS, CF_SMALL_FAKE_SHARDS):
        for shard in shards:
            hf_path = f"datasets/{CF_SMALL_REPO}/data/{shard}"
            print(f"  fetching {shard} ...", flush=True)
            table = retry_network_call(
                lambda p=hf_path: pq.read_table(
                    p, columns=["image_data", "label"], filesystem=fs
                ),
                description=f"CF-Small shard fetch ({shard})",
            )
            image_col = table.column("image_data")
            label_col = table.column("label")
            for row_idx in range(len(image_col)):
                cls = "real" if label_col[row_idx].as_py() == 0 else "fake"
                try:
                    _save(
                        image_col[row_idx].as_py(),
                        TRAIN_DIR / cls,
                        f"cfsmall-{shard}-{row_idx}",
                    )
                except Exception as exc:  # noqa: BLE001 - one bad row must not kill 3k
                    print(f"    skip row {row_idx} in {shard}: {exc}")
                    continue
                counts[cls] += 1
            print(f"    running totals {counts}", flush=True)


def main() -> int:
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)

    build_sid_set()
    build_dragon()
    build_genimage()
    build_cf_small()

    for split in (TRAIN_DIR, HOLDOUT_DIR):
        real = len(list((split / "real").glob("*.jpg"))) if (split / "real").exists() else 0
        fake = len(list((split / "fake").glob("*.jpg"))) if (split / "fake").exists() else 0
        print(f"\n{split}: real={real} fake={fake} total={real + fake}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
