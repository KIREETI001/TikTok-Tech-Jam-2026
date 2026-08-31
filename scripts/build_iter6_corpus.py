"""Materialize the iteration-6 training corpus to local disk: the teammate's
actual iter4 recipe (SID full-synthetic + 17 DRAGON generators +
Community-Forensics-Small) plus this device's GenImage ADM/GLIDE addition
for the pixel-space diffusion gap (see docs/EXPERIMENTS_LOG.md section 11).

Community-Forensics-Small was skipped for iteration 5 -- the HF auto-
converted preview export (refs/convert/parquet) showed only ~10.5k of the
real ~556k rows and looked unreliable under overnight time pressure. That
was a wrong read of a partial preview, not a real data problem: the actual
per-architecture shard files (data/HFCF_small_*.parquet on the main branch)
are complete, cleanly organised (one architecture family per shard), and
this is what iter4's own reported 0.9126 organiser Final Score was built
on (see docs/EXPERIMENTS_LOG.md section 9j). scripts/probe_cf_small_index.py
built the full 186-shard manifest (scripts/cf_small_index.json) this reads.

SID_Set and DRAGON are NOT re-streamed -- ``../iter5_data/train`` already
has both (iter5's corpus = SID_Set + DRAGON-17 + GenImage, iter6's target =
SID_Set + DRAGON-17 + CF-Small + GenImage), and HfFileSystem streaming
doesn't go through huggingface_hub's local disk cache, so a naive re-run
would re-download everything over the network for no reason. Instead this
recomputes the same content-hashed filenames build_iter5_corpus.py used
(index-only metadata, no image bytes) and copies the matching files --
GenImage too, since it's also already local and unchanged for iter6.

Only Community-Forensics-Small is actually fetched fresh here.

Produces:
    ../iter6_data/train/{real,fake}/  -- training corpus

Usage:
    ./.venv/Scripts/python.exe scripts/build_iter6_corpus.py
"""

from __future__ import annotations

import hashlib
import io
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import HfApi, HfFileSystem  # noqa: E402
from PIL import Image  # noqa: E402

from detector.data_sources._network import retry_network_call  # noqa: E402
from detector.data_sources.dragon import (  # noqa: E402
    _load_shard_index as _dragon_load_index,
    _shard_paths as _dragon_shard_paths,
)
from detector.data_sources.sid_set_stream import (  # noqa: E402
    _load_shard_index as _sid_load_index,
    _shard_paths as _sid_shard_paths,
)

IN_ROOT = Path("../iter5_data").resolve()
IN_TRAIN = IN_ROOT / "train"
OUT_ROOT = Path("../iter6_data").resolve()
TRAIN_DIR = OUT_ROOT / "train"

# Must match build_iter5_corpus.py exactly -- these produce the same
# content-hash filenames so the matching files can be located and copied.
SID_SET_TRAIN_SHARDS = 30
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


def _hash_name(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:20] + ".jpg"


def _copy_matching(keys_by_class: dict[str, list[str]]) -> dict[str, int]:
    """Copy files from IN_TRAIN to TRAIN_DIR whose content-hash filename
    matches one of ``keys_by_class``'s recomputed keys. Returns counts of
    how many were found (a miss means the source file was never fetched
    for iter5, e.g. it was dropped as tampered/label-2).
    """
    counts = {"real": 0, "fake": 0, "missing": 0}
    for cls, keys in keys_by_class.items():
        (TRAIN_DIR / cls).mkdir(parents=True, exist_ok=True)
        for key in keys:
            name = _hash_name(key)
            src = IN_TRAIN / cls / name
            if not src.exists():
                counts["missing"] += 1
                continue
            dst = TRAIN_DIR / cls / name
            if not dst.exists():
                shutil.copyfile(src, dst)
            counts[cls] += 1
    return counts


def reuse_sid_set() -> None:
    print(f"\n=== Reusing SID_Set ({SID_SET_TRAIN_SHARDS} shards) from {IN_TRAIN} ===")
    paths = _sid_shard_paths("train", SID_SET_TRAIN_SHARDS)
    keys = {"real": [], "fake": []}
    for i, path in enumerate(paths):
        print(f"  [{i + 1}/{len(paths)}] index: {path}")
        for img_id, raw_label in _sid_load_index(path):
            if raw_label == 2:
                continue
            cls = "real" if raw_label == 0 else "fake"
            keys[cls].append(f"sidset-{path}-{img_id}")
    counts = _copy_matching(keys)
    print(f"  copied: {counts}")


def reuse_dragon() -> None:
    print(f"\n=== Reusing DRAGON 17-generator train split from {IN_TRAIN} ===")
    paths = _dragon_shard_paths("Regular", 10)
    keys: list[str] = []
    for i, path in enumerate(paths):
        print(f"  [{i + 1}/{len(paths)}] index: {path}")
        for row_idx, model_name in enumerate(_dragon_load_index(path)):
            if model_name in DRAGON_HELD_OUT:
                continue
            keys.append(f"dragon-{path}-{row_idx}")
    counts = _copy_matching({"fake": keys})
    print(f"  copied: {counts}")


def reuse_genimage() -> None:
    print(f"\n=== Reusing GenImage ADM/GLIDE from {IN_TRAIN} ===")
    fs = HfFileSystem()
    api = HfApi()
    keys: list[str] = []
    for name, repo_id in GENIMAGE_REPOS.items():
        files = retry_network_call(
            lambda: sorted(f for f in api.list_repo_files(repo_id, repo_type="dataset") if f.endswith(".parquet")),
            description=f"{name} file listing",
        )
        collected = 0
        for path in files:
            if collected >= GENIMAGE_PER_GENERATOR:
                break
            n_rows = retry_network_call(
                lambda p=path: pq.ParquetFile(f"datasets/{repo_id}/{p}", filesystem=fs).metadata.num_rows,
                description=f"{name} row count ({path})",
            )
            for row_idx in range(n_rows):
                if collected >= GENIMAGE_PER_GENERATOR:
                    break
                keys.append(f"genimage-{name}-{path}-{row_idx}")
                collected += 1
    counts = _copy_matching({"fake": keys})
    print(f"  copied: {counts}")


def build_cf_small() -> None:
    print(f"\n=== Community-Forensics-Small: {len(CF_SMALL_REAL_SHARDS)} real + "
          f"{len(CF_SMALL_FAKE_SHARDS)} fake shards (fresh fetch) ===")
    fs = HfFileSystem()
    counts = {"real": 0, "fake": 0}
    for cls, shards in (("real", CF_SMALL_REAL_SHARDS), ("fake", CF_SMALL_FAKE_SHARDS)):
        (TRAIN_DIR / cls).mkdir(parents=True, exist_ok=True)
        for shard in shards:
            hf_path = f"datasets/{CF_SMALL_REPO}/data/{shard}"
            print(f"  fetching {shard} ({cls}) ...", flush=True)
            table = retry_network_call(
                lambda p=hf_path: pq.read_table(p, columns=["image_data", "label"], filesystem=fs),
                description=f"CF-Small shard fetch ({shard})",
            )
            image_col = table.column("image_data")
            label_col = table.column("label")
            n = len(image_col)
            for row_idx in range(n):
                label = label_col[row_idx].as_py()
                row_cls = "real" if label == 0 else "fake"
                image_bytes = image_col[row_idx].as_py()
                key = f"cfsmall-{shard}-{row_idx}"
                name = _hash_name(key)
                out_path = TRAIN_DIR / row_cls / name
                if out_path.exists():
                    continue
                try:
                    with Image.open(io.BytesIO(image_bytes)) as img:
                        img.convert("RGB").save(out_path, format="JPEG", quality=95)
                except Exception as exc:  # noqa: BLE001 - one bad row shouldn't kill 3k
                    print(f"    skip row {row_idx} in {shard}: {exc}")
                    continue
                counts[row_cls] += 1
            print(f"    {shard}: {n} rows processed, running totals {counts}", flush=True)
    print(f"  CF-Small totals: {counts}")


def main() -> int:
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    reuse_sid_set()
    reuse_dragon()
    reuse_genimage()
    build_cf_small()

    real_n = len(list((TRAIN_DIR / "real").glob("*.jpg")))
    fake_n = len(list((TRAIN_DIR / "fake").glob("*.jpg")))
    print(f"\n{TRAIN_DIR}: real={real_n} fake={fake_n} total={real_n + fake_n}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
