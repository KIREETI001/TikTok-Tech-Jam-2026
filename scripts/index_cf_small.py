"""One-time index build over OwensLab/CommunityForensics-Small's real shard
files (data/HFCF_small_*.parquet on the main branch -- NOT the HF auto-
converted refs/convert/parquet preview, which only exposes ~10.5k of the
full ~550k+ rows and is what earlier reconnaissance mistakenly relied on).

Reads only [label, architecture, subset, real_source] columns per shard --
no image bytes -- to build a manifest of which shard holds which
architecture/class, and how many rows, before deciding a sampling plan for
the iter6 corpus. Writes scripts/cf_small_index.json incrementally so a
network hiccup partway through doesn't lose earlier shards' work.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

REPO = "OwensLab/CommunityForensics-Small"
OUT = Path(__file__).resolve().parent / "cf_small_index.json"

fs = HfFileSystem()
api = HfApi()


def main() -> int:
    manifest: dict[str, dict] = {}
    if OUT.exists():
        manifest = json.loads(OUT.read_text(encoding="utf-8"))
        print(f"Resuming: {len(manifest)} shards already indexed.")

    all_files = api.list_repo_files(REPO, repo_type="dataset")
    shards = sorted(f for f in all_files if f.startswith("data/HFCF_small_") and f.endswith(".parquet"))
    print(f"{len(shards)} shard files total.")

    for i, rel_path in enumerate(shards):
        name = Path(rel_path).name
        if name in manifest:
            continue
        hf_path = f"datasets/{REPO}/{rel_path}"
        try:
            t = pq.read_table(hf_path, columns=["label", "architecture", "subset"], filesystem=fs)
        except Exception as exc:  # noqa: BLE001 - log and continue, don't lose progress
            print(f"  [{i+1}/{len(shards)}] {name}: FAILED ({exc})", flush=True)
            continue
        labels = t.column("label").to_pylist()
        archs = t.column("architecture").to_pylist()
        n = len(labels)
        n_real = sum(1 for l in labels if l == 0)
        n_fake = n - n_real
        arch_set = sorted(set(archs))
        manifest[name] = {"rows": n, "real": n_real, "fake": n_fake, "architectures": arch_set}
        print(f"  [{i+1}/{len(shards)}] {name}: rows={n} real={n_real} fake={n_fake} arch={arch_set}", flush=True)
        OUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nDone. Indexed {len(manifest)} shards -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
