"""Measure how much of the real/fake label is recoverable WITHOUT looking at
any generation artifact -- i.e. how much of our accuracy could be a dataset
shortcut rather than detection.

Ported from ziyangchua02/model_training's shortcut_probe.py. The idea: train
gradient-boosted trees on properties that carry no evidence of generation
(file metadata, and global pixel statistics of the exact crop the model is
fed). Whatever those recover is label information available to a detector
without it detecting anything. Chance is 50%.

Two probes, and the distinction between them matters:

  metadata  -- width, height, aspect ratio, file format, file size. These
               reach the model only insofar as they survive preprocessing;
               Resize+CenterCrop erases the geometry before the tensor is
               built, so a high number here flags a corpus-level confound
               (e.g. every fake is a 1024x1024 PNG and every real a 3:2
               JPEG) rather than something the detector is currently using.
  pixels    -- statistics of the 224px centre crop itself: per-channel mean
               and standard deviation, saturation, edge density, and bytes
               per pixel after a fixed-quality JPEG re-encode. This is the
               probe that constrains what our detector could be doing,
               because every one of these properties genuinely reaches it.

Deliberately a crop, not a thumbnail: resizing to a common size would
resample every image by an amount depending on its original resolution,
changing edge density and compressibility -- which would make the probe
measure the resize rather than the corpus.

Diagnostic only. Trains no detector, touches no checkpoint.

Usage:
    ./.venv/Scripts/python.exe scripts/shortcut_probe.py \\
        --root ../Data/test --label "PS5 test" --limit 2000
"""

from __future__ import annotations

import argparse
import io
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score

SEED = 2026
CROP = 224

# Deliberately NOT importing detector.data, which would pull in torch.
#
# The first version of this script did, for ImageRecord and
# load_labeled_root, and it took the machine down: sklearn's joblib workers
# (n_jobs=5 for cross-validation, 4 for permutation importance) each spawn a
# subprocess that re-imports the parent module, so every worker loaded ~2GB
# of CUDA DLLs to use a dataclass and a directory walk. With a training run
# already holding the GPU, that exhausted the Windows paging file
# (WinError 1455) and killed unrelated jobs.
#
# A diagnostic that reads images and fits a small tree model has no business
# depending on the training stack, so it now carries its own 15-line loader
# and runs single-process. Both probes fit in seconds on ~1.5k rows x 10
# features; n_jobs bought nothing.
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})
REAL_FOLDERS = frozenset({"real", "authentic", "non_aigc"})
FAKE_FOLDERS = frozenset({"fake", "ai", "aigc"})


@dataclass(frozen=True)
class ImageRecord:
    """Local stand-in for detector.data.ImageRecord: path and label only.

    No checksum -- this probe does not deduplicate, because it measures the
    corpus as the model sees it, duplicates included.
    """

    path: Path
    label: int


def load_labeled_root(root: str | Path) -> list[ImageRecord]:
    """Find real/ and fake/ (or ai/) subfolders and label their images."""
    root_path = Path(root)
    if not root_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {root_path}")

    records: list[ImageRecord] = []
    for child in sorted(root_path.iterdir()):
        if not child.is_dir():
            continue
        name = child.name.strip().lower().replace("-", "_").replace(" ", "_")
        if name in REAL_FOLDERS:
            label = 0
        elif name in FAKE_FOLDERS:
            label = 1
        else:
            continue
        for path in sorted(child.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(ImageRecord(path=path, label=label))

    if not records:
        raise ValueError(f"No labeled images under {root_path} (need real/ and fake/ subfolders).")
    return records


def balance(records: list[ImageRecord], seed: int = SEED) -> list[ImageRecord]:
    """Equal counts per class, so 50% is genuinely chance.

    Without this, a probe on an unbalanced corpus scores the majority-class
    rate and reads as leakage that is really just the prior.
    """
    rng = random.Random(seed)
    by_label: dict[int, list[ImageRecord]] = {}
    for record in records:
        by_label.setdefault(record.label, []).append(record)
    if len(by_label) < 2:
        raise ValueError(f"Need both classes; found labels {sorted(by_label)}.")
    smallest = min(len(v) for v in by_label.values())
    out: list[ImageRecord] = []
    for label in sorted(by_label):
        pool = by_label[label]
        rng.shuffle(pool)
        out.extend(pool[:smallest])
    rng.shuffle(out)
    return out


def metadata_features(records: list[ImageRecord]) -> tuple[np.ndarray, list[str]]:
    """Properties of the original file, before any preprocessing."""
    formats: dict[str, int] = {}
    rows = []
    for record in records:
        with Image.open(record.path) as image:
            width, height = image.size
            fmt = image.format or "?"
        code = formats.setdefault(fmt, len(formats))
        size_bytes = record.path.stat().st_size
        rows.append(
            [
                float(width),
                float(height),
                width / max(height, 1),
                float(width * height),
                float(code),
                float(size_bytes),
                size_bytes / max(width * height, 1),
            ]
        )
    names = ["width", "height", "aspect", "pixels", "format", "file_bytes", "bytes_per_pixel"]
    return np.array(rows, dtype=np.float64), names


def pixel_features(records: list[ImageRecord]) -> tuple[np.ndarray, list[str]]:
    """Statistics of the exact 224px crop the model is fed."""
    rows = []
    for record in records:
        with Image.open(record.path) as source:
            image = source.convert("RGB")
            width, height = image.size
            if min(width, height) < CROP:
                scale = CROP / min(width, height)
                image = image.resize(
                    (max(CROP, round(width * scale)), max(CROP, round(height * scale))),
                    Image.Resampling.BILINEAR,
                )
                width, height = image.size
            left, top = (width - CROP) // 2, (height - CROP) // 2
            image = image.crop((left, top, left + CROP, top + CROP))

            pixels = np.asarray(image, dtype=np.float32) / 255.0
            gray = pixels.mean(axis=2)
            edges = (
                np.asarray(
                    Image.fromarray((gray * 255).astype(np.uint8)).filter(ImageFilter.FIND_EDGES),
                    dtype=np.float32,
                )
                / 255.0
            )

            # Re-encoding at one fixed quality collapses "how much detail is
            # in this image" into a single number: smooth, low-detail images
            # compress smaller. Generators and cameras differ on this, and it
            # is the feature that carried our measured SID_Set compression gap.
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=75)

            brightest = pixels.max(axis=2)
            darkest = pixels.min(axis=2)
            saturation = (brightest - darkest) / np.maximum(brightest, 1e-6)

            rows.append(
                [
                    *pixels.mean(axis=(0, 1)),
                    *pixels.std(axis=(0, 1)),
                    float(saturation.mean()),
                    float(edges.mean()),
                    float(edges.std()),
                    buffer.getbuffer().nbytes / (CROP * CROP),
                ]
            )
    names = [
        "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
        "saturation", "edge_mean", "edge_std", "recompressed_bpp",
    ]
    return np.array(rows, dtype=np.float64), names


def run_probe(
    name: str, X: np.ndarray, y: np.ndarray, feature_names: list[str], folds: int = 5
) -> dict:
    # n_jobs=1 throughout: see the module-level note on WinError 1455.
    model = HistGradientBoostingClassifier(max_iter=200, random_state=SEED)
    scores = cross_val_score(model, X, y, cv=folds, scoring="accuracy", n_jobs=1)
    accuracy = scores.mean() * 100

    model.fit(X, y)
    importance = permutation_importance(model, X, y, n_repeats=5, random_state=SEED, n_jobs=1)
    ranked = sorted(
        zip(feature_names, importance.importances_mean), key=lambda kv: kv[1], reverse=True
    )

    print(f"\n  {name}: {accuracy:.2f}%  (+/- {scores.std() * 100:.2f})   chance = 50%")
    top = ", ".join(f"{n} {v:+.3f}" for n, v in ranked[:4] if v > 0)
    print(f"    top features: {top or '(none above zero)'}")
    return {
        "accuracy": accuracy,
        "std": scores.std() * 100,
        "top_features": [(n, float(v)) for n, v in ranked[:6]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", required=True, help="labeled folder with real/ and fake/ (or ai/)")
    parser.add_argument("--label", default=None, help="name for this corpus in the output")
    parser.add_argument("--limit", type=int, default=2000, help="max images after balancing")
    parser.add_argument("--out", default=None, help="write JSON here")
    args = parser.parse_args()

    label = args.label or args.root
    print(f"[PROBE] {label}  <- {args.root}")

    records = balance(load_labeled_root(args.root))
    if args.limit and len(records) > args.limit:
        records = records[: args.limit]
    counts = Counter(r.label for r in records)
    print(f"  {len(records)} images, balanced: {counts[0]} real / {counts[1]} fake")

    y = np.array([r.label for r in records])
    meta_X, meta_names = metadata_features(records)
    pixel_X, pixel_names = pixel_features(records)

    result = {
        "corpus": label,
        "root": args.root,
        "n_images": len(records),
        "metadata_probe": run_probe("metadata (pre-preprocessing)", meta_X, y, meta_names),
        "pixel_probe": run_probe("pixel stats (what the model sees)", pixel_X, y, pixel_names),
    }

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\n  written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
