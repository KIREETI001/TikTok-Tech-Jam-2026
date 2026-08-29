"""A-3: how much of the REAL/FAKE label is recoverable from things that are
not generation artifacts?

    python -m detector.shortcut_probe name=dir [name=dir ...]

Three probes, each a gradient-boosted tree with 5-fold CV, on class-balanced
data so 50% is chance:

  metadata   -- original width, height, aspect, megapixels, JPEG size. Our
                materialised sets are all re-saved 448px JPEG, so this mostly
                measures a shortcut already removed; a high number is expected
                and a *low* one is the interesting case.
  pixel      -- per-channel mean/std, saturation, edge density, re-encode
                size of the exact 224 crop the model sees. This is the floor
                on how much a trained detector's score needs no detection at
                all. Above ~65% the corpus teaches something off-task.
  dct_hf     -- mean log-magnitude of the 2D DCT outside a central radius,
                per channel. The specific "real images carry JPEG history,
                fakes don't" bias DDA (NeurIPS 2025) names. A high number here
                means training will learn "high-frequency energy = real/fake"
                rather than an artifact.

A low score rules out these particular shortcuts. It cannot rule out a
pixel-level source fingerprint -- that needs the held-out-generator eval.
"""

from __future__ import annotations

import argparse
import io
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy.fftpack import dct
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

EXT = frozenset({".jpg", ".jpeg", ".png", ".webp"})
SEED = 42
PER_CLASS = 800


def _balanced(root: Path):
    rng = random.Random(SEED)
    out = []
    for cls, label in (("real", 0), ("fake", 1)):
        files = [p for p in (root / cls).glob("*") if p.suffix.lower() in EXT]
        rng.shuffle(files)
        out.append([(p, label) for p in files[:PER_CLASS]])
    n = min(len(out[0]), len(out[1]))
    items = out[0][:n] + out[1][:n]
    rng.shuffle(items)
    return items


def _metadata_feats(path: Path) -> list[float]:
    with Image.open(path) as im:
        w, h = im.size
        fmt = {"JPEG": 0, "PNG": 1, "WEBP": 2}.get(im.format or "", 3)
    return [w, h, w / max(h, 1), w * h / 1e6, path.stat().st_size / 1024, fmt]


def _center_224(im: Image.Image) -> Image.Image:
    im = im.convert("RGB")
    s = 256 / min(im.size)
    im = im.resize((max(256, round(im.width * s)), max(256, round(im.height * s))), Image.BILINEAR)
    left = (im.width - 224) // 2
    top = (im.height - 224) // 2
    return im.crop((left, top, left + 224, top + 224))


def _pixel_feats(path: Path) -> list[float]:
    crop = _center_224(Image.open(path))
    arr = np.asarray(crop, dtype=np.float32) / 255.0
    feats = []
    for c in range(3):
        feats += [float(arr[..., c].mean()), float(arr[..., c].std())]
    mx = arr.max(-1); mn = arr.min(-1)
    feats.append(float((np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)).mean()))  # saturation
    edges = np.asarray(crop.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    feats.append(float(edges.mean()))
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=90)
    feats.append(len(buf.getvalue()) / 1024)
    return feats


def _dct_hf_feats(path: Path, radius_frac: float = 0.25) -> list[float]:
    crop = _center_224(Image.open(path))
    arr = np.asarray(crop, dtype=np.float32)
    feats = []
    yy, xx = np.mgrid[0:224, 0:224]
    outside = (yy + xx) > (224 * radius_frac)  # low-frequency triangle excluded
    for c in range(3):
        d = dct(dct(arr[..., c], axis=0, norm="ortho"), axis=1, norm="ortho")
        feats.append(float(np.log1p(np.abs(d[outside])).mean()))
    return feats


PROBES = {
    "metadata": _metadata_feats,
    "pixel": _pixel_feats,
    "dct_hf": _dct_hf_feats,
}


def run(name: str, root: Path) -> None:
    items = _balanced(root)
    if not items:
        print(f"{name}: no data")
        return
    y = np.array([lbl for _, lbl in items])
    print(f"\n=== {name}  ({(y == 0).sum()} real / {(y == 1).sum()} fake) ===")
    for probe, fn in PROBES.items():
        X = []
        for path, _ in items:
            try:
                X.append(fn(path))
            except Exception:
                X.append([0.0] * len(X[0]) if X else [0.0])
        X = np.asarray(X, dtype=np.float32)
        clf = HistGradientBoostingClassifier(random_state=SEED, max_iter=200)
        acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy").mean()
        flag = "  <- OFF-TASK" if (probe != "metadata" and acc > 0.65) else ""
        print(f"  {probe:<10} {acc * 100:5.1f}%{flag}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("specs", nargs="+", help="name=dir")
    args = ap.parse_args()
    for spec in args.specs:
        name, d = spec.split("=", 1)
        run(name, Path(d))
    print("\n(chance = 50%. metadata high is expected -- our sets are re-encoded. "
          "pixel/dct_hf above ~65% means the corpus teaches a non-artifact cue.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
