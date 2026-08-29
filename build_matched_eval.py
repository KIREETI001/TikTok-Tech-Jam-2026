"""Build a resolution-matched copy of the organisers' evaluation composition.

    python build_matched_eval.py

Why
---
WildFake ships its generated images at fixed sizes (ADM/DDIM/DDPM/VQDM are
256x256; DALLE/Imagen are larger) and the authentic half comes from COCO at
photographic resolutions. On the set as-composed, image size alone separates
four of the six generators perfectly -- every 256x256 fake is smaller than
every COCO real. Any score on that set mixes "can it detect generation" with
"can it read a file header", and the two cannot be told apart.

This writes a copy in which every image is exactly ``SIDE x SIDE``, so the
size term is zero and what remains is detection.

Matching is done by **centre-cropping native pixels**, never by resizing.
Resizing would resample one class more than the other, replacing a size cue
with a resampling cue -- the same mistake in a different coat. Every pixel
that survives here is an original pixel. The cost is field of view on the
larger images, which is the right cost to accept: a generator's fingerprint
is a local statistic.

Both classes then meet the evaluation transform as identically-shaped inputs,
so whatever resampling it does, it does equally to each.

Also writes a size-only AUC for the source and matched sets: a classifier
that sees nothing but the pixel count. 0.5 means size carries no label
information; anything near 0 or 1 means the set is measuring file headers.

*** EVALUATION ONLY -- see EVAL_ONLY_DATASETS.md. ***
"""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

DATA = Path(r"C:\Users\attil\ttj-data")

# The smallest common side across both classes. WildFake's own authentic half
# (Images/Real/coco.zip) ships at 200x200 and its generated images at 256+, so
# 200 is the largest square every image can supply from native pixels. Both
# classes then go through the evaluation transform identically.
SIDE = 200


def center_square(image: Image.Image, side: int) -> Image.Image | None:
    """Largest centred ``side x side`` crop of native pixels, or None if the
    image is too small to give one without upscaling."""

    if min(image.size) < side:
        return None
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    return image.crop((left, top, left + side, top + side))


def build(src_dir: Path, dst_dir: Path, limit: int | None = None) -> tuple[int, int]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    kept = skipped = 0
    for path in sorted(src_dir.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        if limit is not None and kept >= limit:
            break
        with Image.open(path) as source:
            image = source.convert("RGB")
            crop = center_square(image, SIDE)
        if crop is None:
            skipped += 1
            continue
        crop.save(dst_dir / (path.stem + ".jpg"), quality=95)
        kept += 1
    return kept, skipped


def size_only_auc(real_dir: Path, fake_dir: Path) -> float:
    """AUC of a classifier that sees nothing but each image's pixel count."""

    from detector.evaluation import _roc_auc

    def pixels(directory: Path) -> list[float]:
        out = []
        for path in directory.glob("*.jpg"):
            with Image.open(path) as image:
                out.append(float(image.width * image.height))
        return out

    real, fake = pixels(real_dir), pixels(fake_dir)
    return _roc_auc([0] * len(real) + [1] * len(fake), real + fake)


def main() -> None:
    src_real = DATA / "eval_only_wfcoco_native" / "real"
    src_fake = DATA / "eval_only_wildfake_native" / "fake"

    print(f"source size-only AUC: {size_only_auc(src_real, src_fake):.4f}"
          "   (0.5 = size is uninformative)")

    out = DATA / "eval_only_organisers_matched"
    if out.exists():
        shutil.rmtree(out)

    real_kept, real_skipped = build(src_real, out / "real", limit=1200)
    print(f"real:  kept {real_kept}, skipped {real_skipped} (short side < {SIDE})")

    fake_kept, fake_skipped = build(src_fake, out / "fake")
    print(f"fake:  kept {fake_kept}, skipped {fake_skipped} (short side < {SIDE})")

    import collections

    generators = collections.Counter(
        p.name.split("_")[0] for p in (out / "fake").glob("*.jpg")
    )
    print(f"per generator: {dict(sorted(generators.items()))}")
    print(f"matched size-only AUC: {size_only_auc(out / 'real', out / 'fake'):.4f}")
    print(f"\n-> {out}  (every image exactly {SIDE}x{SIDE} of native pixels)")


if __name__ == "__main__":
    main()
