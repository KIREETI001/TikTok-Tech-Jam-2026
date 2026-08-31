"""Build a resolution-matched WildFake+COCO eval set from this device's own
cache, adapting ziyangchua02's build_matched_eval.py to run without their
machine's pre-fetched native-resolution source folders.

Why this exists: WildFake ships its generated images at fixed sizes
(ADM/DDIM/DDPM/VQDM are 256x256; DALLE is larger) and the authentic half
comes from COCO at photographic resolutions. On the set as-composed, image
size alone separates most generators perfectly -- the teammate measured
size-only AUC 1.0000 on their build. Any score on that set mixes "can it
detect generation" with "can it read a file header", and the two cannot be
told apart. This project's earlier WildFake benchmark (experiments.md
section 11, Final 0.8131) did not control for this, so that number needs
re-measuring on the matched set before it means anything.

Source: scripts/evaluate_wildfake.py's cache at runs/wildfake_eval/ --
1,000 native-resolution COCO2017 reals and ~1,000 native-resolution
WildFake fakes across 5 generators (ADM, DALLE, DDIM, DDPM, VQDM), fetched
via ModelScope range-reads earlier this session. Same source data the
teammate's own build used (WildFake's own Images/Real/coco.zip for the
authentic half), just already on this disk instead of needing a fresh
fetch.

Matching is centre-cropping native pixels to the smallest common side,
never resizing -- resizing would resample one class more than the other,
replacing a size cue with a resampling cue, the same mistake in a
different coat.

*** EVALUATION ONLY. Never wired into any data_source used for training. ***
"""

from __future__ import annotations

import collections
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from detector.evaluation import _roc_auc  # noqa: E402

CACHE = Path("runs/wildfake_eval")
GENERATORS = ("adm", "dalle", "ddim", "ddpm", "vqdm")
OUT = Path("runs/wildfake_matched")
PER_GENERATOR = 200


def _sizes(paths: list[Path]) -> list[float]:
    out = []
    for path in paths:
        with Image.open(path) as image:
            out.append(float(image.width * image.height))
    return out


def size_only_auc(real_paths: list[Path], fake_paths: list[Path]) -> float:
    """AUC of a classifier that sees nothing but each image's pixel count."""
    real, fake = _sizes(real_paths), _sizes(fake_paths)
    return _roc_auc([0] * len(real) + [1] * len(fake), real + fake)


def center_square(image: Image.Image, side: int) -> Image.Image | None:
    if min(image.size) < side:
        return None
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    return image.crop((left, top, left + side, top + side))


def main() -> int:
    real_paths = sorted(
        p for p in (CACHE / "real").glob("*") if p.suffix.lower() in {".jpg", ".png"}
    )[: PER_GENERATOR * len(GENERATORS)]  # match total fake count, e.g. 200*5=1000
    fake_paths: list[tuple[str, Path]] = []
    for gen in GENERATORS:
        paths = sorted(p for p in (CACHE / gen).glob("*") if p.suffix.lower() in {".jpg", ".png"})
        for p in paths[:PER_GENERATOR]:
            fake_paths.append((gen, p))

    print(f"source: {len(real_paths)} real, {len(fake_paths)} fake ({len(GENERATORS)} generators)")
    print(
        f"source size-only AUC: {size_only_auc(real_paths, [p for _, p in fake_paths]):.4f}"
        "   (0.5 = size is uninformative)"
    )

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "real").mkdir(parents=True)
    (OUT / "fake").mkdir(parents=True)

    # Smallest common side across BOTH classes -- every crop must fit inside
    # the smallest image in the whole set, real or fake, or that class's
    # side would be padded/upscaled and the size cue would creep back in.
    side = min(_size for path in real_paths for _size in [min(Image.open(path).size)])
    for _gen, p in fake_paths:
        with Image.open(p) as img:
            side = min(side, min(img.size))
    print(f"matching side: {side}px")

    real_kept = real_skipped = 0
    for p in real_paths:
        with Image.open(p) as source:
            crop = center_square(source.convert("RGB"), side)
        if crop is None:
            real_skipped += 1
            continue
        crop.save(OUT / "real" / p.name, quality=95)
        real_kept += 1
    print(f"real:  kept {real_kept}, skipped {real_skipped}")

    fake_kept: collections.Counter = collections.Counter()
    fake_skipped: collections.Counter = collections.Counter()
    for gen, p in fake_paths:
        with Image.open(p) as source:
            crop = center_square(source.convert("RGB"), side)
        if crop is None:
            fake_skipped[gen] += 1
            continue
        crop.save(OUT / "fake" / f"{gen}_{p.name}", quality=95)
        fake_kept[gen] += 1
    print(f"fake:  kept {dict(fake_kept)}, skipped {dict(fake_skipped)}")

    matched_real = sorted((OUT / "real").glob("*.jpg"))
    matched_fake = sorted((OUT / "fake").glob("*.jpg"))
    print(f"matched size-only AUC: {size_only_auc(matched_real, matched_fake):.4f}")
    print(f"\n-> {OUT}  (every image exactly {side}x{side} of native pixels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
