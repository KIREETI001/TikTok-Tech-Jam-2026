"""Training preprocessing and the fixed 15-condition evaluation matrix."""

from __future__ import annotations

import random
from io import BytesIO
from typing import Final

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN: Final = (0.485, 0.456, 0.406)
IMAGENET_STD: Final = (0.229, 0.224, 0.225)

CONDITION_SPECS: Final[tuple[tuple[str, str, float | int | None], ...]] = (
    ("clean", "identity", None),
    ("jpeg_q90", "jpeg", 90),
    ("jpeg_q70", "jpeg", 70),
    ("jpeg_q50", "jpeg", 50),
    ("jpeg_q30", "jpeg", 30),
    ("blur_sigma0.5", "blur", 0.5),
    ("blur_sigma1", "blur", 1.0),
    ("blur_sigma2", "blur", 2.0),
    ("resize_0.5x", "resize", 0.5),
    ("resize_0.25x", "resize", 0.25),
    ("noise_sigma0.02", "noise", 0.02),
    ("noise_sigma0.05", "noise", 0.05),
    ("noise_sigma0.10", "noise", 0.10),
    ("color_jitter_20pct", "color_jitter", 0.20),
    ("center_crop_80pct", "center_crop", 0.80),
)
EVALUATION_CONDITIONS: Final = tuple(name for name, _, _ in CONDITION_SPECS)
_SPEC_BY_NAME = {name: (operation, value) for name, operation, value in CONDITION_SPECS}

# condition -> the real-world effect it is a variant of. The operation name
# already is that grouping, so this is a view of CONDITION_SPECS rather than a
# second list that could drift from it.
#
# Why it matters: the 14 transformed conditions are not evenly spread across
# effects -- JPEG has four severities, centre-crop has one. Averaging the
# conditions flat gives JPEG 4/14 of the robustness score and crop 1/14, which
# weights the metric by how many severities each effect happens to have been
# given rather than by how much each effect matters. Averaging the six groups
# instead gives each effect equal say.
#
# Both are reported because we do not know which the organisers compute, and
# they differ: on the WildFake benchmark our AUC_robust is 0.7767 flat against
# 0.7888 grouped (Final Score 0.8131 vs 0.8192).
CONDITION_GROUPS: Final[dict[str, str]] = {
    name: operation for name, operation, _ in CONDITION_SPECS if operation != "identity"
}
TRANSFORM_GROUPS: Final = tuple(dict.fromkeys(CONDITION_GROUPS.values()))


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, subsampling=2, optimize=False)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def _resize_round_trip(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    small_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small = image.resize(small_size, Image.Resampling.BICUBIC)
    return small.resize((width, height), Image.Resampling.BICUBIC)


def _gaussian_noise(image: Image.Image, sigma: float, seed: int) -> Image.Image:
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    noise = np.random.default_rng(seed).normal(0.0, sigma, size=pixels.shape)
    pixels = np.clip(pixels + noise, 0.0, 1.0)
    return Image.fromarray(np.rint(pixels * 255.0).astype(np.uint8), mode="RGB")


def _color_jitter(image: Image.Image, amount: float, seed: int) -> Image.Image:
    factors = np.random.default_rng(seed).uniform(1.0 - amount, 1.0 + amount, size=3)
    result = ImageEnhance.Brightness(image).enhance(float(factors[0]))
    result = ImageEnhance.Contrast(result).enhance(float(factors[1]))
    return ImageEnhance.Color(result).enhance(float(factors[2]))


def _center_crop(image: Image.Image, fraction: float) -> Image.Image:
    width, height = image.size
    crop_width = max(1, round(width * fraction))
    crop_height = max(1, round(height * fraction))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height), Image.Resampling.BICUBIC)


def apply_condition(image: Image.Image, condition: str = "clean", *, seed: int = 0) -> Image.Image:
    """Apply one deterministic evaluation condition before model preprocessing."""

    try:
        operation, value = _SPEC_BY_NAME[condition]
    except KeyError as exc:
        choices = ", ".join(EVALUATION_CONDITIONS)
        raise ValueError(f"Unknown evaluation condition {condition!r}; choose from {choices}") from exc

    source = image.convert("RGB")
    if operation == "identity":
        return source.copy()
    if operation == "jpeg":
        return _jpeg(source, int(value))
    if operation == "blur":
        return source.filter(ImageFilter.GaussianBlur(radius=float(value)))
    if operation == "resize":
        return _resize_round_trip(source, float(value))
    if operation == "noise":
        return _gaussian_noise(source, float(value), seed)
    if operation == "color_jitter":
        return _color_jitter(source, float(value), seed)
    if operation == "center_crop":
        return _center_crop(source, float(value))
    raise AssertionError(f"Unhandled evaluation operation: {operation}")


class _ApplyCondition:
    def __init__(self, condition: str, seed: int) -> None:
        self.condition = condition
        self.seed = seed

    def __call__(self, image: Image.Image) -> Image.Image:
        return apply_condition(image, self.condition, seed=self.seed)


CROP_POLICIES: Final = ("resize", "native")


class _UpscaleIfSmall:
    """Scale up, preserving aspect ratio, only when an image is smaller than
    the crop window. The one case where resampling is unavoidable.

    Everything larger is left completely untouched -- that is the whole
    point of the ``native`` crop policy (see ``_geometry``).
    """

    def __init__(self, size: int = 224) -> None:
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if min(width, height) >= self.size:
            return image
        scale = self.size / min(width, height)
        return image.resize(
            (max(self.size, round(width * scale)), max(self.size, round(height * scale))),
            Image.Resampling.BILINEAR,
        )


def _geometry(crop_policy: str) -> list:
    """The resize/crop stage, shared by training and evaluation so the two
    cannot drift apart.

    ``resize`` (the original Community Forensics recipe): short edge to 256,
    then a 224 crop. Every image is resampled.

    ``native``: no resampling at all above 224px -- crop the 224 window
    straight out of the image at its own resolution.

    Why ``native`` is worth having, from two independent measurements:

    - ziyangchua02/model_training found resampling low-pass filters the top
      octave, which is exactly where generator fingerprints live, and that a
      fixed resize rule degrades sources unevenly by native resolution
      (their high-resolution generators were blurred while their 1024px ones
      were not, and the blurred ones then scored worst).
    - Our own shortcut probe (scripts/shortcut_probe.py) found the resize
      also makes the corpus *more* trivially separable, consistently across
      all three corpora: PS5 train 72.60% -> 74.47%, PS5 test 74.93% ->
      76.93%, SID_Set 54.93% -> 57.80%, with edge_std's importance roughly
      doubling. Normalising every image to one scale turns edge density into
      a cleanly comparable smoothness measure; at native resolution the same
      feature is confounded with the image's own resolution and is a weaker
      shortcut.

    So the two policies differ on both axes at once -- ``native`` keeps more
    fingerprint signal AND leaks less shortcut -- which is why this is a
    policy to measure rather than a constant to flip.
    """

    if crop_policy not in CROP_POLICIES:
        raise ValueError(
            f"Unknown crop_policy {crop_policy!r}; choose from {', '.join(CROP_POLICIES)}"
        )
    if crop_policy == "resize":
        return [T.Resize(256, interpolation=InterpolationMode.BILINEAR)]
    return [_UpscaleIfSmall(224)]


def _native_tail(crop_policy: str = "resize") -> list:
    """Geometry, then a centre 224 crop, then ImageNet normalization."""

    return [
        *_geometry(crop_policy),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]


def _multi_crop_tail(crop_policy: str = "resize") -> list:
    """Five 224 crops (centre + four corners), stacked as (5, C, H, W) for
    score averaging at evaluation time.

    Ported from ziyangchua02/model_training's ``prepare_crops``: a single
    224px window onto a large image sees a small fraction of it, so one
    crop's score is a high-variance sample of what the model thinks about
    the whole image.

    How much this buys depends entirely on ``crop_policy``. Under
    ``resize`` the image has already been squeezed to a 256 short edge, so
    five 224 windows overlap almost completely and add little. Under
    ``native`` they are five genuinely different regions of a full-resolution
    image, which is the case their result was measured in.
    """

    return [
        *_geometry(crop_policy),
        T.FiveCrop(224),
        T.Lambda(
            lambda crops: torch.stack(
                [T.Normalize(IMAGENET_MEAN, IMAGENET_STD)(T.ToTensor()(crop)) for crop in crops]
            )
        ),
    ]


def build_eval_transform(
    condition: str = "clean", *, seed: int = 0, n_crops: int = 1, crop_policy: str = "resize"
) -> T.Compose:
    """Build model preprocessing preceded by one fixed evaluation condition.

    ``n_crops=5`` switches from a single centre crop to five crops stacked on
    a leading dimension; the caller averages the per-crop scores (see
    detector.evaluation._probabilities).

    ``crop_policy`` must match whatever the checkpoint was trained with --
    see _geometry. Scoring a resize-trained model on native crops (or the
    reverse) pays a train/inference mismatch their row 0b measured at 0.069
    AUC, which is larger than most of the gains being chased here.
    """

    if condition not in EVALUATION_CONDITIONS:
        choices = ", ".join(EVALUATION_CONDITIONS)
        raise ValueError(f"Unknown evaluation condition {condition!r}; choose from {choices}")
    if n_crops not in (1, 5):
        raise ValueError(f"n_crops must be 1 or 5, got {n_crops}.")
    tail = _native_tail(crop_policy) if n_crops == 1 else _multi_crop_tail(crop_policy)
    return T.Compose([_ApplyCondition(condition, seed), *tail])


class _RandomRobustnessAugment:
    """Independently applies each of resize-roundtrip / blur / Gaussian
    noise / JPEG re-encode to a training sample (each at its own randomly
    sampled severity, spanning the same range the fixed-severity evaluation
    matrix (CONDITION_SPECS) tests), so multiple can stack on one image.

    Originally this picked exactly one of the four per sample. Rewritten
    after checking the literature on what actually drives cross-generator
    robustness: Wang et al., "CNN-generated images are surprisingly easy to
    spot...for now" (CVPR 2020) found that blur *and* JPEG applied
    independently (~50% each, so they can co-occur) is what produced
    generalization, not choosing one corruption at a time -- a real
    post-and-reshare image is typically resized *and* re-compressed *and*
    noisy, not exactly one of those. ``probability`` still means "chance at
    least one corruption fires" (so existing config values keep their
    meaning); internally it's converted to an independent per-corruption
    probability so 2+ can stack, matching the validated recipe:
    ``p_each = 1 - (1 - probability) ** (1 / 4)``.

    Order (resize -> blur -> noise -> jpeg) mirrors a plausible real-world
    capture/reshare chain: a resolution change, then optical/compression
    blur, then sensor/channel noise, then the final JPEG save.

    Applied *before* the Resize/RandomCrop step below, mirroring
    build_eval_transform's operation order (apply_condition also runs
    before its own Resize/CenterCrop) -- otherwise the same nominal
    severity (e.g. "blur sigma 2") would mean a different effective amount
    of blur at eval time than at train time, since blurring a 224x224 crop
    is not the same as blurring the original image and then downscaling.
    """

    def __init__(self, probability: float = 0.7) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1.")
        self.probability = probability
        # At least one of 4 independent Bernoulli(p_each) fires with
        # probability `probability`: 1 - (1-p_each)^4 = probability.
        self.per_op_probability = 1.0 - (1.0 - probability) ** 0.25

    def __call__(self, image: Image.Image) -> Image.Image:
        result = image.convert("RGB")
        if random.random() < self.per_op_probability:
            result = _resize_round_trip(result, random.uniform(0.25, 1.0))
        if random.random() < self.per_op_probability:
            result = result.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 2.0)))
        if random.random() < self.per_op_probability:
            result = _gaussian_noise(result, random.uniform(0.0, 0.10), random.randint(0, 2**31 - 1))
        if random.random() < self.per_op_probability:
            result = _jpeg(result, random.randint(30, 100))
        return result


def build_train_transform(
    augment_probability: float = 0.7, *, crop_policy: str = "resize"
) -> T.Compose:
    """Model preprocessing with realistic-corruption augmentation (see
    _RandomRobustnessAugment) plus a small amount of common augmentation.
    Pass ``augment_probability=0.0`` to disable the corruption augmentation
    entirely (e.g. for a fast, deterministic-ish smoke test).

    ``crop_policy`` must match what evaluation and inference will use --
    see _geometry and build_eval_transform.
    """

    return T.Compose(
        [
            _RandomRobustnessAugment(augment_probability),
            *_geometry(crop_policy),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
