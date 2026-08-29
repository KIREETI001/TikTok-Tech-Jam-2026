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


def _motion_blur(image: Image.Image, length: int, angle: float) -> Image.Image:
    """Directional smear (teammate's ``transforms_lib.motion_blur``): a
    camera pan or fast subject produces one constantly, and a detector that
    has never seen a long directional smear reads that smoothness as a
    generator artifact. Done by hand because PIL's kernel filter caps at 5x5.
    """
    import math

    dx, dy = math.cos(angle), math.sin(angle)
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    acc = np.zeros_like(arr)
    for step in range(length):
        offset = step - length // 2
        shifted = np.roll(arr, shift=int(round(offset * dy)), axis=0)
        acc += np.roll(shifted, shift=int(round(offset * dx)), axis=1)
    return Image.fromarray(np.clip(acc / length, 0, 255).astype(np.uint8), mode="RGB")


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


class _UpscaleIfSmall:
    """Scale up, preserving aspect ratio, only when an image is smaller than
    the crop window. The one case where resampling is unavoidable.

    Everything larger is left completely untouched -- that is the whole
    point of the ``crop_from_native`` eval path (see ``_geometry``). Used
    for evaluation only; training's ``crop_from_native`` path instead relies
    on ``RandomCrop(pad_if_needed=True)`` (see ``build_train_transform``),
    since a deterministic evaluation should not be padded with content-free
    black borders on small images, while a small amount of training-time
    padding is a harmless, arguably helpful, mild augmentation.
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


def _geometry(crop_from_native: bool) -> list:
    """The resize/crop stage for evaluation, shared by every eval-only
    caller so they cannot drift apart.

    ``crop_from_native=False`` (the original Community Forensics recipe):
    short edge to 256, then a 224 crop. Every image is resampled.

    ``crop_from_native=True``: no resampling at all above 224px -- crop the
    224 window straight out of the image at its own resolution. Must match
    whatever ``build_train_transform``'s ``crop_from_native`` the checkpoint
    was trained with -- see ``detector.evaluation.evaluate``, which resolves
    this from the checkpoint's own metadata by default. Scoring a
    resize-trained model on native crops (or the reverse) pays a
    train/inference mismatch measured at 0.069 AUC, larger than most of the
    gains being chased here.

    Why ``crop_from_native`` is worth having, from two independent
    measurements:

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

    So the two policies differ on both axes at once -- native crops keep
    more fingerprint signal AND leak less shortcut -- which is why this is a
    setting to measure rather than a constant to flip.
    """

    if not crop_from_native:
        return [T.Resize(256, interpolation=InterpolationMode.BILINEAR)]
    return [_UpscaleIfSmall(224)]


def _native_tail(crop_from_native: bool = False) -> list:
    """Geometry, then a centre 224 crop, then ImageNet normalization."""

    return [
        *_geometry(crop_from_native),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]


def _multi_crop_tail(crop_from_native: bool = False) -> list:
    """Five 224 crops (centre + four corners), stacked as (5, C, H, W) for
    score averaging at evaluation time.

    Ported from ziyangchua02/model_training's ``prepare_crops``: a single
    224px window onto a large image sees a small fraction of it, so one
    crop's score is a high-variance sample of what the model thinks about
    the whole image.

    How much this buys depends entirely on ``crop_from_native``. When False,
    the image has already been squeezed to a 256 short edge, so five 224
    windows overlap almost completely and add little. When True, they are
    five genuinely different regions of a full-resolution image, which is
    the case their result was measured in.
    """

    return [
        *_geometry(crop_from_native),
        T.FiveCrop(224),
        T.Lambda(
            lambda crops: torch.stack(
                [T.Normalize(IMAGENET_MEAN, IMAGENET_STD)(T.ToTensor()(crop)) for crop in crops]
            )
        ),
    ]


def build_eval_transform(
    condition: str = "clean", *, seed: int = 0, n_crops: int = 1, crop_from_native: bool = False
) -> T.Compose:
    """Build model preprocessing preceded by one fixed evaluation condition.

    ``n_crops=5`` switches from a single centre crop to five crops stacked on
    a leading dimension; the caller averages the per-crop scores (see
    detector.evaluation._probabilities).

    ``crop_from_native`` must match what the checkpoint was trained with --
    see _geometry.
    """

    if condition not in EVALUATION_CONDITIONS:
        choices = ", ".join(EVALUATION_CONDITIONS)
        raise ValueError(f"Unknown evaluation condition {condition!r}; choose from {choices}")
    if n_crops not in (1, 5):
        raise ValueError(f"n_crops must be 1 or 5, got {n_crops}.")
    tail = (
        _native_tail(crop_from_native) if n_crops == 1 else _multi_crop_tail(crop_from_native)
    )
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

    def __init__(self, probability: float = 0.7, motion_blur: bool = False) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1.")
        self.probability = probability
        self.motion_blur = motion_blur
        # At least one of 4 independent Bernoulli(p_each) fires with
        # probability `probability`: 1 - (1-p_each)^4 = probability.
        self.per_op_probability = 1.0 - (1.0 - probability) ** 0.25

    def __call__(self, image: Image.Image) -> Image.Image:
        result = image.convert("RGB")
        if random.random() < self.per_op_probability:
            result = _resize_round_trip(result, random.uniform(0.25, 1.0))
        if random.random() < self.per_op_probability:
            result = result.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 2.0)))
        # Gaussian noise gets its own slightly-higher probability and a range
        # that overshoots the eval matrix's max (0.10 -> 0.13): FNR under
        # heavy noise was the single largest residual failure through
        # iterations 2-3 (fakes' scores drift toward "real" as noise rises),
        # so the model needs more exposure to it than the other corruptions.
        if random.random() < min(1.0, self.per_op_probability * 1.5):
            result = _gaussian_noise(result, random.uniform(0.0, 0.13), random.randint(0, 2**31 - 1))
        if self.motion_blur and random.random() < self.per_op_probability * 0.5:
            result = _motion_blur(result, random.randint(5, 25), random.uniform(0, 3.14159))
        if random.random() < self.per_op_probability:
            result = _jpeg(result, random.randint(30, 100))
        return result


class _RandomMask:
    """SAFE (KDD 2025): zero out random square patches of a tensor, up to
    ``max_ratio`` of the area, with probability ``p``. Forces the detector
    onto local statistics -- SAFE's ablation credits it with 2-9 points of
    cross-generator accuracy. Operates on the CHW tensor (post-ToTensor,
    pre-Normalize) so a zeroed patch is a true black square.
    """

    def __init__(self, patch: int = 16, max_ratio: float = 0.75, p: float = 0.5) -> None:
        self.patch, self.max_ratio, self.p = patch, max_ratio, p

    def __call__(self, tensor):
        if random.random() >= self.p:
            return tensor
        _c, h, w = tensor.shape
        ratio = random.uniform(0.0, self.max_ratio)
        n = int(ratio * h * w / (self.patch ** 2))
        for _ in range(n):
            top = random.randint(0, max(0, h - self.patch))
            left = random.randint(0, max(0, w - self.patch))
            tensor[:, top : top + self.patch, left : left + self.patch] = 0.0
        return tensor


def build_train_transform(
    augment_probability: float = 0.7,
    *,
    crop_from_native: bool = False,
    safe_augment: bool = False,
    motion_blur: bool = False,
    windowed: bool = False,
) -> T.Compose:
    """Native-sized preprocessing with realistic-corruption augmentation
    (see _RandomRobustnessAugment) plus a small amount of common
    augmentation. ``augment_probability=0.0`` disables the corruption
    augmentation.

    ``crop_from_native``: skip the ``Resize(256)`` and ``RandomCrop(224)``
    straight from native pixels -- SAFE / briefing-deck slide 10's
    "crop, don't down-sample" (resize low-pass-filters away the artifact).
    Must be kept in sync with whatever ``build_eval_transform`` and
    inference use for the resulting checkpoint -- see ``_geometry``.
    ``safe_augment``: add ``RandomRotation(180)`` and ``RandomMask`` (SAFE).
    ``motion_blur``: add a directional-smear corruption.
    ``windowed``: take a 320px window, degrade *that*, then crop to 224 --
    evaluation degrades the whole image and then crops, so applying JPEG or a
    4x rescale directly to a 224 window is a different operation
    (teammate's ``WindowedAugment``).
    """

    steps: list = []
    if windowed:
        steps.append(T.RandomCrop(320, pad_if_needed=True))
    steps.append(_RandomRobustnessAugment(augment_probability, motion_blur=motion_blur))
    if not crop_from_native:
        steps.append(T.Resize(256, interpolation=InterpolationMode.BILINEAR))
    if safe_augment:
        # fill rotated corners with edge pixels rather than black
        steps.append(T.RandomRotation(180, interpolation=InterpolationMode.BILINEAR))
    steps += [
        T.RandomCrop(224, pad_if_needed=True),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.ToTensor(),
    ]
    if safe_augment:
        steps.append(_RandomMask(patch=16, max_ratio=0.75, p=0.5))
    steps.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return T.Compose(steps)
