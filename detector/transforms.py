"""Training preprocessing and the fixed 15-condition evaluation matrix."""

from __future__ import annotations

from io import BytesIO
from typing import Final

import numpy as np
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


def _native_tail() -> list:
    """Community Forensics: short edge 256, center crop 224, ImageNet norm."""

    return [
        T.Resize(256, interpolation=InterpolationMode.BILINEAR),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]


def build_eval_transform(condition: str = "clean", *, seed: int = 0) -> T.Compose:
    """Build native preprocessing preceded by one fixed evaluation condition."""

    if condition not in EVALUATION_CONDITIONS:
        choices = ", ".join(EVALUATION_CONDITIONS)
        raise ValueError(f"Unknown evaluation condition {condition!r}; choose from {choices}")
    return T.Compose([_ApplyCondition(condition, seed), *_native_tail()])


def build_train_transform() -> T.Compose:
    """Native-sized preprocessing with a small amount of common augmentation."""

    return T.Compose(
        [
            T.Resize(256, interpolation=InterpolationMode.BILINEAR),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
