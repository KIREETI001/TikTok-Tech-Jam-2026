"""Image discovery, deduplication, splitting, and loading."""

from __future__ import annotations

import csv
import hashlib
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})
REAL_FOLDERS = frozenset({"real", "authentic", "non_aigc"})
FAKE_FOLDERS = frozenset({"fake", "ai", "aigc"})


@dataclass(frozen=True)
class ImageRecord:
    """One validated, unique image in a labeled dataset."""

    path: Path
    label: int
    sha256: str
    split: str = ""


def _normalise_folder_name(name: str) -> str:
    return name.strip().casefold().replace("-", "_").replace(" ", "_")


def _label_for_path(path: Path, root: Path) -> int:
    labels: set[int] = set()
    for part in path.relative_to(root).parts[:-1]:
        name = _normalise_folder_name(part)
        if name in REAL_FOLDERS:
            labels.add(0)
        elif name in FAKE_FOLDERS:
            labels.add(1)

    if not labels:
        expected = ", ".join(sorted(REAL_FOLDERS | FAKE_FOLDERS))
        raise ValueError(f"Image is not inside a recognized label folder: {path} (expected {expected})")
    if len(labels) != 1:
        raise ValueError(f"Image path contains conflicting label folders: {path}")
    return labels.pop()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_image(path: Path) -> None:
    try:
        # verify() checks the encoded file; reopening and loading checks that it
        # can actually be decoded into the RGB input expected by the model.
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.convert("RGB").load()
    except Exception as exc:
        raise ValueError(f"Invalid image file: {path}") from exc


EVAL_ONLY_PREFIX = "eval_only_"


def assert_not_eval_only(root: str | Path) -> None:
    """Refuse to use an evaluation-only benchmark as training data.

    The organisers designate WildFake (and anything else staged under an
    ``eval_only_*`` directory -- see EVAL_ONLY_DATASETS.md) as validation
    only. Training on it, or even selecting a threshold on it, would make
    every reported number meaningless, so the training path calls this and
    fails loudly rather than relying on nobody mistyping ``data_dir``.
    """

    path = Path(root).expanduser().resolve()
    for part in path.parts:
        if part.casefold().startswith(EVAL_ONLY_PREFIX):
            raise ValueError(
                f"{path} is an evaluation-only benchmark ('{part}') and must "
                "never be used for training, validation-split selection, or "
                "threshold calibration. See EVAL_ONLY_DATASETS.md."
            )


def load_labeled_root(root: str | Path) -> list[ImageRecord]:
    """Load and validate a ``root/{label}/...`` image tree.

    Exact duplicate files within one label are kept once. If identical bytes
    appear under both labels, ingestion stops because the ground truth is
    contradictory.
    """

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Labeled image root does not exist: {root_path}")

    paths = sorted(
        (
            path
            for path in root_path.rglob("*")
            if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.relative_to(root_path).as_posix().casefold(),
    )
    if not paths:
        raise ValueError(f"No supported images found under {root_path}")

    unique: dict[str, ImageRecord] = {}
    for path in paths:
        label = _label_for_path(path, root_path)
        _validate_image(path)
        checksum = _sha256(path)
        previous = unique.get(checksum)
        if previous is not None:
            if previous.label != label:
                raise ValueError(
                    "Identical image bytes occur under both labels: "
                    f"{previous.path} and {path}"
                )
            continue
        unique[checksum] = ImageRecord(path=path.resolve(), label=label, sha256=checksum)

    records = sorted(unique.values(), key=lambda record: record.path.as_posix().casefold())
    present_labels = {record.label for record in records}
    if present_labels != {0, 1}:
        missing = "real" if 0 not in present_labels else "fake"
        raise ValueError(f"Dataset must contain both labels; no {missing} images were found")
    return records


def stratified_split(
    records: Sequence[ImageRecord],
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Create a repeatable train/validation split independently per label."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    train: list[ImageRecord] = []
    validation: list[ImageRecord] = []
    for label in (0, 1):
        group = sorted(
            (record for record in records if record.label == label),
            key=lambda record: record.path.as_posix().casefold(),
        )
        if len(group) < 2:
            name = "real" if label == 0 else "fake"
            raise ValueError(f"At least two unique {name} images are required for a train/validation split")

        random.Random(seed + label).shuffle(group)
        validation_count = round(len(group) * validation_fraction)
        validation_count = min(len(group) - 1, max(1, validation_count))
        validation.extend(replace(record, split="validation") for record in group[:validation_count])
        train.extend(replace(record, split="train") for record in group[validation_count:])

    def key(record: ImageRecord) -> str:
        return record.path.as_posix().casefold()

    return sorted(train, key=key), sorted(validation, key=key)


def write_manifest(records: Sequence[ImageRecord], manifest_path: str | Path) -> Path:
    """Write records to a CSV that can be loaded by :func:`load_manifest`."""

    destination = Path(manifest_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda record: (record.split, record.path.as_posix().casefold()))
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "label", "sha256", "split"))
        writer.writeheader()
        for record in ordered:
            writer.writerow(
                {
                    "path": str(record.path.resolve()),
                    "label": record.label,
                    "sha256": record.sha256,
                    "split": record.split,
                }
            )
    return destination


def load_manifest(manifest_path: str | Path, *, split: str | None = None) -> list[ImageRecord]:
    """Reload records from a manifest, optionally selecting one split."""

    source = Path(manifest_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {source}")

    records: list[ImageRecord] = []
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"path", "label", "sha256", "split"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Manifest must contain columns: {', '.join(sorted(required))}")
        for row in reader:
            record_split = row["split"].strip()
            if split is not None and record_split != split:
                continue
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"Invalid label {label!r} in {source}")
            image_path = Path(row["path"]).expanduser()
            if not image_path.is_absolute():
                image_path = source.parent / image_path
            records.append(
                ImageRecord(
                    path=image_path.resolve(),
                    label=label,
                    sha256=row["sha256"].strip().casefold(),
                    split=record_split,
                )
            )
    return records


def ingest_training_data(
    root: str | Path,
    manifest_path: str | Path | None = None,
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Validate, deduplicate, split, and manifest a training image tree."""

    root_path = Path(root).expanduser().resolve()
    assert_not_eval_only(root_path)
    records = load_labeled_root(root_path)
    train, validation = stratified_split(
        records,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    write_manifest(train + validation, manifest_path or root_path / "manifest.csv")
    return train, validation


class ImageDataset(Dataset):
    """PyTorch dataset backed by a list of :class:`ImageRecord` objects."""

    def __init__(
        self,
        records: Sequence[ImageRecord],
        transform: Callable | None = None,
        *,
        return_path: bool = False,
        two_view: bool = False,
    ) -> None:
        self.records = list(records)
        if transform is None:
            from .transforms import build_eval_transform

            transform = build_eval_transform()
        self.transform = transform
        self.return_path = return_path
        # two_view: apply the (stochastic) transform twice and return both,
        # for the supervised-contrastive loss (Phase D).
        self.two_view = two_view

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        try:
            with Image.open(record.path) as source:
                image = source.convert("RGB").copy()
        except Exception as exc:
            raise RuntimeError(f"Could not load image from manifest: {record.path}") from exc

        if self.two_view:
            return self.transform(image), self.transform(image), record.label
        image = self.transform(image)
        if self.return_path:
            return image, record.label, str(record.path)
        return image, record.label
