"""Evaluation and folder prediction for the binary image detector."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import ImageDataset, ImageRecord, load_labeled_root
from .model import accelerator_pin_memory, load_checkpoint, resolve_device
from .transforms import CONDITION_GROUPS, EVALUATION_CONDITIONS, build_eval_transform

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})
ERRORS_PER_TYPE_AND_CONDITION = 3

METRIC_FIELDS = (
    "condition",
    "samples",
    "accuracy",
    "f1",
    "roc_auc",
    "tn",
    "fp",
    "fn",
    "tp",
    "fpr",
    "fnr",
    "accuracy_gap_from_clean",
    "f1_gap_from_clean",
    "roc_auc_gap_from_clean",
)

ERROR_FIELDS = (
    "image_path",
    "error_type",
    "true_label",
    "pred",
    "condition",
    "probability_ai",
    "confidence",
)


def _threshold(metadata: Mapping[str, Any], override: float | None) -> float:
    value = metadata.get("threshold", 0.5) if override is None else override
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid prediction threshold: {value!r}") from exc
    if not 0.0 <= result <= 1.0:
        raise ValueError("Prediction threshold must be between 0 and 1.")
    return result


def _probabilities(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """One P(AI) per image.

    Accepts either (B, C, H, W) -- one crop per image -- or (B, N, C, H, W)
    from the multi-crop eval transform, in which case the N crops are scored
    independently and their *probabilities* averaged.

    Averaging probabilities rather than logits is deliberate: the mean of
    sigmoids is not the sigmoid of the mean, and averaging in logit space
    lets one extreme-scoring crop dominate the other four. A crop that lands
    on a flat patch of sky should not be able to veto four that saw detail.
    """

    crops_per_image = 1
    if images.ndim == 5:
        batch, crops_per_image = images.shape[0], images.shape[1]
        images = images.flatten(0, 1)

    logits = model(images)
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits[:, 0]
    if logits.ndim != 1 or logits.shape[0] != images.shape[0]:
        raise RuntimeError(
            "Detector must return one logit per image; "
            f"received shape {tuple(logits.shape)}."
        )

    probabilities = torch.sigmoid(logits)
    if crops_per_image > 1:
        probabilities = probabilities.view(batch, crops_per_image).mean(dim=1)
    return probabilities


def _roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Compute binary ROC-AUC using average ranks, including tied scores."""

    positives = sum(label == 1 for label in labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    ordered = sorted(zip(scores, labels, strict=True), key=lambda item: item[0])
    positive_rank_sum = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(
            label == 1 for _, label in ordered[start:end]
        )
        start = end

    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _metrics(
    condition: str,
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
) -> dict[str, int | float | str | None]:
    predictions = [int(score >= threshold) for score in scores]
    tp = sum(label == 1 and pred == 1 for label, pred in zip(labels, predictions, strict=True))
    tn = sum(label == 0 and pred == 0 for label, pred in zip(labels, predictions, strict=True))
    fp = sum(label == 0 and pred == 1 for label, pred in zip(labels, predictions, strict=True))
    fn = sum(label == 1 and pred == 0 for label, pred in zip(labels, predictions, strict=True))
    samples = len(labels)
    return {
        "condition": condition,
        "samples": samples,
        "accuracy": (tp + tn) / samples,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        "roc_auc": _roc_auc(labels, scores),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "fpr": fp / (fp + tn) if (fp + tn) else None,
        "fnr": fn / (fn + tp) if (fn + tp) else None,
    }


def _difference(clean: object, current: object) -> float | None:
    if clean is None or current is None:
        return None
    return float(clean) - float(current)


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _mean(rows: Sequence[Mapping[str, object]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return sum(values) / len(values) if values else None


def _grouped_robust(transformed: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Mean AUC per transform group, then the mean of those group means.

    Equal weight per real-world effect rather than per severity setting --
    see transforms.CONDITION_GROUPS for why the two differ.
    """

    per_group: dict[str, list[float]] = {}
    for row in transformed:
        group = CONDITION_GROUPS.get(str(row["condition"]))
        value = row.get("roc_auc")
        if group is None or value is None:
            continue
        per_group.setdefault(group, []).append(float(value))

    group_means = {
        group: sum(values) / len(values) for group, values in per_group.items() if values
    }
    overall = sum(group_means.values()) / len(group_means) if group_means else None
    return {"roc_auc": overall, "groups": group_means}


def _worst(rows: Sequence[Mapping[str, object]], field: str) -> dict[str, object] | None:
    available = [row for row in rows if row.get(field) is not None]
    if not available:
        return None
    row = min(available, key=lambda item: (float(item[field]), str(item["condition"])))
    return {"condition": row["condition"], "value": row[field]}


def _worst_high(rows: Sequence[Mapping[str, object]], field: str) -> dict[str, object] | None:
    """Like :func:`_worst` but for fields where higher is worse (FPR, FNR)."""

    available = [row for row in rows if row.get(field) is not None]
    if not available:
        return None
    row = max(available, key=lambda item: (float(item[field]), str(item["condition"])))
    return {"condition": row["condition"], "value": row[field]}


def _representative_errors(errors: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in errors:
        grouped[(str(row["error_type"]), str(row["condition"]))].append(row)

    selected: list[Mapping[str, object]] = []
    for rows in grouped.values():
        rows.sort(key=lambda row: (-float(row["confidence"]), str(row["image_path"])))
        selected.extend(rows[:ERRORS_PER_TYPE_AND_CONDITION])

    condition_order = {name: index for index, name in enumerate(EVALUATION_CONDITIONS)}
    selected.sort(
        key=lambda row: (
            condition_order[str(row["condition"])],
            0 if row["error_type"] == "FP" else 1,
            -float(row["confidence"]),
            str(row["image_path"]),
        )
    )
    return selected


class _ConditionDataset(Dataset):
    """Holds already-decoded RGB PIL images so the 15 evaluation conditions
    each re-run only the (cheap) transform, not a fresh open+decode. Set
    ``.transform`` to the condition's ``build_eval_transform(condition)``
    before iterating.
    """

    def __init__(
        self, images: Sequence["Image.Image"], labels: Sequence[int], paths: Sequence[str]
    ) -> None:
        self.images = list(images)
        self.labels = list(labels)
        self.paths = list(paths)
        self.transform = None

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = self.images[index]
        assert self.transform is not None, "set .transform before iterating"
        return self.transform(image), self.labels[index], self.paths[index]


def _validate_loader_options(batch_size: int, num_workers: int) -> None:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")


def evaluate(
    checkpoint: str | Path,
    data_root: str | Path | None = None,
    output_dir: str | Path = "evaluation",
    batch_size: int = 32,
    num_workers: int = 0,
    device: str | torch.device = "auto",
    threshold: float | None = None,
    *,
    records: Sequence[ImageRecord] | None = None,
    n_crops: int = 1,
    crop_from_native: bool | None = None,
) -> dict[str, object]:
    """Evaluate a checkpoint on clean images and all 14 brief transformations.

    Pass ``records`` to evaluate an already-created validation split. When it is
    supplied, only those records are used, even if ``data_root`` is also set.
    Otherwise ``data_root`` must contain labeled ``real/`` and ``ai/`` folders.

    ``n_crops=5`` averages each image's score over five crops instead of
    scoring one centre crop (see transforms._multi_crop_tail). It is
    inference-only, so it applies to checkpoints already trained.

    ``crop_from_native`` defaults to whatever the checkpoint recorded at
    training time, and that default is the point: scoring a resize-trained
    model on native crops (or the reverse) silently costs more AUC than most
    of the changes being measured here, and nothing in the numbers would say
    so. Pass it explicitly only to deliberately measure that mismatch.
    """

    _validate_loader_options(batch_size, num_workers)
    selected_records = list(records) if records is not None else None
    if selected_records is None:
        if data_root is None:
            raise ValueError("Provide data_root or explicit validation records.")
        selected_records = load_labeled_root(data_root)
    if not selected_records:
        raise ValueError("Evaluation data is empty.")

    target_device = resolve_device(device)
    model, metadata = load_checkpoint(checkpoint, device=target_device)
    cutoff = _threshold(metadata, threshold)
    model.eval()

    # Checkpoints trained before crop_from_native existed carry no such key;
    # they were all trained under the resize path, so False is the correct
    # fallback.
    resolved_crop_from_native = (
        bool(metadata.get("crop_from_native", False)) if crop_from_native is None else crop_from_native
    )
    print(f"[EVALUATE] crop_from_native={resolved_crop_from_native} n_crops={n_crops}")

    metric_rows: list[dict[str, int | float | str | None]] = []
    all_errors: list[dict[str, object]] = []
    pin_memory = accelerator_pin_memory(target_device)

    # Decode every source image once and hold the RGB PIL in memory: the 15
    # conditions otherwise re-open and re-decode the same files 15 times,
    # which dominated evaluate() wall time (~14 img/s). Materialized eval
    # sets are 448px, so ~0.6 MB each decoded -- a few GB for a full set.
    base_dataset = ImageDataset(selected_records, transform=lambda im: im, return_path=True)
    base_images: list[Image.Image] = []
    base_labels: list[int] = []
    base_paths: list[str] = []
    for image, label, path in base_dataset:
        base_images.append(image)
        base_labels.append(int(label))
        base_paths.append(str(path))
    condition_dataset = _ConditionDataset(base_images, base_labels, base_paths)

    for condition in EVALUATION_CONDITIONS:
        condition_dataset.transform = build_eval_transform(
            condition, n_crops=n_crops, crop_from_native=resolved_crop_from_native
        )
        loader = DataLoader(
            condition_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )
        condition_labels: list[int] = []
        condition_scores: list[float] = []

        autocast_ctx = (
            torch.autocast(device_type=target_device.type, dtype=torch.bfloat16)
            if target_device.type in ("cuda", "xpu")
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with torch.inference_mode(), autocast_ctx:
            for images, labels, paths in loader:
                probabilities = _probabilities(
                    model, images.to(target_device, non_blocking=pin_memory)
                ).float().cpu()
                batch_labels = [int(value) for value in labels.tolist()]
                batch_scores = [float(value) for value in probabilities.tolist()]
                if any(label not in {0, 1} for label in batch_labels):
                    raise ValueError("Evaluation labels must be 0 (real) or 1 (AI).")

                condition_labels.extend(batch_labels)
                condition_scores.extend(batch_scores)
                for path, label, score in zip(paths, batch_labels, batch_scores, strict=True):
                    pred = int(score >= cutoff)
                    if pred == label:
                        continue
                    all_errors.append(
                        {
                            "image_path": str(path),
                            "error_type": "FP" if label == 0 else "FN",
                            "true_label": label,
                            "pred": pred,
                            "condition": condition,
                            "probability_ai": score,
                            "confidence": max(score, 1.0 - score),
                        }
                    )

        metric_rows.append(_metrics(condition, condition_labels, condition_scores, cutoff))

    clean = metric_rows[0]
    for row in metric_rows:
        row["accuracy_gap_from_clean"] = _difference(clean["accuracy"], row["accuracy"])
        row["f1_gap_from_clean"] = _difference(clean["f1"], row["f1"])
        row["roc_auc_gap_from_clean"] = _difference(clean["roc_auc"], row["roc_auc"])

    transformed = metric_rows[1:]
    summary: dict[str, object] = {
        "checkpoint": str(Path(checkpoint)),
        "labels": {"real": 0, "ai": 1},
        "threshold": cutoff,
        "images": len(selected_records),
        "conditions_evaluated": len(metric_rows),
        "clean": clean,
        "robust_mean": {
            "accuracy": _mean(transformed, "accuracy"),
            "f1": _mean(transformed, "f1"),
            "roc_auc": _mean(transformed, "roc_auc"),
        },
        # Both averagings, because we do not know which the organisers compute
        # and they give different scores. See transforms.CONDITION_GROUPS: a
        # flat mean weights each effect by how many severities it happens to
        # have (JPEG 4/14, centre-crop 1/14); the grouped mean gives each of
        # the six effects equal say. Measured on the WildFake benchmark:
        # 0.7767 flat vs 0.7888 grouped, i.e. Final Score 0.8131 vs 0.8192.
        "robust_mean_grouped": _grouped_robust(transformed),
        "worst_condition": {
            "accuracy": _worst(transformed, "accuracy"),
            "f1": _worst(transformed, "f1"),
            "roc_auc": _worst(transformed, "roc_auc"),
            "fpr": _worst_high(transformed, "fpr"),
            "fnr": _worst_high(transformed, "fnr"),
        },
        "error_rate_goal": {
            "target": 0.03,
            "clean_fpr": clean["fpr"],
            "clean_fnr": clean["fnr"],
            "worst_fpr_any_condition": _worst_high(metric_rows, "fpr"),
            "worst_fnr_any_condition": _worst_high(metric_rows, "fnr"),
            "all_conditions_within_target": all(
                (row.get("fpr") is not None and float(row["fpr"]) <= 0.03)
                and (row.get("fnr") is not None and float(row["fnr"]) <= 0.03)
                for row in metric_rows
            ),
        },
        "conditions": metric_rows,
        "representative_errors": {
            "selection": "up to 3 highest-confidence FP and FN per condition",
            "count": 0,
        },
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    representatives = _representative_errors(all_errors)
    summary["representative_errors"]["count"] = len(representatives)  # type: ignore[index]
    _write_csv(destination / "metrics.csv", METRIC_FIELDS, metric_rows)
    _write_csv(destination / "errors.csv", ERROR_FIELDS, representatives)
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


class _FolderDataset(Dataset[tuple[torch.Tensor, str]]):
    def __init__(self, images: Sequence[tuple[str, Path]]) -> None:
        self.images = list(images)
        self.transform = build_eval_transform("clean")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        relative_path, path = self.images[index]
        try:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
                tensor = self.transform(image)
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise ValueError(f"Could not read image {relative_path}: {exc}") from exc
        return tensor, relative_path


def _discover_images(input_dir: str | Path) -> list[tuple[str, Path]]:
    root = Path(input_dir)
    if not root.is_dir():
        raise ValueError(f"Input directory does not exist: {root}")
    images = [
        (path.relative_to(root).as_posix(), path)
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    images.sort(key=lambda item: (item[0].casefold(), item[0]))
    if not images:
        raise ValueError(f"No supported images found in: {root}")
    return images


def predict_folder(
    checkpoint: str | Path,
    input_dir: str | Path,
    output_json: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    device: str | torch.device = "auto",
    threshold: float | None = None,
) -> list[dict[str, object]]:
    """Predict a folder recursively and write strict JSON plus a score sidecar."""

    _validate_loader_options(batch_size, num_workers)
    discovered = _discover_images(input_dir)
    target_device = resolve_device(device)
    model, metadata = load_checkpoint(checkpoint, device=target_device)
    cutoff = _threshold(metadata, threshold)
    model.eval()

    pin_memory = accelerator_pin_memory(target_device)
    loader = DataLoader(
        _FolderDataset(discovered),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    strict_records: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for images, paths in loader:
            probabilities = _probabilities(
                model, images.to(target_device, non_blocking=pin_memory)
            ).cpu().tolist()
            for path, probability in zip(paths, probabilities, strict=True):
                score = float(probability)
                pred = int(score >= cutoff)
                # The brief asks for "a confidence score for each image,
                # indicating the likelihood that it is AIGC-generated ... a
                # JSON file containing image_path and pred", so "pred" is the
                # continuous probability, not a rounded label. That is also
                # what the scored metric needs: ROC-AUC over 0/1 predictions
                # collapses into a step function and understates the model.
                # "label" carries the thresholded call for any consumer that
                # wants a decision rather than a score.
                strict_records.append(
                    {"image_path": str(path), "pred": score, "label": pred}
                )
                score_rows.append(
                    {
                        "image_path": str(path),
                        "probability_ai": score,
                        "confidence": max(score, 1.0 - score),
                    }
                )

    destination = Path(output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(strict_records, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    scores_path = (
        destination.with_suffix(".scores.csv")
        if destination.suffix
        else destination.with_name(destination.name + ".scores.csv")
    )
    _write_csv(
        scores_path,
        ("image_path", "probability_ai", "confidence"),
        score_rows,
    )
    return strict_records
