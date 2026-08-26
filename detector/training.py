"""Small, readable training loop for the binary image detector."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import ImageDataset, ImageRecord
from .model import Detector, create_detector, resolve_device, save_checkpoint
from .transforms import build_eval_transform, build_train_transform

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _binary_metrics(
    logits: torch.Tensor, labels: torch.Tensor, threshold: float
) -> dict[str, float]:
    predictions = torch.sigmoid(logits) >= threshold
    targets = labels >= 0.5
    true_positive = int((predictions & targets).sum().item())
    false_positive = int((predictions & ~targets).sum().item())
    false_negative = int((~predictions & targets).sum().item())
    correct = int((predictions == targets).sum().item())

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": correct / max(len(targets), 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _train_epoch(
    model: Detector,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    sample_count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device=device, dtype=torch.float32, non_blocking=True).view(-1)

        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(images), labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.numel()
        total_loss += float(loss.detach().item()) * batch_size
        sample_count += batch_size
    return total_loss / max(sample_count, 1)


@torch.no_grad()
def _validate(
    model: Detector,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    model.eval()
    logits_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    total_loss = 0.0
    sample_count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device=device, dtype=torch.float32, non_blocking=True).view(-1)
        logits = model(images)
        loss = loss_function(logits, labels)

        batch_size = labels.numel()
        total_loss += float(loss.item()) * batch_size
        sample_count += batch_size
        logits_parts.append(logits.cpu())
        label_parts.append(labels.cpu())

    metrics = _binary_metrics(
        torch.cat(logits_parts), torch.cat(label_parts), threshold
    )
    metrics["loss"] = total_loss / max(sample_count, 1)
    return metrics


def train_model(
    train_records: Sequence[ImageRecord],
    val_records: Sequence[ImageRecord],
    output_dir: str | Path,
    *,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    num_workers: int = 0,
    seed: int = 2026,
    device: str | torch.device = "auto",
    pretrained: bool = True,
    local_files_only: bool = False,
    threshold: float = 0.5,
    train_augment_probability: float = 0.7,
) -> Path:
    """Fine-tune the detector on local-disk image records and return the
    best-F1 checkpoint path. Thin wrapper around
    :func:`train_model_from_datasets` for the path-based (``detector.data``)
    data source; a streamed data source (e.g. HTTP-fetched shards with no
    local path) builds its own datasets and calls that function directly.
    """

    if not train_records or not val_records:
        raise ValueError("Training and validation records must both be non-empty.")

    train_dataset = ImageDataset(train_records, build_train_transform(train_augment_probability))
    val_dataset = ImageDataset(val_records, build_eval_transform())
    return train_model_from_datasets(
        train_dataset,
        val_dataset,
        output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_workers=num_workers,
        seed=seed,
        device=device,
        pretrained=pretrained,
        local_files_only=local_files_only,
        threshold=threshold,
        train_count=len(train_records),
        val_count=len(val_records),
    )


def train_model_from_datasets(
    train_dataset: Dataset,
    val_dataset: Dataset,
    output_dir: str | Path,
    *,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    num_workers: int = 0,
    seed: int = 2026,
    device: str | torch.device = "auto",
    pretrained: bool = True,
    local_files_only: bool = False,
    threshold: float = 0.5,
    train_count: int | None = None,
    val_count: int | None = None,
) -> Path:
    """Fine-tune the detector on pre-built, already-transformed datasets.

    ``train_dataset``/``val_dataset`` must each yield ``(image_tensor, label)``
    pairs (i.e. the transform has already been applied) -- this is the
    integration point for data sources that have no local file path to hand
    :class:`detector.data.ImageDataset`, such as one that streams images
    over HTTP.
    """

    if train_count is None:
        train_count = len(train_dataset)  # type: ignore
    if val_count is None:
        val_count = len(val_dataset)  # type: ignore
    if not train_count or not val_count:
        raise ValueError("Training and validation datasets must both be non-empty.")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    if learning_rate <= 0 or weight_decay < 0 or num_workers < 0:
        raise ValueError("Invalid optimizer or worker settings.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    _seed_everything(seed)
    resolved_device = resolve_device(device)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    best_checkpoint = destination / "best.pt"
    history_path = destination / "training.csv"

    generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": resolved_device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_options
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    model = create_detector(
        pretrained=pretrained,
        device=resolved_device,
        local_files_only=local_files_only,
    )
    model.configure_finetuning()
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_function = nn.BCEWithLogitsLoss()
    best_f1 = -1.0

    fields = ["epoch", "train_loss", "val_loss", "accuracy", "precision", "recall", "f1"]
    print(f"Training on {resolved_device} ({train_count} train, {val_count} val)")
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            train_loss = _train_epoch(
                model, train_loader, optimizer, loss_function, resolved_device
            )
            metrics = _validate(
                model, val_loader, loss_function, resolved_device, threshold
            )
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
            writer.writerow(row)
            history_file.flush()
            print(
                f"Epoch {epoch:>2}/{epochs} | train {train_loss:.4f} | "
                f"val {metrics['loss']:.4f} | F1 {metrics['f1']:.4f}"
            )

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                save_checkpoint(
                    best_checkpoint,
                    model,
                    {
                        "epoch": epoch,
                        "threshold": threshold,
                        "f1": metrics["f1"],
                        "validation_metrics": metrics,
                        "seed": seed,
                    },
                )

    print(f"Best validation F1: {best_f1:.4f} | {best_checkpoint}")
    return best_checkpoint
