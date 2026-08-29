"""Small, readable training loop for the binary image detector."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, WeightedRandomSampler

from .data import ImageDataset, ImageRecord
from .model import (
    Detector,
    HybridDetector,
    accelerator_pin_memory,
    create_detector,
    create_hybrid_detector,
    resolve_device,
    save_checkpoint,
    xpu_available,
)
from .transforms import build_eval_transform, build_train_transform

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if xpu_available():
        torch.xpu.manual_seed_all(seed)


def _binary_metrics(
    logits: torch.Tensor, labels: torch.Tensor, threshold: float
) -> dict[str, float]:
    predictions = torch.sigmoid(logits) >= threshold
    targets = labels >= 0.5
    true_positive = int((predictions & targets).sum().item())
    false_positive = int((predictions & ~targets).sum().item())
    false_negative = int((~predictions & targets).sum().item())
    true_negative = int((~predictions & ~targets).sum().item())
    correct = int((predictions == targets).sum().item())

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": correct / max(len(targets), 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": false_positive / max(false_positive + true_negative, 1),
        "fnr": false_negative / max(false_negative + true_positive, 1),
    }


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Binary ROC-AUC via the Mann-Whitney U statistic with tie-averaged
    ranks. The brief scores 0.5*AUC_clean + 0.5*AUC_robust, so this -- not
    F1 at a fixed threshold -- is what checkpoint selection optimizes.
    """

    pos = labels > 0.5
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _best_balanced_threshold(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[float, float, float]:
    """Sweep candidate thresholds and return the one minimizing
    ``max(FPR, FNR)`` on the validation scores, plus that FPR and FNR.

    The competition goal is FPR and FNR both <= 3%; a fixed 0.5 cut on a
    class-imbalanced pool (SID_Set is ~1:2 real:fake) systematically
    trades one error rate against the other, so the operating threshold is
    calibrated here and stored in the checkpoint rather than assumed.
    """

    pos = labels > 0.5
    neg = ~pos
    n_pos = max(int(pos.sum()), 1)
    n_neg = max(int(neg.sum()), 1)
    best = (0.5, 1.0, 1.0)
    best_obj = 2.0
    for thr in np.linspace(0.02, 0.98, 97):
        pred = scores >= thr
        fpr = float((pred & neg).sum()) / n_neg
        fnr = float((~pred & pos).sum()) / n_pos
        obj = max(fpr, fnr)
        if obj < best_obj or (obj == best_obj and abs(fpr - fnr) < abs(best[1] - best[2])):
            best_obj = obj
            best = (float(thr), fpr, fnr)
    return best


def _dataset_labels(dataset: Dataset) -> list[int] | None:
    """Best-effort recovery of per-sample integer labels from the dataset
    wrappers this project uses (ImageDataset, Subset, ConcatDataset, and the
    data-source _TransformSubset shims), for class-balanced sampling.
    Returns None if labels can't be determined without decoding images.
    """

    records = getattr(dataset, "records", None)
    if records is not None:
        try:
            return [int(r.label) for r in records]
        except AttributeError:
            return None
    if isinstance(dataset, Subset):
        parent = _dataset_labels(dataset.dataset)
        return None if parent is None else [parent[i] for i in dataset.indices]
    if isinstance(dataset, ConcatDataset):
        out: list[int] = []
        for part in dataset.datasets:
            part_labels = _dataset_labels(part)
            if part_labels is None:
                return None
            out.extend(part_labels)
        return out
    inner = getattr(dataset, "subset", None)
    if inner is not None:
        return _dataset_labels(inner)
    samples = getattr(dataset, "samples", None)
    if samples is not None:
        try:
            return [int(lbl) for _id, lbl in samples]
        except (TypeError, ValueError):
            return None
    return None


def _balanced_sampler(
    labels: list[int], seed: int, samples_per_epoch: int | None = None
) -> WeightedRandomSampler:
    """Sample so each class is drawn equally often per epoch (SID_Set is
    ~1:2 real:fake; without this the model drifts to a 'predict fake' bias
    that shows up as elevated FPR).

    ``samples_per_epoch`` caps how many images an epoch sees -- a fresh
    random draw from the whole pool each epoch, so a large corpus is used
    for diversity without every epoch touching all of it.
    """

    counts = np.bincount(np.asarray(labels), minlength=2).astype(np.float64)
    counts[counts == 0] = 1.0
    per_class_weight = 1.0 / counts
    weights = torch.tensor([per_class_weight[l] for l in labels], dtype=torch.double)
    gen = torch.Generator().manual_seed(seed)
    n = samples_per_epoch or len(labels)
    return WeightedRandomSampler(weights, num_samples=int(n), replacement=True, generator=gen)


def _autocast(device: torch.device):
    """bf16 autocast on CUDA/XPU, no-op on CPU. bf16 (not fp16) needs no
    GradScaler and is well supported on the Arc iGPU this project runs on.
    """

    if device.type in ("cuda", "xpu"):
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


def _feature_model(model: nn.Module) -> nn.Module:
    """The sub-module exposing ``.features()`` for the contrastive loss."""

    return getattr(model, "vit_detector", model)


def _train_epoch(
    model: Detector,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
    supcon_weight: float = 0.0,
) -> float:
    from .losses import supervised_contrastive_loss

    model.train()
    total_loss = 0.0
    sample_count = 0
    for batch in loader:
        if len(batch) == 3:  # two-view (supcon)
            v1, v2, labels = batch
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
        else:
            (images, labels) = batch
            v1 = images.to(device, non_blocking=True)
            v2 = None
        labels = labels.to(device=device, dtype=torch.float32, non_blocking=True).view(-1)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            logits1 = model(v1)
            loss = loss_function(logits1, labels)
            if v2 is not None:
                loss = loss + loss_function(model(v2), labels)
                if supcon_weight > 0.0:
                    fm = _feature_model(model)
                    f1 = torch.nn.functional.normalize(fm.features(v1).float(), dim=1)
                    f2 = torch.nn.functional.normalize(fm.features(v2).float(), dim=1)
                    loss = loss + supcon_weight * supervised_contrastive_loss(
                        f1, f2, (labels >= 0.5).long()
                    )
        loss.backward()
        # Added after the hybrid (ViT + frequency branch) model showed real
        # training instability in benchmark testing: even with the
        # frequency head's final layer zero-initialized (so the model
        # starts out mathematically identical to the ViT-only model, see
        # HybridDetector's docstring), loss diverged to ~3x higher than the
        # ViT-only baseline within the first epoch. Clipping bounds how
        # much a single early, poorly-scaled gradient step (from the
        # freshly-initialized frequency branch) can perturb the model.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    calibrate_on_noise: bool = True,
) -> dict[str, float]:
    model.eval()
    logits_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    noisy_logits_parts: list[torch.Tensor] = []
    total_loss = 0.0
    sample_count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device=device, dtype=torch.float32, non_blocking=True).view(-1)
        with _autocast(device):
            logits = model(images)
            loss = loss_function(logits, labels)
            if calibrate_on_noise:
                # Score a moderately noise-corrupted copy too, so the stored
                # operating threshold is a compromise that also holds up
                # under noise (the degradation that most hurt FNR in earlier
                # iterations). ~0.35 in normalized space ~= pixel sigma 0.08,
                # i.e. between the eval matrix's 0.05 and 0.10 rows -- strong
                # enough to matter, not so strong it drags the threshold to
                # a clean-FPR-wrecking extreme. Reported AUC/acc stay clean.
                noise = torch.randn_like(images) * 0.35
                noisy_logits_parts.append(model(images + noise).float().cpu())

        batch_size = labels.numel()
        total_loss += float(loss.item()) * batch_size
        sample_count += batch_size
        logits_parts.append(logits.float().cpu())
        label_parts.append(labels.cpu())

    all_logits = torch.cat(logits_parts)
    all_labels = torch.cat(label_parts)
    metrics = _binary_metrics(all_logits, all_labels, threshold)
    metrics["loss"] = total_loss / max(sample_count, 1)

    labels_np = all_labels.numpy()
    scores_np = torch.sigmoid(all_logits).numpy()
    metrics["roc_auc"] = _roc_auc(labels_np, scores_np)

    if noisy_logits_parts:
        noisy_scores = torch.sigmoid(torch.cat(noisy_logits_parts)).numpy()
        # Weight clean 2:1 over noisy -- clean is the primary operating
        # condition; noisy just keeps the threshold from being set so high
        # that heavy noise tanks FNR.
        cal_scores = np.concatenate([scores_np, scores_np, noisy_scores])
        cal_labels = np.concatenate([labels_np, labels_np, labels_np])
    else:
        cal_scores, cal_labels = scores_np, labels_np
    bal_thr, _bf, _bn = _best_balanced_threshold(cal_labels, cal_scores)
    # Report the FPR/FNR this threshold gives on the *clean* val, so the
    # training log stays interpretable.
    clean_pred = scores_np >= bal_thr
    pos = labels_np > 0.5
    bal_fpr = float((clean_pred & ~pos).sum()) / max(int((~pos).sum()), 1)
    bal_fnr = float((~clean_pred & pos).sum()) / max(int(pos.sum()), 1)
    metrics["balanced_threshold"] = bal_thr
    metrics["balanced_fpr"] = bal_fpr
    metrics["balanced_fnr"] = bal_fnr
    metrics["balanced_max_fpfn"] = max(bal_fpr, bal_fnr)
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
    model_type: str = "vit",
    vit_checkpoint: str | Path | None = None,
    branch_kind: str = "wavelet",
    loss_pos_weight: float = 1.0,
    select_metric: str = "roc_auc",
    balance_classes: bool = False,
    lr_schedule: str = "constant",
    warmup_frac: float = 0.05,
    supcon_weight: float = 0.0,
) -> Path:
    """Fine-tune the detector on local-disk image records and return the
    best checkpoint path (selected on ``select_metric``, default validation
    ROC-AUC -- the quantity the brief actually scores). Thin wrapper around
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
        model_type=model_type,
        vit_checkpoint=vit_checkpoint,
        branch_kind=branch_kind,
        loss_pos_weight=loss_pos_weight,
        select_metric=select_metric,
        balance_classes=balance_classes,
        lr_schedule=lr_schedule,
        warmup_frac=warmup_frac,
        supcon_weight=supcon_weight,
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
    model_type: str = "vit",
    vit_checkpoint: str | Path | None = None,
    branch_kind: str = "wavelet",
    loss_pos_weight: float = 1.0,
    select_metric: str = "roc_auc",
    balance_classes: bool = False,
    samples_per_epoch: int | None = None,
    lr_schedule: str = "constant",
    warmup_frac: float = 0.05,
    supcon_weight: float = 0.0,
) -> Path:
    """Fine-tune the detector on pre-built, already-transformed datasets.

    ``train_dataset``/``val_dataset`` must each yield ``(image_tensor, label)``
    pairs (i.e. the transform has already been applied) -- this is the
    integration point for data sources that have no local file path to hand
    :class:`detector.data.ImageDataset`, such as one that streams images
    over HTTP.

    ``model_type``: ``"vit"`` (default, Community Forensics alone) or
    ``"hybrid"`` (Community Forensics + a frequency branch, fused -- see
    detector.model.HybridDetector).
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
        "pin_memory": accelerator_pin_memory(resolved_device),
    }
    if num_workers > 0:
        # The per-sample transform (PIL JPEG re-encode + blur + numpy noise +
        # two resizes) is heavy enough that single-process loading leaves the
        # GPU ~85% idle (measured, iteration 1). Persistent workers amortize
        # the Windows spawn + torch/oneAPI re-import cost across epochs.
        loader_options["persistent_workers"] = True
        loader_options["prefetch_factor"] = 4

    sampler = None
    if balance_classes:
        labels = _dataset_labels(train_dataset)
        if labels is None:
            print("[WARN] balance_classes requested but labels unavailable; using shuffle")
        else:
            sampler = _balanced_sampler(labels, seed, samples_per_epoch)
            pos = int(sum(labels))
            print(
                f"Class-balanced sampling: {len(labels) - pos} real / {pos} fake "
                f"-> drawn 50/50 per epoch"
            )
    train_loader = DataLoader(
        train_dataset,
        shuffle=(sampler is None),
        sampler=sampler,
        generator=generator,
        **loader_options,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    if model_type == "hybrid":
        model: Detector | HybridDetector = create_hybrid_detector(
            pretrained=pretrained,
            device=resolved_device,
            local_files_only=local_files_only,
            vit_checkpoint=vit_checkpoint,
            branch_kind=branch_kind,
        )
    elif model_type == "vit":
        model = create_detector(
            pretrained=pretrained,
            device=resolved_device,
            local_files_only=local_files_only,
        )
    else:
        raise ValueError(f"Unknown model_type {model_type!r}; choose 'vit' or 'hybrid'.")
    model.configure_finetuning()
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    if lr_schedule not in ("constant", "cosine"):
        raise ValueError(f"Unknown lr_schedule {lr_schedule!r}; choose 'constant' or 'cosine'.")
    scheduler = None
    if lr_schedule == "cosine":
        warmup_epochs = max(1, round(epochs * warmup_frac)) if warmup_frac > 0 else 0

        def _lr_lambda(epoch_idx: int) -> float:  # epoch_idx is 0-based
            if epoch_idx < warmup_epochs:
                return (epoch_idx + 1) / (warmup_epochs + 1)
            progress = (epoch_idx - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    if loss_pos_weight and loss_pos_weight != 1.0:
        loss_function: nn.Module = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(float(loss_pos_weight), device=resolved_device)
        )
    else:
        loss_function = nn.BCEWithLogitsLoss()
    # Validation loss stays unweighted so val_loss is comparable across runs.
    eval_loss_function = nn.BCEWithLogitsLoss()
    if select_metric not in ("roc_auc", "f1", "balanced_max_fpfn"):
        raise ValueError(f"Unknown select_metric {select_metric!r}.")
    best_score = -1.0

    fields = [
        "epoch", "train_loss", "val_loss", "accuracy", "precision", "recall",
        "f1", "roc_auc", "fpr", "fnr", "balanced_threshold", "balanced_fpr",
        "balanced_fnr",
    ]
    print(
        f"Training on {resolved_device} ({train_count} train, {val_count} val) "
        f"| select={select_metric} pos_weight={loss_pos_weight}"
    )
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            train_loss = _train_epoch(
                model, train_loader, optimizer, loss_function, resolved_device,
                supcon_weight=supcon_weight,
            )
            if scheduler is not None:
                scheduler.step()
            metrics = _validate(
                model, val_loader, eval_loss_function, resolved_device, threshold
            )
            writer.writerow({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "fpr": metrics["fpr"],
                "fnr": metrics["fnr"],
                "balanced_threshold": metrics["balanced_threshold"],
                "balanced_fpr": metrics["balanced_fpr"],
                "balanced_fnr": metrics["balanced_fnr"],
            })
            history_file.flush()
            print(
                f"Epoch {epoch:>2}/{epochs} | train {train_loss:.4f} | "
                f"val {metrics['loss']:.4f} | AUC {metrics['roc_auc']:.4f} | "
                f"F1 {metrics['f1']:.4f} | bal thr={metrics['balanced_threshold']:.2f} "
                f"FPR={metrics['balanced_fpr']:.3f} FNR={metrics['balanced_fnr']:.3f}"
            )

            score = metrics[select_metric]
            if select_metric == "balanced_max_fpfn":
                score = -score  # lower is better -> maximize negative
            if score > best_score:
                best_score = score
                save_checkpoint(
                    best_checkpoint,
                    model,
                    {
                        "epoch": epoch,
                        # Operating threshold calibrated on validation to
                        # minimize max(FPR, FNR); evaluation/predict read
                        # this from metadata (see evaluation._threshold).
                        "threshold": round(float(metrics["balanced_threshold"]), 4),
                        "fixed_threshold_0.5_metrics": {
                            k: metrics[k] for k in ("accuracy", "f1", "fpr", "fnr")
                        },
                        "roc_auc": metrics["roc_auc"],
                        "f1": metrics["f1"],
                        "select_metric": select_metric,
                        "validation_metrics": metrics,
                        "loss_pos_weight": loss_pos_weight,
                        "seed": seed,
                    },
                )

    print(
        f"Best validation {select_metric}: "
        f"{best_score if select_metric != 'balanced_max_fpfn' else -best_score:.4f} "
        f"| {best_checkpoint}"
    )
    return best_checkpoint
