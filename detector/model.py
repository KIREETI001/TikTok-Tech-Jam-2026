"""Community Forensics image detector used by the training pipeline.

The model returns one logit per image. Positive logits mean AI-generated
(``fake``), while negative logits mean real.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

MODEL_ID = "OwensLab/commfor-model-224"
MODEL_REVISION = "26afc31e6b40c312c3fd42c05a758be62446215b"
MODEL_FILENAME = "model.safetensors"
MODEL_SHA256 = "a6cc439d5a6d2dfadd60c77d27a2838ad55b34e601ecd30f46ad97266d6ac4e0"
TIMM_ARCHITECTURE = "vit_small_patch16_224.augreg_in21k_ft_in1k"
MODEL_PARAMETERS = 21_666_049
FEATURE_WIDTH = 384


class DetectorError(RuntimeError):
    """Raised when the pinned detector cannot be built or loaded."""


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Choose CUDA when available, otherwise CPU."""

    if str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def verify_checkpoint(
    path: str | Path, expected_sha256: str = MODEL_SHA256
) -> str:
    """Verify the official weights before deserializing them."""

    checkpoint = Path(path)
    digest = hashlib.sha256()
    try:
        with checkpoint.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DetectorError(f"Could not read checkpoint {checkpoint}: {exc}") from exc

    observed = digest.hexdigest()
    if observed.lower() != expected_sha256.lower():
        raise DetectorError(
            "Official checkpoint SHA-256 mismatch: "
            f"expected {expected_sha256.lower()}, got {observed}"
        )
    return observed


def _build_vit() -> nn.Module:
    try:
        import timm
    except ImportError as exc:
        raise DetectorError("Install timm to build the detector.") from exc

    vit = timm.create_model(TIMM_ARCHITECTURE, pretrained=False)
    if int(getattr(vit, "num_features", 0)) != FEATURE_WIDTH:
        raise DetectorError(
            f"Unexpected {TIMM_ARCHITECTURE} feature width; expected {FEATURE_WIDTH}."
        )
    vit.head = nn.Linear(FEATURE_WIDTH, 1)
    return vit


class Detector(nn.Module):
    """Thin wrapper around the pinned ViT-S/224 binary classifier."""

    def __init__(self, vit: nn.Module) -> None:
        super().__init__()
        self.vit = vit

    @classmethod
    def from_architecture(
        cls, device: str | torch.device = "auto"
    ) -> "Detector":
        """Create the exact architecture with randomly initialized weights."""

        model = cls(_build_vit())
        observed = parameter_count(model)
        if observed != MODEL_PARAMETERS:
            raise DetectorError(
                f"Unexpected parameter count: expected {MODEL_PARAMETERS:,}, "
                f"got {observed:,}. Check the pinned timm version."
            )
        return model.to(resolve_device(device))

    @classmethod
    def from_pretrained(
        cls,
        device: str | torch.device = "auto",
        *,
        local_files_only: bool = False,
        cache_dir: str | Path | None = None,
    ) -> "Detector":
        """Download, verify, and load the pinned Community Forensics weights."""

        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
        except ImportError as exc:
            raise DetectorError(
                "Install huggingface-hub and safetensors to load pretrained weights."
            ) from exc

        checkpoint = hf_hub_download(
            repo_id=MODEL_ID,
            filename=MODEL_FILENAME,
            revision=MODEL_REVISION,
            local_files_only=local_files_only,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
        verify_checkpoint(checkpoint)

        model = cls.from_architecture(device="cpu")
        model.load_state_dict(load_file(checkpoint, device="cpu"), strict=True)
        return model.to(resolve_device(device)).eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return fake logits with shape ``[batch]``."""

        return self.vit(images).squeeze(-1)

    def configure_finetuning(self) -> None:
        """Train only the classifier head, final transformer block, and norm."""

        self.requires_grad_(False)
        self.vit.head.requires_grad_(True)
        self.vit.blocks[-1].requires_grad_(True)
        self.vit.norm.requires_grad_(True)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]


def create_detector(
    *,
    pretrained: bool = True,
    device: str | torch.device = "auto",
    local_files_only: bool = False,
    cache_dir: str | Path | None = None,
) -> Detector:
    """Create either the official pretrained detector or architecture only."""

    if pretrained:
        return Detector.from_pretrained(
            device=device,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
    return Detector.from_architecture(device=device)


def save_checkpoint(
    path: str | Path,
    model: Detector,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save model weights and small run metadata in one portable file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    details = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "architecture": TIMM_ARCHITECTURE,
        "parameter_count": parameter_count(model),
        "positive_label": "fake",
        **dict(metadata or {}),
    }
    torch.save(
        {
            "format_version": 1,
            "model_state": model.state_dict(),
            "metadata": details,
        },
        destination,
    )
    return destination


def load_checkpoint(
    path: str | Path, device: str | torch.device = "auto"
) -> tuple[Detector, dict[str, Any]]:
    """Load a checkpoint produced by :func:`save_checkpoint`."""

    checkpoint = Path(path)
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise DetectorError(f"Could not load checkpoint {checkpoint}: {exc}") from exc

    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise DetectorError(f"Unsupported checkpoint format: {checkpoint}")
    state = payload.get("model_state")
    metadata = payload.get("metadata", {})
    if not isinstance(state, dict) or not isinstance(metadata, dict):
        raise DetectorError(f"Invalid checkpoint contents: {checkpoint}")
    if metadata.get("architecture") != TIMM_ARCHITECTURE:
        raise DetectorError("Checkpoint architecture does not match this pipeline.")

    model = Detector.from_architecture(device="cpu")
    model.load_state_dict(state, strict=True)
    model.to(resolve_device(device)).eval()
    return model, dict(metadata)
