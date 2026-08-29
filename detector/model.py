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
EMBED_DIM = 256


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

    MODEL_TYPE = "vit"

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


class FrequencyBranch(nn.Module):
    """Log-magnitude 2D FFT spectrum through a shallow CNN, plus an
    azimuthally-averaged radial power profile through a small MLP.

    Ported from the abandoned 4-branch fusion model considered during
    Phase 3 (``upstream/main``'s ``model.py``) -- revived here as the one
    piece of that design worth keeping: a hybrid semantic+spectral model is
    the organizer brief's own stated key insight ("best detectors combine
    high-level semantics + low-level frequency patches"), and it's cheap
    (no pretraining stage, shallow network) unlike the noise/camera and
    CLIP branches, which were left out for that reason.

    Why shallow: frequency artifacts (GAN/diffusion upsampling
    checkerboards, unnatural high-frequency power-spectrum decay) are
    global statistical regularities, not deep spatial hierarchies -- a
    deep network here is wasted capacity and more prone to overfitting the
    spectral signature of whichever generators happen to be in the
    training set, rather than learning the general "natural images follow
    roughly a 1/f^2 power-law decay" cue this branch is meant to capture.
    """

    def __init__(self, embed_dim: int = EMBED_DIM, image_size: int = 224) -> None:
        super().__init__()
        self.register_buffer("radial_bins", self._build_radial_bins(image_size), persistent=False)
        num_radial_bins = int(self.radial_bins.max().item()) + 1

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.radial_mlp = nn.Sequential(nn.Linear(num_radial_bins, 64), nn.ReLU(inplace=True))
        self.project = nn.Linear(64 + 64, embed_dim)

    @staticmethod
    def _build_radial_bins(size: int) -> torch.Tensor:
        # Integer distance-from-center per pixel, used to group FFT
        # magnitude into concentric rings for the azimuthal average -- this
        # collapses the 2D spectrum into a compact 1D "power vs frequency"
        # profile.
        ys, xs = torch.meshgrid(
            torch.arange(size) - size // 2, torch.arange(size) - size // 2, indexing="ij"
        )
        radius = torch.sqrt(xs.float() ** 2 + ys.float() ** 2)
        return radius.round().long()

    def _azimuthal_profile(self, magnitude: torch.Tensor) -> torch.Tensor:
        # magnitude: (B, H, W) log-magnitude spectrum, already averaged
        # over color channels. Returns (B, num_bins) mean magnitude per
        # radial bin, computed with a single vectorized scatter_add (no
        # per-sample Python loop).
        B = magnitude.shape[0]
        bins = self.radial_bins.flatten()
        num_bins = int(bins.max().item()) + 1
        flat = magnitude.reshape(B, -1)

        batch_offset = (torch.arange(B, device=magnitude.device) * num_bins).unsqueeze(1)
        flat_index = (bins.unsqueeze(0) + batch_offset).reshape(-1)

        sums = torch.zeros(B * num_bins, device=magnitude.device, dtype=magnitude.dtype)
        sums.scatter_add_(0, flat_index, flat.reshape(-1))

        counts = torch.zeros(num_bins, device=magnitude.device, dtype=magnitude.dtype)
        counts.scatter_add_(0, bins, torch.ones_like(bins, dtype=magnitude.dtype))
        counts = counts.repeat(B)

        return (sums / counts.clamp(min=1)).reshape(B, num_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fft = torch.fft.fft2(x)
        fft = torch.fft.fftshift(fft, dim=(-2, -1))
        magnitude = torch.log1p(fft.abs())  # log1p for numerical stability (huge raw dynamic range)

        spatial_feat = self.cnn(magnitude).flatten(1)

        radial_profile = self._azimuthal_profile(magnitude.mean(dim=1))
        radial_feat = self.radial_mlp(radial_profile)

        return self.project(torch.cat([spatial_feat, radial_feat], dim=1))


class HybridDetector(nn.Module):
    """Community Forensics ViT (semantic) + a frequency branch (spectral),
    fused as a zero-initialized residual correction to the ViT's own logit.

    Design choice, deliberately conservative given the time available: the
    ViT sub-detector is used as a fixed, already-proven signal, and the
    frequency branch only ever *adds a correction* to its logit -- not the
    full cross-attention fusion the abandoned 4-branch design used.

    First attempt concatenated the ViT's single logit with the frequency
    branch's 256-dim embedding into a freshly-initialized Linear layer.
    Measured, not assumed: on a 3-epoch/2400-image benchmark this failed to
    learn at all (F1 stuck at 0.667, loss converging to ln(2)) while the
    ViT-only model on the same data reached F1 0.83 -- a fresh Linear layer
    treats all 257 input dims symmetrically at init, so the one genuinely
    strong signal (the ViT logit) gets diluted into 256 dims of untrained
    noise rather than preserved. Raising the learning rate 100x didn't fix
    it either (still ~0.66, one epoch's val loss spiked to 0.88) --
    confirming the problem was the fusion design, not the LR.

    Fix: the frequency head's final layer is zero-initialized, so at
    step 0 the hybrid model's output is *exactly* the ViT-only logit
    (freq_adjustment == 0 for every input) -- training only has to learn
    small additive corrections where the frequency signal actually helps,
    rather than rediscover the ViT's already-proven signal from scratch
    inside an untrained fusion layer. This is the standard "zero-init
    residual" pattern for adding a new branch to an already-good model
    without destabilizing it.
    """

    MODEL_TYPE = "hybrid"

    def __init__(self, vit_detector: Detector, freq_branch: FrequencyBranch) -> None:
        super().__init__()
        self.vit_detector = vit_detector
        self.freq_branch = freq_branch
        self.freq_head = nn.Sequential(
            nn.Linear(EMBED_DIM, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
        # Near-zero-init the last layer (NOT exactly zero -- see below) so
        # freq_head(freq_embed) starts close to 0 for any input, meaning
        # forward() below starts out close to the plain ViT detector -- see
        # class docstring.
        #
        # A full 5-epoch/99k-image run (runs/mixed_hybrid_v1) came back with
        # F1 completely flat (0.9405-0.9408) and train loss not decreasing --
        # the branch was learning nothing. Diagnosed with a single
        # forward/backward pass on a real batch: every upstream parameter
        # (freq_branch.cnn.*, .radial_mlp.*, .project.*, freq_head[0]) had
        # gradient EXACTLY 0.0 -- not small, exactly zero. An exact-zero
        # weight on the last Linear multiplicatively gates the entire
        # backward chain through it (d(loss)/d(earlier params) factors
        # through this weight, which is 0), so nothing upstream can learn
        # until this weight itself has moved away from zero -- which under
        # lr=1e-5 with weight_decay=0.01 pulling it back every step, took
        # 18,660 steps to reach only ~0.002 magnitude. The branch was
        # effectively frozen for the whole run.
        # Fix: initialize with a small but genuinely nonzero std so the
        # gate isn't exactly closed -- gradients flow from step 1, while
        # 1e-3 is small enough that the hybrid's initial predictions are
        # still close to the ViT-only checkpoint's (unlike the original
        # concat-fusion design's fresh Linear(257, 64), which distributed
        # the ViT's one strong signal across 256 untrained dims instead of
        # gating a correction term).
        nn.init.normal_(self.freq_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.freq_head[-1].bias)

    @classmethod
    def from_architecture(cls, device: str | torch.device = "auto") -> "HybridDetector":
        """Create the exact architecture with randomly initialized weights
        (used when reloading a saved hybrid checkpoint)."""

        model = cls(Detector.from_architecture(device="cpu"), FrequencyBranch())
        return model.to(resolve_device(device))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        vit_logit = self.vit_detector(images)  # (B,)
        freq_embed = self.freq_branch(images)  # (B, EMBED_DIM)
        freq_adjustment = self.freq_head(freq_embed).squeeze(-1)  # (B,), starts at 0
        return vit_logit + freq_adjustment

    def configure_finetuning(self) -> None:
        """Freeze the ViT sub-detector entirely; train only the new
        frequency branch and fusion head.

        Deliberately different from Detector.configure_finetuning's
        "unfreeze head + last block + norm": that ViT has already been
        fine-tuned across three prior training iterations on this exact
        task and is the strongest single signal we have (checkpoint 6:
        95.15%/90.87% PS5, 86.39%/86.45% SID_Set) -- re-training it further
        here risks destabilizing that proven fit for no benefit, when the
        only genuinely untrained components are the frequency branch and
        fusion head. Also follows Ojha et al. (CVPR 2023)'s finding that
        heavier fine-tuning of the semantic backbone tends to learn
        narrower, more generator-specific shortcuts.
        """

        self.requires_grad_(False)
        self.freq_branch.requires_grad_(True)
        self.freq_head.requires_grad_(True)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]


def create_hybrid_detector(
    *,
    pretrained: bool = True,
    device: str | torch.device = "auto",
    local_files_only: bool = False,
    cache_dir: str | Path | None = None,
    vit_checkpoint: str | Path | None = None,
) -> HybridDetector:
    """Create the hybrid (ViT + frequency branch) detector.

    ``vit_checkpoint``, if given, loads the ViT half from a checkpoint
    already produced by this pipeline (e.g. ``runs/mixed_v2/best.pt``)
    instead of the raw pinned Community Forensics weights -- this matters:
    the raw checkpoint has a strong "predict real" bias on this project's
    data (measured directly: predicts real for 100% of a 480-image
    val set, 50/50 real/fake), since it was never fine-tuned on this task's
    specific data at all. HybridDetector's whole design rationale is
    building on "the model already validated across three training
    iterations" -- omitting this argument silently builds on the much
    weaker raw checkpoint instead, which is *not* that model.
    """

    if vit_checkpoint is not None:
        vit_detector, _metadata = load_checkpoint(vit_checkpoint, device="cpu")
        if not isinstance(vit_detector, Detector):
            raise DetectorError(
                f"{vit_checkpoint} is not a plain ViT checkpoint (found "
                f"model_type={_metadata.get('model_type')!r}); pass a checkpoint "
                "produced with model_type 'vit'."
            )
        return HybridDetector(vit_detector, FrequencyBranch()).to(resolve_device(device))

    vit_detector = create_detector(
        pretrained=pretrained,
        device="cpu",
        local_files_only=local_files_only,
        cache_dir=cache_dir,
    )
    freq_branch = FrequencyBranch()
    return HybridDetector(vit_detector, freq_branch).to(resolve_device(device))


def save_checkpoint(
    path: str | Path,
    model: Detector | HybridDetector,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save model weights and small run metadata in one portable file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    details = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "architecture": TIMM_ARCHITECTURE,
        "model_type": getattr(model, "MODEL_TYPE", "vit"),
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
) -> tuple[Detector | HybridDetector, dict[str, Any]]:
    """Load a checkpoint produced by :func:`save_checkpoint`. Dispatches on
    the ``model_type`` metadata field ("vit" or "hybrid"); absent means
    "vit", so every checkpoint saved before the hybrid model existed still
    loads unchanged.
    """

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

    model_type = metadata.get("model_type", "vit")
    if model_type == "hybrid":
        model: Detector | HybridDetector = HybridDetector.from_architecture(device="cpu")
    elif model_type == "vit":
        model = Detector.from_architecture(device="cpu")
    else:
        raise DetectorError(f"Unknown model_type in checkpoint metadata: {model_type!r}")
    model.load_state_dict(state, strict=True)
    model.to(resolve_device(device)).eval()
    return model, dict(metadata)
