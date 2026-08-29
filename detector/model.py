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


def xpu_available() -> bool:
    """True when a usable Intel GPU (XPU) backend is present."""

    return hasattr(torch, "xpu") and torch.xpu.is_available()


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Choose the best available accelerator, otherwise CPU.

    Priority when ``device`` is ``"auto"``: CUDA, then Intel XPU (Arc), then
    CPU. XPU support was added when this project moved to a machine with an
    Intel Arc iGPU and no NVIDIA GPU; the rest of the pipeline treats an XPU
    device like a CUDA one (autocast, ``pin_memory``, device-side seeding).
    """

    if str(device).lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if xpu_available():
            return torch.device("xpu")
        return torch.device("cpu")
    return torch.device(device)


def accelerator_pin_memory(device: torch.device) -> bool:
    """Whether ``pin_memory`` / ``non_blocking`` transfers help for this device."""

    return device.type in ("cuda", "xpu")


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

    def features(self, images: torch.Tensor) -> torch.Tensor:
        """Pre-head pooled embedding, shape ``[batch, FEATURE_WIDTH]`` --
        the representation a supervised-contrastive loss operates on.
        """

        tokens = self.vit.forward_features(images)
        return self.vit.forward_head(tokens, pre_logits=True)

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


class WaveletBranch(nn.Module):
    """SAFE-style local frequency branch (KDD 2025, arXiv 2408.06741).

    Input is NOT the RGB image: it is the 2D DWT diagonal high-frequency
    sub-band (``bior1.3``, one level), upsampled back to the input size,
    passed through a truncated ResNet stem. SAFE's key results:

    - the diagonal detail band is where generator up-sampling artifacts
      concentrate, and unlike a global FFT it is a *local* statistic that
      survives crops and mild degradation (the briefing deck slide 8:
      "low-level frequency **patches**", not a whole-image spectrum);
    - a model reading this band reaches 82% accuracy on ADM having trained
      only on ProGAN -- i.e. it transfers across the pixel-space vs latent
      diffusion split that our ViT-only model fails on (0.65 AUC on ADM);
    - the branch is tiny: conv stem + two residual stages, ~1.4M parameters.

    Replaces this project's abandoned global-FFT ``FrequencyBranch``, which
    a parallel pipeline measured at 0.457 AUC on unseen generators (below
    chance -- it memorised training-generator spectra).
    """

    def __init__(self, embed_dim: int = EMBED_DIM, wave: str = "bior1.3") -> None:
        super().__init__()
        try:
            from pytorch_wavelets import DWTForward
        except ImportError as exc:
            raise DetectorError(
                "WaveletBranch needs pytorch_wavelets (pip install PyWavelets pytorch_wavelets)."
            ) from exc
        self.dwt = DWTForward(J=1, mode="symmetric", wave=wave)

        def block(cin: int, cout: int, stride: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            )

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            block(32, 64, stride=2),
            block(64, 128, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.project = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DWTForward returns (low, [high]) where high is (B, C, 3, H/2, W/2);
        # index 2 is the diagonal (HH) detail sub-band.
        with torch.autocast(device_type=x.device.type, enabled=False):
            _low, high = self.dwt(x.float())
        diagonal = high[0][:, :, 2, :, :]
        diagonal = nn.functional.interpolate(
            diagonal, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        feat = self.stem(diagonal.to(x.dtype)).flatten(1)
        return self.project(feat)


import os

# CLIP ViT-B/16 by default -- ~4x faster per forward than ViT-L/14 on the
# Intel Arc iGPU (measured ~6 s/step at bs40 for L, the sprint can't afford
# it). Override with TTJ_CLIP_MODEL=vit_large_patch14_clip_224.openai when
# compute allows.
_CLIP_MODEL = os.environ.get("TTJ_CLIP_MODEL", "vit_base_patch16_clip_224.openai")
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_CLIP_MEAN = (0.4814546, 0.4578275, 0.4082107)
_CLIP_STD = (0.2686295, 0.2613026, 0.2757711)


class SemanticBranch(nn.Module):
    """Frozen large vision transformer (OpenAI CLIP ViT-L/14) as an
    additive semantic branch.

    A parallel pipeline measured this as its single biggest architecture
    lever: +0.069 Final Score on unseen generators, and the frozen branch's
    own head (0.897 AUC) beat their whole fused model. The mechanism their
    per-branch numbers showed: a *fine-tuned* backbone loses 0.203 AUC from
    seen to unseen generators, a *frozen* one loses only 0.086 -- fine-tuning
    on our training generators teaches their specific quirks, a frozen model
    that never saw our data cannot overfit to it.

    The encoder is frozen and kept in eval() permanently; only the LayerNorm
    + projection (and the zero-init fusion head in HybridDetector) train.
    The pipeline normalises inputs with ImageNet statistics for the ViT-S
    detector; this branch re-normalises to CLIP's statistics before the
    encoder.
    """

    def __init__(self, embed_dim: int = EMBED_DIM, model_name: str = _CLIP_MODEL) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise DetectorError("SemanticBranch needs timm.") from exc
        self.clip_model = model_name
        self.encoder = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        feature_width = int(self.encoder.num_features)
        self.norm = nn.LayerNorm(feature_width)
        self.project = nn.Linear(feature_width, embed_dim)

        shift = torch.tensor(
            [(i - c) / s for i, c, s in zip(_IMAGENET_MEAN, _CLIP_MEAN, _CLIP_STD)]
        ).view(1, 3, 1, 1)
        scale = torch.tensor(
            [i / c for i, c in zip(_IMAGENET_STD, _CLIP_STD)]
        ).view(1, 3, 1, 1)
        self.register_buffer("renorm_scale", scale, persistent=False)
        self.register_buffer("renorm_shift", shift, persistent=False)

    def train(self, mode: bool = True) -> "SemanticBranch":
        super().train(mode)
        self.encoder.eval()  # never leaves eval
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        clip_input = x * self.renorm_scale + self.renorm_shift
        with torch.no_grad():
            feature = self.encoder(clip_input)
        return self.project(self.norm(feature.float())).to(x.dtype)


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

    @staticmethod
    def _make_head() -> nn.Sequential:
        head = nn.Sequential(
            nn.Linear(EMBED_DIM, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
        # Near-zero-init the last layer (NOT exactly zero) so head(embed)
        # starts close to 0 for any input -- forward() starts out close to
        # the plain ViT detector, per the class docstring's design intent.
        #
        # A full 5-epoch/99k-image run (runs/mixed_hybrid_v1) with an exact
        # zero-init here came back with F1 completely flat (0.9405-0.9408)
        # and train loss not decreasing -- the branch learned nothing.
        # Diagnosed with a single forward/backward pass on a real batch:
        # every upstream parameter (branch.cnn.*, .radial_mlp.*, .project.*,
        # head[0]) had gradient EXACTLY 0.0 -- not small, exactly zero. An
        # exact-zero weight on the last Linear multiplicatively gates the
        # entire backward chain through it (d(loss)/d(earlier params)
        # factors through this weight, which is 0), so nothing upstream can
        # learn until this weight itself moves away from zero -- which
        # under lr=1e-5 with weight_decay=0.01 pulling it back every step,
        # took 18,660 steps to reach only ~0.002 magnitude. The branch was
        # effectively frozen for the whole run.
        #
        # Fix: initialize with a small but genuinely nonzero std so the
        # gate isn't exactly closed -- gradients flow from step 1, while
        # 1e-3 is small enough that the hybrid's initial predictions stay
        # close to the ViT-only checkpoint's.
        nn.init.normal_(head[-1].weight, std=1e-3)
        nn.init.zeros_(head[-1].bias)
        return head

    def __init__(
        self,
        vit_detector: Detector,
        freq_branch: nn.Module | list[nn.Module] | nn.ModuleList,
        branch_kind: str = "fft",
    ) -> None:
        super().__init__()
        self.branch_kind = branch_kind
        self.branch_kinds = [k.strip() for k in branch_kind.split(",") if k.strip()]
        self.vit_detector = vit_detector
        branches = freq_branch if isinstance(freq_branch, (list, nn.ModuleList)) else [freq_branch]
        self.branches = nn.ModuleList(branches)
        self.branch_heads = nn.ModuleList(self._make_head() for _ in self.branches)

    @staticmethod
    def _make_branch(branch_kind: str, clip_model: str | None = None) -> nn.Module:
        if branch_kind == "wavelet":
            return WaveletBranch()
        if branch_kind == "fft":
            return FrequencyBranch()
        if branch_kind == "clip":
            return SemanticBranch(model_name=clip_model or _CLIP_MODEL)
        raise DetectorError(
            f"Unknown hybrid branch_kind {branch_kind!r}; choose from 'wavelet', 'fft', 'clip' "
            "(comma-separate for multiple)."
        )

    @classmethod
    def _make_branches(cls, branch_kind: str, clip_model: str | None = None) -> nn.ModuleList:
        kinds = [k.strip() for k in branch_kind.split(",") if k.strip()]
        return nn.ModuleList(cls._make_branch(k, clip_model) for k in kinds)

    @classmethod
    def from_architecture(
        cls,
        device: str | torch.device = "auto",
        branch_kind: str = "fft",
        clip_model: str | None = None,
    ) -> "HybridDetector":
        """Create the exact architecture (used when reloading a saved hybrid
        checkpoint). Frozen encoder branches (``clip``) load their pretrained
        weights here; only the trained norm/projection/head come from the
        checkpoint."""

        model = cls(
            Detector.from_architecture(device="cpu"),
            cls._make_branches(branch_kind, clip_model),
            branch_kind=branch_kind,
        )
        return model.to(resolve_device(device))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logit = self.vit_detector(images)  # (B,)
        for branch, head in zip(self.branches, self.branch_heads):
            logit = logit + head(branch(images)).squeeze(-1)  # each starts at 0
        return logit

    def features(self, images: torch.Tensor) -> torch.Tensor:
        """Delegate to the ViT sub-detector's pre-head embedding (the
        contrastive loss operates on the semantic representation)."""

        return self.vit_detector.features(images)

    def configure_finetuning(self) -> None:
        """Freeze the ViT sub-detector entirely; train only the new
        branches and fusion heads.

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
        for branch, head in zip(self.branches, self.branch_heads):
            branch.requires_grad_(True)
            head.requires_grad_(True)
        # Frozen encoder branches (CLIP) stay frozen -- only their
        # LayerNorm + projection train.
        for branch in self.branches:
            encoder = getattr(branch, "encoder", None)
            if encoder is not None:
                encoder.requires_grad_(False)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]


def create_hybrid_detector(
    *,
    pretrained: bool = True,
    device: str | torch.device = "auto",
    local_files_only: bool = False,
    cache_dir: str | Path | None = None,
    vit_checkpoint: str | Path | None = None,
    branch_kind: str = "wavelet",
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

    branches = HybridDetector._make_branches(branch_kind)

    if vit_checkpoint is not None:
        vit_detector, _metadata = load_checkpoint(vit_checkpoint, device="cpu")
        if not isinstance(vit_detector, Detector):
            raise DetectorError(
                f"{vit_checkpoint} is not a plain ViT checkpoint (found "
                f"model_type={_metadata.get('model_type')!r}); pass a checkpoint "
                "produced with model_type 'vit'."
            )
        return HybridDetector(vit_detector, branches, branch_kind=branch_kind).to(
            resolve_device(device)
        )

    vit_detector = create_detector(
        pretrained=pretrained,
        device="cpu",
        local_files_only=local_files_only,
        cache_dir=cache_dir,
    )
    return HybridDetector(vit_detector, branches, branch_kind=branch_kind).to(
        resolve_device(device)
    )


def _plain(value: Any) -> Any:
    """Coerce numpy scalars/arrays (and containers of them) to built-in
    Python types so the checkpoint metadata stays loadable under
    ``torch.load(weights_only=True)`` (PyTorch 2.6's default), which rejects
    ``numpy._core.multiarray.scalar``.
    """

    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


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
        "branch_kind": getattr(model, "branch_kind", None),
        "clip_model": next(
            (b.clip_model for b in getattr(model, "branches", []) if hasattr(b, "clip_model")),
            None,
        ),
        "parameter_count": parameter_count(model),
        "positive_label": "fake",
        **_plain(dict(metadata or {})),
    }
    # Frozen encoder branches (CLIP ViT-L, ~304M params) are rebuilt from
    # their pinned pretrained weights on load -- no need to carry ~1.2 GB of
    # unchanged weights in every checkpoint.
    state = {
        key: value
        for key, value in model.state_dict().items()
        if ".encoder." not in key
    }
    torch.save(
        {
            "format_version": 1,
            "model_state": state,
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
        model: Detector | HybridDetector = HybridDetector.from_architecture(
            device="cpu",
            branch_kind=metadata.get("branch_kind", "fft"),
            clip_model=metadata.get("clip_model"),
        )
        # Pre-refactor single-branch checkpoints used ``freq_branch.*`` /
        # ``freq_head.*``; the current model is ``branches.0.*`` /
        # ``branch_heads.0.*``.
        if any(k.startswith("freq_branch.") or k.startswith("freq_head.") for k in state):
            state = {
                k.replace("freq_branch.", "branches.0.").replace("freq_head.", "branch_heads.0."): v
                for k, v in state.items()
            }
        # Frozen encoder weights (``.encoder.*``) are not in the checkpoint --
        # from_architecture already built them from pinned pretrained weights.
        # Non-persistent buffers (DWT filter taps, CLIP renorm constants) are
        # likewise reconstructed by from_architecture and legitimately absent.
        model_buffers = {n for n, _ in model.named_buffers()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        stray = [
            k for k in missing
            if ".encoder." not in k and k not in model_buffers
        ]
        if stray or unexpected:
            raise DetectorError(
                f"Hybrid checkpoint {checkpoint} state mismatch: "
                f"missing {stray[:4]}, unexpected {list(unexpected)[:4]}"
            )
    elif model_type == "vit":
        model = Detector.from_architecture(device="cpu")
        model.load_state_dict(state, strict=True)
    else:
        raise DetectorError(f"Unknown model_type in checkpoint metadata: {model_type!r}")
    model.to(resolve_device(device)).eval()
    return model, dict(metadata)
