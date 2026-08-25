"""Multi-branch AI-vs-real image detector architecture.

Implements the four-branch design from the research proposal:
  Branch A (SpatialBranch)     -- ResNet50 spatial/semantic features
  Branch B (FrequencyBranch)   -- FFT magnitude spectrum + radial power profile
  Branch C (CameraBranch)      -- noise-residual "camera authenticity" cues
  Branch D (VLMConceptBranch)  -- CLIP zero-shot artifact-concept probing

...fused via cross-attention (FusionClassifier) with per-branch auxiliary
heads for deep supervision and an evidential-uncertainty output head.

Honest scope notes (deliberately simplified vs. the full research proposal,
see the project's design discussion for why):
  - Branch C is a single-image noise-authenticity signal pretrained via a
    same-source-patch pretext task (pretrain_camera_branch.py), NOT true
    multi-reference PRNU source attribution -- that needs multiple images
    from a known sensor, which single-image classification doesn't have.
  - Branch D is CLIP zero-shot concept similarity, not full generative VLM
    chain-of-thought reasoning -- current VLMs' physical-plausibility
    reasoning isn't reliable enough yet to trust as a primary signal, so
    this stays cheap and inspectable (a fixed prompt bank + a linear head)
    rather than an expensive, harder-to-validate reasoning loop.
  - There is no domain-adversarial training against generator identity and
    no continual-learning/EWC wiring, because the current dataset has no
    per-generator labels to train either against -- see train.py's
    docstring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from data_utils import IMAGENET_MEAN, IMAGENET_STD

EMBED_DIM = 256


# ---------------------------------------------------------------------------
# Branch A: spatial / semantic
# ---------------------------------------------------------------------------


class SpatialBranch(nn.Module):
    """ResNet50 backbone (ImageNet-pretrained), split into named submodules
    so train.py can freeze/unfreeze stem/layer1/layer2 (always frozen) vs.
    layer3/layer4 (unfrozen in stage 2) exactly like the original
    single-branch model did.
    """

    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(2048, embed_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.project(x)


# ---------------------------------------------------------------------------
# Branch B: frequency
# ---------------------------------------------------------------------------


class FrequencyBranch(nn.Module):
    """Log-magnitude 2D FFT spectrum through a shallow CNN, plus an
    azimuthally-averaged radial power profile through a small MLP.

    Why shallow: frequency artifacts (GAN upsampling checkerboards,
    unnatural high-frequency power-spectrum decay) are global statistical
    regularities, not deep spatial hierarchies -- a deep network here is
    wasted capacity and more prone to overfitting the spectral signature of
    whichever generators happen to be in the training set, rather than
    learning the general "natural images follow roughly a 1/f^2 power-law
    decay" cue this branch is meant to capture.
    """

    def __init__(self, embed_dim=EMBED_DIM, image_size=224):
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
    def _build_radial_bins(size):
        # Integer distance-from-center per pixel, used to group FFT
        # magnitude into concentric rings for the azimuthal average -- this
        # collapses the 2D spectrum into a compact 1D "power vs frequency"
        # profile.
        ys, xs = torch.meshgrid(
            torch.arange(size) - size // 2, torch.arange(size) - size // 2, indexing="ij"
        )
        radius = torch.sqrt(xs.float() ** 2 + ys.float() ** 2)
        return radius.round().long()

    def _azimuthal_profile(self, magnitude):
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

    def forward(self, x):
        fft = torch.fft.fft2(x)
        fft = torch.fft.fftshift(fft, dim=(-2, -1))
        magnitude = torch.log1p(fft.abs())  # log1p for numerical stability (huge raw dynamic range)

        spatial_feat = self.cnn(magnitude).flatten(1)

        radial_profile = self._azimuthal_profile(magnitude.mean(dim=1))
        radial_feat = self.radial_mlp(radial_profile)

        return self.project(torch.cat([spatial_feat, radial_feat], dim=1))


# ---------------------------------------------------------------------------
# Branch C: camera authenticity
# ---------------------------------------------------------------------------


class CameraBranch(nn.Module):
    """Extracts a noise residual (image minus a fixed local-average blur of
    itself) and classifies its statistics with a small CNN.

    The blur kernel is fixed (non-trainable) so the residual step stays
    deterministic and can't be "cheated" by the network learning to smooth
    away the very signal this branch is meant to isolate. See
    pretrain_camera_branch.py for how this encoder is initialized before
    ever seeing a fake/real label.
    """

    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        kernel = torch.ones(3, 1, 5, 5) / 25.0
        self.register_buffer("blur_kernel", kernel, persistent=False)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.project = nn.Linear(64, embed_dim)

    def residual(self, x):
        blurred = F.conv2d(x, self.blur_kernel, padding=2, groups=3)
        return x - blurred

    def forward(self, x):
        feat = self.encoder(self.residual(x)).flatten(1)
        return self.project(feat)


# ---------------------------------------------------------------------------
# Branch D: VLM concept probing
# ---------------------------------------------------------------------------


class VLMConceptBranch(nn.Module):
    """CLIP zero-shot similarity against a fixed bank of artifact-related
    text prompts, through a small trainable projection. CLIP itself is
    frozen -- only the small MLP/projection on top of its (frozen)
    similarity scores is trained.
    """

    ARTIFACT_PROMPTS = [
        "a photo with distorted or malformed hands",
        "a photo with asymmetric or unnatural facial features",
        "a photo with garbled or illegible text",
        "a photo with an impossible or inconsistent shadow",
        "a photo with an unnatural or physically incorrect reflection",
        "a photo with warped or melting background details",
        "a normal, physically realistic photograph",
        "an AI-generated image",
    ]

    def __init__(self, embed_dim=EMBED_DIM, clip_model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        from transformers import CLIPModel, CLIPTokenizerFast

        self.clip = CLIPModel.from_pretrained(clip_model_name)
        for param in self.clip.parameters():
            param.requires_grad = False

        tokenizer = CLIPTokenizerFast.from_pretrained(clip_model_name)
        tokens = tokenizer(self.ARTIFACT_PROMPTS, padding=True, return_tensors="pt")
        with torch.no_grad():
            # NOTE: get_text_features/get_image_features return a
            # BaseModelOutputWithPooling in this transformers version, not
            # a plain tensor -- the pooled embedding is `.pooler_output`.
            text_features = self.clip.get_text_features(**tokens).pooler_output
        text_features = F.normalize(text_features, dim=-1)
        self.register_buffer("text_features", text_features, persistent=True)

        # Our pipeline normalizes with ImageNet stats; CLIP expects its own
        # normalization. Undo one, apply the other, so this branch sees
        # inputs distributed the way CLIP was trained on.
        self.register_buffer("imagenet_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("imagenet_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.register_buffer(
            "clip_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "clip_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1), persistent=False
        )

        self.mlp = nn.Sequential(nn.Linear(len(self.ARTIFACT_PROMPTS), 64), nn.ReLU(inplace=True))
        self.project = nn.Linear(64, embed_dim)

    def forward(self, x):
        x = x * self.imagenet_std + self.imagenet_mean  # undo ImageNet normalization -> [0,1]
        x = (x - self.clip_mean) / self.clip_std  # apply CLIP normalization

        with torch.no_grad():
            image_features = self.clip.get_image_features(pixel_values=x).pooler_output
        image_features = F.normalize(image_features, dim=-1)

        similarity = image_features @ self.text_features.T  # (B, num_prompts)
        return self.project(self.mlp(similarity))


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


class FusionClassifier(nn.Module):
    """Combines all branch embeddings via cross-attention: learned "fusion
    query" tokens attend over the branch tokens, so the model can learn to
    weight each branch's evidence conditionally per image rather than with
    fixed weights. Also exposes per-branch auxiliary classifiers (deep
    supervision, and useful for reporting a per-branch ablation at eval
    time) and an evidential-uncertainty output on the fused prediction
    (see losses.py for the evidential loss and how to read "evidence").
    """

    def __init__(self, embed_dim=EMBED_DIM, num_fusion_queries=4, enable_vlm_branch=True):
        super().__init__()
        self.spatial = SpatialBranch(embed_dim)
        self.frequency = FrequencyBranch(embed_dim)
        self.camera = CameraBranch(embed_dim)
        self.enable_vlm_branch = enable_vlm_branch
        if enable_vlm_branch:
            self.vlm = VLMConceptBranch(embed_dim)

        self.branch_names = ["spatial", "frequency", "camera"] + (["vlm"] if enable_vlm_branch else [])

        # Learned per-branch type embedding, added to that branch's token
        # before fusion so cross-attention can tell which branch each token
        # came from (otherwise all branch tokens look alike beyond content).
        self.branch_type_embed = nn.Parameter(torch.randn(len(self.branch_names), embed_dim) * 0.02)

        self.fusion_queries = nn.Parameter(torch.randn(num_fusion_queries, embed_dim) * 0.02)
        self.fusion_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.fusion_norm = nn.LayerNorm(embed_dim)

        self.aux_heads = nn.ModuleDict({name: nn.Linear(embed_dim, 2) for name in self.branch_names})

        # Outputs 2 "evidence" values, consumed by losses.evidential_loss /
        # losses.evidence_to_probs_and_uncertainty -- NOT raw softmax logits.
        self.fused_head = nn.Sequential(
            nn.Linear(embed_dim * num_fusion_queries, embed_dim), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 2),
        )

    def forward(self, x):
        branch_embeds = {
            "spatial": self.spatial(x),
            "frequency": self.frequency(x),
            "camera": self.camera(x),
        }
        if self.enable_vlm_branch:
            branch_embeds["vlm"] = self.vlm(x)

        aux_logits = {name: self.aux_heads[name](emb) for name, emb in branch_embeds.items()}

        tokens = torch.stack([branch_embeds[name] for name in self.branch_names], dim=1)
        tokens = tokens + self.branch_type_embed.unsqueeze(0)

        B = x.shape[0]
        queries = self.fusion_queries.unsqueeze(0).expand(B, -1, -1)
        fused, attn_weights = self.fusion_attn(queries, tokens, tokens)
        fused = self.fusion_norm(fused + queries)

        evidence = self.fused_head(fused.flatten(1))

        return {
            "evidence": evidence,
            "aux_logits": aux_logits,
            "branch_embeds": branch_embeds,
            "attn_weights": attn_weights,  # (B, num_queries, num_branches) -- which branch each query trusted
        }

    def stage1_trainable_parameters(self):
        """Params trained in both stage 1 (backbone frozen) and stage 2:
        everything except the spatial backbone's layer3/layer4.
        """
        params = list(self.spatial.project.parameters())
        params += list(self.frequency.parameters())
        params += list(self.camera.parameters())
        if self.enable_vlm_branch:
            params += list(self.vlm.mlp.parameters()) + list(self.vlm.project.parameters())
        params += [self.fusion_queries, self.branch_type_embed]
        params += list(self.fusion_attn.parameters()) + list(self.fusion_norm.parameters())
        params += list(self.aux_heads.parameters())
        params += list(self.fused_head.parameters())
        return params

    def set_train_mode(self, is_train, spatial_unfrozen_submodules=()):
        """Puts only the currently-trainable submodules into train() mode;
        everything else (frozen backbone layers, frozen CLIP) stays in
        eval(). Plain model.train(is_train) would put frozen layers'
        BatchNorm into train mode too, letting their running stats drift
        away from pretrained values even though requires_grad=False --
        exactly the bug this project already hit once with the single-
        branch model. See train.py for how this gets called.
        """
        self.eval()
        if not is_train:
            return

        self.frequency.train()
        self.camera.train()
        if self.enable_vlm_branch:
            self.vlm.mlp.train()
            self.vlm.project.train()
            # self.vlm.clip stays in eval() -- it's frozen and pretrained.
        self.fusion_attn.train()
        self.fusion_norm.train()
        self.aux_heads.train()
        self.fused_head.train()

        self.spatial.project.train()
        for name in spatial_unfrozen_submodules:
            getattr(self.spatial, name).train()
