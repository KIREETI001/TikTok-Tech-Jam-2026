"""Supervised contrastive loss (Khosla et al., 2020), used in Phase D to make
the detector degrade *symmetrically* -- recall falling with everything else
under post-processing rather than the model getting trigger-happy as images
blur (the teammate's pipeline credits it with exactly that).

Two independently-augmented views of each training image are pushed to the
same place in embedding space regardless of which degradation each view got,
which trains the representation to be invariant to the JPEG / resize / blur
that breaks fragile artifact-based detectors on real-world images.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """SupCon between two augmented views of one batch. Pulls same-label
    embeddings together and different-label apart, independent of the
    augmentation applied to each view.
    """

    embeddings = F.normalize(torch.cat([embeddings_a, embeddings_b], dim=0), dim=1)
    labels = torch.cat([labels, labels], dim=0)

    sim = embeddings @ embeddings.T / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    n = embeddings.shape[0]
    self_mask = torch.eye(n, dtype=torch.bool, device=embeddings.device)
    positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask

    exp_sim = torch.exp(sim) * (~self_mask)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    num_positives = positive_mask.sum(dim=1).clamp(min=1)
    loss = -(log_prob * positive_mask).sum(dim=1) / num_positives
    return loss.mean()
