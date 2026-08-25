"""Loss functions for the multi-branch detector.

- evidential_loss / evidence_to_probs_and_uncertainty: turns the fused
  head's output into a Dirichlet-distribution-based prediction with a
  calibrated notion of "I don't know", instead of a plain softmax that
  always looks confident even on nonsense input (Sensoy et al., 2018,
  "Evidential Deep Learning to Quantify Classification Uncertainty").
- supervised_contrastive_loss: trains embeddings to be invariant to which
  specific degradation (JPEG level, blur, crop...) was applied, using two
  independently-augmented views of each training image (Khosla et al.,
  2020).
"""

import torch
import torch.nn.functional as F


def evidential_loss(evidence, labels, epoch, num_classes=2, annealing_epochs=10):
    """Primary training loss for the fused head. `evidence` is raw network
    output (any real number); softplus makes it non-negative, and
    evidence + 1 becomes the Dirichlet concentration parameters `alpha`.

    Why this instead of cross-entropy on a softmax: softmax always
    produces *some* confident-looking distribution, even on an input
    completely unlike anything in training (e.g. a new AI generator this
    model has never seen). The evidential formulation lets "low evidence
    for every class" be a distinct, learnable outcome, distinguishable
    from "high evidence, confidently the AI class" -- see
    evidence_to_probs_and_uncertainty() for how that's read back out.
    """
    evidence = F.softplus(evidence)
    alpha = evidence + 1
    strength = alpha.sum(dim=1, keepdim=True)

    one_hot = F.one_hot(labels, num_classes).float()

    # Expected mean-squared error of the Dirichlet mean against the label,
    # plus its variance -- the standard evidential classification loss.
    err = (one_hot - alpha / strength) ** 2
    var = alpha * (strength - alpha) / (strength * strength * (strength + 1))
    mse_loss = (err + var).sum(dim=1)

    # KL regularization pulls evidence for the *wrong* class toward zero.
    # Annealed in over training (annealing_coef ramps 0 -> 1) so the model
    # isn't punished for having any evidence at all before it's learned
    # anything useful yet -- without annealing this term dominates early
    # training and the model never learns to commit to a class at all.
    annealing_coef = min(1.0, epoch / annealing_epochs)
    alpha_tilde = one_hot + (1 - one_hot) * alpha
    kl = _kl_dirichlet(alpha_tilde)

    return (mse_loss + annealing_coef * kl).mean()


def _kl_dirichlet(alpha):
    """KL divergence from Dirichlet(alpha) to the uniform Dirichlet(1,1,...)."""
    beta = torch.ones_like(alpha)
    strength_alpha = alpha.sum(dim=1, keepdim=True)
    strength_beta = beta.sum(dim=1, keepdim=True)

    t1 = torch.lgamma(strength_alpha) - torch.lgamma(strength_beta)
    t2 = -(torch.lgamma(alpha) - torch.lgamma(beta)).sum(dim=1, keepdim=True)
    t3 = ((alpha - beta) * (torch.digamma(alpha) - torch.digamma(strength_alpha))).sum(dim=1, keepdim=True)

    return (t1 + t2 + t3).squeeze(1)


def evidence_to_probs_and_uncertainty(evidence, num_classes=2):
    """Converts raw evidence into (class probabilities, uncertainty in
    [0,1]). uncertainty = num_classes / total_strength: with zero evidence
    for every class, strength = num_classes and uncertainty = 1 (maximum
    "I don't know"); as evidence accumulates for one class, uncertainty
    shrinks toward 0. This is the number predict.py and validate.py surface
    as model confidence/uncertainty.
    """
    evidence = F.softplus(evidence)
    alpha = evidence + 1
    strength = alpha.sum(dim=1, keepdim=True)
    probs = alpha / strength
    uncertainty = num_classes / strength
    return probs, uncertainty.squeeze(1)


def supervised_contrastive_loss(embeddings_a, embeddings_b, labels, temperature=0.1):
    """Supervised contrastive loss between two augmented views of the same
    batch. Pulls same-label embeddings together and different-label
    embeddings apart, regardless of which specific degradation was applied
    to each view -- directly training the representation to be invariant
    to exactly the kind of post-processing (recompression, resizing,
    cropping) that breaks fragile, artifact-based detectors on real-world
    images.
    """
    embeddings = F.normalize(torch.cat([embeddings_a, embeddings_b], dim=0), dim=1)
    labels = torch.cat([labels, labels], dim=0)

    sim = embeddings @ embeddings.T / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()  # numerical stability

    n = embeddings.shape[0]
    self_mask = torch.eye(n, dtype=torch.bool, device=embeddings.device)
    positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask

    exp_sim = torch.exp(sim) * (~self_mask)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    num_positives = positive_mask.sum(dim=1).clamp(min=1)
    loss = -(log_prob * positive_mask).sum(dim=1) / num_positives

    return loss.mean()
