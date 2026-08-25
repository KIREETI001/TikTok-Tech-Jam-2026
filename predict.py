"""Classify a single image as AI-generated or a real photograph, using the
full multi-branch fusion model.

Run with:  python predict.py path/to/image.jpg
           python predict.py path/to/image.jpg --verbose   (per-branch breakdown)

Applies the exact same deterministic preprocessing used for validation in
train.py (resize -> tensor -> ImageNet normalize, no random augmentation) --
a mismatch here is a classic way for a fine model to produce meaningless
predictions at inference time.
"""

import argparse
import os
import sys

import torch
from PIL import Image
from torchvision import transforms

from data_utils import IMAGENET_MEAN, IMAGENET_STD, load_class_to_idx, resolve_class_indices
from model import FusionClassifier
from losses import evidence_to_probs_and_uncertainty
from anomaly import ConvAutoencoder, IMAGE_SIZE as ANOMALY_IMAGE_SIZE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BEST_MODEL_PATH = "ai_detector_best.pth"
ANOMALY_MODEL_PATH = "anomaly_autoencoder_best.pth"
ANOMALY_THRESHOLD_PATH = "anomaly_threshold.txt"
ENABLE_VLM_BRANCH = False  # must match train.py
UNCERTAINTY_WARNING_THRESHOLD = 0.5

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

anomaly_transform = transforms.Compose([
    transforms.Resize((ANOMALY_IMAGE_SIZE, ANOMALY_IMAGE_SIZE)),
    transforms.ToTensor(),
])


def load_model():
    model = FusionClassifier(enable_vlm_branch=ENABLE_VLM_BRANCH)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.set_train_mode(False)
    return model


def load_anomaly_model():
    if not os.path.exists(ANOMALY_MODEL_PATH):
        return None, None
    model = ConvAutoencoder()
    model.load_state_dict(torch.load(ANOMALY_MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE).eval()
    threshold = None
    if os.path.exists(ANOMALY_THRESHOLD_PATH):
        with open(ANOMALY_THRESHOLD_PATH) as f:
            threshold = float(f.read().strip())
    return model, threshold


def predict(model, image, ai_idx, real_idx):
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(tensor)
        probs, uncertainty = evidence_to_probs_and_uncertainty(out["evidence"])

    ai_prob = probs[0, ai_idx].item()
    real_prob = probs[0, real_idx].item()
    uncertainty = uncertainty.item()

    per_branch = {}
    for name, logits in out["aux_logits"].items():
        branch_probs = torch.softmax(logits, dim=1)[0]
        per_branch[name] = {"ai_prob": branch_probs[ai_idx].item(), "real_prob": branch_probs[real_idx].item()}

    return ai_prob, real_prob, uncertainty, per_branch


def compute_anomaly_score(anomaly_model, image):
    tensor = anomaly_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        recon = anomaly_model(tensor)
        error = ((recon - tensor) ** 2).mean().item()
    return error


def main():
    parser = argparse.ArgumentParser(description="Classify an image as AI-generated or real.")
    parser.add_argument("image_path", help="Path to the image to classify")
    parser.add_argument("--verbose", action="store_true", help="Show per-branch probability breakdown")
    args = parser.parse_args()

    print("Using device:", DEVICE)

    class_to_idx = load_class_to_idx()
    ai_idx, real_idx = resolve_class_indices(class_to_idx)

    model = load_model()
    anomaly_model, anomaly_threshold = load_anomaly_model()

    try:
        image = Image.open(args.image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: image not found at {args.image_path}", file=sys.stderr)
        sys.exit(1)

    ai_prob, real_prob, uncertainty, per_branch = predict(model, image, ai_idx, real_idx)
    predicted_label = "AI Generated" if ai_prob > real_prob else "Real Image"

    print(f"\nAI Generated probability: {ai_prob * 100:.1f}%")
    print(f"Real probability: {real_prob * 100:.1f}%")
    print(f"\nPrediction: {predicted_label}")
    print(f"Model uncertainty: {uncertainty:.3f} (0=confident, 1=maximally unsure)")
    if uncertainty > UNCERTAINTY_WARNING_THRESHOLD:
        print(
            "NOTE: high uncertainty -- this input doesn't strongly resemble either "
            "class in a way the model recognizes (could be an unusual real photo, "
            "an unfamiliar generator, or heavy post-processing). Treat the "
            "prediction above with reduced confidence."
        )

    if anomaly_model is not None:
        score = compute_anomaly_score(anomaly_model, image)
        print(f"\nAnomaly score (vs. real-photo manifold): {score:.5f}")
        if anomaly_threshold is not None:
            if score > anomaly_threshold:
                print(
                    f"NOTE: above the calibrated threshold ({anomaly_threshold:.5f}) -- this "
                    "image's low-level statistics look atypical for a real photo, "
                    "independent of the classifier's own prediction above."
                )
            else:
                print(f"Within the typical real-photo range (threshold {anomaly_threshold:.5f}).")
    else:
        print(f"\n(No anomaly score -- run train_anomaly.py to enable it.)")

    if args.verbose:
        print("\nPer-branch breakdown (each branch's own prediction, before fusion):")
        for name, p in per_branch.items():
            print(f"  {name:<10} AI {p['ai_prob'] * 100:5.1f}%   Real {p['real_prob'] * 100:5.1f}%")


if __name__ == "__main__":
    main()
