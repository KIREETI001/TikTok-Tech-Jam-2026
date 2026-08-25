"""Evaluation + robustness + ablation testing for the multi-branch
AI-vs-real image detector.

Run with:  python validate.py

What it does, in order:
  1. Held-out TEST split evaluation: accuracy, precision, recall, F1,
     confusion matrix, mean predictive uncertainty (see losses.py).
  2. Per-branch ablation: each branch's own auxiliary-head accuracy, so you
     can see which branches are actually pulling weight vs. riding along.
  3. Robustness sweep: JPEG/blur/resize/noise/crop degradation, fixed
     levels, to check whether performance holds up on "internet" images.
  4. Anomaly-score check (if train_anomaly.py has been run): does the
     real-only autoencoder's reconstruction error actually separate real
     from AI images in the test set.
  5. Optional: generalization to unseen AI generators, if you provide a
     dataset/../unseen_generators/<name>/*.jpg folder.

Results are written to test_metrics.csv and robustness_report.csv.
"""

import csv
import io
import math
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from data_utils import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    DATASET_PATH,
    get_splits,
    TransformSubset,
    resolve_class_indices,
)
from model import FusionClassifier
from losses import evidence_to_probs_and_uncertainty
from anomaly import ConvAutoencoder, IMAGE_SIZE as ANOMALY_IMAGE_SIZE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BEST_MODEL_PATH = "ai_detector_best.pth"
ANOMALY_MODEL_PATH = "anomaly_autoencoder_best.pth"
ANOMALY_THRESHOLD_PATH = "anomaly_threshold.txt"
ENABLE_VLM_BRANCH = False  # must match train.py

TEST_METRICS_PATH = "test_metrics.csv"
ROBUSTNESS_REPORT_PATH = "robustness_report.csv"
UNSEEN_GENERATORS_PATH = os.path.join(os.path.dirname(DATASET_PATH), "unseen_generators")

base_transform = transforms.Compose([
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


def run_inference(model, loader):
    """Returns (preds, labels, uncertainties, aux_correct) where aux_correct
    is {branch_name: [bool, ...]} for the per-branch ablation.
    """
    all_preds, all_labels, all_unc = [], [], []
    aux_correct = None

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            out = model(images)

            probs, uncertainty = evidence_to_probs_and_uncertainty(out["evidence"])
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_unc.extend(uncertainty.cpu().tolist())

            if aux_correct is None:
                aux_correct = {name: [] for name in out["aux_logits"]}
            for name, logits in out["aux_logits"].items():
                aux_preds = torch.argmax(logits, dim=1)
                aux_correct[name].extend((aux_preds == labels).cpu().tolist())

    return all_preds, all_labels, all_unc, aux_correct


# ---------------------------------------------------------------------------
# 1 & 2. Held-out test set evaluation + per-branch ablation
# ---------------------------------------------------------------------------


def run_test_set_evaluation(model, test_subset, class_to_idx, ai_idx):
    print("\n=== Held-out test set evaluation ===")

    loader = DataLoader(TransformSubset(test_subset, base_transform), batch_size=32)
    preds, labels, uncertainties, aux_correct = run_inference(model, loader)

    accuracy = 100 * sum(p == l for p, l in zip(preds, labels)) / len(labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=ai_idx, zero_division=0
    )
    mean_uncertainty = sum(uncertainties) / len(uncertainties)

    label_order = sorted(class_to_idx, key=class_to_idx.get)
    cm = confusion_matrix(labels, preds)

    print(f"Fused-model test accuracy: {accuracy:.2f}%  (n={len(labels)})")
    print(f"AI-class precision: {precision:.3f}  recall: {recall:.3f}  f1: {f1:.3f}")
    print(f"Mean predictive uncertainty: {mean_uncertainty:.3f} (0=confident, 1=maximally unsure)")
    print(f"Confusion matrix (rows=true, cols=pred), label order = {label_order}")
    print(cm)

    print("\nPer-branch ablation (each branch's OWN auxiliary-head accuracy, not the fused prediction):")
    for name, correct in aux_correct.items():
        acc = 100 * sum(correct) / len(correct)
        print(f"  {name:<10} {acc:.2f}%")
    print(
        "If a branch's standalone accuracy is close to the majority-class "
        "baseline (~75% real here), the fusion attention has little reason "
        "to trust it -- that's a legitimate diagnostic for whether a given "
        "branch is worth its complexity on this dataset."
    )

    with open(TEST_METRICS_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["accuracy", "ai_precision", "ai_recall", "ai_f1", "mean_uncertainty", "n"])
        writer.writerow([accuracy, precision, recall, f1, mean_uncertainty, len(labels)])
    print(f"Saved to {TEST_METRICS_PATH}")


# ---------------------------------------------------------------------------
# 3. Robustness sweep
# ---------------------------------------------------------------------------


class JPEGCompress:
    def __init__(self, quality):
        self.quality = quality

    def __call__(self, img):
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class DownscaleUpscale:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img):
        w, h = img.size
        small = img.resize((max(1, int(w * self.scale)), max(1, int(h * self.scale))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


class GaussianNoise:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, img):
        arr = np.asarray(img).astype(np.float32) / 255.0
        noise = np.random.normal(0, self.sigma, arr.shape)
        noisy = np.clip(arr + noise, 0.0, 1.0) * 255.0
        return Image.fromarray(noisy.astype(np.uint8))


class CenterCropResize:
    def __init__(self, crop_frac):
        self.crop_frac = crop_frac

    def __call__(self, img):
        w, h = img.size
        new_w, new_h = int(w * self.crop_frac), int(h * self.crop_frac)
        left, top = (w - new_w) // 2, (h - new_h) // 2
        cropped = img.crop((left, top, left + new_w, top + new_h))
        return cropped.resize((w, h), Image.BILINEAR)


def gaussian_blur_for_sigma(sigma):
    kernel_size = 2 * math.ceil(3 * sigma) + 1
    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)


def build_conditions():
    conditions = [("clean", lambda img: img)]
    for quality in [90, 70, 50, 30]:
        conditions.append((f"jpeg_q{quality}", JPEGCompress(quality)))
    for sigma in [0.5, 1.0, 2.0]:
        conditions.append((f"blur_sigma{sigma}", gaussian_blur_for_sigma(sigma)))
    for scale in [0.5, 0.25]:
        conditions.append((f"resize_{scale}x", DownscaleUpscale(scale)))
    for sigma in [0.02, 0.05, 0.10]:
        conditions.append((f"noise_sigma{sigma}", GaussianNoise(sigma)))
    conditions.append(("color_jitter_20pct", transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)))
    conditions.append(("center_crop_80pct", CenterCropResize(0.8)))
    return conditions


def run_robustness_sweep(model, test_subset, ai_idx):
    print("\n=== Robustness sweep (degraded test images, fused prediction) ===")

    rows = []
    print(f"{'Condition':<22}{'Accuracy':>10}{'AI Precision':>14}{'AI Recall':>11}{'AI F1':>8}{'MeanUnc':>9}")

    for name, degradation in build_conditions():
        transform = transforms.Compose([
            transforms.Resize((224, 224)), degradation, transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        loader = DataLoader(TransformSubset(test_subset, transform), batch_size=32)
        preds, labels, uncertainties, _ = run_inference(model, loader)

        accuracy = 100 * sum(p == l for p, l in zip(preds, labels)) / len(labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", pos_label=ai_idx, zero_division=0
        )
        mean_unc = sum(uncertainties) / len(uncertainties)

        rows.append({
            "condition": name, "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1, "mean_uncertainty": mean_unc, "n": len(labels),
        })
        print(f"{name:<22}{accuracy:>9.2f}%{precision:>14.3f}{recall:>11.3f}{f1:>8.3f}{mean_unc:>9.3f}")

    with open(ROBUSTNESS_REPORT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["condition", "accuracy", "precision", "recall", "f1", "mean_uncertainty", "n"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved to {ROBUSTNESS_REPORT_PATH}")

    clean_acc = rows[0]["accuracy"]
    worst = min(rows, key=lambda r: r["accuracy"])
    if clean_acc - worst["accuracy"] > 10:
        print(
            f"\nWARNING: accuracy drops {clean_acc - worst['accuracy']:.1f} points "
            f"under '{worst['condition']}' vs clean -- fragile to real-world processing."
        )


# ---------------------------------------------------------------------------
# 4. Anomaly-score sanity check
# ---------------------------------------------------------------------------


def run_anomaly_check(test_subset, ai_idx, real_idx):
    anomaly_model, threshold = load_anomaly_model()
    if anomaly_model is None:
        print(
            f"\n=== Anomaly-score check skipped ===\nNo autoencoder found at "
            f"{ANOMALY_MODEL_PATH}. Run train_anomaly.py first."
        )
        return

    print("\n=== Anomaly-score check (real-only autoencoder reconstruction error) ===")
    loader = DataLoader(TransformSubset(test_subset, anomaly_transform), batch_size=32)

    errors_by_label = {ai_idx: [], real_idx: []}
    flagged = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            recon = anomaly_model(images)
            per_image_error = ((recon - images) ** 2).mean(dim=(1, 2, 3)).cpu().tolist()
            for err, label in zip(per_image_error, labels.tolist()):
                errors_by_label[label].append(err)
                total += 1
                if threshold is not None and err > threshold:
                    flagged += 1

    ai_mean = sum(errors_by_label[ai_idx]) / max(len(errors_by_label[ai_idx]), 1)
    real_mean = sum(errors_by_label[real_idx]) / max(len(errors_by_label[real_idx]), 1)
    print(f"Mean reconstruction error -- AI images: {ai_mean:.5f}   real images: {real_mean:.5f}")
    if threshold is not None:
        print(f"Threshold (95th pct of real val error): {threshold:.5f}  ->  {flagged}/{total} test images flagged as anomalous")

    if ai_mean > real_mean * 1.1:
        print(
            "AI images reconstruct worse than real images on average -- the anomaly "
            "signal carries some real information here, though a full evaluation "
            "would need a much larger test set to trust the gap."
        )
    else:
        print(
            "AI images do NOT reconstruct meaningfully worse than real images on this "
            "test set -- on this dataset size, treat the anomaly score as a weak, "
            "auxiliary signal rather than a strong one. This is consistent with the "
            "research proposal's own caveat that a simple autoencoder is a cruder "
            "density estimate than a proper normalizing flow."
        )


# ---------------------------------------------------------------------------
# 5. Unseen-generator generalization
# ---------------------------------------------------------------------------


def run_unseen_generator_test(model, ai_idx):
    if not os.path.isdir(UNSEEN_GENERATORS_PATH):
        print(
            f"\n=== Unseen-generator test skipped ===\nNo folder found at "
            f"{UNSEEN_GENERATORS_PATH}. Add dataset/../unseen_generators/<generator_name>/"
            "*.jpg (AI images from a generator NOT in your training set) to test "
            "cross-generator generalization."
        )
        return

    print("\n=== Unseen-generator generalization test ===")
    print(f"{'Generator':<20}{'Accuracy (called AI)':>22}{'Mean Uncertainty':>18}{'n':>6}")

    for generator_name in sorted(os.listdir(UNSEEN_GENERATORS_PATH)):
        generator_dir = os.path.join(UNSEEN_GENERATORS_PATH, generator_name)
        if not os.path.isdir(generator_dir):
            continue
        paths = [
            os.path.join(generator_dir, fname)
            for fname in sorted(os.listdir(generator_dir))
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ]
        if not paths:
            continue

        correct = 0
        uncertainties = []
        with torch.no_grad():
            for path in paths:
                img = Image.open(path).convert("RGB")
                tensor = base_transform(img).unsqueeze(0).to(DEVICE)
                out = model(tensor)
                probs, unc = evidence_to_probs_and_uncertainty(out["evidence"])
                pred = torch.argmax(probs, dim=1).item()
                uncertainties.append(unc.item())
                if pred == ai_idx:
                    correct += 1

        accuracy = 100 * correct / len(paths)
        mean_unc = sum(uncertainties) / len(uncertainties)
        print(f"{generator_name:<20}{accuracy:>21.2f}%{mean_unc:>18.3f}{len(paths):>6}")


def main():
    print("Using device:", DEVICE)

    train_subset, val_subset, test_subset, class_to_idx, class_counts = get_splits()
    ai_idx, real_idx = resolve_class_indices(class_to_idx)
    print(f"class_to_idx={class_to_idx}  AI_IDX={ai_idx}  REAL_IDX={real_idx}")
    print(f"Held-out test images: {len(test_subset)}")

    model = load_model()

    run_test_set_evaluation(model, test_subset, class_to_idx, ai_idx)
    run_robustness_sweep(model, test_subset, ai_idx)
    run_anomaly_check(test_subset, ai_idx, real_idx)
    run_unseen_generator_test(model, ai_idx)


if __name__ == "__main__":
    main()
