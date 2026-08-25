"""Trains the real-image-only anomaly autoencoder (see anomaly.py).

Run with: python train_anomaly.py
Produces: anomaly_autoencoder_best.pth
          anomaly_threshold.txt  (95th-percentile real-image reconstruction
                                   error on the validation split, used by
                                   validate.py/predict.py to flag images
                                   that don't look like typical real photos)

This only ever sees REAL images -- by design, it never gets to see what a
fake looks like, which is exactly what makes its reconstruction-error
signal potentially useful against future, unseen generators: it doesn't
need to have seen that generator's output to flag it as unlike the
real-photo manifold it learned.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data_utils import get_splits, resolve_class_indices
from anomaly import ConvAutoencoder, IMAGE_SIZE

MODEL_PATH = "anomaly_autoencoder_best.pth"
THRESHOLD_PATH = "anomaly_threshold.txt"
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


class RealOnlySubset(torch.utils.data.Dataset):
    """Filters a raw Subset (as returned by data_utils.get_splits) down to
    only its real-labeled images, using the underlying ImageFolder's sample
    list to check labels (no image decoding needed just to filter).
    """

    def __init__(self, subset, transform, real_idx):
        self.subset = subset
        self.transform = transform
        self.selected = [
            i for i in range(len(subset))
            if subset.dataset.samples[subset.indices[i]][1] == real_idx
        ]

    def __len__(self):
        return len(self.selected)

    def __getitem__(self, idx):
        image, _ = self.subset[self.selected[idx]]
        return self.transform(image.convert("RGB"))


def main():
    print("Using device:", DEVICE)

    train_subset, val_subset, _, class_to_idx, _ = get_splits()
    _, real_idx = resolve_class_indices(class_to_idx)

    train_dataset = RealOnlySubset(train_subset, transform, real_idx)
    val_dataset = RealOnlySubset(val_subset, transform, real_idx)
    print(f"Real-only images -> train: {len(train_dataset)}  val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = ConvAutoencoder().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False):
            images = images.to(DEVICE)
            optimizer.zero_grad()
            recon = model(images)
            loss = criterion(recon, images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_dataset)

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(DEVICE)
                val_running += criterion(model(images), images).item() * images.size(0)
        val_loss = val_running / len(val_dataset)

        print(f"Epoch {epoch + 1}/{EPOCHS} | train MSE {train_loss:.5f} | val MSE {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  New best autoencoder saved (val MSE {val_loss:.5f})")

    print(f"\nBest val MSE: {best_val_loss:.5f}")

    # Calibrate the "looks anomalous" threshold from the best checkpoint's
    # per-image reconstruction error on real validation images, so
    # predict.py/validate.py have a concrete cutoff instead of an arbitrary
    # constant. 95th percentile: ~5% of genuine real photos will exceed it
    # by chance, which is the false-positive rate this threshold implies.
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    errors = []
    with torch.no_grad():
        for images in val_loader:
            images = images.to(DEVICE)
            recon = model(images)
            per_image = ((recon - images) ** 2).mean(dim=(1, 2, 3))
            errors.extend(per_image.cpu().tolist())

    errors_sorted = sorted(errors)
    threshold = errors_sorted[int(0.95 * (len(errors_sorted) - 1))]
    with open(THRESHOLD_PATH, "w") as f:
        f.write(str(threshold))

    print(f"Anomaly threshold (95th percentile of real val reconstruction error): {threshold:.5f}")
    print("Saved to", MODEL_PATH, "and", THRESHOLD_PATH)


if __name__ == "__main__":
    main()
