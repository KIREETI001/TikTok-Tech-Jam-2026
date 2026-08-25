"""Self-supervised pretraining for the camera-authenticity branch (Branch
C), using only real photos -- no AI-generated images or fake/real labels
involved at all.

Pretext task: given two 96x96 crops, predict whether they came from the
SAME source photo (label 1) or two DIFFERENT source photos (label 0).
Natural sensor noise, optical vignetting, and other camera-authenticity
cues are properties of the whole sensor/capture, so two crops from the
same real photo should share a more consistent noise "signature" than two
crops from different photos. This gives the camera branch's encoder a
meaningful initialization before it ever sees a single fake-vs-real label,
instead of starting from random weights -- the same reasoning as
pretraining on ImageNet, just for a forensic cue instead of object
semantics, and using only real images so it can't accidentally pick up a
"real dataset vs AI dataset" shortcut.

Run with: python pretrain_camera_branch.py
Produces: camera_branch_pretrained.pth  (load into CameraBranch in train.py)
"""

import random

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from data_utils import get_splits, resolve_class_indices, SEED
from model import CameraBranch, EMBED_DIM

CROP_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAINED_PATH = "camera_branch_pretrained.pth"

to_tensor = transforms.ToTensor()


def load_real_image_paths(subset, real_idx):
    """Paths of only the real-labeled images in `subset`, read directly
    from the underlying ImageFolder's sample list (no image decoding, just
    filtering by label).
    """
    return [
        subset.dataset.samples[i][0]
        for i in subset.indices
        if subset.dataset.samples[i][1] == real_idx
    ]


class SameSourcePairDataset(Dataset):
    """Each __getitem__ returns a pair of CROP_SIZE crops and a label: 1 if
    both crops came from the same source image, 0 if from two different,
    randomly chosen images. Images are opened fresh from disk per access
    (not cached in memory) to avoid holding hundreds of file handles open.
    """

    def __init__(self, paths):
        self.paths = paths
        self.rng = random.Random(SEED)

    def __len__(self):
        return len(self.paths)

    def _random_crop(self, img):
        w, h = img.size
        if w < CROP_SIZE or h < CROP_SIZE:
            img = img.resize((max(w, CROP_SIZE), max(h, CROP_SIZE)))
            w, h = img.size
        left = self.rng.randint(0, w - CROP_SIZE)
        top = self.rng.randint(0, h - CROP_SIZE)
        return img.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))

    def __getitem__(self, idx):
        same_source = self.rng.random() < 0.5
        img_a = Image.open(self.paths[idx]).convert("RGB")

        if same_source:
            img_b = img_a
        else:
            other_idx = self.rng.randrange(len(self.paths))
            while other_idx == idx:
                other_idx = self.rng.randrange(len(self.paths))
            img_b = Image.open(self.paths[other_idx]).convert("RGB")

        crop_a = to_tensor(self._random_crop(img_a))
        crop_b = to_tensor(self._random_crop(img_b))
        label = torch.tensor(1.0 if same_source else 0.0)
        return crop_a, crop_b, label


class PairwiseHead(nn.Module):
    """Task-specific head for the pretext task only -- discarded after
    pretraining. Only CameraBranch's encoder weights get saved/reused.
    """

    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, emb_a, emb_b):
        return self.classifier((emb_a - emb_b).abs()).squeeze(1)


def main():
    print("Using device:", DEVICE)

    train_subset, val_subset, _, class_to_idx, _ = get_splits()
    _, real_idx = resolve_class_indices(class_to_idx)

    train_paths = load_real_image_paths(train_subset, real_idx)
    val_paths = load_real_image_paths(val_subset, real_idx)
    print(f"Real images for pretraining -> train: {len(train_paths)}  val: {len(val_paths)}")

    train_loader = DataLoader(SameSourcePairDataset(train_paths), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SameSourcePairDataset(val_paths), batch_size=BATCH_SIZE)

    encoder = CameraBranch(EMBED_DIM).to(DEVICE)
    head = PairwiseHead(EMBED_DIM).to(DEVICE)
    optimizer = optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        encoder.train()
        head.train()
        running_loss, correct, total = 0.0, 0, 0

        for crop_a, crop_b, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [train]", leave=False):
            crop_a, crop_b, labels = crop_a.to(DEVICE), crop_b.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            emb_a, emb_b = encoder(crop_a), encoder(crop_b)
            logits = head(emb_a, emb_b)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = 100 * correct / total

        encoder.eval()
        head.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for crop_a, crop_b, labels in val_loader:
                crop_a, crop_b, labels = crop_a.to(DEVICE), crop_b.to(DEVICE), labels.to(DEVICE)
                logits = head(encoder(crop_a), encoder(crop_b))
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch + 1}/{EPOCHS} | train loss {train_loss:.4f} acc {train_acc:.2f}% | val same-source acc {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(encoder.state_dict(), PRETRAINED_PATH)
            print(f"  New best camera-branch encoder saved (val acc {val_acc:.2f}%)")

    print(f"\nBest same-source-pair validation accuracy: {best_val_acc:.2f}%")
    print(
        "(50% = chance, since positive/negative pairs are balanced. Meaningfully "
        "above 50% means the encoder learned some source-consistent noise signal "
        "to hand off to train.py.)"
    )
    print("Saved encoder weights to", PRETRAINED_PATH)


if __name__ == "__main__":
    main()
