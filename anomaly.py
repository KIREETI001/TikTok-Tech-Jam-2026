"""Real-image-only anomaly detector: a convolutional autoencoder trained
exclusively on real photos (see train_anomaly.py). Its reconstruction error
is fused alongside the classifier's prediction as a second, structurally
different signal -- one that doesn't require having seen any fake at all,
only that a future generator's output still deviates somewhat from the
learned "real photo" manifold.

This is the pragmatic, single-image-friendly stand-in for the research
proposal's normalizing-flow/density-model idea: a normalizing flow is
harder to train stably on a dataset this size (~750 real images) and adds
real implementation risk, while a convolutional autoencoder gives the same
"does this look like a real photo" signal with far less training
instability -- at the cost of being a cruder density estimate.
"""

import torch.nn as nn

IMAGE_SIZE = 128  # smaller than the classifier's 224 -- reconstruction doesn't need full resolution


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),    # 128 -> 64
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),   # 64 -> 32
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(inplace=True),  # 32 -> 16
            nn.Conv2d(128, 128, 4, stride=2, padding=1), nn.ReLU(inplace=True), # 16 -> 8
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid(),  # output in [0,1]
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def reconstruction_error(model, images):
    """Per-image MSE reconstruction error, shape (B,). Higher = looks less
    like the real-photo manifold the autoencoder was trained on.
    """
    recon = model(images)
    return ((recon - images) ** 2).mean(dim=(1, 2, 3))
