"""A small convolutional autoencoder for 128 x 128 RGB reconstruction."""

import torch
from torch import nn


class ConvolutionalAutoencoder(nn.Module):
    """Encode RGB images into spatial features, then reconstruct the images.

    Input and output shapes are (batch_size, 3, 128, 128). The encoder reduces
    spatial resolution to 16 x 16 while increasing the feature channels to 64.
    The decoder restores resolution using bilinear upsampling and convolutions.

    The final convolution has no activation: MSE can compare unrestricted
    predictions with targets in [0, 1] without a sigmoid or clipping.
    """

    def __init__(self) -> None:
        super().__init__()

        # Stride 2 halves height and width at each encoder convolution.
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Upsampling doubles height and width; stride-1 convolutions preserve it.
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return reconstructions of a batch of 128 x 128 RGB images."""
        if images.ndim != 4 or images.shape[1:] != (3, 128, 128):
            raise ValueError("Expected input shape (batch_size, 3, 128, 128).")

        features = self.encoder(images)
        reconstructions = self.decoder(features)
        return reconstructions
