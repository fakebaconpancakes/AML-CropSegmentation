"""A standard randomly initialized U-Net, without pretrained components."""

import torch
from torch import nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        widths = [base_channels * 2 ** i for i in range(5)]
        self.encoders = nn.ModuleList()
        channels = 3
        for width in widths[:-1]:
            self.encoders.append(double_conv(channels, width))
            channels = width
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = double_conv(widths[-2], widths[-1])
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for low, high in zip(reversed(widths[:-1]), reversed(widths[1:])):
            self.ups.append(nn.ConvTranspose2d(high, low, 2, stride=2))
            self.decoders.append(double_conv(2 * low, low))
        self.head = nn.Conv2d(base_channels, 5, 1)

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected [B, 3, H, W] images.")
        if images.shape[-2] % 16 or images.shape[-1] % 16:
            raise ValueError("U-Net spatial dimensions must be divisible by 16.")
        skips = []
        x = images
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = decoder(torch.cat((up(x), skip), dim=1))
        return self.head(x)
