"""ConvAdapter — trainable input adapter.

Maps the CWT input `[in_channels, 32, 32]` to the 3-channel tensor MobileViT
expects `[3, 32, 32]`. The asymmetric (5x1) kernel enhances vertical transients
(the time axis) while preserving frequency resolution (no horizontal blur).
"""
import torch.nn as nn


class ConvAdapter(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)
