"""FrozenMobileViTBackbone — frozen feature extractor (NOT trained).

Upscales the adapter output `[3, 32, 32]` to 256x256 (nearest-neighbour, to
preserve the vertical edge gradients), runs the pretrained MobileViT-small
backbone with all parameters frozen, and mean-pools the last hidden state to a
`feature_dim`-d vector. No gradients flow into MobileViT.
"""
import torch.nn as nn
import torch.nn.functional as F
from transformers import MobileViTModel


class FrozenMobileViTBackbone(nn.Module):
    feature_dim = 640

    def __init__(self, pretrained="apple/mobilevit-small"):
        super().__init__()
        self.backbone = MobileViTModel.from_pretrained(pretrained)
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = F.interpolate(x, size=(256, 256), mode="nearest")
        outputs = self.backbone(pixel_values=x)
        return outputs.last_hidden_state.mean(dim=[2, 3])  # [B, feature_dim]
