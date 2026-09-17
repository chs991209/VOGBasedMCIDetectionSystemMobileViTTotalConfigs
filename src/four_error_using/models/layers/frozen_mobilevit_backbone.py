"""FrozenMobileViTBackbone — frozen feature extractor (NOT trained).

Upscales the adapter output `[3, 32, 32]` to 256x256 (nearest-neighbour, to
preserve the vertical edge gradients), runs a pretrained MobileViT backbone with
all parameters frozen, and mean-pools the last hidden state to a `feature_dim`-d
vector. No gradients flow into the backbone.

The backbone is selectable (MobileViT v1 or the larger MobileViTv2). `feature_dim`
is inferred from a dummy forward, so the classifier head sizes itself to whatever
backbone is chosen:
    mobilevit-small   -> 640-d  (4.94M params, default)
    mobilevitv2-1.0   -> 512-d  (4.39M)
    mobilevitv2-2.0   -> 1024-d (17.4M, the big one)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

# Friendly name -> HF repo id. Full HF ids are also accepted as-is.
BACKBONES = {
    "mobilevit-small": "apple/mobilevit-small",
    "mobilevitv2-1.0": "apple/mobilevitv2-1.0-imagenet1k-256",
    "mobilevitv2-2.0": "apple/mobilevitv2-2.0-imagenet1k-256",
}


class FrozenMobileViTBackbone(nn.Module):
    def __init__(self, pretrained="mobilevit-small"):
        super().__init__()
        repo = BACKBONES.get(pretrained, pretrained)
        self.backbone = AutoModel.from_pretrained(repo)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        # Infer the pooled feature width from a dummy forward (varies by backbone).
        with torch.no_grad():
            probe = self.backbone(pixel_values=torch.zeros(1, 3, 256, 256)).last_hidden_state
        self.feature_dim = int(probe.shape[1])

    def forward(self, x):
        """[B, 3, 32, 32] -> upsample to 256 -> frozen backbone -> [B, feature_dim]."""
        x = F.interpolate(x, size=(256, 256), mode="nearest")
        outputs = self.backbone(pixel_values=x)
        return outputs.last_hidden_state.mean(dim=[2, 3])  # [B, feature_dim]
